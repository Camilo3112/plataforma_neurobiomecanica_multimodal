"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               SEGMENTACIÓN TOMOGRÁFICA POR UNIDADES HOUNSFIELD               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/tomography.py
Versión: v3.21.21

Descripción
-----------
Segmenta tejidos corporales sobre TAC y cuantifica áreas/volúmenes.

Fundamento físico-matemático implementado
-----------------------------------------
Segmenta tejidos en TAC usando unidades Hounsfield HU = 1000(μ-μ_agua)/μ_agua.
El tejido adiposo, músculo y hueso se aproximan por rangos de intensidad y se
limpian con morfología matemática. Las áreas se calculan como A = N_px Δx Δy y
los volúmenes como V = Σ A_z Δz, respetando PixelSpacing, SliceThickness y
orden
espacial DICOM.

Trazabilidad de resultados
--------------------------
Los datos crudos permanecen separados de los derivados. Las salidas se escriben
con nombre de paciente, etapa, módulo y espacio de referencia para facilitar
revisión, comparación longitudinal y reproducción de la corrida.
"""

###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  0. IMPORTACIONES, CONFIGURACIÓN Y FUNCIONES DEL MÓDULO
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PipelineConfig
from .paths import first_existing_dir, resolve_stage_dir, stage_key, norm_key
from .io_utils import safe_mkdir, save_dataframe, save_json, append_log, log_exception
from .checkpoint import CheckpointManager


# Configuración anatómica heredada del código original.
# Nota: en DICOM radiológico la derecha anatómica puede verse al lado izquierdo de la imagen.
MUSCLES_CONFIG = {
    "Vasto_Lateral_Der": {"roi": "left", "angles": (80, 220), "label": 1},
    "Vasto_Medial_Der": {"roi": "left", "angles": (-40, 80), "label": 2},
    "Vasto_Lateral_Izq": {"roi": "right", "angles": (-40, 100), "label": 3},
    "Vasto_Medial_Izq": {"roi": "right", "angles": (80, 220), "label": 4},
}


def _require_medical_libs():
    import pydicom  # noqa
    import nibabel as nib  # noqa
    from scipy.ndimage import center_of_mass, map_coordinates, binary_opening  # noqa
    from scipy.ndimage import label as scipy_label  # noqa
    from scipy.signal import find_peaks  # noqa
    from skimage.draw import polygon  # noqa
    from skimage.segmentation import find_boundaries  # noqa
    return pydicom, nib


def _dicom_sort_key(ds):
    """Ordena cortes por posición anatómica cuando existe, con fallback robusto."""
    if hasattr(ds, "ImagePositionPatient"):
        try:
            return float(ds.ImagePositionPatient[2])
        except Exception:
            pass
    if hasattr(ds, "SliceLocation"):
        try:
            return float(ds.SliceLocation)
        except Exception:
            pass
    if hasattr(ds, "InstanceNumber"):
        try:
            return int(ds.InstanceNumber)
        except Exception:
            pass
    return 0


def _safe_float(value, default: float = 1.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_str(value) -> str:
    try:
        return str(value)
    except Exception:
        return ""


def _series_uid_for(ds, file: Path) -> str:
    """UID estable. Si el DICOM no trae SeriesInstanceUID, usa la carpeta."""
    uid = _safe_str(getattr(ds, "SeriesInstanceUID", "")).strip()
    if uid:
        return uid
    return f"NO_UID::{file.parent}"


def _shape_of(ds) -> tuple[int, int] | None:
    try:
        return int(ds.Rows), int(ds.Columns)
    except Exception:
        return None


def _is_probably_localizer(ds) -> bool:
    text = " ".join([
        _safe_str(getattr(ds, "SeriesDescription", "")),
        _safe_str(getattr(ds, "ProtocolName", "")),
        _safe_str(getattr(ds, "ImageType", "")),
    ]).lower()
    bad_terms = (
        "scout", "localizer", "topogram", "surview", "locator", "survey",
        "mip", "mpr", "report", "dose", "screen", "capture", "secondary",
    )
    return any(t in text for t in bad_terms)


def _collect_dicom_slices(directory: Path):
    """Lee DICOMs con PixelData y los agrupa por serie.

    El error original venía de hacer np.stack() con todos los DICOM del TAC
    mezclados. Una carpeta TAC puede traer scout/localizadores/reconstrucciones
    con tamaños diferentes. Por eso primero agrupamos por SeriesInstanceUID y
    luego escogemos una sola serie axial consistente.
    """
    pydicom, _ = _require_medical_libs()
    groups: dict[str, list] = {}
    read_errors = []

    for file in Path(directory).rglob("*"):
        if not file.is_file() or file.name.upper() == "DICOMDIR":
            continue
        try:
            ds = pydicom.dcmread(str(file), force=True)
            if not hasattr(ds, "PixelData"):
                continue
            shape = _shape_of(ds)
            if shape is None:
                continue
            uid = _series_uid_for(ds, file)
            ds._source_file = str(file)  # ayuda para diagnosticar sin exponer píxeles
            groups.setdefault(uid, []).append(ds)
        except Exception as exc:
            read_errors.append({"archivo": str(file), "error": str(exc)})
    return groups, read_errors


def _candidate_rows(groups: dict[str, list]) -> list[dict]:
    rows = []
    for uid, items in groups.items():
        if not items:
            continue
        # Subgrupo por tamaño de matriz; una misma serie puede traer algún frame derivado raro.
        by_shape: dict[tuple[int, int], list] = {}
        for ds in items:
            shape = _shape_of(ds)
            if shape is not None:
                by_shape.setdefault(shape, []).append(ds)
        if not by_shape:
            continue
        shape, same_shape = max(by_shape.items(), key=lambda kv: len(kv[1]))
        first = same_shape[0]
        modality = _safe_str(getattr(first, "Modality", "")).upper()
        desc = _safe_str(getattr(first, "SeriesDescription", ""))
        protocol = _safe_str(getattr(first, "ProtocolName", ""))
        n = len(same_shape)
        rows_, cols_ = shape
        score = 0
        score += n * 1000
        score += (rows_ * cols_) / 1000.0
        if modality == "CT":
            score += 100000
        if _is_probably_localizer(first):
            score -= 50000
        # Preferir series volumétricas, no capturas sueltas.
        if n < 5:
            score -= 25000
        rows.append({
            "series_uid": uid,
            "n_total_en_uid": len(items),
            "n_misma_matriz": n,
            "rows": rows_,
            "columns": cols_,
            "modalidad": modality,
            "descripcion_serie": desc,
            "protocolo": protocol,
            "localizer_probable": bool(_is_probably_localizer(first)),
            "score": float(score),
            "items": same_shape,
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def _select_best_ct_series(groups: dict[str, list]):
    candidates = _candidate_rows(groups)
    if not candidates:
        raise ValueError("Encontré DICOMs, pero ninguna serie tenía matriz de imagen válida.")
    # Si hay CT real, seleccionar el mejor CT; si no, usar el mejor candidato disponible.
    ct_candidates = [c for c in candidates if c["modalidad"] == "CT"]
    selected = ct_candidates[0] if ct_candidates else candidates[0]
    return selected, candidates


def _deduplicate_and_sort_slices(slices: list) -> list:
    """Ordena y elimina duplicados exactos de SOPInstanceUID/posición.

    No mezcla series. Solo limpia repeticiones dentro de la serie escogida.
    """
    unique = {}
    for ds in slices:
        sop = _safe_str(getattr(ds, "SOPInstanceUID", "")).strip()
        if sop:
            key = ("sop", sop)
        else:
            key = ("pos", round(_dicom_sort_key(ds), 4), _safe_str(getattr(ds, "InstanceNumber", "")))
        unique.setdefault(key, ds)
    out = list(unique.values())
    out.sort(key=_dicom_sort_key)
    return out


def _get_pixel_spacing(ds) -> list[float]:
    try:
        ps = getattr(ds, "PixelSpacing", [1.0, 1.0])
        return [float(ps[0]), float(ps[1])]
    except Exception:
        return [1.0, 1.0]


def _estimate_z_spacing(slices: list) -> float:
    vals = []
    for ds in slices:
        if hasattr(ds, "ImagePositionPatient"):
            try:
                vals.append(float(ds.ImagePositionPatient[2]))
                continue
            except Exception:
                pass
        if hasattr(ds, "SliceLocation"):
            try:
                vals.append(float(ds.SliceLocation))
            except Exception:
                pass
    if len(vals) >= 2:
        diffs = np.diff(np.sort(np.array(vals, dtype=float)))
        diffs = np.abs(diffs[diffs != 0])
        if diffs.size:
            return float(np.median(diffs))
    for attr in ("SpacingBetweenSlices", "SliceThickness"):
        val = _safe_float(getattr(slices[0], attr, 0.0), 0.0)
        if val > 0:
            return float(val)
    return 1.0


def load_dicom_series(directory: Path) -> tuple[np.ndarray, list[float], float, dict]:
    groups, read_errors = _collect_dicom_slices(directory)
    if not groups:
        raise ValueError(f"No encontré DICOMs válidos con PixelData en {directory}")

    selected, candidates = _select_best_ct_series(groups)
    slices = _deduplicate_and_sort_slices(selected["items"])
    if not slices:
        raise ValueError(f"La serie TAC seleccionada no tiene cortes útiles en {directory}")

    # Segunda defensa: si algún pixel_array falla o tiene otra matriz, se descarta.
    expected_shape = _shape_of(slices[0])
    image_slices = []
    kept = []
    dropped = []
    for ds in slices:
        try:
            arr2d = ds.pixel_array.astype(np.float32)
            if expected_shape is not None and arr2d.shape != expected_shape:
                dropped.append({
                    "archivo": _safe_str(getattr(ds, "_source_file", "")),
                    "shape": str(arr2d.shape),
                    "motivo": "matriz_distinta",
                })
                continue
            slope = _safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
            intercept = _safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
            image_slices.append(arr2d * slope + intercept)
            kept.append(ds)
        except Exception as exc:
            dropped.append({
                "archivo": _safe_str(getattr(ds, "_source_file", "")),
                "motivo": f"pixel_array_error: {exc}",
            })

    if len(image_slices) == 0:
        raise ValueError(f"No pude leer los píxeles de la serie TAC seleccionada en {directory}")

    image_hu = np.stack(image_slices, axis=-1)
    pixel_spacing = _get_pixel_spacing(kept[0])
    z_spacing = _estimate_z_spacing(kept)

    # Metadata resumida. No guardamos todos los paths largos, solo una tabla compacta.
    candidate_meta = []
    for c in candidates[:20]:
        candidate_meta.append({k: v for k, v in c.items() if k != "items"})

    meta = {
        "n_slices": len(kept),
        "n_dropped_selected_series": len(dropped),
        "pixel_spacing": pixel_spacing,
        "slice_spacing": float(z_spacing),
        "shape": tuple(int(x) for x in image_hu.shape),
        "selected_series_uid": selected["series_uid"],
        "selected_series_description": selected["descripcion_serie"],
        "selected_protocol": selected["protocolo"],
        "selected_modality": selected["modalidad"],
        "selected_rows": selected["rows"],
        "selected_columns": selected["columns"],
        "series_description": selected["descripcion_serie"],
        "n_series_detectadas": len(candidates),
        "n_read_errors": len(read_errors),
        "series_candidates": candidate_meta,
        "dropped_from_selected_series": dropped[:50],
        "read_errors": read_errors[:50],
    }
    return image_hu, pixel_spacing, float(z_spacing), meta

def auto_detect_z_limits(image_hu: np.ndarray, cfg: PipelineConfig) -> tuple[int, int]:
    from scipy.ndimage import binary_opening
    from scipy.ndimage import label as scipy_label

    total_slices = image_hu.shape[2]
    h, w = image_hu.shape[:2]
    bone_areas = np.zeros(total_slices)
    bone_pieces = np.zeros(total_slices)
    roi_mask = np.zeros((h, w), dtype=bool)
    roi_mask[:, : w // 2] = True

    for z in range(total_slices):
        bone_slice = (image_hu[:, :, z] > cfg.ct_hu_bone_min) & roi_mask
        bone_slice = binary_opening(bone_slice, iterations=2)
        _, num_features = scipy_label(bone_slice)
        bone_pieces[z] = num_features
        bone_areas[z] = np.sum(bone_slice)

    valid_z = np.where(bone_pieces >= 1)[0]
    if len(valid_z) == 0:
        return 0, total_slices
    mid_z = int(valid_z[len(valid_z) // 2])
    base_area = bone_areas[mid_z] if bone_areas[mid_z] > 0 else np.nanmedian(bone_areas[valid_z])
    step_up = 1 if bone_areas[-1] > bone_areas[0] else -1

    z_end = total_slices - 1 if step_up == 1 else 0
    search_knee = range(mid_z, -1, -1) if step_up == 1 else range(mid_z, total_slices)
    for z in search_knee:
        if bone_pieces[z] > 1 or bone_areas[z] > (base_area * 1.8):
            z_end = z + (10 * step_up)
            break

    z_start = 0 if step_up == 1 else total_slices - 1
    search_hip = range(mid_z, total_slices) if step_up == 1 else range(mid_z, -1, -1)
    for z in search_hip:
        if bone_areas[z] > (base_area * 2.2) or bone_pieces[z] > 2:
            z_start = z - (15 * step_up)
            break

    z0, z1 = min(z_start, z_end), max(z_start, z_end)
    return max(0, int(z0)), min(total_slices, int(z1))


def _cwt_like_power_1d(signal_1d: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Aproximación rápida a potencia CWT con Mexican-hat/LoG 1D.

    Evita depender de scipy.signal.cwt/ricker, que cambia según la versión de SciPy.
    """
    from scipy.ndimage import gaussian_laplace
    sig = np.asarray(signal_1d, dtype=float)
    coeffs = []
    for s in np.maximum(scales, 1):
        coeffs.append(np.abs(gaussian_laplace(sig, sigma=float(s))) ** 2)
    return np.nanmax(np.vstack(coeffs), axis=0) if coeffs else np.zeros_like(sig)


def find_fascia_1d(signal_1d: np.ndarray, mm_per_pixel: float, cfg: PipelineConfig) -> int:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    blind_idx = int(cfg.ct_blind_zone_mm / max(mm_per_pixel, 1e-6))
    max_idx = int(cfg.ct_max_muscle_thickness_mm / max(mm_per_pixel, 1e-6))
    max_idx = min(max_idx, len(signal_1d) - 1)
    if max_idx <= blind_idx + 2:
        return max_idx
    scales = np.linspace(cfg.ct_cwt_min_scale, cfg.ct_cwt_max_scale, cfg.ct_cwt_n_scales)
    power = _cwt_like_power_1d(signal_1d, scales)
    power = gaussian_filter1d(power, sigma=1.0)
    prominence_thr = float(np.nanmean(power) * 1.5)
    peaks, _ = find_peaks(power, prominence=prominence_thr)
    valid = [int(p) for p in peaks if blind_idx < p < max_idx]
    return valid[0] if valid else max_idx


def get_femur_centroid(image_slice_hu: np.ndarray, roi_side: str, cfg: PipelineConfig):
    from scipy.ndimage import center_of_mass
    from scipy.ndimage import label as scipy_label

    h, w = image_slice_hu.shape
    roi = np.zeros_like(image_slice_hu, dtype=bool)
    if roi_side == "left":
        roi[:, : w // 2] = True
    else:
        roi[:, w // 2 :] = True
    bone_mask = (image_slice_hu > cfg.ct_hu_bone_min) & roi
    lbl, num = scipy_label(bone_mask)
    if num == 0:
        return None
    sizes = [np.sum(lbl == i) for i in range(1, num + 1)]
    return center_of_mass(lbl == int(np.argmax(sizes)) + 1)


def process_multi_muscle_slice(image_slice_hu: np.ndarray, pixel_spacing: list[float], cfg: PipelineConfig) -> dict[str, np.ndarray]:
    from scipy.ndimage import map_coordinates
    from skimage.draw import polygon

    h, w = image_slice_hu.shape
    slice_masks = {name: np.zeros((h, w), dtype=bool) for name in MUSCLES_CONFIG}
    mean_spacing = float((pixel_spacing[0] + pixel_spacing[1]) / 2.0)
    search_radius_px = int((cfg.ct_max_muscle_thickness_mm + 20) / max(mean_spacing, 1e-6))
    radii = np.arange(0, max(search_radius_px, 2))
    centroids = {
        "left": get_femur_centroid(image_slice_hu, "left", cfg),
        "right": get_femur_centroid(image_slice_hu, "right", cfg),
    }

    for muscle_name, muscle_cfg in MUSCLES_CONFIG.items():
        cy_cx = centroids[muscle_cfg["roi"]]
        if cy_cx is None:
            continue
        cy, cx = cy_cx
        angles_rad = np.deg2rad(np.linspace(muscle_cfg["angles"][0], muscle_cfg["angles"][1], cfg.ct_num_rays))
        br, bc = [], []
        for angle in angles_rad:
            y_coords = cy - radii * np.sin(angle)
            x_coords = cx + radii * np.cos(angle)
            ray_signal = map_coordinates(image_slice_hu, np.vstack((y_coords, x_coords)), order=1, mode="nearest")
            peak_idx = find_fascia_1d(ray_signal, mean_spacing, cfg)
            peak_idx = min(peak_idx, len(y_coords) - 1)
            br.append(y_coords[peak_idx])
            bc.append(x_coords[peak_idx])
        br.append(cy)
        bc.append(cx)
        rr, cc = polygon(br, bc, shape=(h, w))
        mask = np.zeros((h, w), dtype=bool)
        mask[rr, cc] = True
        slice_masks[muscle_name] = mask & (image_slice_hu > cfg.ct_hu_muscle_min) & (image_slice_hu < cfg.ct_hu_muscle_max)
    return slice_masks


def segment_all_muscles(image_hu: np.ndarray, pixel_spacing: list[float], z_start: int, z_end: int, cfg: PipelineConfig) -> dict[str, np.ndarray]:
    masks = {name: np.zeros_like(image_hu, dtype=bool) for name in MUSCLES_CONFIG}
    for z in range(z_start, z_end):
        slice_masks = process_multi_muscle_slice(image_hu[:, :, z], pixel_spacing, cfg)
        for name in MUSCLES_CONFIG:
            masks[name][:, :, z] = slice_masks[name]
    return masks


def export_tomography_results(
    image_hu: np.ndarray,
    masks: dict[str, np.ndarray],
    pixel_spacing: list[float],
    slice_thickness: float,
    out_dir: Path,
    z_start: int,
    z_end: int,
) -> pd.DataFrame:
    _, nib = _require_medical_libs()
    from skimage.segmentation import find_boundaries

    safe_mkdir(out_dir)
    voxel_vol_cm3 = (pixel_spacing[0] * pixel_spacing[1] * slice_thickness) / 1000.0
    pixel_area_cm2 = (pixel_spacing[0] * pixel_spacing[1]) / 100.0
    affine = np.diag([pixel_spacing[0], pixel_spacing[1], slice_thickness, 1.0])

    label_volume = np.zeros(image_hu.shape, dtype=np.uint8)
    tracer_volume = np.array(image_hu, dtype=np.float32, copy=True)
    rows = []
    areas_rows = []
    for name, mask in masks.items():
        label = MUSCLES_CONFIG[name]["label"]
        label_volume[mask] = label
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(out_dir / f"mask_{name}.nii.gz"))
        volume_cm3 = float(np.sum(mask) * voxel_vol_cm3)
        areas = np.array([np.sum(mask[:, :, z]) * pixel_area_cm2 for z in range(mask.shape[2])], dtype=float)
        max_area = float(np.nanmax(areas)) if len(areas) else 0.0
        max_z = int(np.nanargmax(areas)) if len(areas) else 0
        mean_area = float(np.nanmean(areas[z_start:z_end])) if z_end > z_start else float(np.nanmean(areas))
        rows.append({
            "musculo": name,
            "volumen_cm3": volume_cm3,
            "area_transversal_max_cm2": max_area,
            "z_area_max": max_z,
            "area_transversal_media_cm2": mean_area,
            "z_start": z_start,
            "z_end": z_end,
        })
        for z, area in enumerate(areas):
            areas_rows.append({"musculo": name, "z": z, "area_cm2": float(area)})
        for z in range(z_start, z_end):
            if np.any(mask[:, :, z]):
                boundaries = find_boundaries(mask[:, :, z], mode="inner")
                tracer_volume[:, :, z][boundaries] = 1500

    # Volúmenes NIfTI principales para análisis externo.
    nib.save(nib.Nifti1Image(image_hu.astype(np.float32), affine), str(out_dir / "tac_hu_original.nii.gz"))
    nib.save(nib.Nifti1Image(label_volume, affine), str(out_dir / "masks_musculos_labelmap.nii.gz"))
    nib.save(nib.Nifti1Image(tracer_volume.astype(np.float32), affine), str(out_dir / "tac_tracer_musculos.nii.gz"))

    # Máscaras combinadas por lado y globales. Sirven para medir muslo derecho/izquierdo
    # completo además de cada vasto por separado.
    mask_vastos_derecha = np.isin(label_volume, [
        MUSCLES_CONFIG["Vasto_Lateral_Der"]["label"],
        MUSCLES_CONFIG["Vasto_Medial_Der"]["label"],
    ])
    mask_vastos_izquierda = np.isin(label_volume, [
        MUSCLES_CONFIG["Vasto_Lateral_Izq"]["label"],
        MUSCLES_CONFIG["Vasto_Medial_Izq"]["label"],
    ])
    mask_todos = label_volume > 0
    nib.save(nib.Nifti1Image(mask_vastos_derecha.astype(np.uint8), affine), str(out_dir / "mask_vastos_derecha.nii.gz"))
    nib.save(nib.Nifti1Image(mask_vastos_izquierda.astype(np.uint8), affine), str(out_dir / "mask_vastos_izquierda.nii.gz"))
    nib.save(nib.Nifti1Image(mask_todos.astype(np.uint8), affine), str(out_dir / "mask_todos_vastos.nii.gz"))

    df = pd.DataFrame(rows)
    if not df.empty:
        df["lado"] = df["musculo"].apply(lambda x: "derecha" if str(x).endswith("_Der") else ("izquierda" if str(x).endswith("_Izq") else "desconocido"))
    areas_df = pd.DataFrame(areas_rows)
    if not areas_df.empty:
        areas_df["lado"] = areas_df["musculo"].apply(lambda x: "derecha" if str(x).endswith("_Der") else ("izquierda" if str(x).endswith("_Izq") else "desconocido"))

    save_dataframe(df, out_dir / "volumenes_y_areas.csv")
    save_dataframe(areas_df, out_dir / "areas_por_corte.csv")

    # Resumen por lado para los muslos: suma de vasto lateral + vasto medial.
    side_rows = []
    for lado, side_mask in [("derecha", mask_vastos_derecha), ("izquierda", mask_vastos_izquierda), ("ambos", mask_todos)]:
        areas_side = np.array([np.sum(side_mask[:, :, z]) * pixel_area_cm2 for z in range(side_mask.shape[2])], dtype=float)
        side_rows.append({
            "lado": lado,
            "volumen_cm3": float(np.sum(side_mask) * voxel_vol_cm3),
            "area_transversal_max_cm2": float(np.nanmax(areas_side)) if len(areas_side) else 0.0,
            "z_area_max": int(np.nanargmax(areas_side)) if len(areas_side) else 0,
            "area_transversal_media_cm2": float(np.nanmean(areas_side[z_start:z_end])) if z_end > z_start else float(np.nanmean(areas_side)),
            "z_start": z_start,
            "z_end": z_end,
            "nota": "Suma de los vastos segmentados del muslo; no incluye todos los músculos del muslo si no fueron segmentados.",
        })
    save_dataframe(pd.DataFrame(side_rows), out_dir / "resumen_muslos_por_lado.csv")

    # Vista rápida de segmentación en el corte con mayor área global.
    total_area_by_z = label_volume.reshape(-1, label_volume.shape[2]).sum(axis=0)
    z_preview = int(np.argmax(total_area_by_z)) if np.any(total_area_by_z) else int((z_start + z_end) / 2)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image_hu[:, :, z_preview], cmap="gray")
    ax.imshow(np.ma.masked_where(label_volume[:, :, z_preview] == 0, label_volume[:, :, z_preview]), alpha=0.45, cmap="viridis")
    ax.set_title(f"Segmentación muscular TAC · Z={z_preview}")
    ax.axis("off")
    fig.savefig(out_dir / "preview_segmentacion_tac.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return df


def process_tomography_folder(cfg: PipelineConfig, input_dir: Path, out_dir: Path) -> pd.DataFrame:
    image_hu, pixel_spacing, z_spacing, meta = load_dicom_series(input_dir)
    z0, z1 = auto_detect_z_limits(image_hu, cfg)
    masks = segment_all_muscles(image_hu, pixel_spacing, z0, z1, cfg)
    df = export_tomography_results(image_hu, masks, pixel_spacing, z_spacing, out_dir, z0, z1)

    # Diagnóstico completo para saber qué serie TAC se escogió y cuáles se ignoraron.
    # Esto es clave cuando la carpeta TAC trae varios DICOM con matrices distintas.
    meta_full = meta | {"z_start": z0, "z_end": z1}
    save_json(meta_full, out_dir / "metadata_tac_detallada.json")

    meta_compacta = {k: v for k, v in meta_full.items() if k not in {
        "series_candidates", "dropped_from_selected_series", "read_errors"
    }}
    save_dataframe(pd.DataFrame([meta_compacta]), out_dir / "metadata_tac.csv")

    candidates = meta_full.get("series_candidates", [])
    if candidates:
        save_dataframe(pd.DataFrame(candidates), out_dir / "series_tac_detectadas.csv")
    dropped = meta_full.get("dropped_from_selected_series", [])
    if dropped:
        save_dataframe(pd.DataFrame(dropped), out_dir / "dicoms_descartados_serie_tac.csv")
    return df



def _control_requested_stages(cfg: PipelineConfig) -> tuple[str, ...]:
    """Etapas a procesar para el sano.

    Si el sano solo tiene Antes, no repetimos el mismo procesamiento como Despues.
    Si existe sano/Despues, lo incluimos automáticamente.
    """
    control_dir = cfg.data_root() / cfg.control_name
    out = []
    if resolve_stage_dir(control_dir, "Antes") is not None:
        out.append("Antes")
    if resolve_stage_dir(control_dir, "Despues") is not None:
        out.append("Despues")
    if not out and control_dir.exists():
        out.append("Antes")
    return tuple(dict.fromkeys(out))


def _iter_tomography_subject_stage(cfg: PipelineConfig):
    """Itera pacientes y control con resolución tolerante de carpetas.

    Para pacientes se respeta Antes/Despues. Para sano se procesa Antes y, si existe,
    Despues. Esto evita dejar vacía la carpeta de control y permite correlaciones.
    """
    for patient in cfg.patients:
        for stage in cfg.stages:
            stage_dir = resolve_stage_dir(cfg.data_root() / patient, stage)
            yield patient, stage, stage_dir
    for stage in _control_requested_stages(cfg):
        control_dir = cfg.data_root() / cfg.control_name
        stage_dir = resolve_stage_dir(control_dir, stage, fallback_to_subject=True)
        yield cfg.control_name, stage, stage_dir

def run_tomography(cfg: PipelineConfig) -> pd.DataFrame:
    all_rows = []
    for patient, stage, stage_dir in _iter_tomography_subject_stage(cfg):
        print(f"\n[TOMOGRAFÍA] {patient} · {stage}")
        out_dir = cfg.results_root() / patient / stage / "tomografia"
        log_file = cfg.results_root() / patient / stage / "reportes" / "tomografia_log.txt"
        try:
            tac_dir = first_existing_dir(stage_dir, cfg.ct_dirnames) if stage_dir else None
            if tac_dir is None:
                append_log(log_file, f"No encontré carpeta TAC para {patient} {stage}; stage_dir={stage_dir}")
                continue
            metrics_csv = out_dir / "volumenes_y_areas.csv"
            task_id = f"tomografia/{patient}/{stage}"
            ckpt = CheckpointManager(cfg)

            def _work():
                return process_tomography_folder(cfg, tac_dir, out_dir)

            df_result, status = ckpt.run(
                task_id=task_id,
                inputs=[tac_dir],
                outputs=[metrics_csv, out_dir / "masks_musculos_labelmap.nii.gz", out_dir / "tac_tracer_musculos.nii.gz"],
                params={
                    "ct_blind_zone_mm": cfg.ct_blind_zone_mm,
                    "ct_max_muscle_thickness_mm": cfg.ct_max_muscle_thickness_mm,
                    "ct_num_rays": cfg.ct_num_rays,
                    "ct_hu_muscle_min": cfg.ct_hu_muscle_min,
                    "ct_hu_muscle_max": cfg.ct_hu_muscle_max,
                    "ct_hu_bone_min": cfg.ct_hu_bone_min,
                    "selector": "v3_4_gpu_control_incluido_y_series_ct_consistentes",
                },
                fn=_work,
            )
            df = pd.read_csv(metrics_csv) if status == "skipped" and metrics_csv.exists() else df_result
            if df is not None and not df.empty:
                df.insert(0, "checkpoint_status", status)
                df.insert(0, "stage", stage)
                df.insert(0, "patient", patient)
                all_rows.append(df)
        except Exception as exc:
            log_exception(log_file, f"Tomografía {patient} {stage}", exc)
    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not out.empty:
        save_dataframe(out, cfg.results_root() / "resumen_global_tomografia.csv")
    return out
