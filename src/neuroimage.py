"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NEUROIMAGEN NIFTI, AFINES Y REMUESTREO                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/neuroimage.py
Versión: v3.21.21

Descripción
-----------
Gestiona volúmenes, afines, remuestreo y operaciones de neuroimagen.

Fundamento físico-matemático implementado
-----------------------------------------
Opera sobre volúmenes NIfTI como campos escalares discretos I(i,j,k) asociados
a
un sistema físico RAS-mm mediante matriz afín A. Implementa remuestreo,
lectura de
geometría, máscaras, cuantificación volumétrica y comparación espacial. Las
transformaciones se expresan como x' = T x; en etiquetas se usa interpolación
de
vecino más cercano y en intensidades interpolación continua. Las métricas de
cambio se formulan como ΔI = I_post - I_pre y cambios relativos normalizados.

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

import math
import re
from pathlib import Path
from typing import Optional, Iterable, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from .config import PipelineConfig
from .paths import first_existing_dir, resolve_stage_dir, norm_key, stage_key
from .io_utils import safe_mkdir, save_dataframe, append_log, log_exception, save_json
from .checkpoint import CheckpointManager, load_json_if_exists, save_metrics_json
from .gpu import (
    robust_zscore_gpu,
    minmax_gpu,
    resize_to_shape_gpu,
    wavelet_energy_gpu,
    pearson_gpu,
    gpu_info,
)


def _require_neuro_libs():
    import pydicom  # noqa
    import nibabel as nib  # noqa
    from scipy import ndimage  # noqa
    return pydicom, nib, ndimage


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def find_nifti_files(folder: Path) -> list[Path]:
    if not folder or not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.is_file() and is_nifti(p)])


def _to_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or str(value) == "":
            return default
        return float(value)
    except Exception:
        return default


def _list_float(value: Any, default: list[float] | None = None) -> list[float]:
    if default is None:
        default = [1.0, 1.0]
    try:
        if value is None or value == "":
            return default
        return [float(x) for x in value]
    except Exception:
        return default


def _dicom_sort_key(ds):
    """Ordena por posición real cuando existe; si no, por InstanceNumber."""
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


def _dicom_file_path(ds) -> str:
    return str(getattr(ds, "filename", ""))


def _series_text(ds) -> str:
    return " ".join([
        str(getattr(ds, "SeriesDescription", "")),
        str(getattr(ds, "ProtocolName", "")),
    ])


def side_from_text(text: str) -> str:
    """Detecta lateralidad con tolerancia a nombres reales de tus mapas.

    Ejemplos cubiertos:
    - rMotor_izquierda_Rodilla_fwe005.nii
    - rMotor_rod_der_fwe005.nii
    - rMap_MotRodillaDer_fwe005.nii
    - rMap_MotRodillaIzq_fwe005.nii
    - rMOTOR_DERECHA_FWE001.nii
    - rmap_motorIzq_fwe005.nii
    """
    n = norm_key(text)
    compact = re.sub(r"[^a-z0-9]+", "", n)

    derecha_tokens = [
        "derecha", "derecho", "dcha", "right",
        "_der", "-der", " der", "der_", "der-",
        "motorderecha", "motorrodillader", "motrodillader", "rodillader", "motorder",
    ]
    izquierda_tokens = [
        "izquierda", "izquierdo", "izqda", "izq", "left",
        "_izq", "-izq", " izq", "izq_", "izq-",
        "motorizquierda", "motorrodillaizq", "motrodillaizq", "rodillaizq", "motorizq",
    ]

    if any(tok in n for tok in derecha_tokens) or any(tok in compact for tok in derecha_tokens):
        return "derecha"
    if any(tok in n for tok in izquierda_tokens) or any(tok in compact for tok in izquierda_tokens):
        return "izquierda"

    # Sufijos pegados tipo RodillaDer / MotorIzq, sin separadores.
    if compact.endswith(("derecha", "derecho", "right", "dcha", "der")):
        return "derecha"
    if compact.endswith(("izquierda", "izquierdo", "left", "izqda", "izq")):
        return "izquierda"
    return "desconocido"


def side_from_name(path: Path) -> str:
    return side_from_text(path.name)


def classify_series(desc: str, protocol: str) -> dict[str, str]:
    """Clasifica automáticamente las series detectadas en el manifest del resonador.

    Basado en los nombres vistos en tus datos: T1, T2, ep2d_pace_MOTOR derecha/izquierda,
    t-Maps, DTI/difusión, FA/ADC/RD/AD/TRACEW, tractografía y tracto corticoespinal.
    """
    raw = f"{desc} {protocol}"
    t = norm_key(raw)
    side = side_from_text(raw)
    role = "otros"
    subtype = "otros"

    if "tracto corticoespinal" in t or ("cortico" in t and "espinal" in t):
        role, subtype = "vce_explicita", "tracto_corticoespinal"
    elif "tractograf" in t:
        role, subtype = "tractografia", "tractografia"
    elif "difusion" in t or "diffusion" in t or "tensor" in t or "tracew" in t or "colfa" in t:
        role = "dti_difusion"
        if "colfa" in t:
            subtype = "colfa"
        elif re.search(r"\bfa\b", t) or "_fa" in t or "fa_map" in t:
            subtype = "fa"
        elif "adc" in t:
            subtype = "adc"
        elif re.search(r"\bad\b", t) or "ad_map" in t:
            subtype = "ad"
        elif re.search(r"\brd\b", t) or "rd_map" in t:
            subtype = "rd"
        elif "tracew" in t or "bvalue" in t or "b0" in t:
            subtype = "trace_b0"
        elif "tensor" in t:
            subtype = "tensor"
        else:
            subtype = "dwi"
    elif "motor" in t and ("fwe" in t or "map" in t or "mapa" in t or "rodilla" in t):
        # Mapas NIfTI exportados desde ResultadosFuncional/MAPAS, con nombres heterogéneos.
        role, subtype = "fmri_motor", "mapa_activacion"
    elif "ep2d" in t and "motor" in t:
        role = "fmri_motor"
        if "intermediate" in t or "t-map" in t or "tmap" in t or "mean_&_t" in t or "evaseries" in t:
            subtype = "mapa_activacion"
        elif "moco" in t:
            subtype = "moco"
        elif "design" in t:
            subtype = "design"
        elif "startfmri" in t:
            subtype = "startfmri"
        else:
            subtype = "bold_raw"
    elif "resting" in t:
        role, subtype = "resting_state", "resting"
    elif "t1 spc" in t or "mpr" in t:
        role, subtype = "anatomica", "t1"
    elif "t2 tse" in t:
        role, subtype = "anatomica", "t2_tse"
    elif "t2 tirm" in t or "flair" in t:
        role, subtype = "anatomica", "t2_tirm_flair"
    elif "swi" in t or "pha_images" in t or "mag_images" in t or "mip_images(sw)" in t:
        role, subtype = "swi", "swi"
    elif "tof" in t or "vessels" in t:
        role, subtype = "angio_tof", "tof_vessels"
    elif "field_mapping" in t or "fieldmap" in t:
        role, subtype = "fieldmap", "fieldmap"
    elif "postproceso" in t:
        role, subtype = "postproceso", "postproceso"
    elif "scout" in t:
        role, subtype = "scout", "scout"

    return {"series_role": role, "series_subtype": subtype, "series_side": side}


def sanitize_name(text: str, max_len: int = 80) -> str:
    name = re.sub(r"[^A-Za-z0-9_áéíóúÁÉÍÓÚñÑ-]+", "_", str(text)).strip("_")
    name = name.replace("__", "_")
    return (name[:max_len] or "serie")


def collect_dicom_series(directory: Path) -> dict[str, list]:
    pydicom, _, _ = _require_neuro_libs()
    groups: dict[str, list] = {}
    for f in Path(directory).rglob("*"):
        if not f.is_file() or f.name.upper() in {"DICOMDIR", "LOCKFILE", "VERSION"}:
            continue
        try:
            ds = pydicom.dcmread(str(f), force=True)
            if not hasattr(ds, "PixelData"):
                continue
            uid = str(getattr(ds, "SeriesInstanceUID", "serie_sin_uid"))
            # Guardar filename para reportes y trazabilidad.
            try:
                ds.filename = str(f)
            except Exception:
                pass
            groups.setdefault(uid, []).append(ds)
        except Exception:
            pass
    return groups


def dicom_piece_to_array(ds) -> np.ndarray:
    """Convierte pixel_array DICOM a bloque 3D con eje de cortes/frames al final.

    Maneja imágenes RGB postprocesadas de tractografía/tracto corticoespinal convirtiéndolas
    a escala de grises y maneja multiframe como volumen 3D.
    """
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3:
        # RGB/RGBA: filas x columnas x canales.
        if arr.shape[-1] in (3, 4):
            return arr[..., :3].mean(axis=-1)[:, :, None]
        # canales x filas x columnas.
        if arr.shape[0] in (3, 4):
            return arr[:3, ...].mean(axis=0)[:, :, None]
        # multiframe: frames x filas x columnas -> filas x columnas x frames.
        return np.moveaxis(arr, 0, -1)
    if arr.ndim == 4:
        # multiframe RGB: frames x rows x cols x canales.
        if arr.shape[-1] in (3, 4):
            gray = arr[..., :3].mean(axis=-1)
            return np.moveaxis(gray, 0, -1)
        return np.squeeze(arr)
    return np.squeeze(arr)


def dicom_series_to_volume(ds_list: list) -> tuple[np.ndarray, np.ndarray, dict]:
    if not ds_list:
        raise ValueError("Serie DICOM vacía")
    ds_list = sorted(ds_list, key=_dicom_sort_key)
    first = ds_list[0]
    pieces = []
    skipped = 0
    ref_shape = None
    for ds in ds_list:
        try:
            piece = dicom_piece_to_array(ds)
            if piece.ndim == 2:
                piece = piece[:, :, None]
            if piece.ndim != 3:
                skipped += 1
                continue
            if ref_shape is None:
                ref_shape = piece.shape[:2]
            if piece.shape[:2] != ref_shape:
                skipped += 1
                continue
            pieces.append(piece)
        except Exception:
            skipped += 1
    if not pieces:
        raise ValueError("No pude extraer PixelData válido en la serie")
    vol = np.concatenate(pieces, axis=-1).astype(np.float32)
    slope = _to_float(getattr(first, "RescaleSlope", 1.0), 1.0)
    intercept = _to_float(getattr(first, "RescaleIntercept", 0.0), 0.0)
    vol = vol * slope + intercept

    pixel_spacing = _list_float(getattr(first, "PixelSpacing", None), [1.0, 1.0])
    slice_thick = _to_float(getattr(first, "SliceThickness", 1.0), 1.0)
    positions = []
    for ds in ds_list:
        if hasattr(ds, "ImagePositionPatient"):
            try:
                positions.append(float(ds.ImagePositionPatient[2]))
            except Exception:
                pass
    if len(set(np.round(positions, 4))) > 1:
        z_spacing = float(abs(max(positions) - min(positions)) / max(len(set(np.round(positions, 4))) - 1, 1))
    else:
        z_spacing = slice_thick if slice_thick and not math.isnan(slice_thick) else 1.0
    if z_spacing == 0 or math.isnan(z_spacing):
        z_spacing = 1.0
    affine = np.diag([pixel_spacing[0], pixel_spacing[1], z_spacing, 1.0]).astype(float)

    desc = str(getattr(first, "SeriesDescription", ""))
    protocol = str(getattr(first, "ProtocolName", ""))
    cls = classify_series(desc, protocol)
    meta = {
        "series_uid": str(getattr(first, "SeriesInstanceUID", "")),
        "series_description": desc,
        "protocol": protocol,
        "modality": str(getattr(first, "Modality", "")),
        "manufacturer": str(getattr(first, "Manufacturer", "")),
        "model": str(getattr(first, "ManufacturerModelName", "")),
        "magnetic_field": str(getattr(first, "MagneticFieldStrength", "")),
        "tr": str(getattr(first, "RepetitionTime", "")),
        "te": str(getattr(first, "EchoTime", "")),
        "n_dicoms": len(ds_list),
        "n_skipped": skipped,
        "shape": tuple(int(x) for x in vol.shape),
        "pixel_spacing": pixel_spacing,
        "slice_spacing": float(z_spacing),
        "source_example": _dicom_file_path(first),
        **cls,
    }
    return vol, affine, meta


def build_dicom_manifest(series: dict[str, list], out_dir: Path) -> pd.DataFrame:
    rows = []
    for uid, ds_list in sorted(series.items(), key=lambda kv: str(kv[0])):
        if not ds_list:
            continue
        ds = sorted(ds_list, key=_dicom_sort_key)[0]
        desc = str(getattr(ds, "SeriesDescription", ""))
        protocol = str(getattr(ds, "ProtocolName", ""))
        cls = classify_series(desc, protocol)
        rows.append({
            "series_uid": uid,
            "descripcion_serie": desc,
            "protocolo": protocol,
            "modalidad": str(getattr(ds, "Modality", "")),
            "cantidad_archivos": len(ds_list),
            "filas": str(getattr(ds, "Rows", "")),
            "columnas": str(getattr(ds, "Columns", "")),
            "pixel_spacing": str(getattr(ds, "PixelSpacing", "")),
            "slice_thickness": str(getattr(ds, "SliceThickness", "")),
            "fabricante": str(getattr(ds, "Manufacturer", "")),
            "modelo": str(getattr(ds, "ManufacturerModelName", "")),
            "campo_magnetico": str(getattr(ds, "MagneticFieldStrength", "")),
            "tr": str(getattr(ds, "RepetitionTime", "")),
            "te": str(getattr(ds, "EchoTime", "")),
            "ejemplo_archivo": _dicom_file_path(ds),
            **cls,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        save_dataframe(df, out_dir / "manifest_series_detectadas.csv")
        save_dataframe(df.groupby(["series_role", "series_subtype", "series_side"], dropna=False)
                       .agg(series=("series_uid", "count"), archivos=("cantidad_archivos", "sum"))
                       .reset_index(), out_dir / "resumen_series_por_tipo.csv")
    return df


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    _, nib, _ = _require_neuro_libs()
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    cls = classify_series(path.name, "")
    meta = {"source": str(path), "shape": tuple(int(x) for x in data.shape), "kind": "nifti", **cls}
    return data, img.affine, meta


def robust_zscore(vol: np.ndarray, cfg: PipelineConfig | None = None) -> np.ndarray:
    # GPU opcional para volúmenes grandes. Si no hay CuPy/CUDA, cae a CPU sin romper.
    gpu_out = robust_zscore_gpu(vol, cfg)
    if gpu_out is not None:
        return gpu_out.astype(np.float32, copy=False)

    x = np.asarray(vol, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    vals = x[finite]
    lo, hi = np.percentile(vals, [1, 99])
    x = np.clip(x, lo, hi)
    med = np.median(x[finite])
    mad = np.median(np.abs(x[finite] - med))
    return ((x - med) / (1.4826 * mad + 1e-6)).astype(np.float32)


def resize_to_shape(vol: np.ndarray, shape: tuple[int, ...], order: int = 1, cfg: PipelineConfig | None = None) -> np.ndarray:
    _, _, ndimage = _require_neuro_libs()
    if tuple(vol.shape[: len(shape)]) == tuple(shape):
        return vol

    gpu_out = resize_to_shape_gpu(vol, shape, order=order, cfg=cfg)
    if gpu_out is not None:
        return gpu_out

    factors = [shape[i] / vol.shape[i] for i in range(len(shape))]
    return ndimage.zoom(vol, factors, order=order)


def wavelet_energy_volume(vol: np.ndarray, scales: Iterable[float], cfg: PipelineConfig | None = None) -> np.ndarray:
    """Energía multiescala tipo Mexican-hat/LoG para mapas y resonancias.

    En v3.4 intenta usar GPU para volúmenes grandes mediante CuPy/cupyx.scipy.ndimage.
    """
    _, _, ndimage = _require_neuro_libs()
    data = np.asarray(vol, dtype=np.float32)
    data = np.squeeze(data)
    if data.ndim > 3:
        data = np.mean(data, axis=-1)
    if data.ndim == 2:
        data = data[:, :, None]

    gpu_energy = wavelet_energy_gpu(data, scales, cfg)
    if gpu_energy is not None:
        return gpu_energy.astype(np.float32, copy=False)

    data = robust_zscore(data, cfg=cfg)
    acc = np.zeros_like(data, dtype=np.float32)
    count = 0
    for s in scales:
        if s <= 0:
            continue
        response = ndimage.gaussian_laplace(data, sigma=float(s))
        acc += np.abs(response).astype(np.float32)
        count += 1
    return acc / max(count, 1)


def save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    _, nib, _ = _require_neuro_libs()
    safe_mkdir(path.parent)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine), str(path))


def save_orthogonal_png(vol: np.ndarray, path: Path, title: str = "Volumen", cmap: str = "gray") -> None:
    safe_mkdir(path.parent)
    data = np.squeeze(vol)
    if data.ndim > 3:
        data = np.mean(data, axis=-1)
    if data.ndim == 2:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(np.rot90(data), cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    if data.ndim != 3:
        return
    x, y, z = [s // 2 for s in data.shape]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=13)
    axes[0].imshow(np.rot90(data[x, :, :]), cmap=cmap)
    axes[0].set_title("Sagital")
    axes[1].imshow(np.rot90(data[:, y, :]), cmap=cmap)
    axes[1].set_title("Coronal")
    axes[2].imshow(np.rot90(data[:, :, z]), cmap=cmap)
    axes[2].set_title("Axial")
    for ax in axes:
        ax.axis("off")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_topography_3d(vol: np.ndarray, path: Path, title: str = "Topografía 3D") -> None:
    """Gráfico 3D tipo topografía desde el corte axial central."""
    from scipy.ndimage import gaussian_filter

    safe_mkdir(path.parent)
    data = np.squeeze(vol)
    if data.ndim > 3:
        data = np.mean(data, axis=-1)
    if data.ndim == 3:
        z = data.shape[2] // 2
        sl = np.asarray(data[:, :, z], dtype=float)
    elif data.ndim == 2:
        sl = np.asarray(data, dtype=float)
    else:
        return
    sl = np.nan_to_num(sl)
    sl = gaussian_filter(sl, sigma=1.0)
    if np.nanmax(sl) > np.nanmin(sl):
        sl_norm = (sl - np.nanmin(sl)) / (np.nanmax(sl) - np.nanmin(sl))
    else:
        sl_norm = sl
    step = max(1, int(max(sl_norm.shape) / 160))
    y = np.arange(0, sl_norm.shape[0], step)
    x = np.arange(0, sl_norm.shape[1], step)
    X, Y = np.meshgrid(x, y)
    Z = sl_norm[::step, ::step]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cm.terrain, linewidth=0, antialiased=True, alpha=0.92)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Intensidad normalizada")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_vce_heuristic_mask(shape: tuple[int, int, int], side: str, cfg: PipelineConfig) -> np.ndarray:
    """Máscara VCE heurística en coordenadas de imagen.

    Se conserva solo como respaldo. Si existe serie 'TRACTO CORTICOESPINAL' se procesa aparte.
    """
    nx, ny, nz = shape
    mask = np.zeros(shape, dtype=bool)
    x0 = int(nx * 0.42) if side == "izquierda" else int(nx * 0.58)
    y_top = int(ny * 0.38)
    y_bottom = int(ny * 0.55)
    z0 = int(nz * cfg.vce_min_z_fraction)
    z1 = int(nz * cfg.vce_max_z_fraction)
    r = int(cfg.vce_radius_vox)
    yy, xx = np.ogrid[:ny, :nx]
    for zi, z in enumerate(range(z0, z1)):
        frac = zi / max(z1 - z0 - 1, 1)
        x = int(x0 + (nx * 0.05 if side == "izquierda" else -nx * 0.05) * frac)
        y = int(y_top + (y_bottom - y_top) * frac)
        circle = (xx - x) ** 2 + (yy - y) ** 2 <= r**2
        mask[:, :, z] = circle.T
    return mask


def high_activation_mask(vol: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    data = np.asarray(vol, dtype=np.float32)
    data = np.squeeze(data)
    if data.ndim > 3:
        data = np.mean(data, axis=-1)
    if data.ndim == 2:
        data = data[:, :, None]
    finite = np.isfinite(data)
    if not np.any(finite):
        return np.zeros_like(data, dtype=np.uint8)
    thr = np.nanpercentile(data[finite], percentile)
    return (data >= thr).astype(np.uint8)


def vce_metrics(energy: np.ndarray, mask: np.ndarray, side: str, prefix: str = "vce_heuristica") -> dict:
    if mask.shape != energy.shape:
        try:
            mask = resize_to_shape(mask.astype(float), energy.shape, order=0) > 0.5
        except Exception:
            return {f"{prefix}_{side}_mean": math.nan, f"{prefix}_{side}_p95": math.nan, f"{prefix}_{side}_voxels": 0}
    vals = energy[mask.astype(bool)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {f"{prefix}_{side}_mean": math.nan, f"{prefix}_{side}_p95": math.nan, f"{prefix}_{side}_voxels": 0}
    return {
        f"{prefix}_{side}_mean": float(np.mean(vals)),
        f"{prefix}_{side}_median": float(np.median(vals)),
        f"{prefix}_{side}_p95": float(np.percentile(vals, 95)),
        f"{prefix}_{side}_voxels": int(vals.size),
    }




def _voxel_volume_ml_from_affine(affine: np.ndarray) -> float:
    try:
        zoom = np.abs(np.diag(np.asarray(affine, dtype=float))[:3])
        zoom = np.where(np.isfinite(zoom) & (zoom > 0), zoom, 1.0)
        return float(np.prod(zoom) / 1000.0)
    except Exception:
        return 0.001


def _largest_component(mask: np.ndarray) -> np.ndarray:
    _, _, ndimage = _require_neuro_libs()
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return m
    labels, nlab = ndimage.label(m)
    if nlab <= 1:
        return m
    counts = np.bincount(labels.ravel())
    if len(counts) <= 1:
        return m
    counts[0] = 0
    return labels == int(np.argmax(counts))


def create_cortical_shell_mask(vol: np.ndarray, cfg: PipelineConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Crea una máscara de cerebro y una cáscara cortical heurística sobre un volumen 3D.

    Está pensada para mapas de difusividad axial (AD) con fondo cercano a cero.
    Se usa solo como aproximación para resaltar corteza motora primaria/secundaria.
    """
    _, _, ndimage = _require_neuro_libs()
    data = _as_3d(vol)
    vals = data[np.isfinite(data)]
    vals = vals[np.abs(vals) > 1e-8]
    if vals.size == 0:
        brain = np.ones_like(data, dtype=bool)
    else:
        thr = float(np.nanpercentile(vals, 15))
        brain = data > thr
        brain = ndimage.binary_opening(brain, iterations=1)
        brain = ndimage.binary_closing(brain, iterations=2)
        brain = ndimage.binary_fill_holes(brain)
        brain = _largest_component(brain)
    iters = int(getattr(cfg, 'ad_motor_shell_erosion_iters', 2) if cfg is not None else 2)
    eroded = ndimage.binary_erosion(brain, iterations=max(iters, 1))
    shell = brain & (~eroded)
    shell = ndimage.binary_closing(shell, iterations=1)
    return brain.astype(bool), shell.astype(bool)


def motor_cortex_roi_mask(shape: tuple[int, int, int], side: str, region: str) -> np.ndarray:
    """ROI base por coordenadas normalizadas para corteza motora.

    Esta ROI se usa como respaldo y para recortar el atlas probabilístico heurístico.
    """
    nx, ny, nz = shape
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)[:, None, None]
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)[None, :, None]
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)[None, None, :]

    if side == 'izquierda':
        hemi = (x >= 0.08) & (x <= 0.49)
    else:
        hemi = (x >= 0.51) & (x <= 0.92)

    superior = (z >= 0.52) & (z <= 0.97)
    if region == 'm1':
        ap = (y >= 0.47) & (y <= 0.68)
    else:
        ap = (y >= 0.30) & (y < 0.50)
    return np.broadcast_to(hemi & ap & superior, shape)


def motor_cortex_atlas_prior(shape: tuple[int, int, int], side: str, region: str, sigma_scale: float = 1.0) -> np.ndarray:
    """Atlas probabilístico heurístico en coordenadas normalizadas.

    Genera una prior 0..1 para M1/M2 derecha/izquierda usando una elipsoide/gaussiana
    aproximada en espacio de imagen. No sustituye un atlas real MNI, pero es más estable
    que una caja binaria simple.
    """
    nx, ny, nz = shape
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)[:, None, None]
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)[None, :, None]
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)[None, None, :]

    if region == 'm1':
        cy = 0.58
        sy = 0.10
    else:
        cy = 0.41
        sy = 0.10
    cz = 0.80
    sz = 0.12
    cx = 0.33 if side == 'izquierda' else 0.67
    sx = 0.08

    sx *= float(sigma_scale)
    sy *= float(sigma_scale)
    sz *= float(sigma_scale)

    prior = np.exp(-0.5 * (((x - cx) / max(sx, 1e-6)) ** 2 + ((y - cy) / max(sy, 1e-6)) ** 2 + ((z - cz) / max(sz, 1e-6)) ** 2))
    prior *= motor_cortex_roi_mask(shape, side, region).astype(np.float32)
    if np.nanmax(prior) > 0:
        prior = prior / np.nanmax(prior)
    return prior.astype(np.float32)


def _clean_small_mask(mask: np.ndarray, min_voxels: int = 20) -> np.ndarray:
    _, _, ndimage = _require_neuro_libs()
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return m
    labels, nlab = ndimage.label(m)
    if nlab <= 1:
        return m if int(np.sum(m)) >= min_voxels else np.zeros_like(m, dtype=bool)
    counts = np.bincount(labels.ravel())
    keep = np.zeros_like(m, dtype=bool)
    for lab in range(1, len(counts)):
        if counts[lab] >= min_voxels:
            keep |= labels == lab
    if np.any(keep):
        return keep
    counts[0] = 0
    return labels == int(np.argmax(counts))


def identify_motor_cortices_from_ad(
    ad_vol: np.ndarray,
    energy_vol: np.ndarray,
    affine: np.ndarray,
    out_dir: Path,
    base_name: str,
    cfg: PipelineConfig,
) -> dict:
    """Delimita M1/M2 a partir de AD + wavelet con guía de atlas probabilístico heurístico.

    Flujo:
    1) obtiene máscara de cerebro y cáscara cortical;
    2) crea una prior probabilística de atlas para M1/M2 por lado;
    3) combina prior + shell cortical + energía wavelet;
    4) guarda máscaras `.nii.gz` y calcula volúmenes.
    """
    _, _, ndimage = _require_neuro_libs()
    cortex_dir = out_dir / 'corteza_motora_ad'
    safe_mkdir(cortex_dir)

    ad3 = _as_3d(ad_vol)
    energy3 = _as_3d(energy_vol)
    voxel_ml = _voxel_volume_ml_from_affine(affine)
    brain_mask, shell_mask = create_cortical_shell_mask(ad3, cfg)
    save_nifti(brain_mask.astype(np.uint8), affine, cortex_dir / f'{base_name}_brain_mask_heuristica.nii.gz')
    save_nifti(shell_mask.astype(np.uint8), affine, cortex_dir / f'{base_name}_cortical_shell_heuristica.nii.gz')

    # Normalización local para ponderar mejor atlas + wavelet.
    energy_norm = minmax_gpu(energy3, cfg) if cfg is not None else None
    if energy_norm is None:
        e = np.asarray(energy3, dtype=np.float32)
        lo, hi = np.nanpercentile(e[np.isfinite(e)], [1, 99]) if np.any(np.isfinite(e)) else (0.0, 1.0)
        energy_norm = np.clip((e - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)

    rows = []
    labelmap = np.zeros_like(ad3, dtype=np.uint8)
    metrics = {
        'ad_motor_note': 'Delimitación atlas-guided heurística/exploratoria sobre AD + wavelet; validar con atlas/registro si se requiere precisión anatómica.',
        'ad_motor_voxel_volume_ml': voxel_ml,
        'ad_motor_method': 'atlas_guided_wavelet_on_AD',
    }

    label_lut = [
        ('m1', 'primaria', 'derecha', 1),
        ('m1', 'primaria', 'izquierda', 2),
        ('m2', 'secundaria', 'derecha', 3),
        ('m2', 'secundaria', 'izquierda', 4),
    ]
    for region, region_name, side, label_id in label_lut:
        roi = motor_cortex_roi_mask(ad3.shape, side, region)
        prior = motor_cortex_atlas_prior(ad3.shape, side, region, sigma_scale=float(getattr(cfg, 'ad_motor_prior_sigma_scale', 1.0)))
        roi_shell = roi & shell_mask
        if int(np.sum(roi_shell)) < 25:
            roi_shell = roi & brain_mask
        prior_mask = roi_shell & (prior >= float(getattr(cfg, 'ad_motor_prior_threshold', 0.25)))
        if int(np.sum(prior_mask)) < 25:
            prior_mask = roi_shell

        vals = energy_norm[prior_mask]
        vals = vals[np.isfinite(vals)]
        thr_p = float(getattr(cfg, 'ad_motor_wavelet_percentile', 75.0))
        if vals.size == 0:
            thr = math.nan
            refined = np.zeros_like(prior_mask, dtype=bool)
        else:
            thr = float(np.nanpercentile(vals, thr_p))
            weighted = prior * energy_norm
            refined = prior_mask & (weighted >= float(np.nanpercentile(weighted[prior_mask], thr_p)))
            if int(np.sum(refined)) < int(getattr(cfg, 'ad_motor_min_component_voxels', 20)):
                refined = prior_mask & (energy_norm >= thr)
            refined = ndimage.binary_opening(refined, iterations=1)
            refined = ndimage.binary_closing(refined, iterations=1)
            refined = _clean_small_mask(refined, int(getattr(cfg, 'ad_motor_min_component_voxels', 20)))
            if int(np.sum(refined)) < 5:
                refined = _clean_small_mask(prior_mask, 5)

        region_tag = f'corteza_motora_{region_name}_{side}'
        save_nifti(prior.astype(np.float32), affine, cortex_dir / f'{base_name}_{region_tag}_atlas_prior.nii.gz')
        save_nifti(roi_shell.astype(np.uint8), affine, cortex_dir / f'{base_name}_{region_tag}_roi_heuristica.nii.gz')
        save_nifti(refined.astype(np.uint8), affine, cortex_dir / f'{base_name}_{region_tag}_wavelet_mask.nii.gz')
        save_nifti((energy3 * refined.astype(np.float32)), affine, cortex_dir / f'{base_name}_{region_tag}_wavelet_region.nii.gz')
        labelmap[refined] = label_id

        vox = int(np.sum(refined))
        vol_ml = float(vox * voxel_ml)
        energy_vals = energy3[refined]
        energy_vals = energy_vals[np.isfinite(energy_vals)]
        prior_vals = prior[refined]
        mean_energy = float(np.nanmean(energy_vals)) if energy_vals.size else math.nan
        p95_energy = float(np.nanpercentile(energy_vals, 95)) if energy_vals.size else math.nan
        mean_prior = float(np.nanmean(prior_vals)) if prior_vals.size else math.nan
        rows.append({
            'region': region,
            'region_name': region_name,
            'side': side,
            'voxels': vox,
            'volume_ml': vol_ml,
            'wavelet_threshold': thr,
            'energy_mean': mean_energy,
            'energy_p95': p95_energy,
            'atlas_prior_mean_in_mask': mean_prior,
            'roi_voxels': int(np.sum(roi_shell)),
            'mask_path': str(cortex_dir / f'{base_name}_{region_tag}_wavelet_mask.nii.gz'),
            'roi_path': str(cortex_dir / f'{base_name}_{region_tag}_roi_heuristica.nii.gz'),
            'atlas_prior_path': str(cortex_dir / f'{base_name}_{region_tag}_atlas_prior.nii.gz'),
        })
        metrics[f'ad_{region}_{side}_voxels'] = vox
        metrics[f'ad_{region}_{side}_volume_ml'] = vol_ml
        metrics[f'ad_{region}_{side}_wavelet_threshold'] = thr
        metrics[f'ad_{region}_{side}_energy_mean'] = mean_energy
        metrics[f'ad_{region}_{side}_energy_p95'] = p95_energy
        metrics[f'ad_{region}_{side}_atlas_prior_mean_in_mask'] = mean_prior

    save_nifti(labelmap.astype(np.uint8), affine, cortex_dir / f'{base_name}_corteza_motora_labelmap.nii.gz')
    metrics['ad_motor_labelmap'] = str(cortex_dir / f'{base_name}_corteza_motora_labelmap.nii.gz')
    metrics['ad_motor_labelmap_note'] = '1=M1 derecha, 2=M1 izquierda, 3=M2 derecha, 4=M2 izquierda'

    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, 'base_name', base_name)
        save_dataframe(df, cortex_dir / 'metricas_corteza_motora_ad.csv')
    return metrics

def output_dir_for_series(out_dir: Path, meta: dict, base_name: str) -> Path:
    role = meta.get("series_role", "otros")
    subtype = meta.get("series_subtype", "otros")
    side = meta.get("series_side", "desconocido")
    if role == "fmri_motor":
        return out_dir / "fmri_motor" / side / subtype / base_name
    if role == "dti_difusion":
        return out_dir / "dti_difusion" / subtype / base_name
    if role in {"vce_explicita", "tractografia"}:
        return out_dir / "via_corticoespinal" / role / base_name
    return out_dir / role / subtype / base_name



def save_histogram_png(vol: np.ndarray, path: Path, title: str = "Histograma") -> None:
    safe_mkdir(path.parent)
    data = np.asarray(vol, dtype=float).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        return
    lo, hi = np.percentile(data, [1, 99])
    data = data[(data >= lo) & (data <= hi)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=80)
    ax.set_title(title)
    ax.set_xlabel("Intensidad")
    ax.set_ylabel("Voxeles")
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_mip_png(vol: np.ndarray, path: Path, title: str = "MIP") -> None:
    safe_mkdir(path.parent)
    data = np.squeeze(np.asarray(vol, dtype=float))
    if data.ndim > 3:
        data = np.mean(data, axis=-1)
    if data.ndim == 2:
        mip = data
    elif data.ndim == 3:
        mip = np.nanmax(data, axis=2)
    else:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.rot90(mip), cmap="inferno")
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def temporal_stack_metrics(data3: np.ndarray, out_dir: Path, base_name: str, tr_seconds: float | None = None) -> dict:
    """Aprovecha las series fMRI/t-map como stack temporal si el último eje tiene muchos frames.

    Viene de la idea de los códigos originales de correlación fMRI: ACF, PSD y señal media BOLD.
    Si la serie realmente es una pila espacial, el reporte queda marcado como exploratorio.
    """
    data = np.squeeze(np.asarray(data3, dtype=np.float32))
    if data.ndim != 3 or data.shape[2] < 12:
        return {}
    tseries = np.nanmean(data.reshape(-1, data.shape[2]), axis=0)
    tseries = np.nan_to_num(tseries - np.nanmean(tseries))
    tr = float(tr_seconds) if tr_seconds and np.isfinite(tr_seconds) and tr_seconds > 0 else 1.0
    fs = 1.0 / tr
    try:
        from scipy import signal as sp_signal
        acf = sp_signal.correlate(tseries, tseries, mode="full", method="fft")
        lags = sp_signal.correlation_lags(len(tseries), len(tseries), mode="full") * tr
        acf = acf / (np.nanmax(np.abs(acf)) + 1e-12)
        freqs, psd = sp_signal.welch(tseries, fs=fs, nperseg=min(128, len(tseries)))
        peak_freq = float(freqs[int(np.nanargmax(psd))]) if len(freqs) else math.nan

        # Métricas DSP/QC inspiradas en recomendaciones de rs/task-fMRI:
        # ALFF/fALFF, tSNR, DVARS y GCOR aproximado.
        low_mask = (freqs >= 0.01) & (freqs <= 0.08)
        valid_mask = freqs >= 0.01
        alff_mean_signal = float(np.trapz(np.sqrt(np.maximum(psd[low_mask], 0)), freqs[low_mask])) if np.any(low_mask) else math.nan
        denom = float(np.trapz(np.sqrt(np.maximum(psd[valid_mask], 0)), freqs[valid_mask])) if np.any(valid_mask) else math.nan
        falff_mean_signal = float(alff_mean_signal / (denom + 1e-12)) if np.isfinite(denom) else math.nan

        vox_t = data.reshape(-1, data.shape[2]).astype(np.float32)
        finite_rows = np.all(np.isfinite(vox_t), axis=1)
        nz_rows = np.nanstd(vox_t, axis=1) > 1e-8
        vox_t = vox_t[finite_rows & nz_rows]
        if vox_t.shape[0] > 20000:
            # Muestreo determinístico para evitar uso excesivo de RAM en EPI grandes.
            idx = np.linspace(0, vox_t.shape[0] - 1, 20000).astype(int)
            vox_t = vox_t[idx]
        if vox_t.size:
            means = np.nanmean(vox_t, axis=1)
            stds = np.nanstd(vox_t, axis=1) + 1e-12
            tsnr_median = float(np.nanmedian(np.abs(means) / stds))
            diffs = np.diff(vox_t, axis=1)
            dvars_mean = float(np.nanmean(np.sqrt(np.nanmean(diffs ** 2, axis=0)))) if diffs.size else math.nan
            zvox = (vox_t - means[:, None]) / stds[:, None]
            global_z = np.nanmean(zvox, axis=0)
            gcor_approx = float(np.nanmean(global_z ** 2))
        else:
            tsnr_median = math.nan
            dvars_mean = math.nan
            gcor_approx = math.nan

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
        axes[0].plot(np.arange(len(tseries)) * tr, tseries, lw=0.9)
        axes[0].set_title("Señal media del stack fMRI/mapa")
        axes[0].set_xlabel("Tiempo/índice (s o frame)")
        axes[1].plot(lags, acf, lw=0.9)
        axes[1].set_title("Autocorrelación temporal FFT")
        axes[1].set_xlabel("Lag")
        axes[2].semilogy(freqs, psd + 1e-12, lw=0.9)
        axes[2].set_title("PSD temporal")
        axes[2].set_xlabel("Hz")
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.savefig(out_dir / f"{base_name}_fmri_temporal_acf_psd.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {
            "fmri_stack_frames": int(data.shape[2]),
            "fmri_temporal_mean": float(np.nanmean(tseries)),
            "fmri_temporal_std": float(np.nanstd(tseries)),
            "fmri_temporal_peak_freq_hz": peak_freq,
            "fmri_alff_0p01_0p08_mean_signal": alff_mean_signal,
            "fmri_falff_0p01_0p08_mean_signal": falff_mean_signal,
            "fmri_tsnr_median_sampled_voxels": tsnr_median,
            "fmri_dvars_mean_sampled_voxels": dvars_mean,
            "fmri_gcor_approx_sampled_voxels": gcor_approx,
            "fmri_temporal_plot": str(out_dir / f"{base_name}_fmri_temporal_acf_psd.png"),
            "fmri_temporal_nota": "Exploratorio: usa el último eje como tiempo/frames si la serie no trae 4D explícito. ALFF/fALFF se calculan sobre la señal media del stack.",
        }
    except Exception as exc:
        return {"fmri_temporal_error": str(exc)}


def activation_and_volume_metrics(data3: np.ndarray, affine: np.ndarray, role: str, subtype: str) -> dict:
    data = np.squeeze(np.asarray(data3, dtype=np.float32))
    if data.ndim > 3:
        data = np.mean(data, axis=-1)
    if data.ndim == 2:
        data = data[:, :, None]
    finite = np.isfinite(data)
    if not np.any(finite):
        return {}
    zoom = np.abs(np.diag(affine)[:3]) if affine.shape == (4,4) else np.ones(3)
    voxel_ml = float(np.prod(np.where(zoom > 0, zoom, 1.0)) / 1000.0)
    rows = {}
    for pctl in [90, 95, 99]:
        thr = float(np.nanpercentile(data[finite], pctl))
        vox = int(np.sum(data >= thr))
        rows[f"voxels_p{pctl}"] = vox
        rows[f"volume_ml_p{pctl}"] = vox * voxel_ml
        rows[f"threshold_p{pctl}"] = thr
    if role == "dti_difusion" and subtype == "fa":
        # FA suele estar en 0-1; si viene escalada, igual se reporta percentiles.
        rows["fa_mean_nonzero"] = float(np.nanmean(data[(data > 0) & finite])) if np.any((data > 0) & finite) else math.nan
        rows["fa_p50_nonzero"] = float(np.nanpercentile(data[(data > 0) & finite], 50)) if np.any((data > 0) & finite) else math.nan
    return rows

def process_single_volume(
    data: np.ndarray,
    affine: np.ndarray,
    out_dir: Path,
    base_name: str,
    scales: Iterable[float],
    cfg: PipelineConfig,
    meta: Optional[dict] = None,
) -> dict:
    safe_mkdir(out_dir)
    meta = meta or {}
    data3 = np.squeeze(np.asarray(data, dtype=np.float32))
    if data3.ndim > 3:
        data3 = np.mean(data3, axis=-1)
    if data3.ndim == 2:
        data3 = data3[:, :, None]

    save_nifti(data3, affine, out_dir / f"{base_name}_volumen.nii.gz")
    save_orthogonal_png(data3, out_dir / f"{base_name}_ortogonal.png", title=base_name)
    if getattr(cfg, "save_histograms", True):
        save_histogram_png(data3, out_dir / f"{base_name}_histograma.png", title=f"Histograma · {base_name}")
    if getattr(cfg, "save_mip_previews", True):
        save_mip_png(data3, out_dir / f"{base_name}_mip.png", title=f"MIP · {base_name}")

    energy = wavelet_energy_volume(data3, scales, cfg=cfg)
    save_nifti(energy, affine, out_dir / f"{base_name}_wavelet_energy.nii.gz")
    save_orthogonal_png(energy, out_dir / f"{base_name}_wavelet_ortogonal.png", title=f"Wavelet energy · {base_name}", cmap="hot")
    save_topography_3d(energy, out_dir / f"{base_name}_topografia_3d.png", title=f"Topografía 3D · {base_name}")
    save_histogram_png(energy, out_dir / f"{base_name}_wavelet_histograma.png", title=f"Histograma wavelet · {base_name}")

    # Máscaras de alta señal/activación para mapas, DTI y tracto explícito.
    role = meta.get("series_role", "")
    subtype = meta.get("series_subtype", "")
    if role in {"fmri_motor", "dti_difusion", "vce_explicita", "tractografia"} or subtype in {"fa", "adc", "rd", "ad", "trace_b0"}:
        for pctl in (95, 99):
            mask = high_activation_mask(data3, pctl)
            save_nifti(mask, affine, out_dir / f"{base_name}_mask_p{pctl}.nii.gz")

    metrics = {
        "base_name": base_name,
        "shape": tuple(int(x) for x in data3.shape),
        "mean": float(np.nanmean(data3)),
        "std": float(np.nanstd(data3)),
        "p05": float(np.nanpercentile(data3, 5)),
        "p50": float(np.nanpercentile(data3, 50)),
        "p95": float(np.nanpercentile(data3, 95)),
        "energy_mean": float(np.nanmean(energy)),
        "energy_p95": float(np.nanpercentile(energy, 95)),
    }
    metrics.update(activation_and_volume_metrics(data3, affine, role, subtype))
    if role == "fmri_motor":
        tr_ms = _to_float(meta.get("tr", math.nan), math.nan)
        tr_seconds = tr_ms / 1000.0 if np.isfinite(tr_ms) and tr_ms > 50 else tr_ms
        metrics.update(temporal_stack_metrics(data3, out_dir, base_name, tr_seconds=tr_seconds))
    if data3.ndim == 3:
        for side in ["derecha", "izquierda"]:
            mask = create_vce_heuristic_mask(data3.shape, side, cfg)
            save_nifti(mask.astype(np.uint8), affine, out_dir / f"vce_heuristica_{side}.nii.gz")
            metrics.update(vce_metrics(energy, mask, side, prefix="vce_heuristica"))

    # Delimitación heurística de corteza motora primaria/secundaria sobre AD.
    if role == "dti_difusion" and subtype == "ad":
        try:
            metrics.update(identify_motor_cortices_from_ad(data3, energy, affine, out_dir, base_name, cfg))
        except Exception as exc:
            metrics["ad_motor_error"] = str(exc)
    return metrics



def is_probable_functional_map_file(path: Path) -> bool:
    """Filtra NIfTI de mapas funcionales aunque las carpetas/nombres estén desordenados.

    Se diseñó para las rutas reales reportadas:
    - ResultadosFuncional/MAPAS/rMotor_izquierda_Rodilla_fwe005.nii
    - ResultadosFuncional/MAPAS/rMotor_rod_der_fwe005.nii
    - Resultados funcional/MAPAS/MAPAS/rMOTOR_DERECHA_FWE001.nii
    - ResultadosFuncional/Resultados/Resultados/mapas/rmap_motorIzq_fwe005.nii
    """
    if not is_nifti(path):
        return False
    full = norm_key(str(path))
    name = norm_key(path.name)
    compact_full = re.sub(r"[^a-z0-9]+", "", full)
    compact_name = re.sub(r"[^a-z0-9]+", "", name)

    in_functional_context = any(k in compact_full for k in [
        "resultadosfuncional", "resultadofuncional", "funcional", "mapas", "maps"
    ])
    name_says_motor_map = any(k in compact_name for k in [
        "motor", "motrodilla", "rodilla", "fwe", "spmt", "tmap", "activation", "activacion"
    ])
    explicit_map_folder = any(part in {"mapa", "mapas", "maps"} for part in [norm_key(x) for x in path.parts])

    return bool((in_functional_context and name_says_motor_map) or (explicit_map_folder and name_says_motor_map))


def find_map_nifti_files(stage_dir: Optional[Path], cfg: PipelineConfig | None = None) -> list[Path]:
    """Encuentra los mapas funcionales NIfTI de manera recursiva y tolerante.

    No depende de que la carpeta se llame exactamente ResultadosFuncional. Soporta:
    ResultadosFuncional, Resultados funcional, Resultados/Resultados/mapas, MAPAS/MAPAS, etc.
    """
    if stage_dir is None or not stage_dir.exists():
        return []
    niftis = find_nifti_files(stage_dir)
    candidates = [p for p in niftis if is_probable_functional_map_file(p)]

    # Si el usuario guardó los mapas en una carpeta llamada MAPAS con nombres no estándar,
    # incluir todos los NIfTI dentro de esa carpeta.
    for nii in niftis:
        parts = {norm_key(x) for x in nii.parts}
        if {"mapa", "mapas", "maps"} & parts and nii not in candidates:
            candidates.append(nii)

    return sorted(set(candidates))


def common_parent(paths: list[Path]) -> Optional[Path]:
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0].parent
    try:
        import os
        return Path(os.path.commonpath([str(p.parent) for p in paths]))
    except Exception:
        return paths[0].parent

def process_maps_folder(cfg: PipelineConfig, maps_dir: Path, out_dir: Path) -> pd.DataFrame:
    safe_mkdir(out_dir)
    ckpt = CheckpointManager(cfg)
    rows = []

    # Antes se buscaban todos los NIfTI dentro de una carpeta exacta. Ahora se filtra
    # con tolerancia porque tus mapas están en rutas diferentes y con nombres distintos.
    map_files = find_map_nifti_files(maps_dir, cfg)
    if not map_files:
        # Respaldo: si maps_dir ya es una carpeta MAPAS, procesar todos sus NIfTI.
        map_files = find_nifti_files(maps_dir)

    if map_files:
        save_dataframe(pd.DataFrame([{
            "archivo": str(p),
            "nombre": p.name,
            "lado_detectado": side_from_name(p),
            "carpeta": str(p.parent),
        } for p in map_files]), out_dir / "mapas_detectados.csv")

    for nii in map_files:
        try:
            data, affine, meta = load_nifti(nii)
            # Los NIfTI dentro de ResultadosFuncional/MAPAS son mapas de activación motor,
            # aunque el nombre no tenga ep2d ni DICOM metadata.
            meta.update(classify_series(nii.name, "mapa funcional motor"))
            if meta.get("series_role") == "otros":
                meta.update({
                    "series_role": "fmri_motor",
                    "series_subtype": "mapa_activacion",
                    "series_side": side_from_name(nii),
                })
            if meta.get("series_side") in {None, "", "desconocido"}:
                meta["series_side"] = side_from_name(nii)

            base = sanitize_name(nii.name.replace(".nii.gz", "").replace(".nii", ""))
            side = meta.get("series_side") or side_from_name(nii)
            target_dir = out_dir / (side if side != "desconocido" else "sin_lateralidad")
            metrics_json = target_dir / f"{base}_metricas.json"
            task_id = f"mapas/{sanitize_name(str(nii), 120)}"

            def _work():
                metrics = process_single_volume(data, affine, target_dir, base, cfg.map_wavelet_scales, cfg, meta)
                metrics.update({"source": str(nii), "side": side, "modality": "mapas", **meta})
                save_metrics_json(metrics, metrics_json)
                return metrics

            result, status = ckpt.run(
                task_id=task_id,
                inputs=[nii],
                outputs=[metrics_json, target_dir / f"{base}_wavelet_energy.nii.gz"],
                params={"kind": "mapas", "scales": list(cfg.map_wavelet_scales), "detector": "v3_2_rutas_flexibles"},
                fn=_work,
            )
            metrics = load_json_if_exists(metrics_json) if status == "skipped" else result
            if metrics:
                metrics["checkpoint_status"] = status
                rows.append(metrics)
        except Exception as exc:
            append_log(out_dir / "mapas_log.txt", f"Error procesando mapa {nii}: {exc}")
    df = pd.DataFrame(rows)
    if not df.empty:
        save_dataframe(df, out_dir / "metricas_mapas.csv")
    return df

def find_maps_dir(stage_dir: Optional[Path], cfg: PipelineConfig) -> Optional[Path]:
    if stage_dir is None or not stage_dir.exists():
        return None

    # Primero buscar exactamente los NIfTI funcionales de forma recursiva.
    map_files = find_map_nifti_files(stage_dir, cfg)
    if map_files:
        return common_parent(map_files)

    # Compatibilidad con la búsqueda anterior, pero usando comparación tolerante.
    for root_name in cfg.functional_dirnames:
        candidates = []
        direct = stage_dir / root_name
        if direct.exists():
            candidates.append(direct)
        # Windows no distingue mayúsculas, pero Python puede ejecutarse también en entornos
        # sensibles a mayúsculas. Por eso se busca por clave normalizada.
        wanted = norm_key(root_name)
        for child in stage_dir.iterdir():
            if child.is_dir() and norm_key(child.name) == wanted:
                candidates.append(child)
        for root in candidates:
            if find_nifti_files(root):
                return root
            for child in root.rglob("*"):
                if child.is_dir() and norm_key(child.name) in {"mapas", "mapa", "maps"} and find_nifti_files(child):
                    return child

    # Último respaldo: cualquier carpeta con nombre mapa/maps que contenga NIfTI.
    for child in stage_dir.rglob("*"):
        if child.is_dir() and norm_key(child.name) in {"mapas", "mapa", "maps"} and find_nifti_files(child):
            return child
    return None

def process_resonance_folder(cfg: PipelineConfig, resonance_dir: Path, out_dir: Path) -> pd.DataFrame:
    safe_mkdir(out_dir)
    ckpt = CheckpointManager(cfg)
    rows = []

    # 1) NIfTI existentes dentro de resonancias.
    for nii in find_nifti_files(resonance_dir):
        try:
            # Cargar metadatos y forma; si ya está hecho, se leerá solo el JSON.
            data, affine, meta = load_nifti(nii)
            base = sanitize_name(nii.name.replace(".nii.gz", "").replace(".nii", ""))
            target_dir = output_dir_for_series(out_dir, meta, base)
            metrics_json = target_dir / f"{base}_metricas.json"
            task_id = f"resonancias/nifti/{sanitize_name(str(nii), 150)}"

            def _work():
                metrics = process_single_volume(data, affine, target_dir, base, cfg.resonance_wavelet_scales, cfg, meta)
                metrics.update(meta | {"source": str(nii), "source_kind": "nifti", "output_dir": str(target_dir)})
                save_metrics_json(metrics, metrics_json)
                return metrics

            result, status = ckpt.run(
                task_id=task_id,
                inputs=[nii],
                outputs=[metrics_json, target_dir / f"{base}_wavelet_energy.nii.gz"],
                params={"kind": "resonancia_nifti", "scales": list(cfg.resonance_wavelet_scales), "detector": "v3_8_structure_function_qc"},
                fn=_work,
            )
            metrics = load_json_if_exists(metrics_json) if status == "skipped" else result
            if metrics:
                metrics["checkpoint_status"] = status
                rows.append(metrics)
        except Exception as exc:
            append_log(out_dir / "resonancias_log.txt", f"Error NIfTI {nii}: {exc}")

    # 2) DICOM: procesa cada serie completa y la organiza por rol.
    series = collect_dicom_series(resonance_dir)
    manifest_df = build_dicom_manifest(series, out_dir)
    for uid, ds_list in series.items():
        try:
            ds0 = sorted(ds_list, key=_dicom_sort_key)[0]
            desc = str(getattr(ds0, "SeriesDescription", ""))
            protocol = str(getattr(ds0, "ProtocolName", ""))
            cls = classify_series(desc, protocol)
            preview_meta = {
                "series_uid": uid,
                "series_description": desc,
                "protocol": protocol,
                "tr": str(getattr(ds0, "RepetitionTime", "")),
                "te": str(getattr(ds0, "EchoTime", "")),
                **cls,
            }
            base = sanitize_name(f"{cls.get('series_role','serie')}_{cls.get('series_subtype','')}_{desc}_{uid[-6:]}")
            target_dir = output_dir_for_series(out_dir, preview_meta, base)
            metrics_json = target_dir / f"{base}_metricas.json"
            # Firma por archivos originales de esta serie, no por toda la carpeta.
            inputs = [Path(_dicom_file_path(ds)) for ds in ds_list if _dicom_file_path(ds)]
            task_id = f"resonancias/dicom/{uid}"

            def _work():
                vol, affine, meta = dicom_series_to_volume(ds_list)
                # Recalcula por si el primer DICOM no tenía todos los campos.
                target = output_dir_for_series(out_dir, meta, base)
                metrics = process_single_volume(vol, affine, target, base, cfg.resonance_wavelet_scales, cfg, meta)
                metrics.update(meta | {"source": str(resonance_dir), "source_kind": "dicom", "output_dir": str(target)})

                # Máscara especial si el equipo trae serie visual del tracto corticoespinal.
                if meta.get("series_role") == "vce_explicita":
                    mask = high_activation_mask(vol, 90)
                    save_nifti(mask, affine, target / f"{base}_mascara_tracto_corticoespinal_visual.nii.gz")
                    metrics["vce_explicita_voxels_p90"] = int(np.sum(mask > 0))
                    metrics["nota_vce_explicita"] = "Serie visual/postprocesada detectada; validar en visor médico, no sustituye tractografía cuantitativa."

                save_metrics_json(metrics, metrics_json)
                return metrics

            result, status = ckpt.run(
                task_id=task_id,
                inputs=inputs or [resonance_dir],
                outputs=[metrics_json, target_dir / f"{base}_wavelet_energy.nii.gz"],
                params={"kind": "resonancia_dicom", "uid": uid, "scales": list(cfg.resonance_wavelet_scales), "detector": "v3_8_structure_function_qc", **preview_meta},
                fn=_work,
            )
            metrics = load_json_if_exists(metrics_json) if status == "skipped" else result
            if metrics:
                metrics["checkpoint_status"] = status
                rows.append(metrics)
        except Exception as exc:
            append_log(out_dir / "resonancias_log.txt", f"Error serie DICOM {uid}: {exc}")

    df = pd.DataFrame(rows)
    if not df.empty:
        save_dataframe(df, out_dir / "metricas_resonancias.csv")
        # Resúmenes útiles para encontrar rápido DTI, fMRI y VCE.
        try:
            save_dataframe(df.groupby(["series_role", "series_subtype", "series_side"], dropna=False)
                           .agg(series=("series_uid", "count"), energy_mean=("energy_mean", "mean"), energy_p95=("energy_p95", "mean"))
                           .reset_index(), out_dir / "resumen_metricas_por_tipo.csv")
        except Exception:
            pass
        # Resumen específico de corteza motora obtenida desde AD.
        ad_cols = [c for c in df.columns if c.startswith('ad_m1_') or c.startswith('ad_m2_')]
        if ad_cols:
            records = []
            for _, row in df.iterrows():
                if str(row.get('series_role', '')) != 'dti_difusion' or str(row.get('series_subtype', '')) != 'ad':
                    continue
                base_name = row.get('base_name', '')
                for region in ['m1', 'm2']:
                    for side in ['derecha', 'izquierda']:
                        key_vol = f'ad_{region}_{side}_volume_ml'
                        key_vox = f'ad_{region}_{side}_voxels'
                        if key_vol in df.columns or key_vox in df.columns:
                            records.append({
                                'base_name': base_name,
                                'series_uid': row.get('series_uid', ''),
                                'region': region,
                                'side': side,
                                'volume_ml': row.get(key_vol, math.nan),
                                'voxels': row.get(key_vox, math.nan),
                                'energy_mean': row.get(f'ad_{region}_{side}_energy_mean', math.nan),
                                'energy_p95': row.get(f'ad_{region}_{side}_energy_p95', math.nan),
                                'wavelet_threshold': row.get(f'ad_{region}_{side}_wavelet_threshold', math.nan),
                                'source': row.get('source', ''),
                                'output_dir': row.get('output_dir', ''),
                            })
            if records:
                save_dataframe(pd.DataFrame(records), out_dir / 'resumen_corteza_motora_ad.csv')
    return df

def compare_patient_vs_control_volume(
    patient_energy_path: Path,
    control_energy_path: Path,
    out_dir: Path,
    label: str,
) -> dict:
    _, nib, _ = _require_neuro_libs()
    p_img = nib.load(str(patient_energy_path))
    c_img = nib.load(str(control_energy_path))
    p = p_img.get_fdata(dtype=np.float32)
    c = c_img.get_fdata(dtype=np.float32)
    c_resized = resize_to_shape(c, p.shape, order=1)
    pz = robust_zscore(p)
    cz = robust_zscore(c_resized)
    deficit = pz - cz
    safe_mkdir(out_dir)
    save_nifti(deficit, p_img.affine, out_dir / f"deficit_{label}.nii.gz")
    save_orthogonal_png(deficit, out_dir / f"deficit_{label}_ortogonal.png", title=f"Déficit {label}", cmap="coolwarm")
    save_topography_3d(deficit, out_dir / f"deficit_{label}_topografia_3d.png", title=f"Déficit {label}")
    r = float(np.corrcoef(pz.ravel(), cz.ravel())[0, 1]) if pz.size > 2 else math.nan
    return {
        "label": label,
        "patient_energy": str(patient_energy_path),
        "control_energy": str(control_energy_path),
        "pearson_global": r,
        "deficit_mean": float(np.nanmean(deficit)),
        "deficit_p05": float(np.nanpercentile(deficit, 5)),
        "deficit_p95": float(np.nanpercentile(deficit, 95)),
        "out_deficit_nii": str(out_dir / f"deficit_{label}.nii.gz"),
    }


def _subject_stage_dir(cfg: PipelineConfig, subject: str, stage: str) -> Optional[Path]:
    subject_dir = cfg.data_root() / subject
    if subject == cfg.control_name:
        # El sano suele tener solo Antes. Para pedir Despues no debemos caer a la raíz
        # y mezclar resultados; primero buscamos la etapa exacta y luego Antes.
        exact = resolve_stage_dir(subject_dir, stage, fallback_to_subject=False)
        if exact is not None:
            return exact
        before = resolve_stage_dir(subject_dir, "Antes", fallback_to_subject=False)
        if before is not None:
            return before
        return subject_dir if subject_dir.exists() else None
    return resolve_stage_dir(subject_dir, stage, fallback_to_subject=False)


def _control_requested_stages(cfg: PipelineConfig) -> tuple[str, ...]:
    """Etapas reales del sano que deben procesarse para imágenes.

    Si solo existe sano/Antes, no duplicamos ese procesamiento como sano/Despues.
    Si existe sano/Despues, se procesa también y se usa para correlaciones post.
    """
    control_dir = cfg.data_root() / cfg.control_name
    stages = []
    if resolve_stage_dir(control_dir, "Antes", fallback_to_subject=False) is not None:
        stages.append("Antes")
    if resolve_stage_dir(control_dir, "Despues", fallback_to_subject=False) is not None:
        stages.append("Despues")
    if not stages and control_dir.exists():
        stages.append("Antes")
    return tuple(dict.fromkeys(stages))


def _iter_imaging_subject_stage(cfg: PipelineConfig):
    """Pacientes + sano para mapas/resonancias diagnósticas/correlaciones.

    Los pacientes usan todas las etapas. El sano usa sus etapas reales; normalmente Antes.
    """
    for patient in cfg.patients:
        for stage in cfg.stages:
            yield patient, stage, _subject_stage_dir(cfg, patient, stage)
    for stage in _control_requested_stages(cfg):
        yield cfg.control_name, stage, _subject_stage_dir(cfg, cfg.control_name, stage)


def _control_result_stage(cfg: PipelineConfig, stage: str, modality: str) -> Path | None:
    """Devuelve carpeta de resultados del sano para una etapa o cae a sano/Antes.

    modality: mapas, tomografia o resonancias.
    """
    primary = cfg.results_root() / cfg.control_name / stage / modality
    if primary.exists() and any(primary.rglob("*")):
        return primary
    before = cfg.results_root() / cfg.control_name / "Antes" / modality
    if before.exists() and any(before.rglob("*")):
        return before
    return primary if primary.exists() else before


def _process_control_if_needed(cfg: PipelineConfig, stage: str, log_file: Path) -> Path | None:
    control_stage_dir = _subject_stage_dir(cfg, cfg.control_name, stage)
    if not control_stage_dir:
        append_log(log_file, f"No encontré carpeta del control sano para etapa {stage}")
        return None
    out_stage_name = stage if control_stage_dir.name.lower() in {"antes", "despues", "después"} else "Antes"
    control_out = cfg.results_root() / cfg.control_name / out_stage_name / "resonancias"
    if list(control_out.rglob("*_wavelet_energy.nii.gz")):
        return control_out
    res_dir = first_existing_dir(control_stage_dir, cfg.resonance_dirnames)
    if not res_dir and any(norm_key(x) in {"resonancia", "resonancias", "mri", "fmri"} for x in control_stage_dir.parts):
        res_dir = control_stage_dir
    if res_dir:
        try:
            process_resonance_folder(cfg, res_dir, control_out)
            return control_out
        except Exception as exc:
            log_exception(log_file, f"Procesando control sano {stage}", exc)
    return None


def _find_best_energy_by_role(root: Path, role: str, subtype: str = "", side: str = "") -> list[Path]:
    if not root.exists():
        return []
    candidates = sorted(root.rglob("*_wavelet_energy.nii.gz"))
    out = []
    for p in candidates:
        n = norm_key(str(p))
        if role and role not in n:
            continue
        if subtype and subtype not in n:
            continue
        if side and side != "desconocido" and side not in n:
            continue
        out.append(p)
    return out


def run_maps(cfg: PipelineConfig) -> pd.DataFrame:
    rows = []
    for patient, stage, stage_dir in _iter_imaging_subject_stage(cfg):
        print(f"\n[MAPAS] {patient} · {stage}")
        out_dir = cfg.results_root() / patient / stage / "mapas"
        log_file = cfg.results_root() / patient / stage / "reportes" / "mapas_log.txt"
        try:
            maps_dir = find_maps_dir(stage_dir, cfg)
            if maps_dir is None:
                all_nii = find_nifti_files(stage_dir) if stage_dir else []
                append_log(
                    log_file,
                    f"No encontré mapas NIfTI para {patient} {stage}. "
                    f"stage_dir={stage_dir}; niftis_en_etapa={len(all_nii)}. "
                    "Revisa si están bajo ResultadosFuncional/Resultados funcional/MAPAS.",
                )
                continue
            df = process_maps_folder(cfg, maps_dir, out_dir)
            if not df.empty:
                df.insert(0, "stage", stage)
                df.insert(0, "patient", patient)
                rows.append(df)
        except Exception as exc:
            log_exception(log_file, f"Mapas {patient} {stage}", exc)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        save_dataframe(out, cfg.results_root() / "resumen_global_mapas.csv")
    return out


def run_resonances(cfg: PipelineConfig) -> pd.DataFrame:
    rows = []
    for patient, stage, stage_dir in _iter_imaging_subject_stage(cfg):
        print(f"\n[RESONANCIAS] {patient} · {stage}")
        out_dir = cfg.results_root() / patient / stage / "resonancias"
        log_file = cfg.results_root() / patient / stage / "reportes" / "resonancias_log.txt"
        try:
            res_dir = first_existing_dir(stage_dir, cfg.resonance_dirnames) if stage_dir else None
            if res_dir is None:
                append_log(log_file, f"No encontré carpeta RESONANCIA para {patient} {stage}; stage_dir={stage_dir}")
                continue
            df = process_resonance_folder(cfg, res_dir, out_dir)
            if not df.empty:
                df.insert(0, "stage", stage)
                df.insert(0, "patient", patient)
                rows.append(df)
        except Exception as exc:
            log_exception(log_file, f"Resonancias {patient} {stage}", exc)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        save_dataframe(out, cfg.results_root() / "resumen_global_resonancias.csv")
    return out


def run_resonance_diagnostic(cfg: PipelineConfig) -> pd.DataFrame:
    """Genera manifest de resonancias por paciente/etapa sin procesar PixelData completo."""
    rows = []
    for subject in list(cfg.patients) + [cfg.control_name]:
        for stage in cfg.stages:
            stage_dir = _subject_stage_dir(cfg, subject, stage)
            if not stage_dir:
                continue
            res_dir = first_existing_dir(stage_dir, cfg.resonance_dirnames) or stage_dir
            if not res_dir.exists():
                continue
            out_dir = cfg.results_root() / "_diagnostico_resonancias_v2" / subject / stage
            series = collect_dicom_series(res_dir)
            df = build_dicom_manifest(series, out_dir)
            if not df.empty:
                df.insert(0, "stage", stage)
                df.insert(0, "subject", subject)
                rows.append(df)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        save_dataframe(out, cfg.results_root() / "_diagnostico_resonancias_v2" / "manifest_global_resonancias.csv")
    return out



def _as_3d(vol: np.ndarray) -> np.ndarray:
    data = np.squeeze(np.asarray(vol, dtype=np.float32))
    if data.ndim > 3:
        data = np.nanmean(data, axis=-1)
    if data.ndim == 2:
        data = data[:, :, None]
    return np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _safe_pearson(a: np.ndarray, b: np.ndarray, cfg: PipelineConfig | None = None) -> float:
    gpu_val = pearson_gpu(a, b, cfg)
    if gpu_val is not None:
        return gpu_val

    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    finite = np.isfinite(aa) & np.isfinite(bb)
    if np.sum(finite) < 3:
        return math.nan
    aa = aa[finite]
    bb = bb[finite]
    if float(np.nanstd(aa)) < 1e-12 or float(np.nanstd(bb)) < 1e-12:
        return math.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def _slice_similarity_rows(pz: np.ndarray, cz: np.ndarray) -> pd.DataFrame:
    rows = []
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        ssim = None
    for z in range(pz.shape[2]):
        psl = np.asarray(pz[:, :, z], dtype=np.float32)
        csl = np.asarray(cz[:, :, z], dtype=np.float32)
        diff = psl - csl
        row = {
            "z": z,
            "pearson": _safe_pearson(psl, csl),
            "mae": float(np.nanmean(np.abs(diff))),
            "rmse": float(np.sqrt(np.nanmean(diff ** 2))),
            "paciente_mean": float(np.nanmean(psl)),
            "sano_mean": float(np.nanmean(csl)),
            "diff_mean": float(np.nanmean(diff)),
        }
        if ssim is not None:
            try:
                data_range = float(np.nanmax([psl.max(), csl.max()]) - np.nanmin([psl.min(), csl.min()]))
                row["ssim"] = float(ssim(psl, csl, data_range=data_range if data_range > 0 else 1.0))
            except Exception:
                row["ssim"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_slice_correlation(df: pd.DataFrame, out_png: Path, title: str) -> None:
    if df.empty:
        return
    safe_mkdir(out_png.parent)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df["z"], df["pearson"], lw=1.2, label="Pearson por corte")
    ax1.set_xlabel("Corte Z")
    ax1.set_ylabel("Pearson r")
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(df["z"], df["rmse"], lw=1.0, alpha=0.65, label="RMSE")
    ax2.set_ylabel("RMSE normalizado")
    ax1.set_title(title)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_image_to_control_volume(
    patient_path: Path,
    control_path: Path,
    out_dir: Path,
    label: str,
    normalize: str = "robust_zscore",
    cfg: PipelineConfig | None = None,
) -> dict:
    """Correlación imagen-volumen contra sano, con análisis corte por corte.

    Esta función aprovecha la idea del archivo de correlaciones original: correlación espacial
    volumen-a-volumen y comparación por cortes. Sirve para TAC, mapas funcionales y resonancias.
    """
    _, nib, _ = _require_neuro_libs()
    safe_mkdir(out_dir)
    label = sanitize_name(label, 80)
    p_img = nib.load(str(patient_path))
    c_img = nib.load(str(control_path))
    p = _as_3d(p_img.get_fdata(dtype=np.float32))
    c = _as_3d(c_img.get_fdata(dtype=np.float32))
    c_resized = resize_to_shape(c, p.shape, order=1, cfg=cfg)

    if normalize == "robust_zscore":
        p_cmp = robust_zscore(p, cfg=cfg)
        c_cmp = robust_zscore(c_resized, cfg=cfg)
    elif normalize == "minmax":
        def mm(x):
            gpu_out = minmax_gpu(x, cfg)
            if gpu_out is not None:
                return gpu_out
            x = np.asarray(x, dtype=np.float32)
            lo, hi = np.nanpercentile(x, [1, 99])
            return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
        p_cmp, c_cmp = mm(p), mm(c_resized)
    else:
        p_cmp, c_cmp = p, c_resized

    diff = p_cmp - c_cmp
    absdiff = np.abs(diff).astype(np.float32)
    save_nifti(diff, p_img.affine, out_dir / f"deficit_{label}.nii.gz")
    save_nifti(absdiff, p_img.affine, out_dir / f"diferencia_absoluta_{label}.nii.gz")
    save_orthogonal_png(diff, out_dir / f"deficit_{label}_ortogonal.png", title=f"Diferencia paciente - sano · {label}", cmap="coolwarm")
    save_topography_3d(diff, out_dir / f"deficit_{label}_topografia_3d.png", title=f"Topografía déficit · {label}")

    df_slices = _slice_similarity_rows(p_cmp, c_cmp)
    save_dataframe(df_slices, out_dir / f"correlacion_corte_a_corte_{label}.csv")
    _plot_slice_correlation(df_slices, out_dir / f"correlacion_corte_a_corte_{label}.png", f"Correlación corte a corte · {label}")

    metrics = {
        "label": label,
        "patient_volume": str(patient_path),
        "control_volume": str(control_path),
        "normalize": normalize,
        "shape_patient": tuple(int(x) for x in p.shape),
        "shape_control_original": tuple(int(x) for x in c.shape),
        "pearson_global": _safe_pearson(p_cmp, c_cmp, cfg=cfg),
        "gpu_backend": gpu_info().get("backend", "cpu") if cfg is not None else "cpu",
        "mae_global": float(np.nanmean(np.abs(diff))),
        "rmse_global": float(np.sqrt(np.nanmean(diff ** 2))),
        "deficit_mean": float(np.nanmean(diff)),
        "deficit_p05": float(np.nanpercentile(diff, 5)),
        "deficit_p95": float(np.nanpercentile(diff, 95)),
        "slice_pearson_mean": float(np.nanmean(df_slices["pearson"])) if not df_slices.empty else math.nan,
        "slice_pearson_min": float(np.nanmin(df_slices["pearson"])) if not df_slices.empty else math.nan,
        "slice_pearson_max": float(np.nanmax(df_slices["pearson"])) if not df_slices.empty else math.nan,
        "out_deficit_nii": str(out_dir / f"deficit_{label}.nii.gz"),
        "out_absdiff_nii": str(out_dir / f"diferencia_absoluta_{label}.nii.gz"),
        "out_slice_csv": str(out_dir / f"correlacion_corte_a_corte_{label}.csv"),
    }
    save_metrics_json(metrics, out_dir / f"metricas_{label}.json")
    return metrics


def compare_binary_mask_to_control(patient_mask_path: Path, control_mask_path: Path, out_dir: Path, label: str, cfg: PipelineConfig | None = None) -> dict:
    _, nib, _ = _require_neuro_libs()
    safe_mkdir(out_dir)
    label = sanitize_name(label, 80)
    p_img = nib.load(str(patient_mask_path))
    c_img = nib.load(str(control_mask_path))
    p = _as_3d(p_img.get_fdata(dtype=np.float32)) > 0.5
    c = _as_3d(c_img.get_fdata(dtype=np.float32)) > 0.5
    c_resized = resize_to_shape(c.astype(np.float32), p.shape, order=0, cfg=cfg) > 0.5
    inter = int(np.sum(p & c_resized))
    p_sum = int(np.sum(p))
    c_sum = int(np.sum(c_resized))
    union = int(np.sum(p | c_resized))
    dice = float((2 * inter) / (p_sum + c_sum + 1e-12))
    jaccard = float(inter / (union + 1e-12))
    labelmap = p.astype(np.uint8) + (c_resized.astype(np.uint8) * 2)
    save_nifti(labelmap, p_img.affine, out_dir / f"comparacion_mascara_{label}.nii.gz")
    save_orthogonal_png(labelmap, out_dir / f"comparacion_mascara_{label}.png", title=f"Máscara paciente/sano · {label}", cmap="viridis")
    metrics = {
        "label": label,
        "patient_mask": str(patient_mask_path),
        "control_mask": str(control_mask_path),
        "dice": dice,
        "jaccard": jaccard,
        "voxels_patient": p_sum,
        "voxels_control_resized": c_sum,
        "voxels_intersection": inter,
        "voxels_union": union,
        "nota_labelmap": "1=paciente, 2=sano, 3=intersección",
    }
    save_metrics_json(metrics, out_dir / f"metricas_mascara_{label}.json")
    return metrics


def _ensure_maps_result(cfg: PipelineConfig, subject: str, stage: str, log_file: Path) -> Path | None:
    out_dir = cfg.results_root() / subject / stage / "mapas"
    if list(out_dir.rglob("*_wavelet_energy.nii.gz")):
        return out_dir
    stage_dir = _subject_stage_dir(cfg, subject, stage)
    maps_dir = find_maps_dir(stage_dir, cfg)
    if maps_dir is None:
        append_log(log_file, f"No pude preparar mapas para {subject} {stage}; stage_dir={stage_dir}")
        return None
    try:
        process_maps_folder(cfg, maps_dir, out_dir)
    except Exception as exc:
        log_exception(log_file, f"Preparando mapas {subject} {stage}", exc)
    return out_dir if list(out_dir.rglob("*_wavelet_energy.nii.gz")) else None


def _ensure_tomography_result(cfg: PipelineConfig, subject: str, stage: str, log_file: Path) -> Path | None:
    out_dir = cfg.results_root() / subject / stage / "tomografia"
    if (out_dir / "tac_hu_original.nii.gz").exists():
        return out_dir
    stage_dir = _subject_stage_dir(cfg, subject, stage)
    tac_dir = first_existing_dir(stage_dir, cfg.ct_dirnames) if stage_dir else None
    if tac_dir is None:
        append_log(log_file, f"No pude preparar tomografía para {subject} {stage}; stage_dir={stage_dir}")
        return None
    try:
        # Import local para evitar circularidad al cargar módulos.
        from .tomography import process_tomography_folder
        process_tomography_folder(cfg, tac_dir, out_dir)
    except Exception as exc:
        log_exception(log_file, f"Preparando tomografía {subject} {stage}", exc)
    return out_dir if (out_dir / "tac_hu_original.nii.gz").exists() else None


def _ensure_resonance_result(cfg: PipelineConfig, subject: str, stage: str, log_file: Path) -> Path | None:
    out_dir = cfg.results_root() / subject / stage / "resonancias"
    if list(out_dir.rglob("*_wavelet_energy.nii.gz")):
        return out_dir
    stage_dir = _subject_stage_dir(cfg, subject, stage)
    res_dir = first_existing_dir(stage_dir, cfg.resonance_dirnames) if stage_dir else None
    if res_dir is None:
        append_log(log_file, f"No pude preparar resonancias para {subject} {stage}; stage_dir={stage_dir}")
        return None
    try:
        process_resonance_folder(cfg, res_dir, out_dir)
    except Exception as exc:
        log_exception(log_file, f"Preparando resonancias {subject} {stage}", exc)
    return out_dir if list(out_dir.rglob("*_wavelet_energy.nii.gz")) else None


def _control_stage_for_results(cfg: PipelineConfig, stage: str, modality: str) -> str:
    # Si existe sano/Despues con salidas, se usa para Despues. Si no, sano/Antes.
    requested = cfg.results_root() / cfg.control_name / stage / modality
    if requested.exists() and any(requested.rglob("*")):
        return stage
    return "Antes"


def _map_candidates_by_side(root: Path, side: str, prefer_energy: bool = True) -> list[Path]:
    if root is None or not root.exists():
        return []
    suffix = "*_wavelet_energy.nii.gz" if prefer_energy else "*_volumen.nii.gz"
    candidates = sorted(root.rglob(suffix))
    if side == "desconocido":
        return candidates
    out = []
    for p in candidates:
        n = norm_key(str(p))
        if side in n or side_from_text(p.name) == side:
            out.append(p)
    return out


def _compare_maps_for_patient_stage(cfg: PipelineConfig, patient: str, stage: str, rows: list[dict], log_file: Path) -> None:
    patient_root = _ensure_maps_result(cfg, patient, stage, log_file)
    control_stage = _control_stage_for_results(cfg, stage, "mapas")
    control_root = _ensure_maps_result(cfg, cfg.control_name, control_stage, log_file)
    if not patient_root or not control_root:
        append_log(log_file, f"Sin mapas paciente o sano para correlación {patient} {stage}")
        return
    ckpt = CheckpointManager(cfg)
    for side in ["derecha", "izquierda", "desconocido"]:
        p_candidates = _map_candidates_by_side(patient_root, side, prefer_energy=True)
        c_candidates = _map_candidates_by_side(control_root, side, prefer_energy=True)
        if not p_candidates or not c_candidates:
            continue
        corr_dir = cfg.results_root() / patient / stage / "correlaciones" / "mapas" / side
        label = sanitize_name(f"mapas_{side}_vs_sano", 60)
        metrics_json = corr_dir / f"metricas_{label}.json"
        task_id = f"correlaciones/mapas/{patient}/{stage}/{side}"

        def _work():
            metrics = compare_image_to_control_volume(p_candidates[0], c_candidates[0], corr_dir, label, normalize="robust_zscore", cfg=cfg)
            metrics.update({"patient": patient, "stage": stage, "modality": "mapas", "side": side, "control_stage": control_stage})
            save_metrics_json(metrics, metrics_json)
            return metrics

        try:
            result, status = ckpt.run(
                task_id=task_id,
                inputs=[p_candidates[0], c_candidates[0]],
                outputs=[metrics_json, corr_dir / f"deficit_{label}.nii.gz"],
                params={"modality": "mapas", "side": side, "v": "3_8_gpu"},
                fn=_work,
            )
            metrics = load_json_if_exists(metrics_json) if status == "skipped" else result
            if metrics:
                metrics["checkpoint_status"] = status
                rows.append(metrics)
        except Exception as exc:
            log_exception(log_file, f"Correlación mapas {side} {patient} {stage}", exc)


def _compare_tomography_for_patient_stage(cfg: PipelineConfig, patient: str, stage: str, rows: list[dict], log_file: Path) -> None:
    patient_root = _ensure_tomography_result(cfg, patient, stage, log_file)
    control_stage = _control_stage_for_results(cfg, stage, "tomografia")
    control_root = _ensure_tomography_result(cfg, cfg.control_name, control_stage, log_file)
    if not patient_root or not control_root:
        append_log(log_file, f"Sin tomografía paciente o sano para correlación {patient} {stage}")
        return

    p_ct = patient_root / "tac_hu_original.nii.gz"
    c_ct = control_root / "tac_hu_original.nii.gz"
    if not p_ct.exists() or not c_ct.exists():
        append_log(log_file, f"Falta tac_hu_original para correlación {patient} {stage}: p={p_ct.exists()} c={c_ct.exists()}")
        return

    ckpt = CheckpointManager(cfg)
    corr_dir = cfg.results_root() / patient / stage / "correlaciones" / "tomografia" / "imagen_por_imagen"
    label = "tac_hu_vs_sano"
    metrics_json = corr_dir / f"metricas_{label}.json"
    task_id = f"correlaciones/tomografia/volumen/{patient}/{stage}"

    def _work_ct():
        metrics = compare_image_to_control_volume(p_ct, c_ct, corr_dir, label, normalize="robust_zscore", cfg=cfg)
        metrics.update({"patient": patient, "stage": stage, "modality": "tomografia", "control_stage": control_stage})
        save_metrics_json(metrics, metrics_json)
        return metrics

    try:
        result, status = ckpt.run(
            task_id=task_id,
            inputs=[p_ct, c_ct],
            outputs=[metrics_json, corr_dir / f"deficit_{label}.nii.gz", corr_dir / f"correlacion_corte_a_corte_{label}.csv"],
            params={"modality": "tomografia", "kind": "imagen_por_imagen", "v": "3_8_gpu"},
            fn=_work_ct,
        )
        metrics = load_json_if_exists(metrics_json) if status == "skipped" else result
        if metrics:
            metrics["checkpoint_status"] = status
            rows.append(metrics)
    except Exception as exc:
        log_exception(log_file, f"Correlación TAC volumen {patient} {stage}", exc)

    # Comparación de máscaras musculares y de áreas/volúmenes.
    mask_names = [
        "mask_Vasto_Lateral_Der.nii.gz", "mask_Vasto_Medial_Der.nii.gz",
        "mask_Vasto_Lateral_Izq.nii.gz", "mask_Vasto_Medial_Izq.nii.gz",
        "mask_vastos_derecha.nii.gz", "mask_vastos_izquierda.nii.gz", "mask_todos_vastos.nii.gz",
    ]
    mask_rows = []
    for mask_name in mask_names:
        p_mask = patient_root / mask_name
        c_mask = control_root / mask_name
        if not p_mask.exists() or not c_mask.exists():
            continue
        label_mask = sanitize_name(mask_name.replace(".nii.gz", "") + "_vs_sano", 70)
        mask_dir = cfg.results_root() / patient / stage / "correlaciones" / "tomografia" / "mascaras"
        mask_metrics_json = mask_dir / f"metricas_mascara_{label_mask}.json"
        mask_task_id = f"correlaciones/tomografia/mascara/{patient}/{stage}/{label_mask}"

        def _work_mask(p_mask=p_mask, c_mask=c_mask, label_mask=label_mask):
            metrics = compare_binary_mask_to_control(p_mask, c_mask, mask_dir, label_mask, cfg=cfg)
            metrics.update({"patient": patient, "stage": stage, "modality": "tomografia_mascara", "control_stage": control_stage})
            save_metrics_json(metrics, mask_metrics_json)
            return metrics
        try:
            result, status = ckpt.run(
                task_id=mask_task_id,
                inputs=[p_mask, c_mask],
                outputs=[mask_metrics_json, mask_dir / f"comparacion_mascara_{label_mask}.nii.gz"],
                params={"modality": "tomografia_mascara", "mask": mask_name, "v": "3_8_gpu"},
                fn=_work_mask,
            )
            metrics = load_json_if_exists(mask_metrics_json) if status == "skipped" else result
            if metrics:
                metrics["checkpoint_status"] = status
                rows.append(metrics)
                mask_rows.append(metrics)
        except Exception as exc:
            log_exception(log_file, f"Correlación máscara TAC {mask_name} {patient} {stage}", exc)
    if mask_rows:
        save_dataframe(pd.DataFrame(mask_rows), cfg.results_root() / patient / stage / "correlaciones" / "tomografia" / "resumen_mascaras_tac.csv")

    # Comparación de métricas tabulares: volúmenes y áreas por corte.
    try:
        vol_p = patient_root / "volumenes_y_areas.csv"
        vol_c = control_root / "volumenes_y_areas.csv"
        if vol_p.exists() and vol_c.exists():
            dp = pd.read_csv(vol_p)
            dc = pd.read_csv(vol_c)
            merged = dp.merge(dc, on="musculo", suffixes=("_paciente", "_sano"), how="inner")
            if not merged.empty:
                for col in ["volumen_cm3", "area_transversal_max_cm2", "area_transversal_media_cm2"]:
                    a, b = f"{col}_paciente", f"{col}_sano"
                    if a in merged and b in merged:
                        merged[f"delta_{col}"] = merged[a] - merged[b]
                        merged[f"delta_pct_{col}"] = 100 * (merged[a] - merged[b]) / (merged[b].replace(0, np.nan))
                out_tab = cfg.results_root() / patient / stage / "correlaciones" / "tomografia" / "comparacion_volumenes_areas_vs_sano.csv"
                save_dataframe(merged, out_tab)
                rows.append({"patient": patient, "stage": stage, "modality": "tomografia_tabular", "label": "volumenes_areas_vs_sano", "rows": int(len(merged)), "out_csv": str(out_tab), "control_stage": control_stage})
        ar_p = patient_root / "areas_por_corte.csv"
        ar_c = control_root / "areas_por_corte.csv"
        if ar_p.exists() and ar_c.exists():
            dp = pd.read_csv(ar_p)
            dc = pd.read_csv(ar_c)
            area_rows = []
            for musculo in sorted(set(dp.get("musculo", [])) & set(dc.get("musculo", []))):
                ap = dp[dp["musculo"] == musculo].sort_values("z")["area_cm2"].to_numpy(dtype=float)
                ac = dc[dc["musculo"] == musculo].sort_values("z")["area_cm2"].to_numpy(dtype=float)
                if len(ap) and len(ac):
                    # Re-muestrear sano a la cantidad de cortes del paciente.
                    x_old = np.linspace(0, 1, len(ac))
                    x_new = np.linspace(0, 1, len(ap))
                    ac_rs = np.interp(x_new, x_old, ac)
                    area_rows.append({
                        "musculo": musculo,
                        "pearson_area_por_corte": _safe_pearson(ap, ac_rs),
                        "mae_area_cm2": float(np.nanmean(np.abs(ap - ac_rs))),
                        "rmse_area_cm2": float(np.sqrt(np.nanmean((ap - ac_rs) ** 2))),
                        "n_cortes_paciente": int(len(ap)),
                        "n_cortes_sano": int(len(ac)),
                    })
            if area_rows:
                out_area = cfg.results_root() / patient / stage / "correlaciones" / "tomografia" / "correlacion_areas_por_corte_vs_sano.csv"
                save_dataframe(pd.DataFrame(area_rows), out_area)
                rows.append({"patient": patient, "stage": stage, "modality": "tomografia_areas_por_corte", "label": "areas_por_corte_vs_sano", "rows": len(area_rows), "out_csv": str(out_area), "control_stage": control_stage})
    except Exception as exc:
        log_exception(log_file, f"Correlaciones tabulares TAC {patient} {stage}", exc)


def _compare_resonances_for_patient_stage(cfg: PipelineConfig, patient: str, stage: str, rows: list[dict], log_file: Path) -> None:
    patient_root = _ensure_resonance_result(cfg, patient, stage, log_file)
    control_stage = _control_stage_for_results(cfg, stage, "resonancias")
    control_root = _ensure_resonance_result(cfg, cfg.control_name, control_stage, log_file)
    if not patient_root or not control_root:
        append_log(log_file, f"Sin resonancias paciente o sano para correlación {patient} {stage}")
        return

    target_roles = [
        ("vce_explicita", "tracto_corticoespinal"),
        ("tractografia", "tractografia"),
        ("dti_difusion", "fa"),
        ("dti_difusion", "adc"),
        ("dti_difusion", "trace_b0"),
        ("dti_difusion", "rd"),
        ("dti_difusion", "ad"),
        ("fmri_motor", "mapa_activacion"),
        ("fmri_motor", "bold_raw"),
        ("anatomica", "t1"),
    ]
    ckpt = CheckpointManager(cfg)
    for role, subtype in target_roles:
        for side in ["derecha", "izquierda", "desconocido"]:
            p_candidates = _find_best_energy_by_role(patient_root, role, subtype, side)
            c_candidates = _find_best_energy_by_role(control_root, role, subtype, side)
            if not p_candidates or not c_candidates:
                continue
            label = sanitize_name(f"resonancia_{role}_{subtype}_{side}_vs_sano", 80)
            corr_dir = cfg.results_root() / patient / stage / "correlaciones" / "resonancias" / role / subtype / side
            metrics_json = corr_dir / f"metricas_{label}.json"
            task_id = f"correlaciones/resonancias/{patient}/{stage}/{role}/{subtype}/{side}"

            def _work():
                metrics = compare_image_to_control_volume(p_candidates[0], c_candidates[0], corr_dir, label, normalize="robust_zscore", cfg=cfg)
                metrics.update({"patient": patient, "stage": stage, "modality": "resonancias", "series_role": role, "series_subtype": subtype, "side": side, "control_stage": control_stage})
                save_metrics_json(metrics, metrics_json)
                return metrics

            try:
                result, status = ckpt.run(
                    task_id=task_id,
                    inputs=[p_candidates[0], c_candidates[0]],
                    outputs=[metrics_json, corr_dir / f"deficit_{label}.nii.gz"],
                    params={"role": role, "subtype": subtype, "side": side, "v": "3_8_gpu"},
                    fn=_work,
                )
                metrics = load_json_if_exists(metrics_json) if status == "skipped" else result
                if metrics:
                    metrics["checkpoint_status"] = status
                    rows.append(metrics)
            except Exception as exc:
                log_exception(log_file, f"Correlación resonancia {role}/{subtype}/{side} {patient} {stage}", exc)


def _find_ad_motor_cortex_csvs(root: Path) -> list[Path]:
    if root is None or not root.exists():
        return []
    return sorted(root.rglob('metricas_corteza_motora_ad.csv'))


def _compare_ad_motor_cortex_vs_control(cfg: PipelineConfig, patient: str, stage: str, rows: list[dict], log_file: Path) -> None:
    patient_root = _ensure_resonance_result(cfg, patient, stage, log_file)
    control_stage = _control_stage_for_results(cfg, stage, 'resonancias')
    control_root = _ensure_resonance_result(cfg, cfg.control_name, control_stage, log_file)
    if not patient_root or not control_root:
        return
    p_csvs = _find_ad_motor_cortex_csvs(patient_root)
    c_csvs = _find_ad_motor_cortex_csvs(control_root)
    if not p_csvs or not c_csvs:
        append_log(log_file, f'Sin CSV de corteza motora AD paciente o sano para correlación {patient} {stage}: p={len(p_csvs)} c={len(c_csvs)}')
        return
    try:
        dp = pd.concat([pd.read_csv(p) for p in p_csvs], ignore_index=True)
        dc = pd.concat([pd.read_csv(p) for p in c_csvs], ignore_index=True)
    except Exception as exc:
        log_exception(log_file, f'Leyendo CSV corteza motora AD vs sano {patient} {stage}', exc)
        return
    if dp.empty or dc.empty:
        return
    agg_cols = {'volume_ml': 'max', 'voxels': 'max', 'energy_mean': 'mean', 'energy_p95': 'mean'}
    gp = dp.groupby(['region', 'side'], dropna=False).agg(agg_cols).reset_index()
    gc = dc.groupby(['region', 'side'], dropna=False).agg(agg_cols).reset_index()
    merged = gp.merge(gc, on=['region', 'side'], how='outer', suffixes=('_paciente', '_sano'))
    if merged.empty:
        return
    for col in ['volume_ml', 'voxels', 'energy_mean', 'energy_p95']:
        pcol = f'{col}_paciente'
        ccol = f'{col}_sano'
        if pcol in merged.columns and ccol in merged.columns:
            merged[f'delta_{col}_paciente_menos_sano'] = pd.to_numeric(merged[pcol], errors='coerce') - pd.to_numeric(merged[ccol], errors='coerce')
            denom = pd.to_numeric(merged[ccol], errors='coerce').replace(0, np.nan)
            merged[f'delta_pct_{col}'] = 100.0 * merged[f'delta_{col}_paciente_menos_sano'] / denom
    out_dir = cfg.results_root() / patient / stage / 'correlaciones' / 'resonancias' / 'corteza_motora_ad'
    out_csv = out_dir / 'comparacion_corteza_motora_ad_vs_sano.csv'
    save_dataframe(merged, out_csv)
    rows.append({'patient': patient, 'stage': stage, 'modality': 'resonancias_corteza_motora_ad_vs_sano', 'label': 'corteza_motora_ad_vs_sano', 'rows': int(len(merged)), 'out_csv': str(out_csv), 'control_stage': control_stage})


def _compare_internal_morphometry_vs_control(cfg: PipelineConfig, patient: str, stage: str, rows: list[dict], log_file: Path) -> None:
    p_csv = cfg.results_root() / patient / stage / "morfometria" / "resumen_morfometria_interna.csv"
    control_stage = _control_stage_for_results(cfg, stage, "morfometria")
    c_csv = cfg.results_root() / cfg.control_name / control_stage / "morfometria" / "resumen_morfometria_interna.csv"
    if not p_csv.exists() or not c_csv.exists():
        append_log(log_file, f"Sin morfometría interna paciente o sano para correlación {patient} {stage}: p={p_csv.exists()} c={c_csv.exists()}")
        return
    try:
        dp = pd.read_csv(p_csv)
        dc = pd.read_csv(c_csv)
        keys = [k for k in ["region", "region_name", "side"] if k in dp.columns and k in dc.columns]
        if not {"region", "side"}.issubset(set(keys)):
            return
        gp = dp.groupby(keys, dropna=False).agg(volume_ml=("volume_ml", "max"), voxels=("voxels", "max"), energy_mean=("energy_mean", "mean"), energy_p95=("energy_p95", "mean")).reset_index()
        gc = dc.groupby(keys, dropna=False).agg(volume_ml=("volume_ml", "max"), voxels=("voxels", "max"), energy_mean=("energy_mean", "mean"), energy_p95=("energy_p95", "mean")).reset_index()
        merged = gp.merge(gc, on=keys, how="outer", suffixes=("_paciente", "_sano"))
        if merged.empty:
            return
        for col in ["volume_ml", "voxels", "energy_mean", "energy_p95"]:
            pcol = f"{col}_paciente"; ccol = f"{col}_sano"
            if pcol in merged.columns and ccol in merged.columns:
                merged[f"delta_{col}_paciente_menos_sano"] = pd.to_numeric(merged[pcol], errors="coerce") - pd.to_numeric(merged[ccol], errors="coerce")
                denom = pd.to_numeric(merged[ccol], errors="coerce").replace(0, np.nan)
                merged[f"delta_pct_{col}"] = 100.0 * merged[f"delta_{col}_paciente_menos_sano"] / denom
        out_dir = cfg.results_root() / patient / stage / "correlaciones" / "morfometria_interna"
        out_csv = out_dir / "comparacion_morfometria_interna_vs_sano.csv"
        save_dataframe(merged, out_csv)
        rows.append({"patient": patient, "stage": stage, "modality": "morfometria_interna_vs_sano", "label": "morfometria_interna_vs_sano", "rows": int(len(merged)), "out_csv": str(out_csv), "control_stage": control_stage})
    except Exception as exc:
        log_exception(log_file, f"Correlación morfometría interna vs sano {patient} {stage}", exc)


def run_volume_correlations(cfg: PipelineConfig) -> pd.DataFrame:
    """Correlaciones completas contra sano.

    Incluye:
    1) Mapas funcionales paciente vs sano por lado.
    2) TAC de muslos imagen por imagen, máscaras, volúmenes y áreas transversales.
    3) Resonancias por rol: VCE, tractografía, DTI, fMRI motor y T1.

    Si las salidas del sano aún no existen, las calcula antes de correlacionar.
    """
    rows: list[dict] = []
    for patient in cfg.patients:
        for stage in cfg.stages:
            print(f"\n[CORRELACIONES] {patient} · {stage} vs {cfg.control_name}")
            log_file = cfg.results_root() / patient / stage / "reportes" / "correlaciones_log.txt"
            try:
                _compare_maps_for_patient_stage(cfg, patient, stage, rows, log_file)
                _compare_tomography_for_patient_stage(cfg, patient, stage, rows, log_file)
                _compare_resonances_for_patient_stage(cfg, patient, stage, rows, log_file)
                _compare_ad_motor_cortex_vs_control(cfg, patient, stage, rows, log_file)
                _compare_internal_morphometry_vs_control(cfg, patient, stage, rows, log_file)
            except Exception as exc:
                log_exception(log_file, f"Correlaciones globales {patient} {stage}", exc)

    df = pd.DataFrame(rows)
    if not df.empty:
        save_dataframe(df, cfg.results_root() / "resumen_global_correlaciones_volumenes.csv")
        try:
            save_dataframe(df, cfg.results_root() / "resumen_global_correlaciones_volumenes.xlsx")
        except Exception:
            pass
    return df
