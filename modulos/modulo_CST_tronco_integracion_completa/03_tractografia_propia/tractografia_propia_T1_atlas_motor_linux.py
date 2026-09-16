"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   TRACTOGRAFIA PROPIA T1 ATLAS MOTOR LINUX                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: modulos/modulo_CST_tronco_integracion_completa/03_tractografia_propia/tractografia_propia_T1_atlas_motor_linux.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Reconstruye tractografía propia desde DWI y atlas motor en T1. El modelo de
difusión
parte del tensor/campo de orientación estimado con gradientes bvec/bval. Las
streamlines se propagan siguiendo direcciones locales bajo criterios de
anisotropía,
curvatura y longitud, y luego se transforman al espacio anatómico del
paciente.

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

import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label as ndi_label,
)

from dipy.align.imaffine import (
    AffineMap,
    AffineRegistration,
    MutualInformationMetric,
    transform_centers_of_mass,
)
from dipy.align.transforms import (
    AffineTransform3D,
    RigidTransform3D,
    TranslationTransform3D,
)
from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst import dti
from dipy.reconst.csdeconv import (
    ConstrainedSphericalDeconvModel,
    auto_response_ssst,
)
from dipy.segment.mask import median_otsu
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines, transform_streamlines


# ============================================================
# CONFIGURACIÓN DEL USUARIO
# ============================================================

DICOM_ROOT = Path(os.environ.get("VCE_DICOM_ROOT", r"D:\EAFIT\01-2026\proyecto\datos\paciente 6\Antes\RESONANCIA"))

RT1_PATH = Path(os.environ.get("VCE_T1_PATH", r"D:\EAFIT\01-2026\proyecto\datos\paciente 6\Antes\Resultados funcional\REFORMATEO\rT1.nii"))

OUTPUT_DIR = Path(os.environ.get("VCE_TRACTOGRAPHY_DIR", r"D:\EAFIT\01-2026\proyecto\resultados\paciente 6\Antes\tractografia_propia"))

# Si dcm2niix está en el PATH, déjalo en None.
# Si no, pon algo como r"C:\Herramientas\dcm2niix.exe"
DCM2NIIX_EXE = os.environ.get("DCM2NIIX_EXE") or None

# Calidad / tamaño de la tractografía
SEED_DENSITY = 1          # 1 = ligero, 2 = más denso
STEP_SIZE_MM = 0.5
FA_STOP = 0.15
MIN_STREAMLINE_POINTS = 20
MAX_STREAMLINE_ANGLE = 30.0

# Suavizado de la densidad solo para el mapa "bonito" (no para máscara dura)
DENSITY_SMOOTH_SIGMA = 0.8

# Exportar RGB en T1
RGB_SCALE_TO_255 = True

# ------------------------------------------------------------
# ATLAS MOTOR Y ROI PARA EXTRAER CST
# ------------------------------------------------------------
# Prioridad automática:
# 1. ROI explícitas indicadas abajo.
# 2. Máscaras finales FreeSurfer producidas por la suite VCE.
# 3. aparc+aseg de FreeSurfer: M1=precentral; M2 aproximada=
#    caudalmiddlefrontal + superiorfrontal + paracentral.
# 4. Máscaras *_atlas_refinada.nii.gz de la suite.
# 5. Atlas-prior heurístico en el rT1 como respaldo.

AUTO_MOTOR_ATLAS = True
# Pon True para generar/revisar solo M1, M2 y tronco en T1 sin calcular tractografía.
ONLY_PREPARE_MOTOR_ATLAS = False

PROJECT_ROOT = Path(os.environ.get("VCE_PROJECT_ROOT", r"D:\EAFIT\01-2026\proyecto"))
_VCE_PATIENT_ID = os.environ.get("VCE_PATIENT_ID", "paciente 6")
_VCE_STAGE = os.environ.get("VCE_STAGE", "Antes")
ATLAS_SEARCH_ROOTS = [
    PROJECT_ROOT / "resultados" / _VCE_PATIENT_ID / _VCE_STAGE,
    PROJECT_ROOT / "datos" / _VCE_PATIENT_ID / _VCE_STAGE,
    PROJECT_ROOT / "resultados" / "derivados_externos",
    PROJECT_ROOT / "derivatives",
]

# Overrides opcionales. Si existen, tienen prioridad sobre la búsqueda automática.
LEFT_M1_SEED = None
RIGHT_M1_SEED = None
LEFT_M2_SEED = None
RIGHT_M2_SEED = None
BRAINSTEM_WAYPOINT = None
LEFT_CST_INCLUDE = None
RIGHT_CST_INCLUDE = None
EXCLUSION_MASK = None

# La corteza se expande hacia la sustancia blanca para que las streamlines
# puedan tocar la ROI aun cuando se detengan antes de entrar a la sustancia gris.
MOTOR_SEED_DILATION_MM = float(os.environ.get("VCE_MOTOR_SEED_DILATION_MM", "10.0"))
BRAINSTEM_DILATION_MM = float(os.environ.get("VCE_BRAINSTEM_DILATION_MM", "8.0"))
MIN_CST_LENGTH_MM = float(os.environ.get("VCE_MIN_CST_LENGTH_MM", "35.0"))
MIN_DESCENDING_Z_RANGE_MM = float(os.environ.get("VCE_MIN_DESCENDING_Z_RANGE_MM", "22.0"))
MIN_IPSILATERAL_FRACTION = float(os.environ.get("VCE_MIN_IPSILATERAL_FRACTION", "0.45"))

# Rescate anatómico: si el cruce voxel exacto M1+tronco queda en 0 por registro/FOV,
# se acepta cercanía en mm a la ROI cortical y al tronco. Debe validarse visualmente.
CST_RESCUE_ENABLE = os.environ.get("VCE_CST_RESCUE_ENABLE", "1") != "0"
CST_RESCUE_MOTOR_DISTANCE_MM = float(os.environ.get("VCE_CST_RESCUE_MOTOR_DISTANCE_MM", "18.0"))
CST_RESCUE_BRAINSTEM_DISTANCE_MM = float(os.environ.get("VCE_CST_RESCUE_BRAINSTEM_DISTANCE_MM", "20.0"))
CST_RESCUE_TOP_N = int(os.environ.get("VCE_CST_RESCUE_TOP_N", "2500"))

# Etiquetas Desikan-Killiany usadas por la suite VCE.
FS_MOTOR_LABELS = {
    "m1_left": [1024],
    "m1_right": [2024],
    "m2_left": [1003, 1028, 1017],
    "m2_right": [2003, 2028, 2017],
}
FS_BRAINSTEM_LABEL = 16


# ============================================================
# UTILIDADES
# ============================================================

def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def first_existing_path(paths: Iterable[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and Path(path).exists():
            return Path(path)
    return None


def resolve_dcm2niix() -> str:
    if DCM2NIIX_EXE:
        if Path(DCM2NIIX_EXE).exists():
            return str(DCM2NIIX_EXE)
        raise FileNotFoundError(
            f"No existe DCM2NIIX_EXE: {DCM2NIIX_EXE}"
        )

    found = shutil.which("dcm2niix")
    if found:
        return found

    raise FileNotFoundError(
        "No se encontró dcm2niix en el PATH. "
        "Instálalo o define DCM2NIIX_EXE."
    )


def save_nifti_like(
    reference_img: nib.spatialimages.SpatialImage,
    data: np.ndarray,
    out_path: Path,
    dtype=np.float32,
) -> None:
    header = reference_img.header.copy()
    img = nib.Nifti1Image(
        np.asarray(data, dtype=dtype),
        reference_img.affine,
        header,
    )
    nib.save(img, str(out_path))


def load_scalar(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    return img, data




def _safe_binary_from_intensity(vol: np.ndarray, min_voxels: int = 100) -> np.ndarray:
    """Crea una máscara robusta rápida para QC/registro COM.

    No reemplaza skull stripping clínico. Solo sirve para estimar centro de masa
    y detectar desplazamientos globales b0->T1.
    """
    arr = np.asarray(vol, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    positive = arr[arr > 0]
    if positive.size < min_voxels:
        return arr > np.percentile(arr, 90)
    lo = np.percentile(positive, 10)
    hi = np.percentile(positive, 99)
    if hi <= lo:
        mask = arr > lo
    else:
        norm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
        # Umbral conservador: incluye cerebro, excluye gran parte del fondo.
        mask = norm > 0.08
    try:
        mask = binary_closing(mask, iterations=1)
        mask = binary_fill_holes(mask)
        lbl, n = ndi_label(mask)
        if n > 0:
            counts = np.bincount(lbl.ravel())
            counts[0] = 0
            keep = counts.argmax()
            mask = lbl == keep
    except Exception:
        pass
    return np.asarray(mask, dtype=bool)


def _center_of_mass_world(mask: np.ndarray, affine: np.ndarray) -> np.ndarray | None:
    idx = np.argwhere(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return None
    center_vox = idx.mean(axis=0)
    return np.asarray(nib.affines.apply_affine(affine, center_vox), dtype=float)


def _apply_static_translation_to_affine(aff: np.ndarray, delta_world: np.ndarray) -> np.ndarray:
    corr = np.eye(4, dtype=float)
    corr[:3, 3] = np.asarray(delta_world, dtype=float)[:3]
    return corr @ np.asarray(aff, dtype=float)


def _make_affinemap(aff: np.ndarray, t1_img: nib.Nifti1Image, b0_img: nib.Nifti1Image) -> AffineMap:
    return AffineMap(
        np.asarray(aff, dtype=float),
        domain_grid_shape=t1_img.shape[:3],
        domain_grid2world=t1_img.affine,
        codomain_grid_shape=b0_img.shape[:3],
        codomain_grid2world=b0_img.affine,
    )


def _refine_b0_to_t1_affine_by_com(
    initial_affine: np.ndarray,
    b0_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
    static_norm: np.ndarray,
    moving_norm: np.ndarray,
    out_dir: Path,
) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Corrige desplazamiento global residual de b0->T1 por centro de masa.

    Motivo: en algunos rAnatomico/rT1 el registro MI deja las streamlines
    desplazadas superiormente. Si el b0 transformado queda arriba/abajo del T1,
    la CST no cae sobre bulbo/mesencéfalo. Esta corrección estima el centro de
    masa del cerebro T1 y del b0 ya registrado, y aplica una traslación global.

    Variables:
      VCE_REGISTRO_COM_CORREGIR=1/0           (default 1)
      VCE_REGISTRO_COM_EJES=xyz|z|xy|none     (default z)
      VCE_REGISTRO_COM_MAX_MM=60              (default 60)
      VCE_REGISTRO_COM_MIN_MM=3               (default 3)
    """
    enabled = os.environ.get("VCE_REGISTRO_COM_CORREGIR", "1").strip().lower() not in ("0", "false", "no", "off")
    axes = os.environ.get("VCE_REGISTRO_COM_EJES", "z").strip().lower()
    max_mm = float(os.environ.get("VCE_REGISTRO_COM_MAX_MM", "60"))
    min_mm = float(os.environ.get("VCE_REGISTRO_COM_MIN_MM", "3"))

    amap = _make_affinemap(initial_affine, t1_img, b0_img)
    transformed = amap.transform(moving_norm, interpolation="linear")
    transformed_img = nib.Nifti1Image(transformed.astype(np.float32), t1_img.affine, t1_img.header.copy())

    report = {
        "enabled": bool(enabled),
        "axes": axes,
        "max_mm": max_mm,
        "min_mm": min_mm,
        "applied": False,
        "delta_world_mm_raw": None,
        "delta_world_mm_used": [0.0, 0.0, 0.0],
        "note": "",
    }

    if not enabled or axes in ("none", "no", "0"):
        nib.save(transformed_img, str(out_dir / "b0_en_T1.nii.gz"))
        np.savetxt(out_dir / "affine_b0_to_T1.txt", initial_affine)
        (out_dir / "registro_b0_T1_qc.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return np.asarray(initial_affine, dtype=float), transformed_img

    static_mask = _safe_binary_from_intensity(static_norm)
    moving_mask = _safe_binary_from_intensity(transformed)
    c_static = _center_of_mass_world(static_mask, t1_img.affine)
    c_moving = _center_of_mass_world(moving_mask, t1_img.affine)

    if c_static is None or c_moving is None:
        report["note"] = "No se pudo calcular centro de masa para T1 o b0 transformado. Se conserva affine inicial."
        nib.save(transformed_img, str(out_dir / "b0_en_T1.nii.gz"))
        np.savetxt(out_dir / "affine_b0_to_T1.txt", initial_affine)
        (out_dir / "registro_b0_T1_qc.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return np.asarray(initial_affine, dtype=float), transformed_img

    delta = c_static - c_moving
    raw = delta.copy()
    used = np.zeros(3, dtype=float)
    if "x" in axes:
        used[0] = delta[0]
    if "y" in axes:
        used[1] = delta[1]
    if "z" in axes:
        used[2] = delta[2]

    norm = float(np.linalg.norm(used))
    report.update({
        "center_t1_world_mm": [float(x) for x in c_static],
        "center_b0_registered_world_mm": [float(x) for x in c_moving],
        "delta_world_mm_raw": [float(x) for x in raw],
        "delta_world_mm_used": [float(x) for x in used],
        "delta_norm_mm": norm,
    })

    if norm < min_mm:
        report["note"] = "Desplazamiento menor al mínimo; no se aplica corrección COM."
        nib.save(transformed_img, str(out_dir / "b0_en_T1.nii.gz"))
        np.savetxt(out_dir / "affine_b0_to_T1.txt", initial_affine)
        (out_dir / "registro_b0_T1_qc.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return np.asarray(initial_affine, dtype=float), transformed_img

    if norm > max_mm:
        report["note"] = "Desplazamiento supera máximo permitido; posible registro fallido. Se conserva affine inicial."
        nib.save(transformed_img, str(out_dir / "b0_en_T1.nii.gz"))
        np.savetxt(out_dir / "affine_b0_to_T1.txt", initial_affine)
        (out_dir / "registro_b0_T1_qc.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return np.asarray(initial_affine, dtype=float), transformed_img

    refined_affine = _apply_static_translation_to_affine(initial_affine, used)
    refined_map = _make_affinemap(refined_affine, t1_img, b0_img)
    transformed_refined = refined_map.transform(moving_norm, interpolation="linear")
    refined_img = nib.Nifti1Image(transformed_refined.astype(np.float32), t1_img.affine, t1_img.header.copy())

    nib.save(refined_img, str(out_dir / "b0_en_T1.nii.gz"))
    nib.save(refined_img, str(out_dir / "b0_en_T1_refinado_COM.nii.gz"))
    np.savetxt(out_dir / "affine_b0_to_T1.txt", refined_affine)
    np.savetxt(out_dir / "affine_b0_to_T1_inicial.txt", initial_affine)
    np.savetxt(out_dir / "affine_b0_to_T1_correccion_COM.txt", _apply_static_translation_to_affine(np.eye(4), used))

    report["applied"] = True
    report["note"] = "Corrección COM aplicada después de MI/rigid/affine. Validar b0_en_T1 sobre rAnatomico."
    (out_dir / "registro_b0_T1_qc.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return refined_affine, refined_img

def normalize_to_uint8(volume: np.ndarray) -> np.ndarray:
    vol = np.asarray(volume, dtype=np.float32)
    if np.all(vol == 0):
        return np.zeros_like(vol, dtype=np.uint8)

    vmax = np.percentile(vol[vol > 0], 99.5) if np.any(vol > 0) else 1.0
    vmax = max(float(vmax), 1e-6)
    vol = np.clip(vol / vmax, 0.0, 1.0)
    return np.round(vol * 255.0).astype(np.uint8)


@dataclass
class ConvertedStudy:
    dwi_nii: Path
    dwi_bval: Path
    dwi_bvec: Path
    t1_from_dicom: Path | None
    all_niftis: list[Path]


# ============================================================
# CONVERSIÓN DICOM -> NIFTI
# ============================================================

def _nii_sidecars(nii: Path) -> tuple[Path, Path]:
    """Devuelve bval/bvec asociados a .nii o .nii.gz."""
    bval = Path(str(nii).replace(".nii.gz", ".bval").replace(".nii", ".bval"))
    bvec = Path(str(nii).replace(".nii.gz", ".bvec").replace(".nii", ".bvec"))
    return bval, bvec


def _collect_niftis(out_dir: Path) -> list[Path]:
    """Recoge NIfTI generados en salida plana o en subcarpetas de fallback."""
    items = list(out_dir.glob("*.nii")) + list(out_dir.glob("*.nii.gz"))
    items += list(out_dir.rglob("*.nii")) + list(out_dir.rglob("*.nii.gz"))
    # Deduplicar conservando orden
    seen = set()
    out = []
    for f in sorted(items):
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def _dicom_leaf_folders(dicom_root: Path) -> list[Path]:
    """Carpetas hoja con archivos DICOM reales.

    No convierte toda la carpeta RESONANCIA: solo identifica subcarpetas
    candidatas a contener una serie DWI/DTI cruda.
    """
    dicom_root = Path(dicom_root)
    folders: list[Path] = []
    if not dicom_root.exists():
        return folders
    for d in [dicom_root] + [p for p in dicom_root.rglob("*") if p.is_dir()]:
        try:
            files = [f for f in d.iterdir() if f.is_file()]
        except Exception:
            continue
        if not files:
            continue
        d_text = str(d).lower()
        if any(bad in d_text for bad in ["01_nifti_convertidos", "tractografia_propia", "resultados"]):
            continue
        useful = [f for f in files if f.name.upper() not in {"VERSION", "DICOMDIR"}]
        if useful:
            folders.append(d)
    return sorted(set(folders), key=lambda x: str(x))


def _safe_dicom_read(path: Path):
    try:
        import pydicom
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None


def _get_dicom_value(ds, name: str, default=""):
    try:
        value = getattr(ds, name, default)
        return "" if value is None else str(value)
    except Exception:
        return default


def _get_diffusion_bvalue(ds):
    if ds is None:
        return None
    for tag in [(0x0018, 0x9087), (0x0019, 0x100C), (0x0043, 0x1039)]:
        try:
            if tag in ds:
                v = ds[tag].value
                if isinstance(v, (list, tuple)):
                    v = v[0]
                if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
                    v = list(v)[0]
                return float(v)
        except Exception:
            continue
    return None


def _score_dwi_dicom_folder(folder: Path, dicom_root: Path | None = None) -> dict:
    """Puntúa una subcarpeta como posible DWI/DTI cruda.

    Para la tractografía propia se necesita solo DWI/DTI + bval/bvec; el
    rAnatomico/rT1 ya se busca por separado en ResultadosFuncional.
    """
    folder = Path(folder)
    try:
        files = [f for f in folder.iterdir() if f.is_file() and f.name.upper() not in {"VERSION", "DICOMDIR"}]
    except Exception:
        files = []
    headers = []
    for f in files[:min(len(files), 16)]:
        ds = _safe_dicom_read(f)
        if ds is not None:
            headers.append(ds)
    text_parts = [str(folder).lower()]
    rows = cols = None
    series_uid = ""
    bvalues = []
    modality = ""
    for ds in headers:
        modality = modality or _get_dicom_value(ds, "Modality", "")
        series_uid = series_uid or _get_dicom_value(ds, "SeriesInstanceUID", "")
        for attr in ["SeriesDescription", "ProtocolName", "SequenceName", "ScanningSequence", "ImageType"]:
            text_parts.append(_get_dicom_value(ds, attr, "").lower())
        if rows is None:
            try:
                rows = int(getattr(ds, "Rows", 0) or 0)
                cols = int(getattr(ds, "Columns", 0) or 0)
            except Exception:
                rows = cols = None
        bv = _get_diffusion_bvalue(ds)
        if bv is not None:
            bvalues.append(float(bv))
    text = " ".join(text_parts)
    score = 0
    reasons = []
    for kw in ["dwi", "dti", "diff", "difusion", "difusión", "diffusion", "tensor", "30dir", "64dir", "ep2d_diff", "ep2d dti"]:
        if kw in text:
            score += 35; reasons.append(f"kw:{kw}")
    for kw in ["b1000", "b=1000", "trace", "adc", "fa"]:
        if kw in text:
            score += 8; reasons.append(f"kw_weak:{kw}")
    for kw in ["tractografia", "tractography", "syngo", "capture", "secondary", "scout", "localizer", "survey", "mpr", "mip", "bold", "fmri", "rest", "field", "swi", "tof", "t1", "t2", "flair"]:
        if kw in text:
            score -= 25; reasons.append(f"penalty:{kw}")
    if str(modality).upper() == "MR":
        score += 8; reasons.append("MR")
    if len(files) >= 20:
        score += 10; reasons.append("nfiles>=20")
    if len(files) >= 60:
        score += 15; reasons.append("nfiles>=60")
    if len(files) >= 120:
        score += 10; reasons.append("nfiles>=120")
    if bvalues:
        score += 40; reasons.append("dicom_bvalues")
        if max(bvalues) > 50:
            score += 40; reasons.append("b_nonzero")
        if len(set(round(x) for x in bvalues)) >= 2:
            score += 20; reasons.append("b_multiple")
    if rows and cols and max(rows, cols) >= 1000 and len(files) < 80:
        score -= 40; reasons.append("large_secondary_like")
    rel = str(folder)
    if dicom_root is not None:
        try:
            rel = str(folder.relative_to(dicom_root))
        except Exception:
            pass
    return {"folder": str(folder), "relative": rel, "score": int(score), "n_files": int(len(files)), "modality": str(modality), "series_uid": str(series_uid), "rows": rows, "cols": cols, "bvalues_sample": bvalues[:20], "reasons": reasons, "text_sample": text[:500]}


def _rank_dwi_dicom_folders(dicom_root: Path) -> list[dict]:
    forced = os.environ.get("VCE_DWI_DICOM_DIR", "").strip()
    if forced:
        item = _score_dwi_dicom_folder(Path(forced), dicom_root)
        item["forced_by_env"] = True
        item["score"] = 10000
        return [item]
    rows = [_score_dwi_dicom_folder(folder, dicom_root) for folder in _dicom_leaf_folders(dicom_root)]
    rows.sort(key=lambda r: (r.get("score", 0), r.get("n_files", 0)), reverse=True)
    return rows


def _write_json(path: Path, payload) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _score_dwi_series_group(files: list[Path], headers: list) -> dict:
    text_parts = []
    bvalues = []
    rows = cols = None
    uid = ""
    for ds in headers:
        uid = uid or _get_dicom_value(ds, "SeriesInstanceUID", "")
        for attr in ["SeriesDescription", "ProtocolName", "SequenceName", "ScanningSequence", "ImageType"]:
            text_parts.append(_get_dicom_value(ds, attr, "").lower())
        if rows is None:
            try:
                rows = int(getattr(ds, "Rows", 0) or 0)
                cols = int(getattr(ds, "Columns", 0) or 0)
            except Exception:
                rows = cols = None
        bv = _get_diffusion_bvalue(ds)
        if bv is not None:
            bvalues.append(float(bv))

    text = " ".join(text_parts)
    score = 0
    reasons = []
    for kw in ["dwi", "dti", "diff", "difusion", "difusión", "diffusion", "30dir", "64dir", "ep2d_diff", "ep_b1000", "b1000"]:
        if kw in text:
            score += 35; reasons.append(f"kw:{kw}")
    # Queremos la serie direccional cruda, no mapas derivados.
    for bad in ["trace", "tracew", "adc", "fa", "colfa", "tensor", "moco", "rest", "fmri", "motor", "tractografia", "postproceso", "scout", "localizer", "tof", "swi", "field", "t1 spc"]:
        if bad in text:
            score -= 30; reasons.append(f"penalty:{bad}")
    if "derived" in text or "secondary" in text:
        score -= 20; reasons.append("penalty:derived_secondary")
    if "original" in text or "primary" in text:
        score += 10; reasons.append("original_or_primary")
    if len(files) >= 20:
        score += 10; reasons.append("nfiles>=20")
    if len(files) >= 60:
        score += 10; reasons.append("nfiles>=60")
    if bvalues:
        score += 40; reasons.append("dicom_bvalues")
        if max(bvalues) > 50:
            score += 45; reasons.append("b_nonzero")
        if len(set(round(x) for x in bvalues)) >= 2:
            score += 25; reasons.append("b_multiple")
        if sum(1 for x in bvalues if x > 50) >= 6:
            score += 20; reasons.append("many_nonzero_b")
    return {
        "series_uid": uid,
        "score": int(score),
        "n_files": int(len(files)),
        "rows": rows,
        "cols": cols,
        "bvalues_sample": bvalues[:20],
        "reasons": reasons,
        "text_sample": text[:500],
    }


def _prepare_single_series_input(dicom_dir: Path, out_dir: Path, tag: str) -> Path:
    """Si una carpeta mezcla varias series, crea una carpeta temporal solo con la DWI.

    Algunas exportaciones Siemens dejan todas las series en una misma carpeta
    (por ejemplo miles de DICOM mezclados: resting, T1, fMRI, trace, DWI).
    dcm2niix, si recibe esa carpeta completa, convierte demasiado. Aquí se
    agrupa por SeriesInstanceUID y se crea una carpeta de enlaces simbólicos
    solo con la serie DWI/DTI cruda mejor puntuada.
    """
    if os.environ.get("VCE_DWI_FILTRAR_SERIE_UID", "1") != "1":
        return dicom_dir

    try:
        files = [f for f in Path(dicom_dir).iterdir() if f.is_file() and f.name.upper() not in {"VERSION", "DICOMDIR"}]
    except Exception:
        return dicom_dir
    if len(files) < 2:
        return dicom_dir

    groups: dict[str, dict] = {}
    for f in files:
        ds = _safe_dicom_read(f)
        if ds is None:
            continue
        uid = _get_dicom_value(ds, "SeriesInstanceUID", "") or _get_dicom_value(ds, "SOPClassUID", "") or "SIN_UID"
        g = groups.setdefault(uid, {"files": [], "headers": []})
        g["files"].append(f)
        if len(g["headers"]) < 24:
            g["headers"].append(ds)

    if len(groups) <= 1:
        return dicom_dir

    scored = []
    for uid, g in groups.items():
        row = _score_dwi_series_group(g["files"], g["headers"])
        row["series_uid"] = uid
        scored.append((row["score"], uid, row))
    scored.sort(key=lambda x: (x[0], x[2].get("n_files", 0)), reverse=True)

    debug = []
    for _, uid, row in scored:
        r = dict(row)
        r["series_uid"] = uid
        debug.append(r)
    _write_json(out_dir / f"dwi_series_groups_{tag}.json", debug)

    best_score, best_uid, best_row = scored[0]
    if best_score < int(os.environ.get("VCE_DWI_MIN_SERIES_UID_SCORE", "20")):
        log("    ADVERTENCIA: no se pudo aislar una serie DWI por UID con confianza; se usa la carpeta completa candidata.")
        return dicom_dir

    selected_files = groups[best_uid]["files"]
    if len(selected_files) == len(files):
        return dicom_dir

    link_dir = out_dir / f"_input_{tag}_solo_serie"
    if link_dir.exists():
        shutil.rmtree(link_dir, ignore_errors=True)
    ensure_dir(link_dir)
    for idx, src in enumerate(sorted(selected_files, key=lambda p: p.name), start=1):
        dst = link_dir / f"{idx:06d}_{src.name}"
        try:
            os.symlink(str(src), str(dst))
        except Exception:
            shutil.copy2(str(src), str(dst))

    log(f"    Serie DWI aislada por UID: {best_uid} | archivos={len(selected_files)}/{len(files)} | score={best_score}")
    return link_dir


def _run_dcm2niix_once(exe: str, dicom_dir: Path, out_dir: Path, prefix: str, tag: str) -> int:
    ensure_dir(out_dir)
    dicom_input = _prepare_single_series_input(Path(dicom_dir), out_dir, tag)
    safe_prefix = "".join(c if c.isalnum() or c in "_-" else "_" for c in prefix)[:80]
    cmd = [
        exe,
        "-z", "y",
        "-b", "y",
        "-m", "y",
        "-v", "y",
        "-f", f"{safe_prefix}__%p__%s",
        "-o", str(out_dir),
        str(dicom_input),
    ]
    log(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    (out_dir / f"dcm2niix_{tag}_stdout.txt").write_text(result.stdout, encoding="utf-8", errors="ignore")
    (out_dir / f"dcm2niix_{tag}_stderr.txt").write_text(result.stderr, encoding="utf-8", errors="ignore")
    return int(result.returncode)


def _find_dwi_candidates(all_niftis: list[Path]) -> list[tuple]:
    """Selecciona candidatos DWI con bval/bvec, priorizando 4D y direcciones no cero."""
    dwi_candidates = []
    debug_rows = []
    for nii in all_niftis:
        bval, bvec = _nii_sidecars(nii)
        row = {"nii": str(nii), "bval": str(bval), "bvec": str(bvec), "has_bval": bval.exists(), "has_bvec": bvec.exists()}
        if bval.exists() and bvec.exists():
            try:
                img = nib.load(str(nii))
                shape = img.shape
                bvals, bvecs = read_bvals_bvecs(str(bval), str(bvec))
                nonzero_dirs = int(np.sum(np.asarray(bvals) > 50))
                score = 0
                if len(shape) == 4:
                    score += 10
                    score += shape[3]
                score += nonzero_dirs
                # Penalizar mapas derivados no 4D aunque tengan sidecars raros.
                name = nii.name.lower()
                if any(k in name for k in ["adc", "fa", "trace", "colfa", "rgb"]):
                    score -= 10
                dwi_candidates.append((score, nii, bval, bvec, shape, nonzero_dirs))
                row.update({"shape": str(shape), "nonzero_dirs": nonzero_dirs, "score": score})
            except Exception as exc:
                row["error"] = str(exc)
        debug_rows.append(row)
    return dwi_candidates


def run_dcm2niix(dicom_root: Path, out_dir: Path) -> ConvertedStudy:
    """Convierte SOLO la serie DWI necesaria para tractografía.

    No convierte toda RESONANCIA por defecto. Primero identifica carpetas DWI/DTI
    crudas, convierte la mejor candidata y se detiene cuando obtiene una DWI 4D
    con .bval/.bvec. El T1 anatómico se toma de RT1_PATH.
    """
    if out_dir.exists() and os.environ.get("VCE_DWI_LIMPIAR_CONVERTIDOS", "1") == "1":
        shutil.rmtree(out_dir, ignore_errors=True)
    ensure_dir(out_dir)
    exe = resolve_dcm2niix()

    log("\n[1/8] Buscando SOLO la serie DWI/DTI necesaria para tractografía...")
    log(f"Raíz de búsqueda DICOM: {dicom_root}")

    ranked = _rank_dwi_dicom_folders(dicom_root)
    _write_json(out_dir / "dwi_dicom_folder_candidates.json", ranked)

    min_score = int(os.environ.get("VCE_DWI_MIN_FOLDER_SCORE", "20"))
    max_folders = int(os.environ.get("VCE_DWI_MAX_FOLDERS_TO_TRY", "12"))
    candidate_rows = [r for r in ranked if int(r.get("score", 0)) >= min_score]
    if not candidate_rows and os.environ.get("VCE_DWI_PERMITIR_CANDIDATOS_DEBILES", "0") == "1":
        candidate_rows = ranked[:max_folders]
    if not candidate_rows:
        raise RuntimeError(
            "No se encontró carpeta candidata a DWI/DTI crudo. Revisa "
            "dwi_dicom_folder_candidates.json o define manualmente: "
            "export VCE_DWI_DICOM_DIR='/ruta/a/la/serie_DWI'."
        )

    candidate_rows = candidate_rows[:max_folders]
    log(f"Carpetas DWI candidatas a convertir: {len(candidate_rows)}")
    best_candidate = None

    for idx, row in enumerate(candidate_rows, start=1):
        folder = Path(row["folder"])
        rel = row.get("relative", str(folder))
        score = row.get("score", "")
        log(f"  [{idx}/{len(candidate_rows)}] Convirtiendo SOLO candidata DWI score={score}: {rel}")
        sub_out = out_dir / ("DWI_candidata_%03d" % idx)
        code = _run_dcm2niix_once(exe, folder, sub_out, f"dwi{idx:03d}", f"dwi_candidata_{idx:03d}")
        if code != 0:
            log(f"    ADVERTENCIA: dcm2niix devolvió código {code} para esta candidata. Se revisan salidas parciales.")
        all_niftis = _collect_niftis(out_dir)
        dwi_candidates = _find_dwi_candidates(all_niftis)
        _write_json(
            out_dir / "dwi_nifti_candidates.json",
            [{"score": int(sc), "nii": str(nii), "bval": str(bval), "bvec": str(bvec), "shape": str(shape), "nonzero_dirs": int(nd)} for sc, nii, bval, bvec, shape, nd in sorted(dwi_candidates, key=lambda x: x[0], reverse=True)]
        )
        if dwi_candidates:
            dwi_candidates.sort(key=lambda x: x[0], reverse=True)
            cand = dwi_candidates[0]
            _, nii, bval, bvec, shape, nonzero_dirs = cand
            if len(shape) == 4 and nonzero_dirs >= int(os.environ.get("VCE_DWI_MIN_NONZERO_DIRS", "6")):
                best_candidate = cand
                log(f"    OK DWI válida encontrada: {nii.name} | shape={shape} | dirs_no_cero={nonzero_dirs}")
                break
            log(f"    NIfTI insuficiente para tractografía: shape={shape}, dirs_no_cero={nonzero_dirs}. Se prueba otra candidata.")

    if best_candidate is None:
        if os.environ.get("VCE_DWI_ALLOW_ROOT_CONVERSION", "0") == "1":
            log("ADVERTENCIA: VCE_DWI_ALLOW_ROOT_CONVERSION=1. Intentando raíz completa como último recurso.")
            root_out = out_dir / "_ultimo_recurso_raiz_completa"
            _run_dcm2niix_once(exe, dicom_root, root_out, "root", "root_last_resort")
            dwi_candidates = _find_dwi_candidates(_collect_niftis(out_dir))
            if dwi_candidates:
                dwi_candidates.sort(key=lambda x: x[0], reverse=True)
                best_candidate = dwi_candidates[0]
        if best_candidate is None:
            raise RuntimeError(
                "No se pudo obtener una DWI 4D válida con .bval/.bvec convirtiendo solo candidatas DWI. "
                "Revisa dwi_dicom_folder_candidates.json y dwi_nifti_candidates.json. "
                "Si sabes la carpeta exacta, usa VCE_DWI_DICOM_DIR."
            )

    _, dwi_nii, dwi_bval, dwi_bvec, dwi_shape, nonzero_dirs = best_candidate
    log(f"Serie DWI final seleccionada: {dwi_nii} | shape={dwi_shape} | dirs_no_cero={nonzero_dirs}")
    return ConvertedStudy(
        dwi_nii=Path(dwi_nii),
        dwi_bval=Path(dwi_bval),
        dwi_bvec=Path(dwi_bvec),
        t1_from_dicom=None,
        all_niftis=_collect_niftis(out_dir),
    )


# ============================================================
# REGISTRO B0 -> T1
# ============================================================

def register_b0_to_t1(
    b0_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
    out_dir: Path,
) -> tuple[np.ndarray, nib.Nifti1Image]:
    static = np.asarray(t1_img.dataobj, dtype=np.float32)
    moving = np.asarray(b0_img.dataobj, dtype=np.float32)

    # Normalización robusta
    static = (static - np.percentile(static, 1)) / max(np.percentile(static, 99) - np.percentile(static, 1), 1e-6)
    moving = (moving - np.percentile(moving, 1)) / max(np.percentile(moving, 99) - np.percentile(moving, 1), 1e-6)

    metric = MutualInformationMetric(nbins=32, sampling_proportion=None)
    level_iters = [1000, 100, 10]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    affreg = AffineRegistration(
        metric=metric,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors,
    )

    c_of_mass = transform_centers_of_mass(
        static,
        t1_img.affine,
        moving,
        b0_img.affine,
    )

    transform = TranslationTransform3D()
    params0 = None
    translation = affreg.optimize(
        static,
        moving,
        transform,
        params0,
        t1_img.affine,
        b0_img.affine,
        starting_affine=c_of_mass.affine,
    )

    transform = RigidTransform3D()
    rigid = affreg.optimize(
        static,
        moving,
        transform,
        params0,
        t1_img.affine,
        b0_img.affine,
        starting_affine=translation.affine,
    )

    transform = AffineTransform3D()
    affine = affreg.optimize(
        static,
        moving,
        transform,
        params0,
        t1_img.affine,
        b0_img.affine,
        starting_affine=rigid.affine,
    )

    # Transformación inicial por información mutua.
    initial_affine = np.asarray(affine.affine, dtype=float)

    # Corrección automática de desplazamiento global residual.
    # En los rAnatomico de la suite puede quedar un corrimiento superior/inferior
    # que desplaza las fibras hacia arriba y las saca del bulbo/mesencéfalo.
    refined_affine, transformed_img = _refine_b0_to_t1_affine_by_com(
        initial_affine,
        b0_img,
        t1_img,
        static,
        moving,
        out_dir,
    )

    return refined_affine, transformed_img


# ============================================================
# TRACTOGRAFÍA Y MAPAS
# ============================================================

def compute_color_from_streamlines(
    streamlines: Iterable[np.ndarray],
    target_shape: tuple[int, int, int],
    target_affine: np.ndarray,
    density_smooth_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve:
      density   -> (X, Y, Z) float32
      rgb_accum -> (X, Y, Z, 3) float32
    en el espacio del volumen target_affine/target_shape.
    """
    inv_aff = np.linalg.inv(target_affine)

    density = np.zeros(target_shape, dtype=np.float32)
    rgb = np.zeros(target_shape + (3,), dtype=np.float32)

    for sl in streamlines:
        if sl.shape[0] < 2:
            continue

        for i in range(sl.shape[0] - 1):
            p0 = sl[i]
            p1 = sl[i + 1]
            delta = p1 - p0
            seg_len = float(np.linalg.norm(delta))
            if seg_len < 1e-6:
                continue

            # Dirección RGB tipo tractografía: |LR|, |AP|, |SI|
            direction = np.abs(delta / seg_len)

            # Muestreo cada ~0.5 mm
            n_steps = max(2, int(math.ceil(seg_len / 0.5)))
            ts = np.linspace(0.0, 1.0, n_steps)

            for t in ts:
                p = p0 * (1.0 - t) + p1 * t
                vox = nib.affines.apply_affine(inv_aff, p)
                xi, yi, zi = np.round(vox).astype(int)

                if (
                    0 <= xi < target_shape[0]
                    and 0 <= yi < target_shape[1]
                    and 0 <= zi < target_shape[2]
                ):
                    density[xi, yi, zi] += 1.0
                    rgb[xi, yi, zi, :] += direction.astype(np.float32)

    if density_smooth_sigma > 0:
        density = gaussian_filter(density, sigma=density_smooth_sigma)

    mask = density > 0
    if np.any(mask):
        rgb[mask] /= density[mask][..., None]

    return density.astype(np.float32), rgb.astype(np.float32)


def _target_streamlines_tolerant(
    streamlines: Streamlines,
    roi: np.ndarray,
    affine: np.ndarray,
    include: bool = True,
) -> Streamlines:
    """Filtra streamlines por ROI tolerando puntos fuera del volumen.

    DIPY utils.target puede abortar cuando un punto queda apenas fuera de la
    máscara T1 después del registro DWI->T1, por ejemplo índice z=181/183
    con tamaño 181. Eso no invalida toda la tractografía; solo significa que
    algunos puntos finales salen del FOV por redondeo/registro. Esta función
    evalúa únicamente los puntos válidos dentro del volumen y descarta/retiene
    según haya intersección real con la ROI.
    """
    roi_bool = np.asarray(roi).astype(bool)
    if roi_bool.ndim != 3 or not np.any(roi_bool):
        return Streamlines([] if include else list(streamlines))

    inv_aff = np.linalg.inv(affine)
    shape = np.asarray(roi_bool.shape, dtype=np.int64)
    kept = []

    for sl in streamlines:
        arr = np.asarray(sl, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            continue

        vox = nib.affines.apply_affine(inv_aff, arr)
        ijk = np.round(vox).astype(np.int64)
        valid = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < shape[0]) &
            (ijk[:, 1] >= 0) & (ijk[:, 1] < shape[1]) &
            (ijk[:, 2] >= 0) & (ijk[:, 2] < shape[2])
        )

        hit = False
        if np.any(valid):
            v = ijk[valid]
            hit = bool(np.any(roi_bool[v[:, 0], v[:, 1], v[:, 2]]))

        if (include and hit) or ((not include) and (not hit)):
            kept.append(arr)

    return Streamlines(kept)


def extract_streamlines_through_roi(
    streamlines: Streamlines,
    roi: np.ndarray,
    affine: np.ndarray,
    mode: str = "any",
) -> Streamlines:
    return _target_streamlines_tolerant(streamlines, roi, affine, include=True)


def exclude_streamlines_by_roi(
    streamlines: Streamlines,
    roi: np.ndarray,
    affine: np.ndarray,
) -> Streamlines:
    return _target_streamlines_tolerant(streamlines, roi, affine, include=False)


def voxelize_streamlines(
    streamlines: Streamlines,
    vol_shape: tuple[int, int, int],
    affine: np.ndarray,
) -> np.ndarray:
    dmap = utils.density_map(streamlines, affine=affine, vol_dims=vol_shape)
    return (dmap > 0).astype(np.uint8)


def save_streamlines_trk(
    streamlines: Streamlines,
    reference_img: nib.Nifti1Image,
    out_path: Path,
) -> None:
    sft = StatefulTractogram(streamlines, reference_img, Space.RASMM)
    save_trk(sft, str(out_path), bbox_valid_check=False)


def maybe_load_roi(path_like) -> tuple[nib.Nifti1Image | None, np.ndarray | None]:
    if path_like is None:
        return None, None
    path = Path(path_like)
    if not path.exists():
        return None, None
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    return img, data


def resample_roi_to_t1(
    roi_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
) -> np.ndarray:
    resampled = resample_from_to(roi_img, t1_img, order=0)
    data = np.asarray(resampled.dataobj)
    return data > 0.5


# ============================================================
# ATLAS MOTOR DE LA SUITE VCE + EXTRACCIÓN CST
# ============================================================

def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndi_label(np.asarray(mask, dtype=bool))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def _remove_tiny_components(mask: np.ndarray, min_voxels: int = 20) -> np.ndarray:
    labels, n = ndi_label(np.asarray(mask, dtype=bool))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    keep = np.zeros_like(mask, dtype=bool)
    for idx in range(1, len(counts)):
        if counts[idx] >= min_voxels:
            keep |= labels == idx
    return keep if np.any(keep) else _largest_component(mask)


def _brain_mask_from_t1(t1_data: np.ndarray) -> np.ndarray:
    """Máscara cerebral simple para restringir priors heurísticas."""
    _, mask = median_otsu(
        np.asarray(t1_data, dtype=np.float32),
        median_radius=4,
        numpass=4,
        autocrop=False,
        dilate=1,
    )
    mask = binary_closing(mask.astype(bool), iterations=2)
    mask = binary_fill_holes(mask)
    return _largest_component(mask)


def _voxel_sizes(affine: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.asarray(affine[:3, :3], dtype=float) ** 2, axis=0))


def _expand_mask_mm(mask: np.ndarray, affine: np.ndarray, distance_mm: float) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if distance_mm <= 0 or not np.any(mask):
        return mask
    dist = distance_transform_edt(~mask, sampling=_voxel_sizes(affine))
    return dist <= float(distance_mm)


def _world_midline_x(brain_mask: np.ndarray, affine: np.ndarray) -> float:
    pts = np.argwhere(brain_mask)
    if pts.size == 0:
        center_vox = (np.asarray(brain_mask.shape, dtype=float) - 1.0) / 2.0
        return float(nib.affines.apply_affine(affine, center_vox)[0])
    # Centro entre los extremos izquierda/derecha en coordenadas RAS.
    sample = pts[::max(1, len(pts) // 20000)]
    world = nib.affines.apply_affine(affine, sample)
    return float((np.min(world[:, 0]) + np.max(world[:, 0])) / 2.0)


def _candidate_niftis(roots: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for pattern in ("*.nii", "*.nii.gz", "*.mgz"):
            for path in root.rglob(pattern):
                key = str(path.resolve())
                if key not in seen:
                    out.append(path)
                    seen.add(key)
    return out


def _score_motor_mask(path: Path, region: str, side: str) -> int:
    name = str(path).lower().replace("á", "a").replace("í", "i")
    side_tokens = {
        "left": ("izquierda", "izq", "left", "_lh", "-lh"),
        "right": ("derecha", "der", "right", "_rh", "-rh"),
    }[side]
    region_tokens = {
        "m1": ("m1", "primaria", "precentral"),
        "m2": ("m2", "secundaria", "premotora", "premotor", "sma"),
    }[region]

    if not any(token in name for token in side_tokens):
        return -10_000
    if not any(token in name for token in region_tokens):
        return -10_000

    score = 0
    if "roi_motor_final" in name:
        score += 140
    if "freesurfer_native" in name:
        score += 120
    if "atlas_refinada" in name:
        score += 90
    if "wavelet_mask" in name:
        score += 55
    if "mask" in name or "mascara" in name:
        score += 30
    if "atlas_prior" in name and "refinada" not in name:
        score -= 25
    if "target" in name or "shell" in name or "preview" in name:
        score -= 35
    return score


def _find_best_motor_mask(
    candidates: list[Path],
    region: str,
    side: str,
) -> Path | None:
    scored = [(_score_motor_mask(path, region, side), path) for path in candidates]
    scored = [item for item in scored if item[0] > -10_000]
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], -len(str(item[1]))), reverse=True)
    return scored[0][1]


def _find_aparc_aseg(roots: Iterable[Path]) -> Path | None:
    names = ("aparc+aseg.mgz", "aparc+aseg.nii.gz", "aparc+aseg.nii")
    found: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for name in names:
            found.extend(root.rglob(name))
    if not found:
        return None
    # Favorecer el archivo dentro de una carpeta de paciente/Antes y rutas cortas.
    found.sort(
        key=lambda p: (
            0 if "paciente 6" in str(p).lower() else 1,
            0 if "antes" in str(p).lower() else 1,
            len(str(p)),
        )
    )
    return found[0]


def _resample_mask_path_to_t1(path: Path, t1_img: nib.Nifti1Image) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"ROI no 3D: {path} shape={data.shape}")

    # Binaria o probabilística.
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError(f"ROI sin valores finitos: {path}")
    if np.nanmax(finite) <= 1.0 and len(np.unique(finite[:min(len(finite), 20000)])) > 3:
        threshold = 0.10
    else:
        threshold = 0.5
    binary = data > threshold

    binary_img = nib.Nifti1Image(binary.astype(np.uint8), img.affine, img.header.copy())
    resampled = resample_from_to(binary_img, t1_img, order=0)
    mask = np.asarray(resampled.dataobj) > 0.5
    return _remove_tiny_components(mask, min_voxels=20)


def _atlas_prior_motor_mask(
    brain_mask: np.ndarray,
    affine: np.ndarray,
    side: str,
    region: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Respaldo atlas-prior, adaptado al rT1 RAS y a la caja cerebral."""
    pts = np.argwhere(brain_mask)
    if pts.size == 0:
        raise ValueError("No se pudo crear atlas-prior: máscara cerebral vacía.")

    lo = pts.min(axis=0).astype(float)
    hi = pts.max(axis=0).astype(float)
    span = np.maximum(hi - lo, 1.0)

    grid = np.indices(brain_mask.shape, dtype=np.float32)
    x = (grid[0] - lo[0]) / span[0]
    y = (grid[1] - lo[1]) / span[1]
    z = (grid[2] - lo[2]) / span[2]

    # En RAS: x aumenta hacia derecha, y hacia anterior, z hacia superior.
    cx = 0.30 if side == "left" else 0.70
    cy = 0.56 if region == "m1" else 0.70
    cz = 0.80
    sx = 0.085
    sy = 0.09 if region == "m1" else 0.13
    sz = 0.14

    prior = np.exp(
        -0.5 * (
            ((x - cx) / sx) ** 2
            + ((y - cy) / sy) ** 2
            + ((z - cz) / sz) ** 2
        )
    ).astype(np.float32)

    inside_distance = distance_transform_edt(brain_mask, sampling=_voxel_sizes(affine))
    cortical_shell = brain_mask & (inside_distance <= 6.0)
    hemi = x < 0.50 if side == "left" else x > 0.50
    prior *= (cortical_shell & hemi).astype(np.float32)

    if np.max(prior) > 0:
        prior /= np.max(prior)

    fraction = 0.0025 if region == "m1" else 0.0040
    target_voxels = max(400, int(np.sum(brain_mask) * fraction))
    valid = np.argwhere(prior > 0)
    mask = np.zeros_like(brain_mask, dtype=bool)
    if valid.size:
        values = prior[tuple(valid.T)]
        k = min(target_voxels, len(values))
        idx = np.argpartition(values, -k)[-k:]
        mask[tuple(valid[idx].T)] = True
    mask = _remove_tiny_components(mask, min_voxels=20)
    return mask, prior


def _heuristic_brainstem_mask(brain_mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    pts = np.argwhere(brain_mask)
    if pts.size == 0:
        return np.zeros_like(brain_mask, dtype=bool)
    lo = pts.min(axis=0).astype(float)
    hi = pts.max(axis=0).astype(float)
    span = np.maximum(hi - lo, 1.0)
    grid = np.indices(brain_mask.shape, dtype=np.float32)
    x = (grid[0] - lo[0]) / span[0]
    y = (grid[1] - lo[1]) / span[1]
    z = (grid[2] - lo[2]) / span[2]
    mask = (
        brain_mask
        & (x >= 0.40) & (x <= 0.60)
        & (y >= 0.30) & (y <= 0.58)
        & (z >= 0.08) & (z <= 0.34)
    )
    mask = binary_closing(mask, iterations=2)
    return _largest_component(mask)


def prepare_motor_atlas_rois(
    t1_img: nib.Nifti1Image,
    t1_data: np.ndarray,
    out_dir: Path,
) -> dict:
    """Localiza M1/M2 con la misma jerarquía usada por la suite VCE."""
    ensure_dir(out_dir)
    brain_mask = _brain_mask_from_t1(t1_data)
    save_nifti_like(t1_img, brain_mask.astype(np.uint8), out_dir / "brain_mask_T1_atlas.nii.gz", dtype=np.uint8)

    candidates = _candidate_niftis(ATLAS_SEARCH_ROOTS)
    aparc_path = _find_aparc_aseg(ATLAS_SEARCH_ROOTS)

    aparc_data_t1 = None
    if aparc_path is not None:
        aparc_img = nib.load(str(aparc_path))
        aparc_res = resample_from_to(aparc_img, t1_img, order=0)
        aparc_data_t1 = np.rint(np.asarray(aparc_res.dataobj)).astype(np.int32)

    explicit = {
        "m1_left": LEFT_M1_SEED,
        "m1_right": RIGHT_M1_SEED,
        "m2_left": LEFT_M2_SEED,
        "m2_right": RIGHT_M2_SEED,
    }

    rois: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    prior_paths: dict[str, str] = {}

    for key in ("m1_left", "m1_right", "m2_left", "m2_right"):
        region, side = key.split("_")
        explicit_path = Path(explicit[key]) if explicit[key] is not None else None

        if explicit_path is not None and explicit_path.exists():
            mask = _resample_mask_path_to_t1(explicit_path, t1_img)
            source = f"explicit:{explicit_path}"
        else:
            best = _find_best_motor_mask(candidates, region, side)
            if best is not None and _score_motor_mask(best, region, side) >= 110:
                mask = _resample_mask_path_to_t1(best, t1_img)
                source = f"suite_mask:{best}"
            elif aparc_data_t1 is not None:
                mask = np.isin(aparc_data_t1, FS_MOTOR_LABELS[key])
                mask = _remove_tiny_components(mask, min_voxels=20)
                source = f"freesurfer_aparc_aseg:{aparc_path};labels={FS_MOTOR_LABELS[key]}"
            elif best is not None:
                mask = _resample_mask_path_to_t1(best, t1_img)
                source = f"suite_mask_fallback:{best}"
            else:
                mask, prior = _atlas_prior_motor_mask(brain_mask, t1_img.affine, side, region)
                prior_path = out_dir / f"atlas_prior_{key}_T1.nii.gz"
                save_nifti_like(t1_img, prior, prior_path, dtype=np.float32)
                prior_paths[key] = str(prior_path)
                source = "heuristic_atlas_prior_RAS_fallback"

        mask &= brain_mask
        mask = _remove_tiny_components(mask, min_voxels=20)
        rois[key] = mask
        sources[key] = source
        save_nifti_like(t1_img, mask.astype(np.uint8), out_dir / f"{key}_T1.nii.gz", dtype=np.uint8)

    # Tronco encefálico: explícito > aseg label 16 > máscara encontrada > heurístico.
    brainstem_source = ""
    if BRAINSTEM_WAYPOINT is not None and Path(BRAINSTEM_WAYPOINT).exists():
        brainstem = _resample_mask_path_to_t1(Path(BRAINSTEM_WAYPOINT), t1_img)
        brainstem_source = f"explicit:{BRAINSTEM_WAYPOINT}"
    elif aparc_data_t1 is not None and np.any(aparc_data_t1 == FS_BRAINSTEM_LABEL):
        brainstem = aparc_data_t1 == FS_BRAINSTEM_LABEL
        brainstem_source = f"freesurfer_aparc_aseg:{aparc_path};label={FS_BRAINSTEM_LABEL}"
    else:
        brainstem_candidates = [
            p for p in candidates
            if any(t in p.name.lower() for t in ("brainstem", "brain_stem", "tronco", "tallo"))
        ]
        if brainstem_candidates:
            brainstem = _resample_mask_path_to_t1(brainstem_candidates[0], t1_img)
            brainstem_source = f"found_mask:{brainstem_candidates[0]}"
        else:
            brainstem = _heuristic_brainstem_mask(brain_mask, t1_img.affine)
            brainstem_source = "heuristic_brainstem_RAS_fallback"

    brainstem &= brain_mask
    brainstem = _remove_tiny_components(brainstem, min_voxels=20)
    rois["brainstem"] = brainstem
    sources["brainstem"] = brainstem_source
    save_nifti_like(t1_img, brainstem.astype(np.uint8), out_dir / "brainstem_T1.nii.gz", dtype=np.uint8)

    # Labelmap fácil de revisar en ITK-SNAP.
    labelmap = np.zeros(t1_img.shape[:3], dtype=np.uint8)
    labelmap[rois["m1_left"]] = 1
    labelmap[rois["m1_right"]] = 2
    labelmap[rois["m2_left"]] = 3
    labelmap[rois["m2_right"]] = 4
    labelmap[rois["brainstem"]] = 5
    save_nifti_like(t1_img, labelmap, out_dir / "atlas_motor_labelmap_T1.nii.gz", dtype=np.uint8)

    report = {
        "method_priority": [
            "explicit_roi",
            "suite_roi_motor_final_or_freesurfer_native",
            "freesurfer_aparc_aseg",
            "suite_atlas_refinada",
            "heuristic_atlas_prior_RAS_fallback",
        ],
        "freesurfer_labels": FS_MOTOR_LABELS,
        "brainstem_label": FS_BRAINSTEM_LABEL,
        "aparc_aseg": str(aparc_path) if aparc_path else None,
        "sources": sources,
        "prior_paths": prior_paths,
        "voxel_counts": {key: int(np.sum(mask)) for key, mask in rois.items()},
        "warning": (
            "M1 FreeSurfer=precentral. M2 es aproximación de la suite: "
            "caudalmiddlefrontal+superiorfrontal+paracentral. "
            "El respaldo heurístico no sustituye segmentación FreeSurfer real."
        ),
    }
    (out_dir / "atlas_motor_reporte.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"rois": rois, "sources": sources, "brain_mask": brain_mask, "report": report}


def _streamline_length_mm(streamline: np.ndarray) -> float:
    if len(streamline) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(streamline, axis=0), axis=1)))


def _filter_descending_motor_streamlines(
    streamlines: Streamlines,
    side: str,
    midline_x: float,
) -> Streamlines:
    kept = []
    for sl in streamlines:
        sl = np.asarray(sl, dtype=np.float32)
        if len(sl) < 2:
            continue
        if _streamline_length_mm(sl) < MIN_CST_LENGTH_MM:
            continue
        if float(np.ptp(sl[:, 2])) < MIN_DESCENDING_Z_RANGE_MM:
            continue
        if side == "left":
            fraction = float(np.mean(sl[:, 0] <= midline_x + 5.0))
        else:
            fraction = float(np.mean(sl[:, 0] >= midline_x - 5.0))
        if fraction < MIN_IPSILATERAL_FRACTION:
            continue
        kept.append(sl)
    return Streamlines(kept)


def _valid_voxel_indices_for_streamline(sl: np.ndarray, affine: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Convierte puntos RAS-mm de una streamline a índices válidos dentro del volumen."""
    if sl is None:
        return np.zeros((0, 3), dtype=np.int64)
    arr = np.asarray(sl, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int64)
    inv_aff = np.linalg.inv(affine)
    ijk = np.round(nib.affines.apply_affine(inv_aff, arr)).astype(np.int64)
    shape_arr = np.asarray(shape, dtype=np.int64)
    valid = np.all((ijk >= 0) & (ijk < shape_arr), axis=1)
    return ijk[valid]


def _min_distance_to_mask_mm(sl: np.ndarray, dist_map: np.ndarray, affine: np.ndarray) -> float:
    idx = _valid_voxel_indices_for_streamline(sl, affine, dist_map.shape)
    if idx.size == 0:
        return float("inf")
    vals = dist_map[idx[:, 0], idx[:, 1], idx[:, 2]]
    if vals.size == 0:
        return float("inf")
    return float(np.nanmin(vals))


def _filter_streamlines_by_roi_distance(
    streamlines: Streamlines,
    cortical_roi: np.ndarray,
    brainstem_roi: np.ndarray,
    t1_img: nib.Nifti1Image,
    side: str,
    midline_x: float,
    motor_distance_mm: float,
    brainstem_distance_mm: float,
    top_n: int = 2500,
) -> Streamlines:
    """Rescate anatómico CST por proximidad en mm.

    No inventa fibras: siempre parte de wholebrain_T1.trk. Solo evita que el CST
    quede vacío cuando el registro DWI->T1 deja fibras a pocos mm de la máscara
    cortical/tronco y el cruce voxel exacto no detecta intersección.
    """
    cortical = np.asarray(cortical_roi, dtype=bool)
    brainstem = np.asarray(brainstem_roi, dtype=bool)
    if not np.any(cortical) or not np.any(brainstem):
        return Streamlines([])

    zooms = _voxel_sizes(t1_img.affine)
    d_cort = distance_transform_edt(~cortical, sampling=zooms)
    d_stem = distance_transform_edt(~brainstem, sampling=zooms)

    accepted = []
    scored = []
    for sl in streamlines:
        arr = np.asarray(sl, dtype=np.float32)
        if len(arr) < 2:
            continue
        if _streamline_length_mm(arr) < MIN_CST_LENGTH_MM:
            continue
        if float(np.ptp(arr[:, 2])) < MIN_DESCENDING_Z_RANGE_MM:
            continue
        if side == "left":
            frac = float(np.mean(arr[:, 0] <= midline_x + 5.0))
        else:
            frac = float(np.mean(arr[:, 0] >= midline_x - 5.0))
        if frac < MIN_IPSILATERAL_FRACTION:
            continue

        dc = _min_distance_to_mask_mm(arr, d_cort, t1_img.affine)
        ds = _min_distance_to_mask_mm(arr, d_stem, t1_img.affine)
        score = dc + ds
        if dc <= motor_distance_mm and ds <= brainstem_distance_mm:
            accepted.append(arr)
        else:
            scored.append((score, dc, ds, arr))

    if accepted:
        return Streamlines(accepted)

    # Último respaldo: no dejamos sin archivo CST si hay fibras descendentes cercanas.
    # Se marca en el resumen como rescate por proximidad; debe validarse en ITK-SNAP/TrackVis.
    scored = [x for x in scored if np.isfinite(x[0])]
    scored.sort(key=lambda x: x[0])
    rescue = [x[3] for x in scored[:max(0, int(top_n))]]
    return Streamlines(rescue)


def _extract_motor_bundle(
    wholebrain_t1: Streamlines,
    cortical_roi: np.ndarray,
    brainstem_roi: np.ndarray,
    t1_img: nib.Nifti1Image,
    side: str,
    midline_x: float,
    include_roi: np.ndarray | None = None,
    exclusion_roi: np.ndarray | None = None,
) -> Streamlines:
    cortical_target = _expand_mask_mm(cortical_roi, t1_img.affine, MOTOR_SEED_DILATION_MM)
    brainstem_target = _expand_mask_mm(brainstem_roi, t1_img.affine, BRAINSTEM_DILATION_MM)

    # 1) Método principal: intersección tolerante con máscaras dilatadas.
    bundle = extract_streamlines_through_roi(wholebrain_t1, cortical_target, t1_img.affine)
    bundle = extract_streamlines_through_roi(bundle, brainstem_target, t1_img.affine)

    if include_roi is not None and np.any(include_roi):
        bundle = extract_streamlines_through_roi(bundle, include_roi, t1_img.affine)
    if exclusion_roi is not None and np.any(exclusion_roi):
        bundle = exclude_streamlines_by_roi(bundle, exclusion_roi, t1_img.affine)

    bundle = _filter_descending_motor_streamlines(bundle, side, midline_x)
    if len(bundle) > 0 or not CST_RESCUE_ENABLE:
        return bundle

    # 2) Rescate anatómico: proximidad a M1/M2 y tronco, sin exigir cruce voxel exacto.
    bundle = _filter_streamlines_by_roi_distance(
        wholebrain_t1,
        cortical_roi,
        brainstem_roi,
        t1_img,
        side=side,
        midline_x=midline_x,
        motor_distance_mm=CST_RESCUE_MOTOR_DISTANCE_MM,
        brainstem_distance_mm=CST_RESCUE_BRAINSTEM_DISTANCE_MM,
        top_n=CST_RESCUE_TOP_N,
    )

    if include_roi is not None and np.any(include_roi) and len(bundle) > 0:
        # Para el rescate usamos una ROI hemisférica amplia; no volvemos a exigir cruce exacto.
        pass
    if exclusion_roi is not None and np.any(exclusion_roi) and len(bundle) > 0:
        bundle = exclude_streamlines_by_roi(bundle, exclusion_roi, t1_img.affine)

    return _filter_descending_motor_streamlines(bundle, side, midline_x)


def _save_bundle_outputs(
    streamlines: Streamlines,
    name: str,
    t1_img: nib.Nifti1Image,
    out_dir: Path,
) -> dict:
    trk_path = out_dir / f"{name}_T1.trk"
    density_path = out_dir / f"{name}_density_T1.nii.gz"
    rgb_path = out_dir / f"{name}_rgb_T1.nii.gz"
    mask_path = out_dir / f"{name}_mask_T1.nii.gz"

    save_streamlines_trk(streamlines, t1_img, trk_path)
    density, rgb = compute_color_from_streamlines(
        streamlines,
        t1_img.shape[:3],
        t1_img.affine,
        density_smooth_sigma=DENSITY_SMOOTH_SIGMA,
    )
    save_nifti_like(t1_img, density, density_path)
    save_nifti_like(t1_img, (density > 0).astype(np.uint8), mask_path, dtype=np.uint8)

    # Carpeta visible de CST para que el usuario encuentre las vías sin buscar en la raíz.
    if name.startswith("cst_") or name.startswith("motor_"):
        vis_dir = out_dir / "cst_visualizacion_ventral_medial"
        vis_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(trk_path, vis_dir / trk_path.name)
            shutil.copy2(density_path, vis_dir / density_path.name)
            shutil.copy2(mask_path, vis_dir / mask_path.name)
        except Exception:
            pass

    if RGB_SCALE_TO_255:
        rgb_u8 = np.stack(
            [
                normalize_to_uint8(rgb[..., 0] * density),
                normalize_to_uint8(rgb[..., 1] * density),
                normalize_to_uint8(rgb[..., 2] * density),
            ],
            axis=3,
        )
        save_nifti_like(t1_img, rgb_u8, rgb_path, dtype=np.uint8)
    else:
        save_nifti_like(t1_img, rgb, rgb_path, dtype=np.float32)

    return {
        "name": name,
        "streamline_count": int(len(streamlines)),
        "trk": str(trk_path),
        "density": str(density_path),
        "rgb": str(rgb_path),
        "mask": str(mask_path),
    }


# ============================================================
# FLUJO PRINCIPAL
# ============================================================

def main() -> int:
    if not DICOM_ROOT.exists():
        raise FileNotFoundError(f"No existe DICOM_ROOT: {DICOM_ROOT}")
    if not RT1_PATH.exists():
        raise FileNotFoundError(f"No existe RT1_PATH: {RT1_PATH}")

    ensure_dir(OUTPUT_DIR)

    # Modo rápido: solo atlas motor para comprobarlo en ITK-SNAP.
    if ONLY_PREPARE_MOTOR_ATLAS:
        log("\n[MODO ATLAS] Generando M1/M2 y tronco en el espacio del rT1...")
        t1_img_only = nib.load(str(RT1_PATH))
        t1_data_only = np.asarray(t1_img_only.dataobj, dtype=np.float32)
        atlas_dir_only = OUTPUT_DIR / "atlas_motor_rois"
        atlas_result_only = prepare_motor_atlas_rois(
            t1_img_only,
            t1_data_only,
            atlas_dir_only,
        )
        log(f"Atlas motor listo: {atlas_dir_only}")
        log(f"Revisa: {atlas_dir_only / 'atlas_motor_labelmap_T1.nii.gz'}")
        log(f"Fuentes: {atlas_result_only['sources']}")
        return 0

    conv_dir = OUTPUT_DIR / "01_nifti_convertidos"
    ensure_dir(conv_dir)

    # --------------------------------------------------------
    # 1) Conversión
    # --------------------------------------------------------
    converted = run_dcm2niix(DICOM_ROOT, conv_dir)

    # --------------------------------------------------------
    # 2) Carga DWI + T1
    # --------------------------------------------------------
    log("\n[2/8] Cargando DWI y rT1...")
    dwi_img = nib.load(str(converted.dwi_nii))
    dwi_data = np.asarray(dwi_img.dataobj, dtype=np.float32)

    bvals, bvecs = read_bvals_bvecs(str(converted.dwi_bval), str(converted.dwi_bvec))
    gtab = gradient_table(bvals, bvecs)

    t1_img = nib.load(str(RT1_PATH))
    t1_data = np.asarray(t1_img.dataobj, dtype=np.float32)

    if dwi_data.ndim != 4:
        raise RuntimeError(f"La DWI seleccionada no es 4D. shape={dwi_data.shape}")

    log(f"DWI shape: {dwi_data.shape}")
    log(f"T1  shape: {t1_data.shape}")

    # --------------------------------------------------------
    # 3) Máscara y b0
    # --------------------------------------------------------
    log("\n[3/8] Generando máscara cerebral y b0...")
    b0_mask = gtab.b0s_mask
    mean_b0 = np.mean(dwi_data[..., b0_mask], axis=3)
    b0_img = nib.Nifti1Image(mean_b0.astype(np.float32), dwi_img.affine, dwi_img.header.copy())
    nib.save(b0_img, str(OUTPUT_DIR / "mean_b0_dwi.nii.gz"))

    masked_data, brain_mask = median_otsu(
        dwi_data,
        vol_idx=np.where(b0_mask)[0],
        median_radius=4,
        numpass=4,
        dilate=1,
        autocrop=False,
    )
    save_nifti_like(dwi_img, brain_mask.astype(np.uint8), OUTPUT_DIR / "brain_mask_dwi.nii.gz", dtype=np.uint8)

    # --------------------------------------------------------
    # 4) DTI para FA y ColorFA
    # --------------------------------------------------------
    log("\n[4/8] Ajustando DTI para FA / ColorFA...")
    tenmodel = dti.TensorModel(gtab, fit_method="WLS")
    tenfit = tenmodel.fit(dwi_data, mask=brain_mask)

    evals = np.clip(tenfit.evals, 0, None)
    fa = dti.fractional_anisotropy(evals)
    fa = np.nan_to_num(fa, nan=0.0, posinf=0.0, neginf=0.0)
    fa[~brain_mask] = 0.0
    save_nifti_like(dwi_img, fa, OUTPUT_DIR / "FA_dwi.nii.gz")

    color_fa = dti.color_fa(fa, tenfit.evecs)
    color_fa = np.nan_to_num(color_fa, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    save_nifti_like(dwi_img, color_fa, OUTPUT_DIR / "ColorFA_dwi.nii.gz")

    # --------------------------------------------------------
    # 5) CSD / fallback a tensor peaks y tractografía
    # --------------------------------------------------------
    log("\n[5/8] Preparando orientación de fibras...")
    peaks = None
    csd_ok = False
    response = None

    try:
        n_nonzero = int(np.sum(bvals > 50))
        if n_nonzero >= 20:
            response, ratio = auto_response_ssst(gtab, dwi_data, roi_radii=10, fa_thr=0.7)
            csd_model = ConstrainedSphericalDeconvModel(gtab, response)
            peaks = peaks_from_model(
                model=csd_model,
                data=dwi_data,
                sphere=default_sphere,
                relative_peak_threshold=0.5,
                min_separation_angle=25,
                mask=brain_mask,
                return_sh=False,
                parallel=False,
                normalize_peaks=True,
            )
            csd_ok = True
            log(f"CSD listo. Ratio respuesta = {ratio}")
    except Exception as exc:
        log(f"CSD no fue posible, se usará fallback tensorial. Motivo: {exc}")

    if peaks is None:
        peaks = peaks_from_model(
            model=tenmodel,
            data=dwi_data,
            sphere=default_sphere,
            relative_peak_threshold=0.5,
            min_separation_angle=25,
            mask=brain_mask,
            return_sh=False,
            parallel=False,
            normalize_peaks=True,
        )

    log("Generando semillas y ejecutando tractografía...")
    stop_criterion = ThresholdStoppingCriterion(fa, FA_STOP)
    wm_seed_mask = brain_mask.astype(bool) & (fa >= max(FA_STOP, 0.12))
    save_nifti_like(dwi_img, wm_seed_mask.astype(np.uint8), OUTPUT_DIR / "wm_seed_mask_dwi.nii.gz", dtype=np.uint8)
    seeds = utils.seeds_from_mask(wm_seed_mask, affine=dwi_img.affine, density=SEED_DENSITY)

    streamline_generator = LocalTracking(
        peaks,
        stop_criterion,
        seeds,
        affine=dwi_img.affine,
        step_size=STEP_SIZE_MM,
        max_cross=2 if csd_ok else 1,
        return_all=False,
    )

    wholebrain = Streamlines(
        sl for sl in streamline_generator
        if len(sl) >= MIN_STREAMLINE_POINTS
    )

    log(f"Streamlines completas generadas: {len(wholebrain)}")
    if len(wholebrain) == 0:
        raise RuntimeError("No se generaron streamlines. Revisa el umbral FA o la conversión DWI.")

    save_streamlines_trk(wholebrain, dwi_img, OUTPUT_DIR / "wholebrain_dwi_space.trk")

    # --------------------------------------------------------
    # 6) Registro b0 -> T1 y transformación de tractografía
    # --------------------------------------------------------
    log("\n[6/8] Registrando b0 al rT1...")
    b0_to_t1_affine, _ = register_b0_to_t1(b0_img, t1_img, OUTPUT_DIR)

    log("Transformando streamlines al espacio del rT1...")
    wholebrain_t1 = Streamlines(transform_streamlines(wholebrain, b0_to_t1_affine))
    save_streamlines_trk(wholebrain_t1, t1_img, OUTPUT_DIR / "wholebrain_T1.trk")

    # Transformar FA y ColorFA con la MISMA transformación b0->T1.
    dwi_to_t1_map = AffineMap(
        b0_to_t1_affine,
        domain_grid_shape=t1_img.shape[:3],
        domain_grid2world=t1_img.affine,
        codomain_grid_shape=dwi_img.shape[:3],
        codomain_grid2world=dwi_img.affine,
    )
    fa_t1 = dwi_to_t1_map.transform(fa, interpolation="linear")
    save_nifti_like(t1_img, fa_t1, OUTPUT_DIR / "FA_en_T1.nii.gz")

    color_fa_t1 = np.stack(
        [
            dwi_to_t1_map.transform(color_fa[..., ch], interpolation="linear")
            for ch in range(3)
        ],
        axis=3,
    ).astype(np.float32)
    save_nifti_like(t1_img, color_fa_t1, OUTPUT_DIR / "ColorFA_en_T1.nii.gz")

    # --------------------------------------------------------
    # 7) Densidad + RGB "bonitos" en T1
    # --------------------------------------------------------
    log("\n[7/8] Voxelizando tractografía completa en el espacio del rT1...")
    density_t1, rgb_t1 = compute_color_from_streamlines(
        wholebrain_t1,
        target_shape=t1_img.shape[:3],
        target_affine=t1_img.affine,
        density_smooth_sigma=DENSITY_SMOOTH_SIGMA,
    )

    save_nifti_like(t1_img, density_t1, OUTPUT_DIR / "wholebrain_density_T1.nii.gz")

    if RGB_SCALE_TO_255:
        rgb_save = np.stack(
            [
                normalize_to_uint8(rgb_t1[..., 0] * density_t1),
                normalize_to_uint8(rgb_t1[..., 1] * density_t1),
                normalize_to_uint8(rgb_t1[..., 2] * density_t1),
            ],
            axis=3,
        )
        save_nifti_like(t1_img, rgb_save, OUTPUT_DIR / "wholebrain_rgb_T1.nii.gz", dtype=np.uint8)
    else:
        save_nifti_like(t1_img, rgb_t1, OUTPUT_DIR / "wholebrain_rgb_T1.nii.gz", dtype=np.float32)

    wholebrain_mask = (density_t1 > 0).astype(np.uint8)
    save_nifti_like(t1_img, wholebrain_mask, OUTPUT_DIR / "wholebrain_mask_T1.nii.gz", dtype=np.uint8)

    # --------------------------------------------------------
    # 8) CST opcional con ROIs
    # --------------------------------------------------------
    log("\n[8/8] Localizando M1/M2 con atlas de la suite y extrayendo vías motoras...")

    atlas_dir = OUTPUT_DIR / "atlas_motor_rois"
    atlas_result = prepare_motor_atlas_rois(t1_img, t1_data, atlas_dir)
    motor_rois = atlas_result["rois"]
    midline_x = _world_midline_x(atlas_result["brain_mask"], t1_img.affine)

    excl_img, _ = maybe_load_roi(EXCLUSION_MASK)
    left_inc_img, _ = maybe_load_roi(LEFT_CST_INCLUDE)
    right_inc_img, _ = maybe_load_roi(RIGHT_CST_INCLUDE)

    exclusion = resample_roi_to_t1(excl_img, t1_img) if excl_img is not None else None
    left_include = resample_roi_to_t1(left_inc_img, t1_img) if left_inc_img is not None else None
    right_include = resample_roi_to_t1(right_inc_img, t1_img) if right_inc_img is not None else None

    # CST estricto: M1/precentral + tronco encefálico.
    cst_left = _extract_motor_bundle(
        wholebrain_t1,
        motor_rois["m1_left"],
        motor_rois["brainstem"],
        t1_img,
        side="left",
        midline_x=midline_x,
        include_roi=left_include,
        exclusion_roi=exclusion,
    )
    cst_right = _extract_motor_bundle(
        wholebrain_t1,
        motor_rois["m1_right"],
        motor_rois["brainstem"],
        t1_img,
        side="right",
        midline_x=midline_x,
        include_roi=right_include,
        exclusion_roi=exclusion,
    )

    # Vías descendentes secundarias/premotoras: M2 aproximada de la suite.
    m2_left = _extract_motor_bundle(
        wholebrain_t1,
        motor_rois["m2_left"],
        motor_rois["brainstem"],
        t1_img,
        side="left",
        midline_x=midline_x,
        include_roi=left_include,
        exclusion_roi=exclusion,
    )
    m2_right = _extract_motor_bundle(
        wholebrain_t1,
        motor_rois["m2_right"],
        motor_rois["brainstem"],
        t1_img,
        side="right",
        midline_x=midline_x,
        include_roi=right_include,
        exclusion_roi=exclusion,
    )

    motor_left = Streamlines(list(cst_left) + list(m2_left))
    motor_right = Streamlines(list(cst_right) + list(m2_right))

    motor_outputs = {
        "cst_left": _save_bundle_outputs(cst_left, "cst_m1_left", t1_img, OUTPUT_DIR),
        "cst_right": _save_bundle_outputs(cst_right, "cst_m1_right", t1_img, OUTPUT_DIR),
        "m2_left": _save_bundle_outputs(m2_left, "motor_m2_left", t1_img, OUTPUT_DIR),
        "m2_right": _save_bundle_outputs(m2_right, "motor_m2_right", t1_img, OUTPUT_DIR),
        "motor_left": _save_bundle_outputs(motor_left, "motor_combined_left", t1_img, OUTPUT_DIR),
        "motor_right": _save_bundle_outputs(motor_right, "motor_combined_right", t1_img, OUTPUT_DIR),
    }

    cst_summary = {
        "executed": True,
        "atlas_sources": atlas_result["sources"],
        "atlas_report": str(atlas_dir / "atlas_motor_reporte.json"),
        "midline_x_ras_mm": midline_x,
        "rescue_enable": bool(CST_RESCUE_ENABLE),
        "rescue_motor_distance_mm": CST_RESCUE_MOTOR_DISTANCE_MM,
        "rescue_brainstem_distance_mm": CST_RESCUE_BRAINSTEM_DISTANCE_MM,
        "left_streamlines": len(cst_left),
        "right_streamlines": len(cst_right),
        "m2_left_streamlines": len(m2_left),
        "m2_right_streamlines": len(m2_right),
        "outputs": motor_outputs,
        "interpretation": (
            "cst_m1_* usa M1/precentral como semilla cortical y tronco encefálico como waypoint. "
            "motor_m2_* representa vías descendentes desde la aproximación M2 de la suite."
        ),
    }

    # --------------------------------------------------------
    # Resumen final
    # --------------------------------------------------------
    summary = {
        "dwi_nifti": str(converted.dwi_nii),
        "dwi_bval": str(converted.dwi_bval),
        "dwi_bvec": str(converted.dwi_bvec),
        "t1_reference": str(RT1_PATH),
        "wholebrain_streamlines_dwi": len(wholebrain),
        "wholebrain_streamlines_t1": len(wholebrain_t1),
        "csd_used": csd_ok,
        "fa_stop": FA_STOP,
        "seed_density": SEED_DENSITY,
        "step_size_mm": STEP_SIZE_MM,
        "cst": cst_summary,
        "outputs": {
            "wholebrain_trk_dwi": str(OUTPUT_DIR / "wholebrain_dwi_space.trk"),
            "wholebrain_trk_t1": str(OUTPUT_DIR / "wholebrain_T1.trk"),
            "wholebrain_density_t1": str(OUTPUT_DIR / "wholebrain_density_T1.nii.gz"),
            "wholebrain_rgb_t1": str(OUTPUT_DIR / "wholebrain_rgb_T1.nii.gz"),
            "wholebrain_mask_t1": str(OUTPUT_DIR / "wholebrain_mask_T1.nii.gz"),
            "fa_dwi": str(OUTPUT_DIR / "FA_dwi.nii.gz"),
            "fa_t1": str(OUTPUT_DIR / "FA_en_T1.nii.gz"),
            "colorfa_dwi": str(OUTPUT_DIR / "ColorFA_dwi.nii.gz"),
            "colorfa_t1": str(OUTPUT_DIR / "ColorFA_en_T1.nii.gz"),
            "b0_t1": str(OUTPUT_DIR / "b0_en_T1.nii.gz"),
            "affine_txt": str(OUTPUT_DIR / "affine_b0_to_T1.txt"),
        },
    }

    (OUTPUT_DIR / "resumen_tractografia.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    readme = f"""
LISTO REY

Salidas principales:
- wholebrain_T1.trk
- wholebrain_density_T1.nii.gz
- wholebrain_rgb_T1.nii.gz
- FA_en_T1.nii.gz
- ColorFA_en_T1.nii.gz

Vías motoras atlas-guiadas:
- cst_m1_left_T1.trk / cst_m1_right_T1.trk
- cst_m1_left_density_T1.nii.gz / cst_m1_right_density_T1.nii.gz
- cst_m1_left_rgb_T1.nii.gz / cst_m1_right_rgb_T1.nii.gz
- motor_m2_left_T1.trk / motor_m2_right_T1.trk
- motor_combined_left_T1.trk / motor_combined_right_T1.trk
- atlas_motor_rois/atlas_motor_labelmap_T1.nii.gz

NOTAS IMPORTANTES
1. El archivo .trk conserva las streamlines reales.
2. El archivo .nii.gz de densidad es la versión voxelizada.
3. El archivo .nii.gz RGB es la versión "bonita" de colores alineada al rT1.
4. Si tu software no interpreta el RGB 4D automáticamente, usa:
   - rT1.nii como fondo
   - wholebrain_density_T1.nii.gz o FA_en_T1.nii.gz como overlay
   - wholebrain_T1.trk como tractografía real

Directorio de salida:
{OUTPUT_DIR}
""".strip()

    (OUTPUT_DIR / "LEEME_resultados.txt").write_text(readme, encoding="utf-8")

    log("\nTERMINADO.")
    log(f"Resultados en: {OUTPUT_DIR}")
    log("Archivo clave para ver en el software:")
    log(f"  {OUTPUT_DIR / 'wholebrain_T1.trk'}")
    log(f"  {OUTPUT_DIR / 'wholebrain_rgb_T1.nii.gz'}")
    log(f"  {OUTPUT_DIR / 'wholebrain_density_T1.nii.gz'}")

    if cst_summary["executed"]:
        log("También se extrajeron CST M1 y vías motoras M2 atlas-guiadas.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
