"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       MORFOMETRÍA INTERNA DEL PACIENTE                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/internal_morphometry.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Extrae morfometría interna por ROI del paciente. Las máscaras binarias definen
regiones Ω y el volumen se estima como V(Ω)=Σ_{v∈Ω}|det(A_3x3)|. Las
comparaciones
por lateralidad usan derecha, izquierda, total y asimetría para evaluar
cambios
anatómicos del sistema motor.

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
from typing import Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .paths import resolve_stage_dir, norm_key
from .io_utils import safe_mkdir, save_dataframe, append_log, log_exception, save_json
from .neuroimage import (
    _require_neuro_libs,
    _as_3d,
    save_nifti,
    save_orthogonal_png,
    _voxel_volume_ml_from_affine,
    resize_to_shape,
    robust_zscore,
    wavelet_energy_volume,
    create_cortical_shell_mask,
    motor_cortex_atlas_prior,
    motor_cortex_roi_mask,
    minmax_gpu,
    sanitize_name,
)


def _subject_stage_dir(cfg: PipelineConfig, subject: str, stage: str) -> Optional[Path]:
    subject_dir = cfg.data_root() / subject
    if subject == cfg.control_name:
        exact = resolve_stage_dir(subject_dir, stage, fallback_to_subject=False)
        if exact is not None:
            return exact
        before = resolve_stage_dir(subject_dir, "Antes", fallback_to_subject=False)
        if before is not None:
            return before
        return subject_dir if subject_dir.exists() else None
    return resolve_stage_dir(subject_dir, stage, fallback_to_subject=False)


def _real_stages_for_subject(cfg: PipelineConfig, subject: str) -> list[str]:
    if subject != cfg.control_name:
        return list(cfg.stages)
    out = []
    for st in cfg.stages:
        if _subject_stage_dir(cfg, subject, st) is not None:
            out.append(st)
    return out or ["Antes"]


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _find_reformat_niftis(stage_dir: Optional[Path]) -> list[Path]:
    """Busca NIfTI de reformateo/coregistro dentro de la etapa.

    Estos archivos NO equivalen a FreeSurfer/CAT12, pero sí pueden servir como
    referencia anatómica/coregistrada para crear una morfometría interna proxy.
    """
    if stage_dir is None or not stage_dir.exists():
        return []
    candidates = []
    tokens = [
        "reform", "reformat", "reformateo", "reformateado", "reformatted",
        "coreg", "coregister", "registr", "registered", "normalized", "normalizado",
        "wra", "ra", "mean", "anat", "t1"
    ]
    reject = ["mask_p", "wavelet", "labelmap", "metricas", "deficit", "diferencia"]
    for p in stage_dir.rglob("*"):
        if not p.is_file() or not _is_nifti(p):
            continue
        n = norm_key(str(p))
        if any(r in n for r in reject):
            continue
        if any(t in n for t in tokens):
            candidates.append(p)
    return sorted(set(candidates))


def _find_processed_t1(stage_out: Path) -> Optional[Path]:
    res = stage_out / "resonancias"
    if not res.exists():
        return None
    cands = [p for p in res.rglob("*_volumen.nii.gz") if p.is_file()]
    if not cands:
        return None
    def score(p: Path) -> tuple[int, int, int]:
        n = norm_key(str(p))
        s = 0
        for tok in ["anatomica", "t1", "spc", "mpr", "vol"]:
            if tok in n:
                s += 3
        for bad in ["fmri", "motor", "dti", "difusion", "swi", "tof", "scout", "fieldmap", "mapa"]:
            if bad in n:
                s -= 3
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        return (s, size, -len(str(p)))
    return sorted(cands, key=score, reverse=True)[0]


def _find_best_structural_reference(cfg: PipelineConfig, subject: str, stage: str, stage_out: Path, log_file: Path) -> tuple[Optional[Path], str]:
    stage_dir = _subject_stage_dir(cfg, subject, stage)
    reform = _find_reformat_niftis(stage_dir)
    if reform:
        def score(p: Path) -> tuple[int, int, int]:
            n = norm_key(str(p))
            s = 0
            for tok in ["t1", "anat", "reform", "reformat", "reformateo", "coreg", "registr"]:
                if tok in n:
                    s += 4
            for bad in ["motor", "fwe", "spm", "map", "activation", "funcional"]:
                if bad in n:
                    s -= 2
            try: size = p.stat().st_size
            except Exception: size = 0
            return (s, size, -len(str(p)))
        best = sorted(reform, key=score, reverse=True)[0]
        return best, "nii_reformateo_coregistrado_en_datos"
    t1 = _find_processed_t1(stage_out)
    if t1:
        return t1, "t1_procesado_desde_resonancia"
    append_log(log_file, f"No encontré NIfTI de reformateo ni T1 procesado para morfometría interna {subject} {stage}")
    return None, "no_detectado"


def _find_ad_energy(stage_out: Path) -> Optional[Path]:
    res = stage_out / "resonancias"
    if not res.exists():
        return None
    cands = []
    for p in res.rglob("*_wavelet_energy.nii.gz"):
        n = norm_key(str(p))
        if "dti_difusion" in n and ("/ad/" in n.replace("\\", "/") or "_ad_" in n or "ad_map" in n):
            cands.append(p)
    if not cands:
        cands = [p for p in res.rglob("*_wavelet_energy.nii.gz") if "dti_difusion" in norm_key(str(p))]
    if not cands:
        return None
    def score(p: Path) -> tuple[int, int]:
        n = norm_key(str(p))
        s = 0
        for tok in ["ad", "ad_map", "difusion", "dti"]:
            if tok in n:
                s += 2
        try: size = p.stat().st_size
        except Exception: size = 0
        return (s, size)
    return sorted(cands, key=score, reverse=True)[0]


def _load_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    _, nib, _ = _require_neuro_libs()
    img = nib.load(str(path))
    return _as_3d(img.get_fdata(dtype=np.float32)), img.affine


def _normalize01(x: np.ndarray, cfg: PipelineConfig | None = None) -> np.ndarray:
    gpu = minmax_gpu(x, cfg) if cfg is not None else None
    if gpu is not None:
        return gpu.astype(np.float32, copy=False)
    arr = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.nanpercentile(arr[finite], [1, 99])
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def _make_internal_motor_masks(
    cfg: PipelineConfig,
    ref_vol: np.ndarray,
    affine: np.ndarray,
    ref_path: Path,
    ref_kind: str,
    stage_out: Path,
    out_dir: Path,
    log_file: Path,
) -> pd.DataFrame:
    _, _, ndimage = _require_neuro_libs()
    safe_mkdir(out_dir)
    ref3 = _as_3d(ref_vol)
    voxel_ml = _voxel_volume_ml_from_affine(affine)

    # Shell cortical desde referencia. Si el reformateo no es anatómico perfecto,
    # esto se reporta como proxy y queda trazable.
    brain, shell = create_cortical_shell_mask(ref3, cfg)
    save_nifti(ref3.astype(np.float32), affine, out_dir / "referencia_morfometria_interna.nii.gz")
    save_nifti(brain.astype(np.uint8), affine, out_dir / "brain_mask_interna.nii.gz")
    save_nifti(shell.astype(np.uint8), affine, out_dir / "cortex_gray_internal_shell_mask.nii.gz")
    save_orthogonal_png(shell.astype(np.float32), out_dir / "cortex_gray_internal_shell_mask.png", title="Shell cortical interna")

    ad_energy_path = _find_ad_energy(stage_out)
    if ad_energy_path is not None:
        ad_energy, _ = _load_volume(ad_energy_path)
        ad_rs = resize_to_shape(ad_energy.astype(np.float32), ref3.shape, order=1, cfg=cfg)
        energy_source = str(ad_energy_path)
        energy = _normalize01(ad_rs, cfg)
    else:
        energy = _normalize01(wavelet_energy_volume(ref3, cfg.resonance_wavelet_scales, cfg), cfg)
        energy_source = "wavelet_referencia_interna"
    save_nifti(energy.astype(np.float32), affine, out_dir / "energia_ad_o_referencia_en_espacio_interno.nii.gz")

    labelmap = np.zeros_like(ref3, dtype=np.uint8)
    rows = []
    label_lut = [
        ("m1", "primaria", "derecha", 1),
        ("m1", "primaria", "izquierda", 2),
        ("m2", "secundaria", "derecha", 3),
        ("m2", "secundaria", "izquierda", 4),
    ]
    for region, region_name, side, code in label_lut:
        roi = motor_cortex_roi_mask(ref3.shape, side, region)
        prior = motor_cortex_atlas_prior(ref3.shape, side, region, sigma_scale=float(getattr(cfg, "ad_motor_prior_sigma_scale", 1.0)))
        prior_mask = (prior >= float(getattr(cfg, "ad_motor_prior_threshold", 0.25))) & roi & shell
        if int(np.sum(prior_mask)) < 20:
            prior_mask = roi & shell
        vals = energy[prior_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            thr = float(np.nanpercentile(vals, float(getattr(cfg, "ad_motor_wavelet_percentile", 75.0))))
            weighted = prior * energy
            refined = prior_mask & (weighted >= float(np.nanpercentile(weighted[prior_mask], float(getattr(cfg, "ad_motor_wavelet_percentile", 75.0)))))
            if int(np.sum(refined)) < int(getattr(cfg, "ad_motor_min_component_voxels", 20)):
                refined = prior_mask & (energy >= thr)
        else:
            thr = math.nan
            refined = prior_mask
        refined = ndimage.binary_closing(refined, iterations=1)
        refined = ndimage.binary_fill_holes(refined)
        if int(np.sum(refined)) < 5:
            refined = prior_mask
        tag = f"internal_corteza_motora_{region_name}_{side}"
        save_nifti(prior.astype(np.float32), affine, out_dir / f"{tag}_atlas_prior.nii.gz")
        save_nifti((roi & shell).astype(np.uint8), affine, out_dir / f"{tag}_roi_shell.nii.gz")
        mask_path = out_dir / f"{tag}.nii.gz"
        save_nifti(refined.astype(np.uint8), affine, mask_path)
        save_nifti((energy * refined.astype(np.float32)), affine, out_dir / f"{tag}_wavelet_region.nii.gz")
        labelmap[refined] = code
        vox = int(np.sum(refined))
        ev = energy[refined]
        ev = ev[np.isfinite(ev)]
        rows.append({
            "tool": "MorfometriaInterna",
            "source_type": ref_kind,
            "region": region,
            "region_name": region_name,
            "side": side,
            "voxels": vox,
            "volume_ml": float(vox * voxel_ml),
            "energy_mean": float(np.nanmean(ev)) if ev.size else math.nan,
            "energy_p95": float(np.nanpercentile(ev, 95)) if ev.size else math.nan,
            "wavelet_threshold": thr,
            "reference": str(ref_path),
            "energy_source": energy_source,
            "mask_path": str(mask_path),
            "note": "Morfometría interna proxy: usa NIfTI de reformateo/T1 + shell cortical + prior M1/M2 + AD/wavelet. No equivale a FreeSurfer/CAT12, pero permite correr el flujo completo con tus datos actuales.",
        })
    save_nifti(labelmap.astype(np.uint8), affine, out_dir / "internal_corteza_motora_labelmap.nii.gz")
    save_json({"1":"M1 derecha", "2":"M1 izquierda", "3":"M2 derecha", "4":"M2 izquierda"}, out_dir / "internal_corteza_motora_labelmap_lut.json")
    df = pd.DataFrame(rows)
    if not df.empty:
        save_dataframe(df, out_dir / "metricas_morfometria_interna.csv")
    return df


def run_internal_morphometry_integration(cfg: PipelineConfig) -> pd.DataFrame:
    """Genera derivados morfométricos internos usando datos existentes.

    Usa NIfTI de reformateo si existen en ResultadosFuncional; si no, usa T1 procesado
    desde resonancias. Esto evita depender de FreeSurfer/CAT12/fMRIPrep para que el
    flujo completo corra, pero mantiene trazabilidad y etiqueta el resultado como proxy.
    """
    all_rows = []
    subjects = list(cfg.patients) + [cfg.control_name]
    for subject in subjects:
        for stage in _real_stages_for_subject(cfg, subject):
            print(f"\n[MORFOMETRÍA INTERNA] {subject} · {stage}")
            stage_out = cfg.results_root() / subject / stage
            out_dir = stage_out / "morfometria" / "interna"
            log_file = stage_out / "reportes" / "morfometria_interna_log.txt"
            try:
                ref, kind = _find_best_structural_reference(cfg, subject, stage, stage_out, log_file)
                if ref is None:
                    continue
                vol, affine = _load_volume(ref)
                df = _make_internal_motor_masks(cfg, vol, affine, ref, kind, stage_out, out_dir, log_file)
                if not df.empty:
                    df.insert(0, "stage", stage)
                    df.insert(0, "subject", subject)
                    save_dataframe(df, stage_out / "morfometria" / "resumen_morfometria_interna.csv")
                    save_dataframe(df, out_dir / "resumen_morfometria_interna.csv")
                    all_rows.append(df)
            except Exception as exc:
                log_exception(log_file, f"Morfometría interna {subject} {stage}", exc)
    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not out.empty:
        save_dataframe(out, cfg.results_root() / "resumen_global_morfometria_interna.csv")
    return out
