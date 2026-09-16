"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         RELACIÓN ESTRUCTURA-FUNCIÓN                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/structure_function.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Integra variables estructurales y funcionales mediante asociaciones
cuantitativas.
El modelo compara dominios anatómicos, tractográficos y biomecánicos usando
correlaciones, deltas normalizados y matrices paciente×variable. La lectura
matemática se basa en covariación: corr(X,Y)=cov(X,Y)/(σ_X σ_Y).

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
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .paths import resolve_stage_dir, first_existing_dir, norm_key
from .io_utils import safe_mkdir, save_dataframe, append_log, log_exception, save_json
from .checkpoint import CheckpointManager, load_json_if_exists, save_metrics_json
from .neuroimage import (
    _require_neuro_libs,
    _as_3d,
    resize_to_shape,
    robust_zscore,
    save_nifti,
    save_orthogonal_png,
    create_cortical_shell_mask,
    _voxel_volume_ml_from_affine,
    _safe_pearson,
    compare_image_to_control_volume,
    sanitize_name,
    side_from_text,
    run_maps,
    run_resonances,
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


def _result_stage_dir(cfg: PipelineConfig, subject: str, stage: str) -> Path:
    if subject == cfg.control_name:
        candidate = cfg.results_root() / subject / stage
        if candidate.exists() and any(candidate.rglob("*")):
            return candidate
        return cfg.results_root() / subject / "Antes"
    return cfg.results_root() / subject / stage


def _best_file(root: Path, patterns: list[str], prefer: list[str] | None = None) -> Optional[Path]:
    if not root.exists():
        return None
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(root.rglob(pat))
    candidates = [p for p in sorted(set(candidates)) if p.is_file()]
    if not candidates:
        return None
    prefer = prefer or []
    def score(p: Path) -> tuple[int, int, int]:
        n = norm_key(str(p))
        s = sum(1 for token in prefer if token in n)
        # Prefer larger NIfTI-like volumes, not tiny masks, when names tie.
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        return (s, size, -len(str(p)))
    return sorted(candidates, key=score, reverse=True)[0]


def _list_files(root: Path, patterns: list[str]) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for pat in patterns:
        out.extend(root.rglob(pat))
    return sorted(set([p for p in out if p.is_file()]))


def _load_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    _, nib, _ = _require_neuro_libs()
    img = nib.load(str(path))
    data = _as_3d(img.get_fdata(dtype=np.float32))
    return (data > 0.5), img.affine


def _load_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    _, nib, _ = _require_neuro_libs()
    img = nib.load(str(path))
    return _as_3d(img.get_fdata(dtype=np.float32)), img.affine


def _dice_jaccard(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int, int, int]:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    inter = int(np.sum(aa & bb))
    union = int(np.sum(aa | bb))
    a_sum = int(np.sum(aa))
    b_sum = int(np.sum(bb))
    dice = float((2 * inter) / (a_sum + b_sum + 1e-12))
    jaccard = float(inter / (union + 1e-12))
    return dice, jaccard, inter, a_sum, b_sum


def _ensure_neuro_results(cfg: PipelineConfig, subject: str, stage: str, log_file: Path) -> Path:
    stage_out = _result_stage_dir(cfg, subject, stage)
    # No llama pesado si ya hay salidas mínimas.
    if not list((stage_out / "mapas").rglob("*_wavelet_energy.nii.gz")):
        try:
            run_maps(cfg)
        except Exception as exc:
            log_exception(log_file, f"Procesando mapas para acoplamiento estructura-función {subject} {stage}", exc)
    if not list((stage_out / "resonancias").rglob("*_wavelet_energy.nii.gz")):
        try:
            run_resonances(cfg)
        except Exception as exc:
            log_exception(log_file, f"Procesando resonancias para acoplamiento estructura-función {subject} {stage}", exc)
    return stage_out


def _qc_neuroimage(stage_out: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_name in ["metricas_mapas.csv", "metricas_resonancias.csv", "resumen_corteza_motora_ad.csv"]:
        for p in stage_out.rglob(csv_name):
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if df.empty:
                continue
            for _, r in df.iterrows():
                rows.append({
                    "source_csv": str(p),
                    "base_name": r.get("base_name", ""),
                    "series_role": r.get("series_role", ""),
                    "series_subtype": r.get("series_subtype", ""),
                    "side": r.get("series_side", r.get("side", "")),
                    "shape": r.get("shape", ""),
                    "mean": r.get("mean", math.nan),
                    "std": r.get("std", math.nan),
                    "p95": r.get("p95", math.nan),
                    "energy_p95": r.get("energy_p95", math.nan),
                    "fmri_frames": r.get("fmri_stack_frames", math.nan),
                    "fmri_alff": r.get("fmri_alff_0p01_0p08_mean_signal", math.nan),
                    "fmri_falff": r.get("fmri_falff_0p01_0p08_mean_signal", math.nan),
                    "fmri_tsnr": r.get("fmri_tsnr_median_sampled_voxels", math.nan),
                    "fmri_dvars": r.get("fmri_dvars_mean_sampled_voxels", math.nan),
                    "fmri_gcor_approx": r.get("fmri_gcor_approx_sampled_voxels", math.nan),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        save_dataframe(df, out_dir / "qc_neuroimagen_estructura_funcion.csv")
    return df



def _external_cortex_mask(stage_out: Path) -> Optional[Path]:
    morf = stage_out / "morfometria"
    # Prioridad: FreeSurfer real; respaldo: shell cortical interno desde reformateo/T1.
    candidates = sorted(morf.rglob("cortex_gray_freesurfer_aparc_mask.nii.gz"))
    if candidates:
        return candidates[0]
    candidates = sorted(morf.rglob("cortex_gray_internal_shell_mask.nii.gz"))
    return candidates[0] if candidates else None


def _external_motor_masks(stage_out: Path) -> list[Path]:
    morf = stage_out / "morfometria"
    patterns = ["freesurfer_corteza_motora_*.nii.gz", "internal_corteza_motora_*.nii.gz"]
    out=[]
    for pat in patterns:
        out.extend([p for p in morf.rglob(pat) if "labelmap" not in p.name.lower() and "cortex_gray" not in p.name.lower() and "prior" not in p.name.lower() and "region" not in p.name.lower() and "roi" not in p.name.lower()])
    return sorted(set(out))


def _external_motor_function_coupling(cfg: PipelineConfig, subject: str, stage: str, stage_out: Path, out_dir: Path, log_file: Path) -> list[dict]:
    rows=[]
    maps_root=stage_out/"mapas"
    map_masks=[p for p in _list_files(maps_root, ["*_mask_p95.nii.gz"]) if "p95" in p.name.lower()]
    ext_masks=_external_motor_masks(stage_out)
    if not map_masks or not ext_masks:
        return rows
    for mm in map_masks:
        try:
            map_mask, map_aff = _load_mask(mm)
            side_map=side_from_text(str(mm))
            for em in ext_masks:
                side_ext=side_from_text(str(em))
                if side_map != "desconocido" and side_ext != "desconocido" and side_map != side_ext:
                    continue
                ext_mask, _ = _load_mask(em)
                ext_rs = resize_to_shape(ext_mask.astype(np.float32), map_mask.shape, order=0, cfg=cfg) > 0.5
                dice,jaccard,inter,map_vox,ext_vox=_dice_jaccard(map_mask, ext_rs)
                n=norm_key(str(em))
                region="m1" if "primaria" in n or "m1" in n else ("m2" if "secundaria" in n or "m2" in n else "motor")
                voxel_ml=_voxel_volume_ml_from_affine(map_aff)
                overlap=map_mask & ext_rs
                label=sanitize_name(f"{region}_{side_map}_{mm.stem}_x_freesurfer_{em.stem}",90)
                out_mask=out_dir/"acoplamiento_mapa_funcional_morfometria_externa"/side_map/f"overlap_{label}.nii.gz"
                save_nifti(overlap.astype(np.uint8), map_aff, out_mask)
                rows.append({
                    "subject":subject,"stage":stage,"analysis":"functional_map_vs_external_motor_cortex",
                    "region":region,"side":side_map,"functional_map_mask":str(mm),"external_motor_mask":str(em),
                    "dice":dice,"jaccard":jaccard,"overlap_voxels":inter,"functional_voxels":map_vox,
                    "external_voxels_resized":ext_vox,"overlap_volume_ml":float(inter*voxel_ml),"out_overlap_mask":str(out_mask),
                    "note":"Acoplamiento mapa funcional vs M1/M2 externa FreeSurfer/CAT12 o morfometría interna cuando está disponible."
                })
        except Exception as exc:
            log_exception(log_file, f"Acoplamiento mapa funcional/morfometría externa/interna {mm}", exc)
    return rows

def _active_cortex_from_maps(cfg: PipelineConfig, subject: str, stage: str, stage_out: Path, out_dir: Path, log_file: Path) -> list[dict]:
    _, nib, _ = _require_neuro_libs()
    rows: list[dict] = []
    res_root = stage_out / "resonancias"
    maps_root = stage_out / "mapas"
    t1 = _best_file(res_root, ["*_volumen.nii.gz"], prefer=["anatomica", "t1"])
    if t1 is None:
        append_log(log_file, f"No encontré T1/volumen anatómico para corteza activa {subject} {stage}")
        return rows
    t1_vol, t1_affine = _load_volume(t1)
    external_cortex = _external_cortex_mask(stage_out)
    if external_cortex is not None:
        shell_ext, shell_aff = _load_mask(external_cortex)
        shell = resize_to_shape(shell_ext.astype(np.float32), t1_vol.shape, order=0, cfg=cfg) > 0.5
        shell_source = str(external_cortex)
        shell_name = "cortical_shell_externa_freesurfer_resampled_to_t1.nii.gz"
    else:
        brain, shell = create_cortical_shell_mask(t1_vol, cfg)
        shell_source = "heuristica_T1"
        shell_name = "t1_cortical_shell_heuristica.nii.gz"
    voxel_ml = _voxel_volume_ml_from_affine(t1_affine)
    save_nifti(shell.astype(np.uint8), t1_affine, out_dir / shell_name)

    map_masks = _list_files(maps_root, ["*_mask_p95.nii.gz", "*_mask_p99.nii.gz"])
    # Procesar p95 primero; p99 queda como análisis de sensibilidad.
    for mp in map_masks:
        try:
            m, _ = _load_mask(mp)
            m_rs = resize_to_shape(m.astype(np.float32), t1_vol.shape, order=0, cfg=cfg) > 0.5
            active_shell = m_rs & shell
            side = side_from_text(str(mp))
            label = sanitize_name(mp.name.replace(".nii.gz", ""), 80)
            out_mask = out_dir / "corteza_activa_guiada_por_mapas" / side / f"{label}_en_shell_t1.nii.gz"
            save_nifti(active_shell.astype(np.uint8), t1_affine, out_mask)
            save_orthogonal_png(active_shell.astype(np.float32), out_mask.with_suffix(".png"), title=f"Corteza activa {subject} {stage} {side}")
            rows.append({
                "subject": subject,
                "stage": stage,
                "analysis": "active_cortex_functional_roi_on_t1_shell",
                "side": side,
                "map_mask": str(mp),
                "t1_reference": str(t1),
                "voxels_active_cortex": int(np.sum(active_shell)),
                "volume_ml_active_cortex": float(np.sum(active_shell) * voxel_ml),
                "out_mask": str(out_mask),
                "shell_source": shell_source,
                "note": "Corteza activa = mapa funcional p95/p99 remuestreado a T1 ∩ shell cortical externa FreeSurfer si existe; si no, shell heurística T1.",
            })
        except Exception as exc:
            log_exception(log_file, f"Corteza activa guiada por mapa {mp}", exc)
    return rows


def _motor_ad_function_coupling(cfg: PipelineConfig, subject: str, stage: str, stage_out: Path, out_dir: Path, log_file: Path) -> list[dict]:
    rows: list[dict] = []
    maps_root = stage_out / "mapas"
    res_root = stage_out / "resonancias"
    map_masks = [p for p in _list_files(maps_root, ["*_mask_p95.nii.gz"]) if "p95" in p.name.lower()]
    ad_masks = _list_files(res_root, ["*corteza_motora_*_wavelet_mask.nii.gz"])
    if not map_masks or not ad_masks:
        append_log(log_file, f"Faltan mapas funcionales o máscaras AD motoras para acoplamiento {subject} {stage}: mapas={len(map_masks)} ad={len(ad_masks)}")
        return rows
    for mm in map_masks:
        try:
            map_mask, map_aff = _load_mask(mm)
            side_map = side_from_text(str(mm))
            for ad in ad_masks:
                n = norm_key(str(ad))
                side_ad = side_from_text(str(ad))
                if side_map != "desconocido" and side_ad != "desconocido" and side_map != side_ad:
                    continue
                region = "m1" if "primaria" in n else ("m2" if "secundaria" in n else "motor")
                ad_mask, _ = _load_mask(ad)
                ad_rs = resize_to_shape(ad_mask.astype(np.float32), map_mask.shape, order=0, cfg=cfg) > 0.5
                dice, jaccard, inter, map_vox, ad_vox = _dice_jaccard(map_mask, ad_rs)
                voxel_ml = _voxel_volume_ml_from_affine(map_aff)
                overlap = map_mask & ad_rs
                label = sanitize_name(f"{region}_{side_map}_{mm.stem}_x_{ad.stem}", 90)
                out_mask = out_dir / "acoplamiento_mapa_funcional_ad" / side_map / f"overlap_{label}.nii.gz"
                save_nifti(overlap.astype(np.uint8), map_aff, out_mask)
                rows.append({
                    "subject": subject,
                    "stage": stage,
                    "analysis": "functional_map_vs_ad_motor_cortex",
                    "region": region,
                    "side": side_map,
                    "functional_map_mask": str(mm),
                    "ad_motor_mask": str(ad),
                    "dice": dice,
                    "jaccard": jaccard,
                    "overlap_voxels": inter,
                    "functional_voxels": map_vox,
                    "ad_voxels_resized": ad_vox,
                    "overlap_volume_ml": float(inter * voxel_ml),
                    "out_overlap_mask": str(out_mask),
                })
        except Exception as exc:
            log_exception(log_file, f"Acoplamiento mapa funcional/AD {mm}", exc)
    return rows


def _ad_motor_vs_control(cfg: PipelineConfig, patient: str, stage: str, rows: list[dict], log_file: Path) -> None:
    stage_out = _result_stage_dir(cfg, patient, stage)
    control_stage = stage if (_result_stage_dir(cfg, cfg.control_name, stage) / "resonancias").exists() else "Antes"
    control_out = _result_stage_dir(cfg, cfg.control_name, control_stage)
    p_csvs = sorted((stage_out / "resonancias").rglob("metricas_corteza_motora_ad.csv"))
    c_csvs = sorted((control_out / "resonancias").rglob("metricas_corteza_motora_ad.csv"))
    if not p_csvs or not c_csvs:
        return
    try:
        dp = pd.concat([pd.read_csv(p) for p in p_csvs], ignore_index=True)
        dc = pd.concat([pd.read_csv(p) for p in c_csvs], ignore_index=True)
        gp = dp.groupby(["region", "side"], dropna=False).agg(volume_ml=("volume_ml", "max"), voxels=("voxels", "max"), energy_mean=("energy_mean", "mean")).reset_index()
        gc = dc.groupby(["region", "side"], dropna=False).agg(volume_ml=("volume_ml", "max"), voxels=("voxels", "max"), energy_mean=("energy_mean", "mean")).reset_index()
        merged = gp.merge(gc, on=["region", "side"], suffixes=("_paciente", "_sano"), how="outer")
        for col in ["volume_ml", "voxels", "energy_mean"]:
            a = f"{col}_paciente"; b = f"{col}_sano"
            merged[f"delta_{col}_paciente_menos_sano"] = pd.to_numeric(merged[a], errors="coerce") - pd.to_numeric(merged[b], errors="coerce")
            merged[f"delta_pct_{col}"] = 100.0 * merged[f"delta_{col}_paciente_menos_sano"] / pd.to_numeric(merged[b], errors="coerce").replace(0, np.nan)
        out_csv = cfg.results_root() / patient / stage / "correlaciones" / "estructura_funcion" / "corteza_motora_ad_vs_sano.csv"
        save_dataframe(merged, out_csv)
        rows.append({"subject": patient, "stage": stage, "analysis": "ad_motor_cortex_vs_control", "control_stage": control_stage, "out_csv": str(out_csv), "rows": int(len(merged))})
    except Exception as exc:
        log_exception(log_file, f"AD motor vs sano estructura_funcion {patient} {stage}", exc)


def _write_method_note(cfg: PipelineConfig) -> Path:
    out = cfg.results_root() / "_metodologia_estructura_funcion.md"
    safe_mkdir(out.parent)
    out.write_text(
        "# Metodología estructura-función incorporada\n\n"
        "Esta versión evita interpretar fMRI BOLD como volumen cortical directo. "
        "El flujo mide morfometría/volumen en T1 o en máscaras corticales derivadas de AD y luego cruza esas regiones con mapas funcionales.\n\n"
        "## Salidas principales\n"
        "- Corteza activa guiada por mapas funcionales sobre shell cortical T1.\n"
        "- Acoplamiento mapa funcional vs corteza motora AD M1/M2.\n"
        "- QC funcional con ALFF/fALFF, tSNR, DVARS y GCOR aproximado cuando hay stack temporal fMRI.\n"
        "- Comparación paciente vs sano y antes vs después mediante CSV y NIfTI.\n\n"
        "## Limitación\n"
        "La segmentación cortical usa FreeSurfer/CAT12 o morfometría interna/fMRIPrep cuando esos derivados existen; si no, cae a máscaras heurísticas. "
        "Para publicación o conclusión clínica se recomienda registro T1↔EPI/AD, QC visual y segmentación anatómica validada.\n",
        encoding="utf-8",
    )
    return out


def run_structure_function_coupling(cfg: PipelineConfig) -> pd.DataFrame:
    """Acoplamiento estructura-función basado en T1, mapas funcionales y AD.

    Implementa la recomendación práctica: no inferir volumen cortical desde BOLD solo,
    sino medir regiones anatómicas/AD y cruzarlas con activación funcional.
    """
    all_rows: list[dict] = []
    subjects = list(cfg.patients) + [cfg.control_name]
    for subject in subjects:
        stages = cfg.stages
        if subject == cfg.control_name:
            stages = tuple(["Antes"] + [s for s in cfg.stages if s != "Antes" and (_subject_stage_dir(cfg, subject, s) is not None)])
        for stage in stages:
            print(f"\n[ESTRUCTURA-FUNCIÓN] {subject} · {stage}")
            out_dir = cfg.results_root() / subject / stage / "estructura_funcion"
            log_file = cfg.results_root() / subject / stage / "reportes" / "estructura_funcion_log.txt"
            try:
                stage_out = _ensure_neuro_results(cfg, subject, stage, log_file)
                _qc_neuroimage(stage_out, out_dir)
                rows = []
                rows.extend(_active_cortex_from_maps(cfg, subject, stage, stage_out, out_dir, log_file))
                rows.extend(_motor_ad_function_coupling(cfg, subject, stage, stage_out, out_dir, log_file))
                rows.extend(_external_motor_function_coupling(cfg, subject, stage, stage_out, out_dir, log_file))
                for r in rows:
                    all_rows.append(r)
                if rows:
                    save_dataframe(pd.DataFrame(rows), out_dir / "resumen_acoplamiento_estructura_funcion.csv")
            except Exception as exc:
                log_exception(log_file, f"Estructura-función global {subject} {stage}", exc)

    # Pacientes vs sano para AD motor.
    for patient in cfg.patients:
        for stage in cfg.stages:
            log_file = cfg.results_root() / patient / stage / "reportes" / "estructura_funcion_log.txt"
            _ad_motor_vs_control(cfg, patient, stage, all_rows, log_file)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        save_dataframe(df, cfg.results_root() / "resumen_global_estructura_funcion.csv")
        try:
            save_dataframe(df, cfg.results_root() / "resumen_global_estructura_funcion.xlsx")
        except Exception:
            pass
    _write_method_note(cfg)
    return df
