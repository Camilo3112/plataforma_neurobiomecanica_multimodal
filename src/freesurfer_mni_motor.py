"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       FREESURFER, MNI Y CORTEZA MOTORA                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/freesurfer_mni_motor.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Relaciona espacio nativo del paciente con referencias atlas/MNI. La
transferencia
de etiquetas anatómicas usa composición de transformaciones T_native<-MNI y
remuestreo categórico. La segmentación cortical deriva de reconstrucciones
FreeSurfer y mapas como aparc+aseg, aseg y superficies corticales. Las
métricas
volumétricas siguen V = N_vox * |det(A_3x3)|, donde N_vox es el número de
voxeles
incluidos en la ROI.

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
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .paths import resolve_stage_dir, norm_key
from .io_utils import safe_mkdir, save_dataframe, save_json, log_exception
from .checkpoint import CheckpointManager
from .external_morphometry import (
    _find_freesurfer_subject_dirs,
    _process_fs_aparc_masks,
    _fs_stats_motor_summary,
)

MOTOR_PROTOCOL_NOTE = (
    "FreeSurfer principal: M1=precentral; M2 aproximada/premotora-SMA="
    "caudalmiddlefrontal+superiorfrontal+paracentral. "
    "MNI se usa como respaldo/validacion si existen transformaciones fMRIPrep/ANTs; "
    "la medicion final recomendada se hace en espacio nativo FreeSurfer/paciente."
)


def _control_stages(cfg: PipelineConfig) -> list[str]:
    control_dir = cfg.data_root() / cfg.control_name
    stages = [st for st in cfg.stages if resolve_stage_dir(control_dir, st, fallback_to_subject=False) is not None]
    if not stages and control_dir.exists():
        stages = [cfg.stages[0]]
    return stages


def _iter_subject_stage(cfg: PipelineConfig):
    for patient in cfg.patients:
        for stage in cfg.stages:
            yield patient, stage
    for stage in _control_stages(cfg):
        yield cfg.control_name, stage


def _find_mni_transform_candidates(cfg: PipelineConfig, subject: str, stage: str) -> list[Path]:
    roots: list[Path] = []
    roots.append(cfg.results_root() / "derivados_externos" / "fmriprep")
    roots.append(cfg.results_root() / "derivados_externos")
    roots.append(cfg.project_root / "derivatives")
    roots.append(cfg.data_root() / "derivatives")
    roots = [r for r in roots if r.exists()]
    subj_norm = norm_key(subject).replace(" ", "")
    st_norm = norm_key(stage).replace("é", "e")
    patterns = [
        "*from-MNI*to-T1w*xfm*", "*from-T1w*to-MNI*xfm*",
        "*from-MNI*to-anat*xfm*", "*from-anat*to-MNI*xfm*",
        "*MNI*xfm*", "*mni*xfm*", "*.h5", "*.mat",
    ]
    out=[]; seen=set()
    for root in roots:
        for pat in patterns:
            for p in root.rglob(pat):
                if not p.is_file():
                    continue
                s = norm_key(str(p)).replace(" ", "")
                if subj_norm in s or st_norm in s or subject == cfg.control_name:
                    rp=str(p.resolve())
                    if rp not in seen:
                        out.append(p); seen.add(rp)
    return sorted(out)


def _copy_key_outputs_to_final(mask_rows: pd.DataFrame, final_dir: Path) -> pd.DataFrame:
    safe_mkdir(final_dir)
    rows=[]
    if mask_rows is None or mask_rows.empty or "mask_path" not in mask_rows.columns:
        return pd.DataFrame()
    for _, r in mask_rows.iterrows():
        src = Path(str(r.get("mask_path", "")))
        if not src.exists():
            continue
        region = str(r.get("region", "motor"))
        region_name = str(r.get("region_name", r.get("region", "motor")))
        side = str(r.get("side", "desconocido"))
        dst = final_dir / f"corteza_motora_{region}_{region_name}_{side}_freesurfer_native.nii.gz"
        try:
            shutil.copy2(src, dst)
            row = dict(r)
            row["final_mask_path"] = str(dst)
            row["recommended_for_protocol"] = "si"
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


def _add_asymmetry_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "volume_ml" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["volume_ml"] = pd.to_numeric(d["volume_ml"], errors="coerce")
    if "source_type" in d.columns:
        d = d[d["source_type"].astype(str).str.contains("mask|stats|aparc", case=False, na=False)]
    rows=[]
    for (subject, stage, region), g in d.groupby(["subject", "stage", "region"], dropna=False):
        gd = g.dropna(subset=["volume_ml"])
        if gd.empty or "side" not in gd.columns:
            continue
        by_side = gd.groupby("side")["volume_ml"].sum()
        left = float(by_side.get("izquierda", np.nan))
        right = float(by_side.get("derecha", np.nan))
        if np.isfinite(left) and np.isfinite(right) and (left + right) > 0:
            rows.append({
                "subject": subject,
                "stage": stage,
                "region": region,
                "metric_name": f"asimetria_{region}_pct",
                "metric_value": 100.0 * abs(right - left) / ((right + left) / 2.0),
                "left_volume_ml": left,
                "right_volume_ml": right,
                "interpretation": "Asimetría derecha/izquierda; menor sugiere mayor simetría anatómica de la ROI.",
            })
    return pd.DataFrame(rows)


def _stability_across_patients(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "volume_ml" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["volume_ml"] = pd.to_numeric(d["volume_ml"], errors="coerce")
    d = d.dropna(subset=["volume_ml"])
    rows=[]
    keys = [c for c in ["stage", "region", "region_name", "side", "source_type"] if c in d.columns]
    if not keys:
        return pd.DataFrame()
    for key, g in d.groupby(keys, dropna=False):
        vals = g["volume_ml"].astype(float).to_numpy()
        if len(vals) < 2:
            continue
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(keys, key_tuple))
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else math.nan
        rows.append({
            **key_dict,
            "n": int(len(vals)),
            "mean_volume_ml": mean,
            "std_volume_ml": std,
            "cv_volume_pct": float(100.0 * std / mean) if abs(mean) > 1e-12 else math.nan,
            "min_volume_ml": float(np.nanmin(vals)),
            "max_volume_ml": float(np.nanmax(vals)),
            "interpretation": "CV bajo en volumen de corteza motora sugiere mayor estabilidad del protocolo de ROI entre sujetos.",
        })
    return pd.DataFrame(rows)


def _before_after_change(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "volume_ml" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["volume_ml"] = pd.to_numeric(d["volume_ml"], errors="coerce")
    d = d.dropna(subset=["volume_ml"])
    keys=["subject", "region", "region_name", "side", "source_type"]
    keys=[k for k in keys if k in d.columns]
    if not keys:
        return pd.DataFrame()
    means = d.groupby(keys + ["stage"], dropna=False)["volume_ml"].mean().reset_index()
    piv = means.pivot_table(index=keys, columns="stage", values="volume_ml", aggfunc="mean").reset_index()
    if "Antes" not in piv.columns or "Despues" not in piv.columns:
        return pd.DataFrame()
    piv["delta_volume_ml_despues_menos_antes"] = piv["Despues"] - piv["Antes"]
    piv["delta_volume_pct"] = 100.0 * piv["delta_volume_ml_despues_menos_antes"] / piv["Antes"].replace(0, np.nan)
    piv["interpretation"] = "Cambio longitudinal de volumen atlas-guiado; cambios muy grandes sugieren revisar registro/segmentación."
    return piv


def process_subject_stage_freesurfer_motor(cfg: PipelineConfig, subject: str, stage: str) -> dict:
    out_dir = cfg.results_root() / subject / stage / "morfometria" / "freesurfer_mni_motor"
    safe_mkdir(out_dir)
    log_file = cfg.results_root() / subject / stage / "reportes" / "freesurfer_mni_motor_log.txt"
    fs_dirs = _find_freesurfer_subject_dirs(cfg, subject, stage)
    mni_candidates = _find_mni_transform_candidates(cfg, subject, stage)
    save_dataframe(pd.DataFrame({
        "freesurfer_subject_dir": [str(p) for p in fs_dirs] or [""],
        "status": ["detectado"] * len(fs_dirs) if fs_dirs else ["no_detectado"],
    }), out_dir / "freesurfer_subjects_detectados.csv")
    if mni_candidates:
        save_dataframe(pd.DataFrame([{"mni_transform_candidate": str(p)} for p in mni_candidates]), out_dir / "mni_transformaciones_detectadas.csv")
    else:
        save_dataframe(pd.DataFrame([{"status": "sin_transformaciones_mni_detectadas", "nota": "Para usar MNI real se requieren transformaciones MNI<->T1w de fMRIPrep/ANTs."}]), out_dir / "mni_transformaciones_detectadas.csv")
    if not fs_dirs:
        status = {
            "subject": subject,
            "stage": stage,
            "status": "sin_freesurfer",
            "out_dir": str(out_dir),
            "next_step": "Ejecutar ./run_freesurfer_mni_motor_linux.sh después de instalar FreeSurfer y license.txt.",
            "note": MOTOR_PROTOCOL_NOTE,
        }
        save_json(status, out_dir / "estado_freesurfer_mni_motor.json")
        save_dataframe(pd.DataFrame([status]), out_dir / "metricas_corteza_motora_fs_mni.csv")
        return status
    all_dfs=[]
    fs_dir = fs_dirs[0]
    try:
        mask_df = _process_fs_aparc_masks(cfg, fs_dir, out_dir)
        if mask_df is not None and not mask_df.empty:
            mask_df.insert(0, "stage", stage)
            mask_df.insert(0, "subject", subject)
            mask_df["method_priority"] = "principal_freesurfer_native"
            all_dfs.append(mask_df)
            final_df = _copy_key_outputs_to_final(mask_df, out_dir / "roi_motor_final")
            if not final_df.empty:
                all_dfs.append(final_df)
        stats_df = _fs_stats_motor_summary(fs_dir)
        if stats_df is not None and not stats_df.empty:
            stats_df.insert(0, "stage", stage)
            stats_df.insert(0, "subject", subject)
            stats_df["method_priority"] = "principal_freesurfer_stats"
            all_dfs.append(stats_df)
    except Exception as exc:
        log_exception(log_file, f"FreeSurfer/MNI motor {subject} {stage} {fs_dir}", exc)
    df = pd.concat(all_dfs, ignore_index=True, sort=False) if all_dfs else pd.DataFrame()
    if not df.empty:
        df["protocol_note"] = MOTOR_PROTOCOL_NOTE
        save_dataframe(df, out_dir / "metricas_corteza_motora_fs_mni.csv")
        asym = _add_asymmetry_metrics(df)
        if not asym.empty:
            save_dataframe(asym, out_dir / "metricas_asimetria_corteza_motora.csv")
    else:
        save_dataframe(pd.DataFrame([{"subject": subject, "stage": stage, "status": "freesurfer_detectado_sin_metricas", "fs_dir": str(fs_dir)}]), out_dir / "metricas_corteza_motora_fs_mni.csv")
    status={
        "subject": subject,
        "stage": stage,
        "status": "ok" if not df.empty else "sin_metricas",
        "fs_dir": str(fs_dir),
        "out_dir": str(out_dir),
        "n_metric_rows": int(len(df)),
        "mni_transform_candidates": [str(p) for p in mni_candidates],
        "note": MOTOR_PROTOCOL_NOTE,
    }
    save_json(status, out_dir / "estado_freesurfer_mni_motor.json")
    return status


def run_freesurfer_mni_motor(cfg: PipelineConfig) -> pd.DataFrame:
    ckpt = CheckpointManager(cfg)
    statuses=[]
    for subject, stage in _iter_subject_stage(cfg):
        print(f"\n[FS/MNI MOTOR] {subject} · {stage}")
        task_id=f"freesurfer_mni_motor/{subject}/{stage}"
        out_dir = cfg.results_root() / subject / stage / "morfometria" / "freesurfer_mni_motor"
        def _work(subject=subject, stage=stage):
            return process_subject_stage_freesurfer_motor(cfg, subject, stage)
        result, status = ckpt.run(
            task_id=task_id,
            inputs=[cfg.results_root()/"derivados_externos"],
            outputs=[out_dir / "metricas_corteza_motora_fs_mni.csv"],
            params={"version": "v3_21_freesurfer_mni_motor", "note": MOTOR_PROTOCOL_NOTE},
            fn=_work,
        )
        result = result or {}
        result["checkpoint_status"] = status
        statuses.append(result)
    dfs=[]
    for subject, stage in _iter_subject_stage(cfg):
        p = cfg.results_root() / subject / stage / "morfometria" / "freesurfer_mni_motor" / "metricas_corteza_motora_fs_mni.csv"
        if p.exists():
            try:
                d=pd.read_csv(p)
                if not d.empty:
                    dfs.append(d)
            except Exception:
                pass
    global_df=pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame(statuses)
    if not global_df.empty:
        save_dataframe(global_df, cfg.results_root() / "resumen_global_corteza_motora_freesurfer_mni.csv")
        stab=_stability_across_patients(global_df)
        if not stab.empty:
            save_dataframe(stab, cfg.results_root() / "qc_estabilidad_volumen_corteza_motora_freesurfer.csv")
        change=_before_after_change(global_df)
        if not change.empty:
            save_dataframe(change, cfg.results_root() / "comparacion_antes_despues_volumen_corteza_motora_freesurfer.csv")
    return global_df
