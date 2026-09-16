"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ANÁLISIS LONGITUDINAL PRE/POST                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/longitudinal.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Evalúa evolución Antes/Después con diferencias absolutas, relativas y métricas
normalizadas. Para cada variable x se calcula Δx=x_post-x_pre y, cuando
aplica,
Δ%=100(x_post-x_pre)/|x_pre|. También organiza pares temporales para comparar
recuperación hacia referencias internas o externas.

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PipelineConfig
from .paths import first_existing_dir, find_file_by_side_and_test, resolve_stage_dir
from .io_utils import safe_mkdir, save_dataframe, append_log, log_exception
from .checkpoint import CheckpointManager, load_json_if_exists, save_metrics_json
from .signals import analyze_pair
from .neuroimage import (
    _require_neuro_libs,
    _as_3d,
    robust_zscore,
    resize_to_shape,
    save_nifti,
    save_orthogonal_png,
    save_topography_3d,
    sanitize_name,
    _slice_similarity_rows,
    _plot_slice_correlation,
    _safe_pearson,
    _ensure_maps_result,
    _ensure_tomography_result,
    _ensure_resonance_result,
    _map_candidates_by_side,
    _find_best_energy_by_role,
    compare_image_to_control_volume,
    gpu_info,
)


def _stage_dir(cfg: PipelineConfig, patient: str, stage: str) -> Optional[Path]:
    return resolve_stage_dir(cfg.data_root() / patient, stage)


def _modality_dir(stage_dir: Optional[Path], names: tuple[str, ...] | list[str]) -> Optional[Path]:
    if stage_dir is None:
        return None
    return first_existing_dir(stage_dir, names)


def _comparison_root(cfg: PipelineConfig, patient: str) -> Path:
    return safe_mkdir(cfg.results_root() / patient / "comparacion" / "antes_vs_despues")


def _delta_metrics(metrics: dict) -> dict:
    """Agrega deltas simples g - f para métricas generadas por analyze_pair.

    En las comparaciones longitudinales f=Antes y g=Despues.
    """
    out = dict(metrics)
    for key, f_val in list(metrics.items()):
        if not key.startswith("f_"):
            continue
        g_key = "g_" + key[2:]
        if g_key not in metrics:
            continue
        try:
            fv = float(f_val)
            gv = float(metrics[g_key])
            if np.isfinite(fv) and np.isfinite(gv):
                base = key[2:]
                out[f"delta_{base}"] = gv - fv
                out[f"delta_pct_{base}"] = 100.0 * (gv - fv) / (fv if abs(fv) > 1e-12 else np.nan)
        except Exception:
            pass
    return out


def compare_volume_before_after(
    before_path: Path,
    after_path: Path,
    out_dir: Path,
    label: str,
    normalize: str = "robust_zscore",
    cfg: PipelineConfig | None = None,
) -> dict:
    """Compara dos volúmenes del mismo paciente: Después - Antes.

    Guarda cambio, diferencia absoluta, correlación global y correlación corte a corte.
    """
    _, nib, _ = _require_neuro_libs()
    safe_mkdir(out_dir)
    label = sanitize_name(label, 80)

    b_img = nib.load(str(before_path))
    a_img = nib.load(str(after_path))
    b = _as_3d(b_img.get_fdata(dtype=np.float32))
    a = _as_3d(a_img.get_fdata(dtype=np.float32))
    b_resized = resize_to_shape(b, a.shape, order=1, cfg=cfg)

    if normalize == "robust_zscore":
        before_cmp = robust_zscore(b_resized, cfg=cfg)
        after_cmp = robust_zscore(a, cfg=cfg)
    elif normalize == "minmax":
        def mm(x):
            x = np.asarray(x, dtype=np.float32)
            lo, hi = np.nanpercentile(x, [1, 99])
            return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
        before_cmp, after_cmp = mm(b_resized), mm(a)
    else:
        before_cmp, after_cmp = b_resized, a

    change = after_cmp - before_cmp
    absdiff = np.abs(change).astype(np.float32)
    save_nifti(change, a_img.affine, out_dir / f"cambio_{label}.nii.gz")
    save_nifti(absdiff, a_img.affine, out_dir / f"diferencia_absoluta_{label}.nii.gz")
    save_orthogonal_png(change, out_dir / f"cambio_{label}_ortogonal.png", title=f"Cambio longitudinal · Después - Antes · {label}", cmap="coolwarm")
    save_topography_3d(change, out_dir / f"cambio_{label}_topografia_3d.png", title=f"Topografía cambio · Después - Antes · {label}")

    df_slices = _slice_similarity_rows(after_cmp, before_cmp)
    # Renombrar columnas para que el CSV no diga sano.
    rename = {
        "paciente_mean": "despues_mean",
        "sano_mean": "antes_mean",
        "diff_mean": "cambio_mean_despues_menos_antes",
    }
    df_slices = df_slices.rename(columns=rename)
    save_dataframe(df_slices, out_dir / f"correlacion_corte_a_corte_{label}.csv")
    _plot_slice_correlation(df_slices, out_dir / f"correlacion_corte_a_corte_{label}.png", f"Correlación corte a corte · Antes vs Después · {label}")

    metrics = {
        "label": label,
        "before_volume": str(before_path),
        "after_volume": str(after_path),
        "normalize": normalize,
        "shape_before_original": tuple(int(x) for x in b.shape),
        "shape_after": tuple(int(x) for x in a.shape),
        "pearson_global_antes_vs_despues": _safe_pearson(after_cmp, before_cmp, cfg=cfg),
        "mae_global": float(np.nanmean(np.abs(change))),
        "rmse_global": float(np.sqrt(np.nanmean(change ** 2))),
        "cambio_mean_despues_menos_antes": float(np.nanmean(change)),
        "cambio_p05": float(np.nanpercentile(change, 5)),
        "cambio_p95": float(np.nanpercentile(change, 95)),
        "slice_pearson_mean": float(np.nanmean(df_slices["pearson"])) if not df_slices.empty else math.nan,
        "slice_pearson_min": float(np.nanmin(df_slices["pearson"])) if not df_slices.empty else math.nan,
        "slice_pearson_max": float(np.nanmax(df_slices["pearson"])) if not df_slices.empty else math.nan,
        "gpu_backend": gpu_info().get("backend", "cpu") if cfg is not None else "cpu",
        "out_cambio_nii": str(out_dir / f"cambio_{label}.nii.gz"),
        "out_absdiff_nii": str(out_dir / f"diferencia_absoluta_{label}.nii.gz"),
        "out_slice_csv": str(out_dir / f"correlacion_corte_a_corte_{label}.csv"),
        "interpretacion_cambio": "Los volúmenes de cambio se calculan como Después - Antes.",
    }
    save_metrics_json(metrics, out_dir / f"metricas_{label}.json")
    return metrics


def compare_binary_mask_before_after(before_mask_path: Path, after_mask_path: Path, out_dir: Path, label: str, cfg: PipelineConfig | None = None) -> dict:
    _, nib, _ = _require_neuro_libs()
    safe_mkdir(out_dir)
    label = sanitize_name(label, 80)
    b_img = nib.load(str(before_mask_path))
    a_img = nib.load(str(after_mask_path))
    b = _as_3d(b_img.get_fdata(dtype=np.float32)) > 0.5
    a = _as_3d(a_img.get_fdata(dtype=np.float32)) > 0.5
    b_rs = resize_to_shape(b.astype(np.float32), a.shape, order=0, cfg=cfg) > 0.5
    inter = int(np.sum(a & b_rs))
    a_sum = int(np.sum(a))
    b_sum = int(np.sum(b_rs))
    union = int(np.sum(a | b_rs))
    dice = float((2 * inter) / (a_sum + b_sum + 1e-12))
    jaccard = float(inter / (union + 1e-12))
    labelmap = b_rs.astype(np.uint8) + (a.astype(np.uint8) * 2)
    save_nifti(labelmap, a_img.affine, out_dir / f"comparacion_mascara_{label}.nii.gz")
    save_orthogonal_png(labelmap, out_dir / f"comparacion_mascara_{label}.png", title=f"Máscara antes/después · {label}", cmap="viridis")
    metrics = {
        "label": label,
        "before_mask": str(before_mask_path),
        "after_mask": str(after_mask_path),
        "dice_antes_vs_despues": dice,
        "jaccard_antes_vs_despues": jaccard,
        "voxels_antes_resized": b_sum,
        "voxels_despues": a_sum,
        "delta_voxels_despues_menos_antes": a_sum - b_sum,
        "delta_pct_voxels": 100.0 * (a_sum - b_sum) / (b_sum if b_sum else np.nan),
        "voxels_intersection": inter,
        "voxels_union": union,
        "nota_labelmap": "1=Antes, 2=Después, 3=intersección",
    }
    save_metrics_json(metrics, out_dir / f"metricas_mascara_{label}.json")
    return metrics


def _compare_signals_before_after(cfg: PipelineConfig, patient: str, rows: list[dict], log_file: Path) -> None:
    antes_dir = _stage_dir(cfg, patient, "Antes")
    despues_dir = _stage_dir(cfg, patient, "Despues")
    if antes_dir is None or despues_dir is None:
        append_log(log_file, f"Sin carpeta Antes o Despues para señales {patient}: antes={antes_dir}, despues={despues_dir}")
        return

    emg_antes = _modality_dir(antes_dir, [cfg.emg_dirname, "emg"])
    emg_despues = _modality_dir(despues_dir, [cfg.emg_dirname, "emg"])
    din_antes = _modality_dir(antes_dir, [cfg.dyn_dirname, "dinamometria", "dinamometría", "DIN", "Dinamometry"])
    din_despues = _modality_dir(despues_dir, [cfg.dyn_dirname, "dinamometria", "dinamometría", "DIN", "Dinamometry"])
    ckpt = CheckpointManager(cfg)
    comp_root = _comparison_root(cfg, patient)

    # EMG: por prueba, lado y músculo.
    if emg_antes and emg_despues:
        for test in cfg.tests:
            for side in ["derecha", "izquierda"]:
                p_before = find_file_by_side_and_test(emg_antes, side, test, suffix=".csv")
                p_after = find_file_by_side_and_test(emg_despues, side, test, suffix=".csv")
                if not p_before or not p_after:
                    append_log(log_file, f"[EMG longitudinal omitido] {patient} prueba {test} {side}: antes={p_before}, despues={p_after}")
                    continue
                for muscle in ["lateral", "medial"]:
                    out_dir = comp_root / "emg" / f"prueba_{test}" / side / muscle
                    label = f"emg_{side}_{muscle}_prueba_{test}_antes_vs_despues"
                    out_png = out_dir / f"{label}.png"
                    metrics_json = out_dir / f"metricas_{label}.json"
                    task_id = f"comparacion/antes_vs_despues/emg/{patient}/prueba_{test}/{side}/{muscle}"

                    def _work_emg(p_before=p_before, p_after=p_after, side=side, muscle=muscle, out_png=out_png, label=label):
                        metrics = analyze_pair(
                            path_f=p_before,
                            kind_f="EMG",
                            side_f=side,
                            muscle_f=muscle,
                            label_f=f"Antes EMG {side} {muscle}",
                            path_g=p_after,
                            kind_g="EMG",
                            side_g=side,
                            muscle_g=muscle,
                            label_g=f"Después EMG {side} {muscle}",
                            title=f"{patient} · Antes vs Después · EMG {side} {muscle} · Prueba {test}",
                            out_png=out_png,
                            max_freq_plot_hz=cfg.max_freq_plot_hz,
                        )
                        metrics = _delta_metrics(metrics)
                        metrics.update({"patient": patient, "comparison": "Antes_vs_Despues", "modality": "emg", "test": test, "side": side, "muscle": muscle})
                        save_metrics_json(metrics, metrics_json)
                        return metrics
                    try:
                        result, status = ckpt.run(
                            task_id=task_id,
                            inputs=[p_before, p_after],
                            outputs=[out_png, metrics_json],
                            params={"modality": "emg", "test": test, "side": side, "muscle": muscle, "v": "3_7_longitudinal"},
                            fn=_work_emg,
                        )
                        metrics = load_json_if_exists(metrics_json) if status == "skipped" else result
                        if metrics:
                            metrics["checkpoint_status"] = status
                            rows.append(metrics)
                    except Exception as exc:
                        log_exception(log_file, f"Comparación EMG antes/después {patient} prueba {test} {side} {muscle}", exc)

    # Dinamometría: solo una medición antes/después por lado.
    if din_antes and din_despues:
        for side in ["derecha", "izquierda"]:
            p_before = find_file_by_side_and_test(din_antes, side, 1, suffix=".csv")
            p_after = find_file_by_side_and_test(din_despues, side, 1, suffix=".csv")
            if not p_before or not p_after:
                append_log(log_file, f"[DIN longitudinal omitido] {patient} {side}: antes={p_before}, despues={p_after}")
                continue
            out_dir = comp_root / "dinamometria" / "prueba_1" / side
            label = f"dinamometria_{side}_antes_vs_despues"
            out_png = out_dir / f"{label}.png"
            metrics_json = out_dir / f"metricas_{label}.json"
            task_id = f"comparacion/antes_vs_despues/dinamometria/{patient}/{side}"

            def _work_din(p_before=p_before, p_after=p_after, side=side, out_png=out_png, label=label):
                metrics = analyze_pair(
                    path_f=p_before,
                    kind_f="DIN",
                    side_f=side,
                    muscle_f=None,
                    label_f=f"Antes DIN {side}",
                    path_g=p_after,
                    kind_g="DIN",
                    side_g=side,
                    muscle_g=None,
                    label_g=f"Después DIN {side}",
                    title=f"{patient} · Antes vs Después · Dinamometría {side}",
                    out_png=out_png,
                    max_freq_plot_hz=cfg.max_freq_plot_hz,
                )
                metrics = _delta_metrics(metrics)
                metrics.update({"patient": patient, "comparison": "Antes_vs_Despues", "modality": "dinamometria", "test": 1, "side": side})
                save_metrics_json(metrics, metrics_json)
                return metrics
            try:
                result, status = ckpt.run(
                    task_id=task_id,
                    inputs=[p_before, p_after],
                    outputs=[out_png, metrics_json],
                    params={"modality": "dinamometria", "side": side, "v": "3_7_longitudinal"},
                    fn=_work_din,
                )
                metrics = load_json_if_exists(metrics_json) if status == "skipped" else result
                if metrics:
                    metrics["checkpoint_status"] = status
                    rows.append(metrics)
            except Exception as exc:
                log_exception(log_file, f"Comparación DIN antes/después {patient} {side}", exc)


def _compare_biomech_phase_tables(cfg: PipelineConfig, patient: str, rows: list[dict], log_file: Path) -> None:
    before_csv = cfg.results_root() / patient / "Antes" / "reportes" / "biomecanica_metricas_todas_las_pruebas.csv"
    after_csv = cfg.results_root() / patient / "Despues" / "reportes" / "biomecanica_metricas_todas_las_pruebas.csv"
    if not before_csv.exists() or not after_csv.exists():
        append_log(log_file, f"No hay tablas de fases biomecánicas para comparación longitudinal: antes={before_csv.exists()}, despues={after_csv.exists()}")
        return
    try:
        db = pd.read_csv(before_csv)
        da = pd.read_csv(after_csv)
        keys = [k for k in ["test", "phase", "folder", "phase_name"] if k in db.columns and k in da.columns]
        if not keys:
            return
        merged = db.merge(da, on=keys, suffixes=("_antes", "_despues"), how="inner")
        preferred = [
            "pearson_aligned", "rmse_zscore", "xcorr_max_norm", "xcorr_lag_s",
            "coherence_mean", "plv", "f_rms", "g_rms", "f_peak_abs", "g_peak_abs",
            "f_iemg", "g_iemg", "f_max_force", "g_max_force", "f_mean_force", "g_mean_force",
        ]
        for col in preferred:
            a, d = f"{col}_antes", f"{col}_despues"
            if a in merged.columns and d in merged.columns:
                merged[f"delta_{col}"] = pd.to_numeric(merged[d], errors="coerce") - pd.to_numeric(merged[a], errors="coerce")
                denom = pd.to_numeric(merged[a], errors="coerce").replace(0, np.nan)
                merged[f"delta_pct_{col}"] = 100.0 * merged[f"delta_{col}"] / denom
        out_dir = _comparison_root(cfg, patient) / "biomecanica" / "fases_16"
        out_csv = out_dir / "metricas_fases_antes_vs_despues.csv"
        save_dataframe(merged, out_csv)
        try:
            save_dataframe(merged, out_dir / "metricas_fases_antes_vs_despues.xlsx")
        except Exception:
            pass
        rows.append({"patient": patient, "comparison": "Antes_vs_Despues", "modality": "biomecanica_fases_16", "rows": int(len(merged)), "out_csv": str(out_csv)})
    except Exception as exc:
        log_exception(log_file, f"Comparación tablas biomecánicas antes/después {patient}", exc)


def _compare_maps_before_after(cfg: PipelineConfig, patient: str, rows: list[dict], log_file: Path) -> None:
    before_root = _ensure_maps_result(cfg, patient, "Antes", log_file)
    after_root = _ensure_maps_result(cfg, patient, "Despues", log_file)
    if not before_root or not after_root:
        append_log(log_file, f"Sin mapas Antes o Después para comparación longitudinal {patient}")
        return
    ckpt = CheckpointManager(cfg)
    for side in ["derecha", "izquierda", "desconocido"]:
        b_candidates = _map_candidates_by_side(before_root, side, prefer_energy=True)
        a_candidates = _map_candidates_by_side(after_root, side, prefer_energy=True)
        if not b_candidates or not a_candidates:
            continue
        out_dir = _comparison_root(cfg, patient) / "mapas" / side
        label = sanitize_name(f"mapas_{side}_despues_vs_antes", 70)
        metrics_json = out_dir / f"metricas_{label}.json"
        task_id = f"comparacion/antes_vs_despues/mapas/{patient}/{side}"

        def _work():
            metrics = compare_volume_before_after(b_candidates[0], a_candidates[0], out_dir, label, normalize="robust_zscore", cfg=cfg)
            metrics.update({"patient": patient, "comparison": "Antes_vs_Despues", "modality": "mapas", "side": side})
            save_metrics_json(metrics, metrics_json)
            return metrics
        try:
            result, status = ckpt.run(
                task_id=task_id,
                inputs=[b_candidates[0], a_candidates[0]],
                outputs=[metrics_json, out_dir / f"cambio_{label}.nii.gz"],
                params={"modality": "mapas", "side": side, "v": "3_7_longitudinal"},
                fn=_work,
            )
            metrics = load_json_if_exists(metrics_json) if status == "skipped" else result
            if metrics:
                metrics["checkpoint_status"] = status
                rows.append(metrics)
        except Exception as exc:
            log_exception(log_file, f"Comparación mapas antes/después {patient} {side}", exc)


def _compare_tomography_before_after(cfg: PipelineConfig, patient: str, rows: list[dict], log_file: Path) -> None:
    before_root = _ensure_tomography_result(cfg, patient, "Antes", log_file)
    after_root = _ensure_tomography_result(cfg, patient, "Despues", log_file)
    if not before_root or not after_root:
        append_log(log_file, f"Sin tomografía Antes o Después para comparación longitudinal {patient}")
        return
    b_ct = before_root / "tac_hu_original.nii.gz"
    a_ct = after_root / "tac_hu_original.nii.gz"
    if not b_ct.exists() or not a_ct.exists():
        append_log(log_file, f"Falta tac_hu_original para comparación longitudinal {patient}: antes={b_ct.exists()}, despues={a_ct.exists()}")
        return

    ckpt = CheckpointManager(cfg)
    out_dir = _comparison_root(cfg, patient) / "tomografia" / "imagen_por_imagen"
    label = "tac_hu_despues_vs_antes"
    metrics_json = out_dir / f"metricas_{label}.json"
    task_id = f"comparacion/antes_vs_despues/tomografia/volumen/{patient}"

    def _work_ct():
        metrics = compare_volume_before_after(b_ct, a_ct, out_dir, label, normalize="robust_zscore", cfg=cfg)
        metrics.update({"patient": patient, "comparison": "Antes_vs_Despues", "modality": "tomografia"})
        save_metrics_json(metrics, metrics_json)
        return metrics
    try:
        result, status = ckpt.run(
            task_id=task_id,
            inputs=[b_ct, a_ct],
            outputs=[metrics_json, out_dir / f"cambio_{label}.nii.gz", out_dir / f"correlacion_corte_a_corte_{label}.csv"],
            params={"modality": "tomografia", "kind": "imagen_por_imagen", "v": "3_7_longitudinal"},
            fn=_work_ct,
        )
        metrics = load_json_if_exists(metrics_json) if status == "skipped" else result
        if metrics:
            metrics["checkpoint_status"] = status
            rows.append(metrics)
    except Exception as exc:
        log_exception(log_file, f"Comparación TAC volumen antes/después {patient}", exc)

    mask_names = [
        "mask_Vasto_Lateral_Der.nii.gz", "mask_Vasto_Medial_Der.nii.gz",
        "mask_Vasto_Lateral_Izq.nii.gz", "mask_Vasto_Medial_Izq.nii.gz",
        "mask_vastos_derecha.nii.gz", "mask_vastos_izquierda.nii.gz", "mask_todos_vastos.nii.gz",
    ]
    mask_rows = []
    for mask_name in mask_names:
        b_mask = before_root / mask_name
        a_mask = after_root / mask_name
        if not b_mask.exists() or not a_mask.exists():
            continue
        label_mask = sanitize_name(mask_name.replace(".nii.gz", "") + "_despues_vs_antes", 70)
        mask_dir = _comparison_root(cfg, patient) / "tomografia" / "mascaras"
        mask_metrics_json = mask_dir / f"metricas_mascara_{label_mask}.json"
        mask_task_id = f"comparacion/antes_vs_despues/tomografia/mascara/{patient}/{label_mask}"

        def _work_mask(b_mask=b_mask, a_mask=a_mask, label_mask=label_mask):
            metrics = compare_binary_mask_before_after(b_mask, a_mask, mask_dir, label_mask, cfg=cfg)
            metrics.update({"patient": patient, "comparison": "Antes_vs_Despues", "modality": "tomografia_mascara"})
            save_metrics_json(metrics, mask_metrics_json)
            return metrics
        try:
            result, status = ckpt.run(
                task_id=mask_task_id,
                inputs=[b_mask, a_mask],
                outputs=[mask_metrics_json, mask_dir / f"comparacion_mascara_{label_mask}.nii.gz"],
                params={"modality": "tomografia_mascara", "mask": mask_name, "v": "3_7_longitudinal"},
                fn=_work_mask,
            )
            metrics = load_json_if_exists(mask_metrics_json) if status == "skipped" else result
            if metrics:
                metrics["checkpoint_status"] = status
                rows.append(metrics)
                mask_rows.append(metrics)
        except Exception as exc:
            log_exception(log_file, f"Comparación máscara TAC antes/después {mask_name} {patient}", exc)
    if mask_rows:
        save_dataframe(pd.DataFrame(mask_rows), _comparison_root(cfg, patient) / "tomografia" / "resumen_mascaras_antes_vs_despues.csv")

    # Tablas de volumen/área: Después - Antes.
    try:
        vol_b = before_root / "volumenes_y_areas.csv"
        vol_a = after_root / "volumenes_y_areas.csv"
        if vol_b.exists() and vol_a.exists():
            db = pd.read_csv(vol_b)
            da = pd.read_csv(vol_a)
            merged = db.merge(da, on="musculo", suffixes=("_antes", "_despues"), how="inner")
            if not merged.empty:
                for col in ["volumen_cm3", "area_transversal_max_cm2", "area_transversal_media_cm2"]:
                    b, a = f"{col}_antes", f"{col}_despues"
                    if b in merged and a in merged:
                        merged[f"delta_{col}_despues_menos_antes"] = pd.to_numeric(merged[a], errors="coerce") - pd.to_numeric(merged[b], errors="coerce")
                        denom = pd.to_numeric(merged[b], errors="coerce").replace(0, np.nan)
                        merged[f"delta_pct_{col}"] = 100.0 * merged[f"delta_{col}_despues_menos_antes"] / denom
                out_tab = _comparison_root(cfg, patient) / "tomografia" / "comparacion_volumenes_areas_antes_vs_despues.csv"
                save_dataframe(merged, out_tab)
                rows.append({"patient": patient, "comparison": "Antes_vs_Despues", "modality": "tomografia_tabular", "label": "volumenes_areas_antes_vs_despues", "rows": int(len(merged)), "out_csv": str(out_tab)})
        ar_b = before_root / "areas_por_corte.csv"
        ar_a = after_root / "areas_por_corte.csv"
        if ar_b.exists() and ar_a.exists():
            db = pd.read_csv(ar_b)
            da = pd.read_csv(ar_a)
            area_rows = []
            for musculo in sorted(set(db.get("musculo", [])) & set(da.get("musculo", []))):
                ab = db[db["musculo"] == musculo].sort_values("z")["area_cm2"].to_numpy(dtype=float)
                aa = da[da["musculo"] == musculo].sort_values("z")["area_cm2"].to_numpy(dtype=float)
                if len(ab) and len(aa):
                    x_old = np.linspace(0, 1, len(ab))
                    x_new = np.linspace(0, 1, len(aa))
                    ab_rs = np.interp(x_new, x_old, ab)
                    area_rows.append({
                        "musculo": musculo,
                        "pearson_area_por_corte_antes_vs_despues": _safe_pearson(aa, ab_rs),
                        "mae_area_cm2": float(np.nanmean(np.abs(aa - ab_rs))),
                        "rmse_area_cm2": float(np.sqrt(np.nanmean((aa - ab_rs) ** 2))),
                        "delta_area_media_cm2": float(np.nanmean(aa) - np.nanmean(ab)),
                        "n_cortes_antes": int(len(ab)),
                        "n_cortes_despues": int(len(aa)),
                    })
            if area_rows:
                out_area = _comparison_root(cfg, patient) / "tomografia" / "correlacion_areas_por_corte_antes_vs_despues.csv"
                save_dataframe(pd.DataFrame(area_rows), out_area)
                rows.append({"patient": patient, "comparison": "Antes_vs_Despues", "modality": "tomografia_areas_por_corte", "label": "areas_por_corte_antes_vs_despues", "rows": len(area_rows), "out_csv": str(out_area)})
    except Exception as exc:
        log_exception(log_file, f"Correlaciones tabulares TAC antes/después {patient}", exc)


def _compare_resonances_before_after(cfg: PipelineConfig, patient: str, rows: list[dict], log_file: Path) -> None:
    before_root = _ensure_resonance_result(cfg, patient, "Antes", log_file)
    after_root = _ensure_resonance_result(cfg, patient, "Despues", log_file)
    if not before_root or not after_root:
        append_log(log_file, f"Sin resonancias Antes o Después para comparación longitudinal {patient}")
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
            b_candidates = _find_best_energy_by_role(before_root, role, subtype, side)
            a_candidates = _find_best_energy_by_role(after_root, role, subtype, side)
            if not b_candidates or not a_candidates:
                continue
            out_dir = _comparison_root(cfg, patient) / "resonancias" / role / subtype / side
            label = sanitize_name(f"resonancia_{role}_{subtype}_{side}_despues_vs_antes", 80)
            metrics_json = out_dir / f"metricas_{label}.json"
            task_id = f"comparacion/antes_vs_despues/resonancias/{patient}/{role}/{subtype}/{side}"

            def _work():
                metrics = compare_volume_before_after(b_candidates[0], a_candidates[0], out_dir, label, normalize="robust_zscore", cfg=cfg)
                metrics.update({"patient": patient, "comparison": "Antes_vs_Despues", "modality": "resonancias", "series_role": role, "series_subtype": subtype, "side": side})
                save_metrics_json(metrics, metrics_json)
                return metrics
            try:
                result, status = ckpt.run(
                    task_id=task_id,
                    inputs=[b_candidates[0], a_candidates[0]],
                    outputs=[metrics_json, out_dir / f"cambio_{label}.nii.gz"],
                    params={"role": role, "subtype": subtype, "side": side, "v": "3_7_longitudinal"},
                    fn=_work,
                )
                metrics = load_json_if_exists(metrics_json) if status == "skipped" else result
                if metrics:
                    metrics["checkpoint_status"] = status
                    rows.append(metrics)
            except Exception as exc:
                log_exception(log_file, f"Comparación resonancia antes/después {role}/{subtype}/{side} {patient}", exc)



def _find_ad_motor_cortex_csvs(root: Path) -> list[Path]:
    if root is None or not root.exists():
        return []
    return sorted(root.rglob('metricas_corteza_motora_ad.csv'))


def _compare_ad_motor_cortex_before_after(cfg: PipelineConfig, patient: str, rows: list[dict], log_file: Path) -> None:
    before_root = _ensure_resonance_result(cfg, patient, 'Antes', log_file)
    after_root = _ensure_resonance_result(cfg, patient, 'Despues', log_file)
    if not before_root or not after_root:
        return
    before_csvs = _find_ad_motor_cortex_csvs(before_root)
    after_csvs = _find_ad_motor_cortex_csvs(after_root)
    if not before_csvs or not after_csvs:
        append_log(log_file, f'Sin CSV de corteza motora AD para comparación longitudinal {patient}: antes={len(before_csvs)}, despues={len(after_csvs)}')
        return
    try:
        db = pd.concat([pd.read_csv(p) for p in before_csvs], ignore_index=True)
        da = pd.concat([pd.read_csv(p) for p in after_csvs], ignore_index=True)
    except Exception as exc:
        log_exception(log_file, f'Leyendo CSV corteza motora AD {patient}', exc)
        return

    if db.empty or da.empty:
        return

    # Agrega por región/lado para evitar depender de nombres de serie idénticos.
    agg_cols = {'volume_ml': 'max', 'voxels': 'max', 'energy_mean': 'mean', 'energy_p95': 'mean', 'wavelet_threshold': 'mean'}
    gb = db.groupby(['region', 'side'], dropna=False).agg(agg_cols).reset_index()
    ga = da.groupby(['region', 'side'], dropna=False).agg(agg_cols).reset_index()
    merged = gb.merge(ga, on=['region', 'side'], how='outer', suffixes=('_antes', '_despues'))
    if merged.empty:
        return
    for col in ['volume_ml', 'voxels', 'energy_mean', 'energy_p95']:
        b = f'{col}_antes'
        a = f'{col}_despues'
        if b in merged.columns and a in merged.columns:
            merged[f'delta_{col}_despues_menos_antes'] = pd.to_numeric(merged[a], errors='coerce') - pd.to_numeric(merged[b], errors='coerce')
            denom = pd.to_numeric(merged[b], errors='coerce').replace(0, np.nan)
            merged[f'delta_pct_{col}'] = 100.0 * merged[f'delta_{col}_despues_menos_antes'] / denom
    merged['aumento_volumen_despues'] = merged.get('delta_volume_ml_despues_menos_antes', np.nan) > 0
    out_dir = _comparison_root(cfg, patient) / 'resonancias' / 'corteza_motora_ad'
    out_csv = out_dir / 'comparacion_corteza_motora_ad_antes_vs_despues.csv'
    save_dataframe(merged, out_csv)
    rows.append({
        'patient': patient,
        'comparison': 'Antes_vs_Despues',
        'modality': 'resonancias_corteza_motora_ad',
        'label': 'corteza_motora_ad_antes_vs_despues',
        'rows': int(len(merged)),
        'out_csv': str(out_csv),
    })



def _compare_external_morphometry_before_after(cfg: PipelineConfig, patient: str, rows: list[dict], log_file: Path) -> None:
    before_csv = cfg.results_root() / patient / "Antes" / "morfometria" / "resumen_morfometria_externa.csv"
    after_csv = cfg.results_root() / patient / "Despues" / "morfometria" / "resumen_morfometria_externa.csv"
    if not before_csv.exists() or not after_csv.exists():
        append_log(log_file, f"Sin morfometría externa Antes/Después para {patient}: antes={before_csv.exists()}, despues={after_csv.exists()}")
        return
    try:
        db = pd.read_csv(before_csv)
        da = pd.read_csv(after_csv)
        # Mantener solo métricas regionales motoras cuando existen.
        keys=[k for k in ["tool","source_type","region","region_name","region_detail","side"] if k in db.columns and k in da.columns]
        if not {"region","side"}.issubset(set(keys)):
            return
        gb = db.groupby(keys, dropna=False).agg(volume_ml=("volume_ml", "max"), voxels=("voxels", "max") if "voxels" in db.columns else ("volume_ml", "count")).reset_index()
        ga = da.groupby(keys, dropna=False).agg(volume_ml=("volume_ml", "max"), voxels=("voxels", "max") if "voxels" in da.columns else ("volume_ml", "count")).reset_index()
        merged = gb.merge(ga, on=keys, how="outer", suffixes=("_antes", "_despues"))
        if merged.empty:
            return
        for col in ["volume_ml", "voxels"]:
            b=f"{col}_antes"; a=f"{col}_despues"
            if b in merged.columns and a in merged.columns:
                merged[f"delta_{col}_despues_menos_antes"] = pd.to_numeric(merged[a], errors="coerce") - pd.to_numeric(merged[b], errors="coerce")
                denom=pd.to_numeric(merged[b], errors="coerce").replace(0, np.nan)
                merged[f"delta_pct_{col}"] = 100.0 * merged[f"delta_{col}_despues_menos_antes"] / denom
        out_dir = _comparison_root(cfg, patient) / "morfometria_externa"
        out_csv = out_dir / "comparacion_morfometria_externa_antes_vs_despues.csv"
        save_dataframe(merged, out_csv)
        rows.append({"patient": patient, "comparison":"Antes_vs_Despues", "modality":"morfometria_externa", "label":"morfometria_externa_antes_vs_despues", "rows": int(len(merged)), "out_csv": str(out_csv)})
    except Exception as exc:
        log_exception(log_file, f"Comparación morfometría externa antes/después {patient}", exc)

def _compare_internal_morphometry_before_after(cfg: PipelineConfig, patient: str, rows: list[dict], log_file: Path) -> None:
    before_csv = cfg.results_root() / patient / "Antes" / "morfometria" / "resumen_morfometria_interna.csv"
    after_csv = cfg.results_root() / patient / "Despues" / "morfometria" / "resumen_morfometria_interna.csv"
    if not before_csv.exists() or not after_csv.exists():
        append_log(log_file, f"Sin morfometría interna Antes/Después para {patient}: antes={before_csv.exists()}, despues={after_csv.exists()}")
        return
    try:
        db = pd.read_csv(before_csv)
        da = pd.read_csv(after_csv)
        keys = [k for k in ["region", "region_name", "side"] if k in db.columns and k in da.columns]
        if not {"region", "side"}.issubset(set(keys)):
            return
        gb = db.groupby(keys, dropna=False).agg(volume_ml=("volume_ml", "max"), voxels=("voxels", "max"), energy_mean=("energy_mean", "mean"), energy_p95=("energy_p95", "mean")).reset_index()
        ga = da.groupby(keys, dropna=False).agg(volume_ml=("volume_ml", "max"), voxels=("voxels", "max"), energy_mean=("energy_mean", "mean"), energy_p95=("energy_p95", "mean")).reset_index()
        merged = gb.merge(ga, on=keys, how="outer", suffixes=("_antes", "_despues"))
        if merged.empty:
            return
        for col in ["volume_ml", "voxels", "energy_mean", "energy_p95"]:
            b = f"{col}_antes"; a = f"{col}_despues"
            if b in merged.columns and a in merged.columns:
                merged[f"delta_{col}_despues_menos_antes"] = pd.to_numeric(merged[a], errors="coerce") - pd.to_numeric(merged[b], errors="coerce")
                denom = pd.to_numeric(merged[b], errors="coerce").replace(0, np.nan)
                merged[f"delta_pct_{col}"] = 100.0 * merged[f"delta_{col}_despues_menos_antes"] / denom
        merged["aumento_volumen_despues"] = merged.get("delta_volume_ml_despues_menos_antes", np.nan) > 0
        out_dir = _comparison_root(cfg, patient) / "morfometria_interna"
        out_csv = out_dir / "comparacion_morfometria_interna_antes_vs_despues.csv"
        save_dataframe(merged, out_csv)
        rows.append({"patient": patient, "comparison": "Antes_vs_Despues", "modality": "morfometria_interna", "label": "morfometria_interna_antes_vs_despues", "rows": int(len(merged)), "out_csv": str(out_csv)})
    except Exception as exc:
        log_exception(log_file, f"Comparación morfometría interna antes/después {patient}", exc)


def run_before_after_comparisons(cfg: PipelineConfig) -> pd.DataFrame:
    """Comparación longitudinal Antes vs Después por paciente.

    Crea resultados/<paciente>/comparacion/antes_vs_despues con:
    - emg: correlación señal cruda antes/después por prueba, lado y músculo.
    - dinamometria: correlación de fuerza antes/después por lado.
    - biomecanica/fases_16: deltas de las 16 fases si ya existen sus tablas.
    - mapas: cambios NIfTI/PNG/CSV por lateralidad.
    - tomografia: TAC HU imagen por imagen, máscaras, volúmenes y áreas.
    - resonancias: VCE, tractografía, DTI, fMRI y T1 si existen ambos tiempos.
    """
    all_rows: list[dict] = []
    for patient in cfg.patients:
        print(f"\n[COMPARACIÓN ANTES VS DESPUÉS] {patient}")
        comp_root = _comparison_root(cfg, patient)
        log_file = comp_root / "comparacion_antes_vs_despues_log.txt"
        try:
            _compare_signals_before_after(cfg, patient, all_rows, log_file)
            _compare_biomech_phase_tables(cfg, patient, all_rows, log_file)
            _compare_maps_before_after(cfg, patient, all_rows, log_file)
            _compare_tomography_before_after(cfg, patient, all_rows, log_file)
            _compare_resonances_before_after(cfg, patient, all_rows, log_file)
            _compare_ad_motor_cortex_before_after(cfg, patient, all_rows, log_file)
            _compare_external_morphometry_before_after(cfg, patient, all_rows, log_file)
            _compare_internal_morphometry_before_after(cfg, patient, all_rows, log_file)
        except Exception as exc:
            log_exception(log_file, f"Comparación longitudinal global {patient}", exc)

        df_patient = pd.json_normalize([r for r in all_rows if r.get("patient") == patient]) if all_rows else pd.DataFrame()
        if not df_patient.empty:
            save_dataframe(df_patient, comp_root / "resumen_comparacion_antes_vs_despues.csv")
            try:
                save_dataframe(df_patient, comp_root / "resumen_comparacion_antes_vs_despues.xlsx")
            except Exception:
                pass

    df = pd.json_normalize(all_rows) if all_rows else pd.DataFrame()
    if not df.empty:
        save_dataframe(df, cfg.results_root() / "resumen_global_comparacion_antes_vs_despues.csv")
        try:
            save_dataframe(df, cfg.results_root() / "resumen_global_comparacion_antes_vs_despues.xlsx")
        except Exception:
            pass
    return df
