"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  CONSOLIDACIÓN MULTIMODAL DE BIOMARCADORES                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/consolidation.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Consolida métricas multimodales en tablas comparables. El modelo transforma
datos
heterogéneos en variables indexadas por paciente, etapa, dominio y
biomarcador.
Permite construir matrices X_{paciente,variable}, calcular deltas y preparar
correlaciones o reportes sin alterar los archivos fuente.

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
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .io_utils import safe_mkdir, save_dataframe, append_log


EXCLUDE_DIR_NAMES = {
    "consolidado",
    "__pycache__",
}

EXCLUDE_FILE_TOKENS = {
    "checkpoints",
    "pipeline_eventos",
    "labelmap_lut",
}


METRIC_EXPLANATIONS: list[tuple[str, str, str]] = [
    (r"^(patient|subject|sujeto)$", "Identificador del sujeto/paciente dentro del proyecto.", "texto"),
    (r"^stage$", "Etapa de adquisición: Antes o Despues.", "texto"),
    (r"^comparison$", "Tipo de comparación realizada, por ejemplo Antes_vs_Despues o Paciente_vs_Sano.", "texto"),
    (r"^modality$", "Módulo o modalidad que generó la métrica: biomecánica, tomografía, mapas, resonancias, morfometría, estructura-función, etc.", "texto"),
    (r"^source|source$|archivo|path|ruta", "Ruta o archivo fuente usado para generar la métrica. Sirve para trazabilidad.", "ruta/texto"),
    (r"^test$|prueba", "Número de prueba usada en EMG/dinamometría.", "índice"),
    (r"^phase$|fase", "Fase biomecánica del protocolo de 16 fases.", "índice/texto"),
    (r"pearson|corr", "Correlación lineal. Valores cercanos a 1 indican alta similitud; 0 poca relación lineal; negativos relación inversa.", "adimensional"),
    (r"spearman", "Correlación por rangos, menos sensible a no linealidades y valores extremos que Pearson.", "adimensional"),
    (r"dice", "Coeficiente Dice de solapamiento entre dos máscaras: 1 perfecto, 0 sin solapamiento.", "0-1"),
    (r"jaccard", "Índice Jaccard de solapamiento entre máscaras: intersección/unión.", "0-1"),
    (r"mae", "Error absoluto medio entre dos señales/volúmenes/curvas.", "depende de la variable"),
    (r"rmse", "Raíz del error cuadrático medio; penaliza más los errores grandes.", "depende de la variable"),
    (r"delta_pct|porcentaje|percent", "Cambio porcentual respecto al valor de referencia.", "%"),
    (r"^delta|cambio|diff|diferencia|deficit", "Cambio o diferencia entre condiciones. En comparaciones longitudinales suele ser Después - Antes.", "depende de la variable"),
    (r"volume_ml|volumen_ml|volume_cm3|volumen_cm3", "Volumen estimado de la región/máscara/tejido.", "mL o cm³"),
    (r"voxels|voxeles", "Cantidad de voxeles incluidos en una máscara o región.", "voxeles"),
    (r"area_transversal|area_cm2|area", "Área transversal estimada, normalmente por corte o región muscular/cortical.", "cm² o unidades de imagen"),
    (r"energy_mean|energy_p95|wavelet|energia", "Energía multiescala tipo wavelet/LoG; resalta bordes, activación o cambios espaciales según el volumen analizado.", "intensidad normalizada"),
    (r"threshold|umbral", "Umbral usado para crear máscara o seleccionar región de alta señal/energía.", "intensidad"),
    (r"mean|media|promedio", "Promedio de la variable en la señal, volumen o región.", "depende de la variable"),
    (r"median|mediana|p50", "Mediana o percentil 50 de la distribución.", "depende de la variable"),
    (r"std|desv", "Desviación estándar; mide dispersión de la señal o intensidades.", "depende de la variable"),
    (r"p05|p5", "Percentil 5 de la distribución.", "depende de la variable"),
    (r"p95", "Percentil 95 de la distribución.", "depende de la variable"),
    (r"p99", "Percentil 99 de la distribución, usado para alta señal/activación.", "depende de la variable"),
    (r"rms", "Valor cuadrático medio. En EMG/fuerza resume amplitud efectiva.", "depende de la señal"),
    (r"iemg|integral", "EMG integrado o área bajo la curva de activación muscular.", "amplitud·tiempo"),
    (r"peak|pico|max", "Valor máximo o pico de la señal/volumen/región.", "depende de la variable"),
    (r"coherence|coherencia", "Coherencia espectral entre dos señales; mide acoplamiento por frecuencia.", "0-1"),
    (r"plv", "Phase Locking Value; estabilidad del acople de fase entre señales.", "0-1"),
    (r"xcorr|crosscorr|lag", "Correlación cruzada o desfase temporal entre señales.", "adimensional/segundos"),
    (r"entropy|entropia", "Entropía de la señal; mayor valor sugiere más complejidad/irregularidad.", "adimensional"),
    (r"hurst", "Exponente de Hurst; mide persistencia o memoria de largo plazo en una señal.", "adimensional"),
    (r"alff", "Amplitude of Low Frequency Fluctuations: potencia de baja frecuencia en fMRI de reposo/stack temporal.", "intensidad/potencia"),
    (r"falff", "Fracción ALFF: proporción de potencia de baja frecuencia respecto a la potencia total.", "0-1"),
    (r"tsnr", "Temporal signal-to-noise ratio: estabilidad temporal de una serie fMRI.", "adimensional"),
    (r"dvars", "Cambio temporal global entre volúmenes fMRI consecutivos; útil para detectar ruido/movimiento.", "intensidad"),
    (r"fd|framewise_displacement", "Desplazamiento framewise; métrica de movimiento en fMRI.", "mm"),
    (r"gcor", "Correlación global residual aproximada; ayuda a valorar señal global/ruido compartido.", "adimensional"),
    (r"fa_", "Métrica de anisotropía fraccional derivada de DTI.", "0-1 aprox."),
    (r"adc|ad_|rd_|trace", "Métrica de difusión: ADC/AD/RD/TRACE según subtipo de DTI.", "unidades DTI"),
    (r"status|estado", "Estado de procesamiento o validación de la tarea.", "texto"),
    (r"note|nota|interpretacion", "Nota interpretativa o advertencia metodológica asociada a la métrica.", "texto"),
]


def _stage_alias(stage: str) -> str:
    s = str(stage or "").lower().replace("é", "e")
    return "Despues" if "desp" in s or "post" in s else "Antes"


def _metric_info(metric: str) -> tuple[str, str]:
    key = str(metric or "").lower()
    for pattern, explanation, unit in METRIC_EXPLANATIONS:
        try:
            if re.search(pattern, key, re.IGNORECASE):
                return explanation, unit
        except re.error:
            continue
    return "Métrica exportada por la suite. Revisar source_file y modality para interpretar el contexto exacto.", "variable"


def _is_excluded(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    if any(x in parts for x in EXCLUDE_DIR_NAMES):
        return True
    name = path.name.lower()
    return any(tok in name for tok in EXCLUDE_FILE_TOKENS)


def _infer_stage_from_path(subject_root: Path, path: Path) -> str:
    rel = path.relative_to(subject_root)
    parts = [p.lower().replace("é", "e") for p in rel.parts]
    if any(p in {"despues", "después", "post", "post-qx"} for p in parts):
        return "Despues"
    if any(p in {"antes", "pre", "pre-qx"} for p in parts):
        return "Antes"
    if "comparacion" in parts or "antes_vs_despues" in parts:
        return "Antes_vs_Despues"
    return "No_especificado"


def _infer_modality_from_path(path: Path) -> str:
    parts = [p.lower().replace("í", "i").replace("é", "e") for p in path.parts]
    priority = [
        "biomecanica", "emg", "dinamometria", "tomografia", "mapas", "resonancias",
        "morfometria", "estructura_funcion", "correlaciones", "comparacion", "reportes",
    ]
    for p in priority:
        if p in parts:
            return p
    name = path.name.lower()
    for p in priority:
        if p in name:
            return p
    return "otros"


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    try:
        s = str(value).strip().replace(",", ".")
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_json(v, key))
    elif isinstance(obj, list):
        if len(obj) <= 20 and all(not isinstance(x, (dict, list)) for x in obj):
            out[prefix] = "; ".join(str(x) for x in obj)
        else:
            out[f"{prefix}.n_items"] = len(obj)
    else:
        out[prefix] = obj
    return out


def _read_csv_rows(path: Path) -> tuple[list[dict], list[dict]]:
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, sep=";")
        except Exception as exc:
            return [], [{"source_file": str(path), "error": str(exc), "n_rows": 0, "n_columns": 0}]
    wide_rows = []
    long_rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        record = row.to_dict()
        record["source_row"] = int(idx)
        wide_rows.append(record)
        for col, value in record.items():
            if col == "source_row":
                continue
            number = _to_number(value)
            explanation, unit = _metric_info(col)
            long_rows.append({
                "source_row": int(idx),
                "metric_name": str(col),
                "metric_value": number,
                "metric_value_text": "" if pd.isna(value) else str(value),
                "unit_or_scale": unit,
                "metric_explanation": explanation,
            })
    info = [{"source_file": str(path), "n_rows": int(len(df)), "n_columns": int(len(df.columns)), "columns": "; ".join(map(str, df.columns))}]
    return long_rows, wide_rows, info


def _read_json_rows(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        flat = _flatten_json(data)
    except Exception as exc:
        return [], [], [{"source_file": str(path), "error": str(exc), "n_rows": 0, "n_columns": 0}]
    long_rows = []
    wide = {"source_row": 0, **flat}
    for col, value in flat.items():
        number = _to_number(value)
        explanation, unit = _metric_info(col)
        long_rows.append({
            "source_row": 0,
            "metric_name": str(col),
            "metric_value": number,
            "metric_value_text": "" if value is None else str(value),
            "unit_or_scale": unit,
            "metric_explanation": explanation,
        })
    return long_rows, [wide], [{"source_file": str(path), "n_rows": 1, "n_columns": len(flat), "columns": "; ".join(flat.keys())}]


def _iter_metric_files(subject_root: Path) -> Iterable[Path]:
    if not subject_root.exists():
        return []
    files = []
    for ext in ("*.csv", "*.json"):
        for p in subject_root.rglob(ext):
            if p.is_file() and not _is_excluded(p):
                files.append(p)
    return sorted(files)


def consolidate_subject_metrics(cfg: PipelineConfig, subject: str) -> dict[str, Any]:
    subject_root = cfg.results_root() / subject
    out_dir = subject_root / "consolidado"
    safe_mkdir(out_dir)
    log_file = out_dir / "consolidado_log.txt"

    all_long: list[dict] = []
    all_wide: list[dict] = []
    sources: list[dict] = []
    metric_files = list(_iter_metric_files(subject_root))

    for path in metric_files:
        try:
            rel = str(path.relative_to(subject_root))
            stage = _infer_stage_from_path(subject_root, path)
            modality = _infer_modality_from_path(path)
            if path.suffix.lower() == ".csv":
                long_rows, wide_rows, info_rows = _read_csv_rows(path)
            elif path.suffix.lower() == ".json":
                long_rows, wide_rows, info_rows = _read_json_rows(path)
            else:
                continue
            for r in long_rows:
                r.update({
                    "subject": subject,
                    "stage": stage,
                    "modality": modality,
                    "source_file": rel,
                    "source_file_abs": str(path),
                })
            for r in wide_rows:
                r.update({
                    "subject": subject,
                    "stage": stage,
                    "modality": modality,
                    "source_file": rel,
                    "source_file_abs": str(path),
                })
            for r in info_rows:
                r.update({
                    "subject": subject,
                    "stage": stage,
                    "modality": modality,
                    "source_file": rel,
                    "source_file_abs": str(path),
                })
            all_long.extend(long_rows)
            all_wide.extend(wide_rows)
            sources.extend(info_rows)
        except Exception as exc:
            append_log(log_file, f"[ERROR] consolidando {path}: {exc}")

    df_long = pd.DataFrame(all_long)
    df_wide = pd.DataFrame(all_wide)
    df_sources = pd.DataFrame(sources)

    if not df_long.empty:
        cols = ["subject", "stage", "modality", "source_file", "source_row", "metric_name", "metric_value", "metric_value_text", "unit_or_scale", "metric_explanation", "source_file_abs"]
        df_long = df_long[[c for c in cols if c in df_long.columns] + [c for c in df_long.columns if c not in cols]]
        save_dataframe(df_long, out_dir / f"{subject.replace(' ', '_')}_todos_los_datos_largo.csv")

        dict_df = (
            df_long[["metric_name", "unit_or_scale", "metric_explanation"]]
            .drop_duplicates()
            .sort_values(["metric_name"])
            .reset_index(drop=True)
        )
        save_dataframe(dict_df, out_dir / f"{subject.replace(' ', '_')}_diccionario_metricas.csv")
    else:
        dict_df = pd.DataFrame(columns=["metric_name", "unit_or_scale", "metric_explanation"])
        save_dataframe(df_long, out_dir / f"{subject.replace(' ', '_')}_todos_los_datos_largo.csv")
        save_dataframe(dict_df, out_dir / f"{subject.replace(' ', '_')}_diccionario_metricas.csv")

    if not df_wide.empty:
        lead = ["subject", "stage", "modality", "source_file", "source_row"]
        df_wide = df_wide[[c for c in lead if c in df_wide.columns] + [c for c in df_wide.columns if c not in lead]]
    save_dataframe(df_wide, out_dir / f"{subject.replace(' ', '_')}_tabla_ancha_metricas.csv")
    save_dataframe(df_sources, out_dir / f"{subject.replace(' ', '_')}_resumen_fuentes.csv")

    # Resumen compacto por etapa/modalidad.
    if not df_long.empty:
        numeric = df_long.dropna(subset=["metric_value"]).copy()
        if not numeric.empty:
            summary = numeric.groupby(["subject", "stage", "modality", "metric_name"], dropna=False).agg(
                n=("metric_value", "count"),
                mean=("metric_value", "mean"),
                median=("metric_value", "median"),
                min=("metric_value", "min"),
                max=("metric_value", "max"),
            ).reset_index()
            # Agregar explicación a cada métrica.
            summary["metric_explanation"] = summary["metric_name"].map(lambda x: _metric_info(str(x))[0])
            summary["unit_or_scale"] = summary["metric_name"].map(lambda x: _metric_info(str(x))[1])
        else:
            summary = pd.DataFrame()
    else:
        summary = pd.DataFrame()
    save_dataframe(summary, out_dir / f"{subject.replace(' ', '_')}_resumen_estadistico_metricas.csv")

    # Excel opcional, resumido para consulta rápida.
    try:
        xlsx = out_dir / f"{subject.replace(' ', '_')}_consolidado_metricas.xlsx"
        with pd.ExcelWriter(xlsx) as writer:
            df_sources.to_excel(writer, sheet_name="fuentes"[:31], index=False)
            dict_df.to_excel(writer, sheet_name="diccionario"[:31], index=False)
            summary.to_excel(writer, sheet_name="resumen_estadistico"[:31], index=False)
            # Excel tiene límite de filas; guardar solo muestra si el largo es enorme.
            df_long.head(200000).to_excel(writer, sheet_name="datos_largo_muestra"[:31], index=False)
            df_wide.head(50000).to_excel(writer, sheet_name="tabla_ancha_muestra"[:31], index=False)
    except Exception as exc:
        append_log(log_file, f"No se pudo guardar Excel consolidado: {exc}")

    return {
        "subject": subject,
        "n_metric_files": len(metric_files),
        "n_long_rows": int(len(df_long)),
        "n_wide_rows": int(len(df_wide)),
        "n_sources": int(len(df_sources)),
        "out_long": str(out_dir / f"{subject.replace(' ', '_')}_todos_los_datos_largo.csv"),
        "out_wide": str(out_dir / f"{subject.replace(' ', '_')}_tabla_ancha_metricas.csv"),
        "out_dictionary": str(out_dir / f"{subject.replace(' ', '_')}_diccionario_metricas.csv"),
        "out_sources": str(out_dir / f"{subject.replace(' ', '_')}_resumen_fuentes.csv"),
    }


def run_patient_consolidation(cfg: PipelineConfig) -> pd.DataFrame:
    """Crea CSV consolidados por sujeto con todas las métricas generadas.

    Incluye pacientes seleccionados y control sano. Si el sano tiene Antes y Despues,
    ambas etapas quedan consolidadas porque se leen desde resultados/sano/Antes y
    resultados/sano/Despues.
    """
    rows = []
    subjects = list(cfg.patients) + [cfg.control_name]
    for subject in subjects:
        print(f"\n[CONSOLIDADO] {subject}")
        rows.append(consolidate_subject_metrics(cfg, subject))
    df = pd.DataFrame(rows)
    if not df.empty:
        save_dataframe(df, cfg.results_root() / "resumen_global_consolidado_pacientes.csv")
        # Diccionario global único.
        dicts = []
        for subject in subjects:
            p = cfg.results_root() / subject / "consolidado" / f"{subject.replace(' ', '_')}_diccionario_metricas.csv"
            if p.exists():
                try:
                    d = pd.read_csv(p)
                    d.insert(0, "subject", subject)
                    dicts.append(d)
                except Exception:
                    pass
        if dicts:
            dg = pd.concat(dicts, ignore_index=True).drop_duplicates(subset=["metric_name", "unit_or_scale", "metric_explanation"])
            save_dataframe(dg, cfg.results_root() / "diccionario_global_metricas.csv")
    return df
