"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     ORQUESTACIÓN DE TOMOGRAFÍA AVANZADA                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/tomography_advanced_runner.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.


-----------------------------------------
Orquesta el análisis TAC avanzado por paciente. Conserva metadatos DICOM,
selecciona
series anatómicas, ejecuta segmentación por HU y consolida métricas de área,
volumen y proporción tisular. Las comparaciones pre/post se basan en
diferencias
absolutas Δx y relativas 100·(post-pre)/pre cuando existe denominador válido.

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

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelineConfig
from .paths import first_existing_dir, resolve_stage_dir
from .io_utils import safe_mkdir, save_dataframe, append_log, log_exception
from .checkpoint import CheckpointManager


def _control_requested_stages(cfg: PipelineConfig):
    control_dir = cfg.data_root() / cfg.control_name
    for stage in cfg.stages:
        if (control_dir / stage).exists():
            yield stage
    if not any((control_dir / st).exists() for st in cfg.stages) and control_dir.exists():
        yield cfg.stages[0]


def _iter_subject_stage(cfg: PipelineConfig):
    for patient in cfg.patients:
        for stage in cfg.stages:
            stage_dir = resolve_stage_dir(cfg.data_root() / patient, stage)
            yield patient, stage, stage_dir
    for stage in _control_requested_stages(cfg):
        control_dir = cfg.data_root() / cfg.control_name
        stage_dir = resolve_stage_dir(control_dir, stage, fallback_to_subject=True)
        yield cfg.control_name, stage, stage_dir


def _parse_float(text: str):
    m = re.search(r"[-+]?\d+(?:[\.,]\d+)?", str(text))
    if not m:
        return None
    try:
        return float(m.group(0).replace(',', '.'))
    except Exception:
        return None


def parse_advanced_tomography_report(report_path: Path, patient: str, stage: str) -> pd.DataFrame:
    if not report_path.exists():
        return pd.DataFrame()
    text = report_path.read_text(encoding='utf-8', errors='ignore')
    rows: list[dict[str, Any]] = []
    current_section = "general"
    current_entity = "global"
    current_side = ""
    section_headers = {
        "LANDMARKS": "landmarks",
        "COMPOSICIÓN CORPORAL": "composicion_corte_medio",
        "GRASA INTRAMUSCULAR": "grasa_intramuscular_volumetria",
        "RESUMEN POR MIEMBRO": "resumen_por_miembro",
        "RESUMEN GLOBAL": "resumen_global",
    }
    key_map = {
        "Perímetro externo del miembro": ("limb_perimeter", "cm/mm"),
        "Área de grasa subcutánea": ("subcutaneous_fat_area_cm2", "cm2"),
        "Área muscular total": ("total_muscle_area_cm2", "cm2"),
        "Relación grasa/músculo": ("fat_to_muscle_ratio", "ratio"),
        "Relación músculo/grasa": ("muscle_to_fat_ratio", "ratio"),
        "Grasa respecto a grasa + músculo": ("fat_fraction_percent", "%"),
        "Área de banda dérmica estimada": ("dermis_band_area_cm2", "cm2"),
        "Espesor cutáneo medio estimado": ("mean_skin_thickness_mm", "mm"),
        "Espesor subcutáneo medio estimado": ("mean_subcutaneous_thickness_mm", "mm"),
        "Volumen muscular segmentado": ("segmented_muscle_volume_cm3", "cm3"),
        "Volumen de grasa intramuscular": ("intramuscular_fat_volume_cm3", "cm3"),
        "Relación grasa IM / volumen músculo": ("intramuscular_fat_to_muscle_ratio", "ratio"),
        "Área máxima de grasa IM": ("max_intramuscular_fat_area_cm2", "cm2"),
        "Área media no nula de grasa IM": ("mean_nonzero_intramuscular_fat_area_cm2", "cm2"),
        "Volumen muscular total del lado": ("side_muscle_volume_cm3", "cm3"),
        "Volumen grasa intramuscular": ("side_intramuscular_fat_volume_cm3", "cm3"),
        "Volumen grasa subcutánea": ("side_subcutaneous_fat_volume_cm3", "cm3"),
        "Volumen total de grasa": ("side_total_fat_volume_cm3", "cm3"),
        "Relación grasa IM / músculo": ("side_intramuscular_fat_to_muscle_ratio", "ratio"),
        "Relación grasa subcutánea / músculo": ("side_subcutaneous_fat_to_muscle_ratio", "ratio"),
        "Relación grasa total / músculo": ("side_total_fat_to_muscle_ratio", "ratio"),
        "Volumen muscular total": ("global_muscle_volume_cm3", "cm3"),
        "Volumen total de grasa intramuscular": ("global_intramuscular_fat_volume_cm3", "cm3"),
        "Volumen grasa subcutánea / músculos": ("global_subcutaneous_fat_volume_cm3", "cm3"),
        "Volumen total de grasa": ("global_total_fat_volume_cm3", "cm3"),
        "Relación grasa total / músculo": ("global_total_fat_to_muscle_ratio", "ratio"),
        "Distancia entre landmarks": ("landmark_distance_mm", "mm"),
        "CORTE TRANSVERSAL MEDIO": ("midpoint_slice_z", "slice"),
        "Área combinada de los dos vastos": ("midpoint_total_vasti_area_cm2", "cm2"),
    }
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        up = line.upper()
        for marker, sec in section_headers.items():
            if marker in up:
                current_section = sec
        if line.startswith('MIEMBRO '):
            current_entity = line.replace('MIEMBRO ', '').strip().lower().capitalize()
            current_side = current_entity
            continue
        if line.startswith('MÚSCULO ') or line.startswith('MUSCULO '):
            current_entity = line.split(' ', 1)[1].strip()
            current_side = ""
            continue
        if line.startswith('RESUMEN GLOBAL'):
            current_entity = 'global'
            current_side = ''
            current_section = 'resumen_global'
            continue
        if line.startswith('-') and ':' in line:
            key, val = line.lstrip('-').split(':', 1)
            key, val = key.strip(), val.strip()
            number = _parse_float(val)
            if number is None:
                continue
            metric_name, unit = key_map.get(key, (re.sub(r'[^A-Za-z0-9_]+', '_', key.lower()).strip('_'), 'auto'))
            rows.append({
                'patient': patient,
                'subject': patient,
                'stage': stage,
                'modality': 'tomografia_avanzada',
                'section': current_section,
                'entity': current_entity,
                'side': current_side,
                'metric_name': metric_name,
                'metric_value': number,
                'unit_or_scale': unit,
                'raw_line': line,
                'source_file': str(report_path),
            })
    return pd.DataFrame(rows)


def summarize_intramuscular_area_csv(csv_path: Path, patient: str, stage: str) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()
    rows = []
    for col in df.columns:
        if col == 'Z':
            continue
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        if vals.empty:
            continue
        base = col.replace('_cm2', '')
        for stat_name, value, unit in [
            ('mean_area_cm2', vals.mean(), 'cm2'),
            ('max_area_cm2', vals.max(), 'cm2'),
            ('sum_area_cm2', vals.sum(), 'cm2*slices'),
        ]:
            rows.append({
                'patient': patient,
                'subject': patient,
                'stage': stage,
                'modality': 'tomografia_avanzada',
                'section': 'grasa_intramuscular_por_corte',
                'entity': base,
                'side': '',
                'metric_name': f'{base}_{stat_name}',
                'metric_value': float(value),
                'unit_or_scale': unit,
                'source_file': str(csv_path),
            })
    return pd.DataFrame(rows)



def read_adipose_metrics_csv(csv_path: Path, patient: str, stage: str) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()
    if df.empty or 'metric_name' not in df.columns or 'metric_value' not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df.insert(0, 'patient', patient)
    df.insert(1, 'subject', patient)
    df.insert(2, 'stage', stage)
    df.insert(3, 'modality', 'tomografia_avanzada_tejido_adiposo')
    df['metric_value'] = pd.to_numeric(df['metric_value'], errors='coerce')
    df['source_file'] = str(csv_path)
    return df.dropna(subset=['metric_value'])


def build_adipose_file_manifest(out_dir: Path, patient: str, stage: str) -> pd.DataFrame:
    expected = [
        ('labelmap_composicion_corte_medio', 'Composicion_Corporal_CorteMedio.nii'),
        ('overlay_composicion_corte_medio', 'TAC_Composicion_Corporal_CorteMedio.nii'),
        ('labelmap_grasa_intramuscular', 'Grasa_Intramuscular_Muscular.nii'),
        ('overlay_grasa_intramuscular', 'TAC_Grasa_Intramuscular_Muscular.nii'),
        ('labelmap_grasa_subcutanea_3d', 'Grasa_Subcutanea_Volumen.nii.gz'),
        ('labelmap_tejido_adiposo_total_3d', 'Tejido_Adiposo_Total_Volumen.nii.gz'),
        ('metricas_tejido_adiposo', 'Metricas_Tejido_Adiposo.csv'),
        ('metricas_resumen_avanzado', 'resumen_metricas_tomografia_avanzada.csv'),
        ('areas_grasa_intramuscular_por_corte', 'Areas_Grasa_Intramuscular_Por_Corte.csv'),
        ('reporte_morfometrico', 'Reporte_Morfometrico_Completo.txt'),
        ('etiquetas_tejido_adiposo_total', 'Etiquetas_Tejido_Adiposo_Total.txt'),
    ]
    rows = []
    for kind, filename in expected:
        path = out_dir / filename
        rows.append({
            'patient': patient,
            'subject': patient,
            'stage': stage,
            'modality': 'tomografia_avanzada',
            'file_kind': kind,
            'filename': filename,
            'exists': bool(path.exists()),
            'size_bytes': int(path.stat().st_size) if path.exists() else 0,
            'path': str(path),
        })
    return pd.DataFrame(rows)

def _script_path() -> Path:
    return Path(__file__).resolve().parents[1] / 'tools' / 'tomografia_avanzada_core.py'


def process_advanced_tomography_folder(cfg: PipelineConfig, tac_dir: Path, out_dir: Path, patient: str, stage: str) -> pd.DataFrame:
    safe_mkdir(out_dir)
    env = os.environ.copy()
    env['VCE_TAC_INPUT_DIR'] = str(tac_dir)
    env['VCE_TAC_OUTPUT_DIR'] = str(out_dir)
    cmd = [sys.executable, str(_script_path())]
    log_path = out_dir / 'tomografia_avanzada_stdout.log'
    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = log_path.read_text(encoding='utf-8', errors='ignore')[-4000:]
        raise RuntimeError(f'Tomografía avanzada falló con código {proc.returncode}. Últimas líneas:\n{tail}')
    report = out_dir / 'Reporte_Morfometrico_Completo.txt'
    adipose_metrics_csv = out_dir / 'Metricas_Tejido_Adiposo.csv'
    rows = [
        parse_advanced_tomography_report(report, patient, stage),
        read_adipose_metrics_csv(adipose_metrics_csv, patient, stage),
        summarize_intramuscular_area_csv(out_dir / 'Areas_Grasa_Intramuscular_Por_Corte.csv', patient, stage),
    ]
    summary = pd.concat([x for x in rows if x is not None and not x.empty], ignore_index=True) if any(x is not None and not x.empty for x in rows) else pd.DataFrame()
    if not summary.empty:
        save_dataframe(summary, out_dir / 'resumen_metricas_tomografia_avanzada.csv')

    manifest = build_adipose_file_manifest(out_dir, patient, stage)
    save_dataframe(manifest, out_dir / 'manifiesto_archivos_tejido_adiposo.csv')

    missing = manifest[(manifest['exists'] == False) | (manifest['size_bytes'] <= 0)]
    if not missing.empty:
        missing_names = ', '.join(missing['filename'].astype(str).tolist())
        raise RuntimeError(f'La tomografía avanzada terminó, pero faltan salidas obligatorias de tejido adiposo: {missing_names}')

    return summary


def run_advanced_tomography(cfg: PipelineConfig) -> pd.DataFrame:
    all_rows = []
    for patient, stage, stage_dir in _iter_subject_stage(cfg):
        print(f"\n[TOMOGRAFÍA AVANZADA] {patient} · {stage}")
        out_dir = cfg.results_root() / patient / stage / 'tomografia' / 'avanzada_tejido_adiposo_landmarks'
        log_file = cfg.results_root() / patient / stage / 'reportes' / 'tomografia_avanzada_log.txt'
        try:
            tac_dir = first_existing_dir(stage_dir, cfg.ct_dirnames) if stage_dir else None
            if tac_dir is None:
                append_log(log_file, f'No encontré carpeta TAC para {patient} {stage}; stage_dir={stage_dir}')
                continue
            metrics_csv = out_dir / 'resumen_metricas_tomografia_avanzada.csv'
            task_id = f'tomografia_avanzada/{patient}/{stage}'
            ckpt = CheckpointManager(cfg)
            def _work():
                return process_advanced_tomography_folder(cfg, tac_dir, out_dir, patient, stage)
            df_result, status = ckpt.run(
                task_id=task_id,
                inputs=[tac_dir, _script_path()],
                outputs=[
                    metrics_csv,
                    out_dir / 'Reporte_Morfometrico_Completo.txt',
                    out_dir / 'Metricas_Tejido_Adiposo.csv',
                    out_dir / 'Grasa_Subcutanea_Volumen.nii.gz',
                    out_dir / 'Grasa_Intramuscular_Muscular.nii',
                    out_dir / 'Tejido_Adiposo_Total_Volumen.nii.gz',
                    out_dir / 'manifiesto_archivos_tejido_adiposo.csv',
                ],
                params={'selector': 'v3_19_tejido_adiposo_nii_metricas_completas'},
                fn=_work,
            )
            df = pd.read_csv(metrics_csv) if status == 'skipped' and metrics_csv.exists() else df_result
            if df is not None and not df.empty:
                df.insert(0, 'checkpoint_status', status)
                all_rows.append(df)
        except Exception as exc:
            log_exception(log_file, f'Tomografía avanzada {patient} {stage}', exc)
    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not out.empty:
        save_dataframe(out, cfg.results_root() / 'resumen_global_tomografia_avanzada.csv')
    return out
