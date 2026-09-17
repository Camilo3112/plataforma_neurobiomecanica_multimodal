"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       EJECUCIÓN DE ROI CORTICAL MOTORA                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/cortical_roi_runner.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.


-----------------------------------------
Ejecuta la corrección de ROI cortical en espacio del paciente. El modelo
combina
máscaras binarias, shell cortical y transformaciones geométricas. Las
operaciones
clave son dilatación/erosión, intersección de conjuntos y búsqueda espacial de
alineación. La ROI final conserva estructura anatómica al restringirse a
voxeles
compatibles con la corteza estimada.

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
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .paths import resolve_stage_dir
from .io_utils import safe_mkdir, save_dataframe, append_log, log_exception
from .checkpoint import CheckpointManager
from .motor_atlas_refinement import refine_cortical_mask_with_atlas_prior

BRAIN_PATTERNS = ['**/Brain00mm.nii', '**/brain00mm.nii', '**/Brain*.nii', '**/brain*.nii', '**/*REFORMATEO*/*.nii', '**/*reformateo*/*.nii']
MASK_PATTERNS = ['*corteza_motora*_roi_shell.nii.gz', '*corteza_motora*.nii.gz', '*internal_corteza_motora*.nii.gz']


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


def find_brain(stage_dir: Path | None) -> Path | None:
    if stage_dir is None or not stage_dir.exists():
        return None
    candidates = []
    for pat in BRAIN_PATTERNS:
        candidates.extend(stage_dir.glob(pat))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    def score(p: Path):
        s = str(p).lower()
        val = 0
        if 'brain00mm' in s: val += 100
        if 'reformateo' in s: val += 30
        if 'resultadosfuncional' in s.replace(' ', ''): val += 20
        return -val, len(str(p))
    return sorted(candidates, key=score)[0]


def find_masks(results_stage_dir: Path) -> list[Path]:
    masks = []
    for root in [results_stage_dir / 'morfometria' / 'interna', results_stage_dir / 'resonancias', results_stage_dir]:
        if not root.exists():
            continue
        for pat in MASK_PATTERNS:
            masks.extend(root.rglob(pat))
    out, seen = [], set()
    for p in masks:
        name = p.name.lower()
        if 'corregida' in name or 'preview' in name:
            continue
        if p.is_file() and p not in seen:
            out.append(p); seen.add(p)
    return out


def _script_path() -> Path:
    return Path(__file__).resolve().parents[1] / 'tools' / 'corregir_roi_cortical_automatico.py'


def run_one_cortical_fix(brain: Path, mask: Path, output: Path, max_mm: float = 65.0, shell_mm: float = 5.0, auto_scale: bool = False) -> dict:
    safe_mkdir(output.parent)
    cmd = [sys.executable, str(_script_path()), '--brain', str(brain), '--mask', str(mask), '--output', str(output), '--max-mm', str(max_mm), '--shell-mm', str(shell_mm)]
    if auto_scale:
        cmd.append('--auto-scale')
    log_path = output.with_name(output.name.replace('.nii.gz','').replace('.nii','') + '_stdout.log')
    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = log_path.read_text(encoding='utf-8', errors='ignore')[-4000:]
        raise RuntimeError(f'Corrección cortical falló con código {proc.returncode}. Últimas líneas:\n{tail}')
    reports = sorted(output.parent.glob(output.name.split('.nii')[0] + '*reporte.json'))
    data, report = {}, None
    if reports:
        report = reports[0]
        try:
            data = json.loads(report.read_text(encoding='utf-8'))
        except Exception:
            data = {}
    atlas_report = {}
    try:
        atlas_report = refine_cortical_mask_with_atlas_prior(
            brain_path=brain,
            corrected_mask_path=output,
            output_dir=output.parent,
            shell_mm=shell_mm,
        )
    except Exception as exc:
        atlas_report = {'atlas_refinement_error': str(exc)}

    return {
        'brain_path': str(brain), 'mask_path': str(mask), 'output_path': str(output),
        'report_path': str(report) if report else '', 'quality': data.get('quality', ''),
        'score': data.get('automatic_result', {}).get('score'),
        'shell_overlap': data.get('automatic_result', {}).get('shell_overlap'),
        'inside_fraction': data.get('automatic_result', {}).get('inside_fraction'),
        'translation_mm_RAS': json.dumps(data.get('translation_mm_RAS', [])),
        'mask_voxels_corrected': data.get('mask_voxels_corrected'),
        'atlas_refined_mask_path': atlas_report.get('refined_mask_path', ''),
        'atlas_prior_path': atlas_report.get('atlas_prior_path', ''),
        'atlas_target_path': atlas_report.get('atlas_target_path', ''),
        'atlas_refinement_method': atlas_report.get('method', ''),
        'atlas_refinement_decision': atlas_report.get('decision', ''),
        'atlas_refinement_quality': atlas_report.get('quality_hint', ''),
        'atlas_refinement_warning': atlas_report.get('warning', atlas_report.get('atlas_refinement_error', '')),
        'atlas_prior_mean_original_mask': atlas_report.get('prior_mean_original_mask'),
        'atlas_prior_mean_refined_mask': atlas_report.get('prior_mean_refined_mask'),
        'atlas_shell_fraction_refined': atlas_report.get('shell_fraction_refined'),
    }


def run_cortical_roi_corrections(cfg: PipelineConfig, auto_scale: bool = False) -> pd.DataFrame:
    rows = []
    for subject, stage, stage_dir in _iter_subject_stage(cfg):
        print(f"\n[CORTEZA FIX] {subject} · {stage}")
        results_stage = cfg.results_root() / subject / stage
        log_file = results_stage / 'reportes' / 'corteza_fix_log.txt'
        try:
            brain = find_brain(stage_dir)
            if brain is None:
                append_log(log_file, f'No encontré Brain00mm/T1 para {subject} {stage}; stage_dir={stage_dir}')
                continue
            masks = find_masks(results_stage)
            if not masks:
                append_log(log_file, f'No encontré máscaras de corteza motora en {results_stage}')
                continue
            out_root = results_stage / 'morfometria' / 'interna' / 'corregidas_corticales'
            safe_mkdir(out_root)
            ckpt = CheckpointManager(cfg)
            for mask in masks:
                out = out_root / f"{mask.name.replace('.nii.gz','').replace('.nii','')}_corregida_auto.nii.gz"
                task_id = f'corteza_fix/{subject}/{stage}/{mask.name}'
                def _work(mask=mask, out=out, brain=brain):
                    return run_one_cortical_fix(brain, mask, out, auto_scale=auto_scale)
                result, status = ckpt.run(task_id=task_id, inputs=[brain, mask, _script_path()], outputs=[out], params={'selector': 'v3_20_correccion_roi_cortical_shell_mas_atlas_prior', 'auto_scale': auto_scale}, fn=_work)
                if status == 'skipped' and out.exists():
                    result = {'brain_path': str(brain), 'mask_path': str(mask), 'output_path': str(out), 'quality': 'skipped'}
                result = result or {}
                result.update({'patient': subject, 'subject': subject, 'stage': stage, 'checkpoint_status': status})
                rows.append(result)
        except Exception as exc:
            log_exception(log_file, f'Corrección cortical {subject} {stage}', exc)
    df = pd.DataFrame(rows)
    if not df.empty:
        save_dataframe(df, cfg.results_root() / 'resumen_global_correccion_corteza.csv')
    return df
