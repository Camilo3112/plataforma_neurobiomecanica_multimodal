"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       BIOMECÁNICA Y MÉTRICAS DE FUERZA                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/biomechanics.py
Versión: v3.21.21

Descripción
-----------
Calcula métricas biomecánicas, asimetrías y cambios longitudinales.

Fundamento físico-matemático implementado
-----------------------------------------
Calcula métricas biomecánicas desde fuerza, torque y señales normalizadas.
Incluye
asimetría bilateral AI = 100(R-L)/((R+L)/2), cambios longitudinales Δ = post-
pre,
porcentajes de recuperación y relaciones estructura-función. Las magnitudes se
mantienen trazables por lado, etapa y fuente para evitar mezclar unidades
físicas.

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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import PipelineConfig
from .paths import first_existing_dir, find_file_by_side_and_test, resolve_stage_dir
from .signals import analyze_pair
from .io_utils import safe_mkdir, save_dataframe, save_json, append_log, log_exception
from .checkpoint import CheckpointManager, load_json_if_exists, save_metrics_json


@dataclass
class PhaseSpec:
    idx: int
    name: str
    folder: str  # emg o dinamometria
    f_source: str
    f_kind: str
    f_side: str
    f_muscle: Optional[str]
    f_label: str
    g_source: str
    g_kind: str
    g_side: str
    g_muscle: Optional[str]
    g_label: str


PHASES_16: list[PhaseSpec] = [
    PhaseSpec(1, "EMG bilateral lateral derecho vs izquierdo", "emg", "pac", "EMG", "derecha", "lateral", "Vasto lateral der", "pac", "EMG", "izquierda", "lateral", "Vasto lateral izq"),
    PhaseSpec(2, "EMG bilateral medial derecho vs izquierdo", "emg", "pac", "EMG", "derecha", "medial", "Vasto medial der", "pac", "EMG", "izquierda", "medial", "Vasto medial izq"),
    PhaseSpec(3, "EMG ipsilateral derecho lateral vs medial", "emg", "pac", "EMG", "derecha", "lateral", "Vasto lateral der", "pac", "EMG", "derecha", "medial", "Vasto medial der"),
    PhaseSpec(4, "EMG ipsilateral izquierdo lateral vs medial", "emg", "pac", "EMG", "izquierda", "lateral", "Vasto lateral izq", "pac", "EMG", "izquierda", "medial", "Vasto medial izq"),
    PhaseSpec(5, "Normativo EMG lateral derecho paciente vs sano", "emg", "pac", "EMG", "derecha", "lateral", "VL der paciente", "sano", "EMG", "derecha", "lateral", "VL der sano"),
    PhaseSpec(6, "Normativo EMG lateral izquierdo paciente vs sano", "emg", "pac", "EMG", "izquierda", "lateral", "VL izq paciente", "sano", "EMG", "izquierda", "lateral", "VL izq sano"),
    PhaseSpec(7, "Normativo EMG medial derecho paciente vs sano", "emg", "pac", "EMG", "derecha", "medial", "VM der paciente", "sano", "EMG", "derecha", "medial", "VM der sano"),
    PhaseSpec(8, "Normativo EMG medial izquierdo paciente vs sano", "emg", "pac", "EMG", "izquierda", "medial", "VM izq paciente", "sano", "EMG", "izquierda", "medial", "VM izq sano"),
    PhaseSpec(9, "Acople electromecánico derecho DIN vs VL", "dinamometria", "pac", "DIN", "derecha", None, "DIN der paciente", "pac", "EMG", "derecha", "lateral", "VL der paciente"),
    PhaseSpec(10, "Acople electromecánico izquierdo DIN vs VL", "dinamometria", "pac", "DIN", "izquierda", None, "DIN izq paciente", "pac", "EMG", "izquierda", "lateral", "VL izq paciente"),
    PhaseSpec(11, "Simetría cinética paciente derecha vs izquierda", "dinamometria", "pac", "DIN", "derecha", None, "DIN der paciente", "pac", "DIN", "izquierda", None, "DIN izq paciente"),
    PhaseSpec(12, "Sano acople derecho DIN vs VL", "dinamometria", "sano", "DIN", "derecha", None, "DIN der sano", "sano", "EMG", "derecha", "lateral", "VL der sano"),
    PhaseSpec(13, "Sano acople izquierdo DIN vs VL", "dinamometria", "sano", "DIN", "izquierda", None, "DIN izq sano", "sano", "EMG", "izquierda", "lateral", "VL izq sano"),
    PhaseSpec(14, "Sano simetría cinética derecha vs izquierda", "dinamometria", "sano", "DIN", "derecha", None, "DIN der sano", "sano", "DIN", "izquierda", None, "DIN izq sano"),
    PhaseSpec(15, "Normativo DIN derecho paciente vs sano", "dinamometria", "pac", "DIN", "derecha", None, "DIN der paciente", "sano", "DIN", "derecha", None, "DIN der sano"),
    PhaseSpec(16, "Normativo DIN izquierdo paciente vs sano", "dinamometria", "pac", "DIN", "izquierda", None, "DIN izq paciente", "sano", "DIN", "izquierda", None, "DIN izq sano"),
]


def _subject_stage_dirs(cfg: PipelineConfig, patient: str, stage: str) -> tuple[Optional[Path], Optional[Path]]:
    data_root = cfg.data_root()
    pac_dir = resolve_stage_dir(data_root / patient, stage)
    control_subject = data_root / cfg.control_name
    # Para sano, si no existe Despues, cae a Antes, y si no, a la raíz sano.
    control_stage = resolve_stage_dir(control_subject, stage)
    if control_stage is None:
        control_stage = resolve_stage_dir(control_subject, "Antes", fallback_to_subject=True)
    return pac_dir, control_stage


def _modality_dirs(cfg: PipelineConfig, stage_dir: Optional[Path]) -> dict[str, Optional[Path]]:
    if stage_dir is None:
        return {"EMG": None, "DIN": None}
    return {
        "EMG": first_existing_dir(stage_dir, [cfg.emg_dirname, "emg"]),
        "DIN": first_existing_dir(stage_dir, [cfg.dyn_dirname, "dinamometria", "dinamometría", "DIN", "Dinamometry"]),
    }


def _file_for_signal(dirs: dict[str, Optional[Path]], kind: str, side: str, test: int) -> Optional[Path]:
    folder = dirs.get("EMG" if kind.upper() == "EMG" else "DIN")
    if folder is None:
        return None
    # EMG tiene 1-3. Dinamometría solo 1 antes/después: usar derecha_1/izquierda_1 para todas las pruebas.
    effective_test = test if kind.upper() == "EMG" else 1
    return find_file_by_side_and_test(folder, side, effective_test, suffix=".csv")


def run_biomechanics_for_patient_stage(cfg: PipelineConfig, patient: str, stage: str) -> pd.DataFrame:
    print(f"\n[BIOMECÁNICA] {patient} · {stage}")
    ckpt = CheckpointManager(cfg)
    log_file = cfg.results_root() / patient / stage / "reportes" / "biomecanica_log.txt"
    pac_stage, sano_stage = _subject_stage_dirs(cfg, patient, stage)
    pac_dirs = _modality_dirs(cfg, pac_stage)
    sano_dirs = _modality_dirs(cfg, sano_stage)
    all_rows: list[dict] = []

    if pac_stage is None:
        append_log(log_file, f"No encontré carpeta de etapa para {patient} {stage}")
        return pd.DataFrame()

    for test in cfg.tests:
        print(f"  -> Prueba {test}")
        for folder in ["emg", "dinamometria"]:
            safe_mkdir(cfg.results_root() / patient / stage / folder / f"prueba_{test}")

        rows_this_test: list[dict] = []
        for phase in PHASES_16:
            try:
                dirs_f = pac_dirs if phase.f_source == "pac" else sano_dirs
                dirs_g = pac_dirs if phase.g_source == "pac" else sano_dirs
                path_f = _file_for_signal(dirs_f, phase.f_kind, phase.f_side, test)
                path_g = _file_for_signal(dirs_g, phase.g_kind, phase.g_side, test)
                if path_f is None or path_g is None:
                    append_log(log_file, f"[OMITIDO] Prueba {test} fase {phase.idx:02d}: falta archivo f={path_f} g={path_g}")
                    continue

                out_dir = cfg.results_root() / patient / stage / phase.folder / f"prueba_{test}"
                out_png = out_dir / f"{phase.idx:02d}_{phase.name.lower().replace(' ', '_').replace('·', '').replace('/', '_')}.png"
                metrics_json = out_dir / f"{phase.idx:02d}_metricas.json"
                task_id = f"biomecanica/{patient}/{stage}/prueba_{test}/fase_{phase.idx:02d}"

                def _work():
                    metrics = analyze_pair(
                        path_f=path_f,
                        kind_f=phase.f_kind,  # type: ignore[arg-type]
                        side_f=phase.f_side,  # type: ignore[arg-type]
                        muscle_f=phase.f_muscle,  # type: ignore[arg-type]
                        label_f=phase.f_label,
                        path_g=path_g,
                        kind_g=phase.g_kind,  # type: ignore[arg-type]
                        side_g=phase.g_side,  # type: ignore[arg-type]
                        muscle_g=phase.g_muscle,  # type: ignore[arg-type]
                        label_g=phase.g_label,
                        title=f"{patient} · {stage} · Prueba {test} · Fase {phase.idx:02d}: {phase.name}",
                        out_png=out_png,
                        max_freq_plot_hz=cfg.max_freq_plot_hz,
                    )
                    metrics.update({
                        "patient": patient,
                        "stage": stage,
                        "test": test,
                        "phase": phase.idx,
                        "phase_name": phase.name,
                        "folder": phase.folder,
                        "status": "ok",
                    })
                    save_metrics_json(metrics, metrics_json)
                    return metrics

                result, status = ckpt.run(
                    task_id=task_id,
                    inputs=[path_f, path_g],
                    outputs=[out_png, metrics_json],
                    params={"phase": phase.__dict__, "max_freq_plot_hz": cfg.max_freq_plot_hz},
                    fn=_work,
                )
                metrics = load_json_if_exists(metrics_json) if status == "skipped" else result
                if metrics:
                    metrics["checkpoint_status"] = status
                    rows_this_test.append(metrics)
                    all_rows.append(metrics)
            except Exception as exc:
                log_exception(log_file, f"Prueba {test} fase {phase.idx:02d} {phase.name}", exc)

        if rows_this_test:
            df_test = pd.json_normalize(rows_this_test)
            # Guarda copia en ambas carpetas principales de la prueba para facilitar búsqueda.
            save_dataframe(df_test, cfg.results_root() / patient / stage / "emg" / f"prueba_{test}" / "metricas_fases.csv")
            save_dataframe(df_test, cfg.results_root() / patient / stage / "dinamometria" / f"prueba_{test}" / "metricas_fases.csv")

    df = pd.json_normalize(all_rows) if all_rows else pd.DataFrame()
    if not df.empty:
        save_dataframe(df, cfg.results_root() / patient / stage / "reportes" / "biomecanica_metricas_todas_las_pruebas.csv")
        try:
            save_dataframe(df, cfg.results_root() / patient / stage / "reportes" / "biomecanica_metricas_todas_las_pruebas.xlsx")
        except Exception as exc:
            append_log(log_file, f"No se pudo guardar XLSX de biomecánica: {exc}")
    return df


def _control_stages_for_biomechanics(cfg: PipelineConfig) -> list[str]:
    """Etapas reales del sano para EMG/DIN.

    Ahora se procesa sano/Antes y sano/Despues si existen, porque el protocolo
    necesita el control completo para comparación y consolidado.
    """
    stages = []
    control_dir = cfg.data_root() / cfg.control_name
    for st in cfg.stages:
        if resolve_stage_dir(control_dir, st, fallback_to_subject=False) is not None:
            stages.append(st)
    if not stages and control_dir.exists():
        stages.append("Antes")
    return stages


def run_biomechanics(cfg: PipelineConfig) -> pd.DataFrame:
    all_df = []
    subjects_and_stages = [(p, list(cfg.stages)) for p in cfg.patients]
    subjects_and_stages.append((cfg.control_name, _control_stages_for_biomechanics(cfg)))
    for patient, stages in subjects_and_stages:
        for stage in stages:
            all_df.append(run_biomechanics_for_patient_stage(cfg, patient, stage))
    out = pd.concat([d for d in all_df if d is not None and not d.empty], ignore_index=True) if any(not d.empty for d in all_df if d is not None) else pd.DataFrame()
    if not out.empty:
        save_dataframe(out, cfg.results_root() / "resumen_global_biomecanica.csv")
    return out
