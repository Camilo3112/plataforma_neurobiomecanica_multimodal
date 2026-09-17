"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       EJECUCIÓN INTERACTIVA MULTIMODAL                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: run_todo_interactivo.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.


-----------------------------------------
Permite selección interactiva de pacientes, etapas y secciones. Representa la
corrida como un conjunto de tareas T={(paciente, etapa, sección)} y ejecuta
cada
módulo conservando logs, rutas y resultados intermedios.

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
import shutil
import subprocess
from pathlib import Path

from src.config import PipelineConfig, DEFAULT_PROJECT_ROOT
from src.paths import ensure_result_tree
from src.biomechanics import run_biomechanics
from src.tomography import run_tomography
from src.neuroimage import run_maps, run_resonances, run_volume_correlations
from src.structure_function import run_structure_function_coupling
from src.external_morphometry import run_external_morphometry_integration
from src.internal_morphometry import run_internal_morphometry_integration
from src.real_derivatives import run_real_derivatives_preparation
from src.longitudinal import run_before_after_comparisons
from src.reports import build_global_report, write_readme_results
from src.gpu import configure_gpu
from src.consolidation import run_patient_consolidation


def normalize_patient_token(token: str) -> str:
    t = str(token or "").strip().strip(",;")
    if not t:
        return ""
    low = t.lower().replace("_", " ")
    if low in {"y", "e", "and", "or", "o"}:
        return ""
    m = re.search(r"(?:paciente\s*)?(\d+)", low)
    if m:
        return f"paciente {int(m.group(1))}"
    if low == "sano":
        return "sano"
    if not low.startswith("paciente"):
        return f"paciente {t}"
    return " ".join(t.split())


def parse_patients(raw: str) -> tuple[str, ...]:
    raw = re.sub(r"paciente\s*(\d+)", r"\1", raw or "", flags=re.IGNORECASE)
    tokens = [x for x in re.split(r"[;,\s]+", raw or "") if x.strip()]
    out: list[str] = []
    for tok in tokens:
        p = normalize_patient_token(tok)
        if p and p != "sano" and p != "paciente" and p not in out:
            out.append(p)
    return tuple(out)


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        value = ""
    return value or default


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    d = "s" if default else "n"
    value = ask(prompt + " (s/n)", d).lower()
    return value in {"s", "si", "sí", "y", "yes", "1", "true"}


def run_bat(path: Path) -> int:
    if os.name != "nt":
        print(f"[EXTERNO OMITIDO] Este .bat solo se ejecuta en Windows: {path}")
        return 0
    if not path.exists():
        print(f"[EXTERNO OMITIDO] No existe: {path}")
        return 0
    print("\n" + "=" * 90)
    print(f"Ejecutando: {path}")
    print("=" * 90)
    return subprocess.call(["cmd", "/c", str(path)])


def command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def main() -> int:
    print("=" * 90)
    print("TRABAJO COMPLETO INTERACTIVO - SUITE INTEGRADA VCE v3.16 Linux consolidado")
    print("=" * 90)

    project_root = Path(ask("Ruta del proyecto", str(DEFAULT_PROJECT_ROOT)))
    patients_raw = ask("Pacientes a correr. Puedes escribir: 3, 6, 7", "3, 6, 7")
    patients = parse_patients(patients_raw)
    if not patients:
        patients = ("paciente 3", "paciente 6", "paciente 7")

    stages_raw = ask("Etapas", "Antes, Despues")
    stages = tuple(x.strip() for x in re.split(r"[,;]+", stages_raw) if x.strip()) or ("Antes", "Despues")
    gpu = ask("GPU", "auto").lower()
    if gpu not in {"auto", "on", "off"}:
        gpu = "auto"
    force = ask_yes_no("¿Reprocesar con --force? Recomendado si cambiaste versión", True)
    skip_stuck = ask_yes_no("¿Saltar tareas pegadas/fallidas y continuar?", True)
    run_real = ask_yes_no("¿Preparar y tratar de generar derivados reales FreeSurfer/fMRIPrep nativo si hay licencia?", True)

    cfg = PipelineConfig(
        project_root=project_root,
        patients=patients,
        control_name="sano",
        stages=stages,
        resume=True,
        force=force,
        skip_stuck=skip_stuck,
        gpu=gpu,
    )

    gpu_status = configure_gpu(cfg)
    cfg.results_root().mkdir(parents=True, exist_ok=True)
    for patient in list(cfg.patients) + [cfg.control_name]:
        ensure_result_tree(cfg.results_root(), patient, cfg.stages, cfg.tests)

    print("\nCONFIGURACIÓN")
    print(f"Proyecto: {cfg.project_root}")
    print(f"Pacientes: {', '.join(cfg.patients)}")
    print(f"Etapas: {', '.join(cfg.stages)}")
    print(f"Control: {cfg.control_name}")
    print(f"GPU: {gpu_status}")

    derivative_workspace = None
    if run_real:
        print("\n[1/10] Preparando BIDS mínimo y scripts de derivados reales...")
        derivative_workspace = run_real_derivatives_preparation(cfg)
        print(f"Workspace: {derivative_workspace}")

        license_ok = (cfg.project_root / "license.txt").exists()
        recon_ok = command_exists("recon-all")
        fmriprep_ok = command_exists("fmriprep")
        dcm2niix_ok = command_exists("dcm2niix")

        print(f"recon-all FreeSurfer: {'OK' if recon_ok else 'NO'} | license.txt: {'OK' if license_ok else 'NO'} | fMRIPrep: {'OK' if fmriprep_ok else 'NO'} | dcm2niix: {'OK' if dcm2niix_ok else 'NO'}")
        print("En Linux sin Docker/MATLAB: si quieres correr recon-all nativo completo, usa ./run_derivados_nativos_linux.sh.")
        print("Este flujo continúa con análisis interno y consolidación; integrará derivados reales si ya existen en resultados/derivados_externos.")
    else:
        print("\n[1/10] Saltando preparación/ejecución de derivados reales externos.")

    print("\n[2/10] Biomecánica EMG + dinamometría...")
    run_biomechanics(cfg)

    print("\n[3/10] Tomografía...")
    run_tomography(cfg)

    print("\n[4/10] Mapas funcionales...")
    run_maps(cfg)

    print("\n[5/10] Resonancias...")
    run_resonances(cfg)

    print("\n[6/10] Morfometría interna por reformateo/T1...")
    run_internal_morphometry_integration(cfg)

    print("\n[7/10] Integración de derivados externos reales si existen...")
    run_external_morphometry_integration(cfg)

    print("\n[8/10] Acoplamiento estructura-función...")
    run_structure_function_coupling(cfg)

    print("\n[9/10] Correlaciones paciente vs sano...")
    run_volume_correlations(cfg)

    print("\n[10/11] Comparación Antes vs Después...")
    run_before_after_comparisons(cfg)

    print("\n[11/11] Consolidado CSV por paciente y sano...")
    run_patient_consolidation(cfg)

    report = build_global_report(cfg)
    readme = write_readme_results(cfg)

    print("\n" + "=" * 90)
    print("FINALIZADO")
    print(f"Pacientes corridos: {', '.join(cfg.patients)}")
    print(f"Resultados: {cfg.results_root()}")
    if derivative_workspace:
        print(f"Derivados reales / scripts: {derivative_workspace}")
    if report:
        print(f"Reporte global: {report}")
    print(f"Guía de resultados: {readme}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
