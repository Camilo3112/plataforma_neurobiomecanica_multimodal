"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TRACTOGRAFÍA DE LA VÍA CORTICOESPINAL                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/cst_tronco_runner.py
Versión: v3.21.21

Descripción
-----------
Coordina la reconstrucción CST, la segmentación del tronco y las salidas para
Slicer/ITK-SNAP.

Fundamento físico-matemático implementado
-----------------------------------------
Reconstruye la vía corticoespinal a partir de difusión y anatomía T1. El flujo
parte del modelo DWI S(b)=S0 exp(-b g^T D g) para estimar anisotropía y
orientar
streamlines. Las fibras se expresan como curvas discretas r(s) en espacio RAS-
mm
y se filtran por waypoints anatómicos: corteza motora, mesencéfalo, puente y
bulbo. La corrección de registro usa matrices afines 4x4 y desplazamientos de
centro de masa Δc = c_T1 - c_b0 para reducir errores globales DWI->T1. La
salida
fusiona hemisferios y genera densidad voxelizada ρ(v)=n_streamlines(v).

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
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import PipelineConfig
from .paths import resolve_stage_dir
from .io_utils import safe_mkdir, save_dataframe, append_log, log_exception
from .brainstem_freesurfer import run_brainstem_segmentation


def _module_root() -> Path:
    return Path(__file__).resolve().parents[1] / "modulos" / "modulo_CST_tronco_integracion_completa"


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _t1_candidates(stage_dir: Path) -> list[Path]:
    """Busca la referencia anatómica T1 ya reformateada del paciente.

    En los datos reales el archivo puede llamarse `rT1.nii` o `rAnatomico.nii`
    y puede estar en rutas distintas, por ejemplo:

    - ResultadosFuncional/REFORMATEO/rAnatomico.nii
    - ResultadosFuncional/Resultados/Resultados/rAnatomico.nii
    - Resultados funcional/REFORMATEO/rT1.nii

    Por eso se priorizan rutas conocidas y después se hace una búsqueda
    recursiva dentro de carpetas de resultados funcionales y resonancia.
    """
    stage_dir = Path(stage_dir)

    explicit = [
        # Nombre real encontrado en varios pacientes
        stage_dir / "ResultadosFuncional" / "REFORMATEO" / "rAnatomico.nii",
        stage_dir / "Resultados funcional" / "REFORMATEO" / "rAnatomico.nii",
        stage_dir / "ResultadosFuncional" / "Resultados" / "Resultados" / "rAnatomico.nii",
        stage_dir / "Resultados funcional" / "Resultados" / "Resultados" / "rAnatomico.nii",

        # Nombre usado en versiones previas
        stage_dir / "ResultadosFuncional" / "REFORMATEO" / "rT1.nii",
        stage_dir / "Resultados funcional" / "REFORMATEO" / "rT1.nii",
        stage_dir / "ResultadosFuncional" / "Resultados" / "Resultados" / "rT1.nii",
        stage_dir / "Resultados funcional" / "Resultados" / "Resultados" / "rT1.nii",

        # Alternativas por si el reformateo quedó bajo resonancia
        stage_dir / "RESONANCIA" / "REFORMATEO" / "rAnatomico.nii",
        stage_dir / "RESONANCIA" / "REFORMATEO" / "rT1.nii",
        stage_dir / "Resonancia" / "REFORMATEO" / "rAnatomico.nii",
        stage_dir / "Resonancia" / "REFORMATEO" / "rT1.nii",
    ]

    roots = [
        stage_dir / "ResultadosFuncional",
        stage_dir / "Resultados funcional",
        stage_dir / "ResultadosFuncionales",
        stage_dir / "RESONANCIA",
        stage_dir / "Resonancia",
        stage_dir,
    ]

    recursive: list[Path] = []
    exact_names = {
        "ranatomico.nii", "ranatomico.nii.gz",
        "rt1.nii", "rt1.nii.gz",
        "anatomico.nii", "anatomico.nii.gz",
        "t1.nii", "t1.nii.gz",
    }

    for root in roots:
        if not root.exists():
            continue
        try:
            for f in root.rglob("*.nii*"):
                name = f.name.lower()
                parent_text = str(f.parent).lower()

                # Prioridad alta: archivos ya reformateados/registrados usados por SPM/ResultadosFuncional
                if name in exact_names:
                    recursive.append(f)
                    continue

                # Prioridad media: nombres anatómicos/T1, evitando mapas funcionales y máscaras
                if ("anatom" in name or "t1" in name) and not any(
                    bad in name for bad in [
                        "mask", "roi", "motor", "map", "bold", "fmri", "func", "activation",
                        "fa", "adc", "dwi", "colfa", "trace", "field", "swi"
                    ]
                ):
                    recursive.append(f)
                    continue

                # Si está dentro de REFORMATEO y empieza por r, suele ser una referencia registrada
                if "reformateo" in parent_text and name.startswith("r") and ("anat" in name or "t1" in name):
                    recursive.append(f)
        except Exception:
            # No se debe detener la ejecución por un error de permisos/lectura en una subcarpeta.
            continue

    # Quitar duplicados conservando orden: explícitos primero, luego hallazgos recursivos.
    ordered: list[Path] = []
    seen: set[str] = set()
    for f in explicit + recursive:
        key = str(f.resolve()) if f.exists() else str(f)
        if key not in seen:
            seen.add(key)
            ordered.append(f)
    return ordered


def _dwi_dicom_candidates(stage_dir: Path) -> list[Path]:
    """Raíces probables donde están los DICOM crudos de difusión.

    Se usa una lista amplia porque la organización del resonador puede variar
    entre pacientes y etapas. dcm2niix escanea recursivamente la carpeta dada.
    """
    return [
        stage_dir / "RESONANCIA",
        stage_dir / "Resonancia",
        stage_dir / "RM",
        stage_dir / "DIFUSION",
        stage_dir / "DWI",
        stage_dir / "RESONANCIA" / "DICOMDIR",
        stage_dir / "RESONANCIA" / "DICOM",
    ]


def _mkdir_tree_tractografia(out_base: Path) -> None:
    """Crea siempre la estructura esperada del anexo de tractografía.

    Así, aunque falte DWI, rT1 o falle el cálculo, queda una carpeta trazable
    con logs y un estado en resumen_modulo_cst_tronco.csv.
    """
    for rel in [
        ".",
        "logs_cst_tronco",
        "01_nifti_convertidos",
        "atlas_motor_rois",
        "brainstem_substructures_validated",
        "cst_visualizacion_ventral_medial",
    ]:
        safe_mkdir(out_base / rel)


def _run_subprocess(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    """Ejecuta un comando mostrando el avance en consola y guardando log.

    En versiones anteriores el proceso corría, pero la salida quedaba solo en
    el archivo de log. Para recon-all eso era confuso porque podía tardar horas
    sin mostrar actividad. Esta versión hace tee en tiempo real: todo lo que
    imprime recon-all, segment_subregions, dcm2niix o los scripts Python se ve
    en pantalla y también queda guardado.
    """
    safe_mkdir(log_path.parent)
    print("", flush=True)
    print("============================================================", flush=True)
    print("EJECUTANDO SUBPROCESO", flush=True)
    print("Comando:", " ".join(map(str, cmd)), flush=True)
    print("Directorio:", str(cwd), flush=True)
    print("Log:", str(log_path), flush=True)
    print("============================================================", flush=True)

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMANDO: " + " ".join(map(str, cmd)) + "\n")
        log.write("DIRECTORIO: " + str(cwd) + "\n")
        log.write("============================================================\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        proc.wait()
        log.write("\n============================================================\n")
        log.write(f"CODIGO_SALIDA: {proc.returncode}\n")
        log.flush()
    print("", flush=True)
    print(f"Subproceso terminado con codigo {proc.returncode}", flush=True)
    return int(proc.returncode or 0)


def run_cst_tronco_module(cfg: PipelineConfig) -> pd.DataFrame:
    """Integra el módulo tronco/CST como etapa explícita de la suite.

    Esta función no corre por defecto en `--only todo` porque FreeSurfer/recon-all
    puede tardar horas. Se ejecuta con `--only cst_tronco`.
    """
    module_root = _module_root()
    cst_script = module_root / "03_cst_visualizacion" / "FINAL_V4_CORONAL" / "extraer_visualizar_cst_colores_V4_coronal.py"
    tract_script = module_root / "03_tractografia_propia" / "tractografia_propia_T1_atlas_motor_linux.py"
    brainstem_ps1 = module_root / "02_segmentacion_tronco" / "FINAL_V9" / "segmentar_tronco_todo_windows_V9.ps1"
    brainstem_validator_linux = module_root / "02_segmentacion_tronco" / "FINAL_LINUX" / "validar_brainstem_substructures_T1_linux.py"

    rows: list[dict] = []
    if not module_root.exists():
        raise FileNotFoundError(f"No existe el módulo CST integrado: {module_root}")
    if not cst_script.exists():
        raise FileNotFoundError(f"No existe script CST V4: {cst_script}")
    if not tract_script.exists():
        raise FileNotFoundError(f"No existe script de tractografía propia: {tract_script}")

    for patient in cfg.patients:
        for stage in cfg.stages:
            stage_dir = resolve_stage_dir(cfg.data_root() / patient, stage)
            out_base = cfg.results_root() / patient / stage / "tractografia_propia"
            # Alias opcional para usuarios que busquen el nombre en plural.
            out_alias_plural = cfg.results_root() / patient / stage / "tractografias_propias"
            cst_out = out_base / "cst_visualizacion_ventral_medial"
            log_dir = out_base / "logs_cst_tronco"
            _mkdir_tree_tractografia(out_base)
            try:
                if not out_alias_plural.exists():
                    out_alias_plural.symlink_to(out_base, target_is_directory=True)
            except Exception:
                # No es crítico: el nombre oficial del pipeline es tractografia_propia.
                pass
            row = {
                "patient": patient,
                "subject": patient,
                "stage": stage,
                "modality": "tractografia_cst_tronco",
                "module_version": "cst_tronco_v9_v8_python_main_documentado_suite_v3_21_20",
                "status": "pendiente",
                "t1_path": "",
                "wholebrain_trk": str(out_base / "wholebrain_T1.trk"),
                "brainstem_dir": str(out_base / "brainstem_substructures_validated"),
                "cst_output_dir": str(cst_out),
                "source_module": str(module_root),
            }

            try:
                t1_candidates = _t1_candidates(stage_dir)
                t1 = _first_existing(t1_candidates)
                row["t1_candidates_checked"] = ";".join(str(p) for p in t1_candidates[:25])
                if t1 is None:
                    row["status"] = "omitido_sin_rT1_o_rAnatomico"
                    row["note"] = "No se encontró rAnatomico.nii/rT1.nii mediante rutas conocidas ni búsqueda recursiva."
                    rows.append(row)
                    continue
                row["t1_path"] = str(t1)
                print("", flush=True)
                print("============================================================", flush=True)
                print(f"PACIENTE {patient} / {stage}", flush=True)
                print("============================================================", flush=True)
                print(f"T1 encontrado: {t1}", flush=True)
                print(f"Salida tractografia: {out_base}", flush=True)

                wholebrain = out_base / "wholebrain_T1.trk"
                if not wholebrain.exists():
                    print("[TRACTOGRAFIA] No existe wholebrain_T1.trk. Se generará desde DWI crudo.", flush=True)
                    # v3.21.7: el anexo ya no solo consume wholebrain_T1.trk;
                    # si no existe, intenta generar la tractografía propia desde DWI crudo.
                    dwi_root = _first_existing(_dwi_dicom_candidates(stage_dir))
                    if dwi_root is None:
                        row["status"] = "omitido_sin_dicom_dwi"
                        row["note"] = "No se encontró carpeta DICOM de difusión. Se creó la estructura tractografia_propia, pero no se pudo calcular wholebrain_T1.trk."
                        rows.append(row)
                        continue

                    env_tract = os.environ.copy()
                    env_tract.update({
                        "VCE_PROJECT_ROOT": str(cfg.project_root),
                        "VCE_PATIENT_ID": patient,
                        "VCE_STAGE": stage,
                        "VCE_DICOM_ROOT": str(dwi_root),
                        "VCE_T1_PATH": str(t1),
                        "VCE_TRACTOGRAPHY_DIR": str(out_base),
                    })
                    print(f"[TRACTOGRAFIA] DWI root seleccionado: {dwi_root}", flush=True)
                    code = _run_subprocess([sys.executable, str(tract_script)], cwd=tract_script.parent, env=env_tract, log_path=log_dir / "tractografia_propia_stdout.log")
                    if code != 0:
                        row["status"] = f"fallo_tractografia_propia_codigo_{code}"
                        row["dwi_root"] = str(dwi_root)
                        rows.append(row)
                        continue
                    if not wholebrain.exists():
                        row["status"] = "fallo_tractografia_no_genero_wholebrain_T1_trk"
                        row["dwi_root"] = str(dwi_root)
                        rows.append(row)
                        continue
                    row["tractografia_generada_en_ejecucion"] = True
                    row["dwi_root"] = str(dwi_root)

                # Antes de procesar mesencéfalo, se exige recon-all completo y aparc+aseg.mgz.
                # Si las máscaras de tronco ya existen pero no hay aparc+aseg verificable, se fuerza
                # el paso FreeSurfer para evitar usar una segmentación heurística o incompleta.
                brainstem_dir = out_base / "brainstem_substructures_validated"
                subject_id = f"{patient.replace(' ', '')}_{stage}"
                subjects_dir = Path(os.environ.get("VCE_FREESURFER_SUBJECTS_DIR", str(Path.home() / "freesurfer_subjects")))
                fs_subject_dir = subjects_dir / subject_id
                aparc_aseg = fs_subject_dir / "mri" / "aparc+aseg.mgz"
                aseg = fs_subject_dir / "mri" / "aseg.mgz"
                recon_done = fs_subject_dir / "scripts" / "recon-all.done"
                row["freesurfer_subject_id"] = subject_id
                row["subjects_dir"] = str(subjects_dir)
                row["aparc_aseg_path"] = str(aparc_aseg)
                row["aparc_aseg_exists_pre"] = bool(aparc_aseg.exists())
                required_brainstem = [
                    brainstem_dir / "midbrain_T1.nii.gz",
                    brainstem_dir / "pons_T1.nii.gz",
                    brainstem_dir / "medulla_T1.nii.gz",
                ]
                require_aparc_gate = os.environ.get("VCE_REQUIRE_APARC_ASEG_BEFORE_MIDBRAIN", "1") != "0"
                brainstem_already_ok = all(p.exists() for p in required_brainstem)
                freesurfer_ready = recon_done.exists() and aparc_aseg.exists() and aseg.exists()
                if require_aparc_gate and not freesurfer_ready:
                    brainstem_already_ok = False
                    row["note_aparc_gate"] = "Se fuerza/reanuda recon-all porque falta recon-all.done, aseg.mgz o aparc+aseg.mgz antes del mesencéfalo."

                print("", flush=True)
                print("[FREESURFER] Verificación antes del mesencéfalo", flush=True)
                print(f"  SUBJECTS_DIR: {subjects_dir}", flush=True)
                print(f"  subject_id: {subject_id}", flush=True)
                print(f"  recon-all.done: {recon_done} -> {recon_done.exists()}", flush=True)
                print(f"  aseg.mgz: {aseg} -> {aseg.exists()}", flush=True)
                print(f"  aparc+aseg.mgz: {aparc_aseg} -> {aparc_aseg.exists()}", flush=True)
                if not brainstem_already_ok:
                    print("[FREESURFER] Falta recon-all/aparc+aseg o no está validado. Se ejecutará/reanudará recon-all y luego se segmentará mesencéfalo.", flush=True)
                    license_path = Path(os.environ.get("FS_LICENSE", str(cfg.project_root / "license.txt")))
                    if not brainstem_ps1.exists():
                        row["status"] = "omitido_sin_script_segmentacion_tronco"
                        rows.append(row)
                        continue
                    if not license_path.exists():
                        row["status"] = "pendiente_license_freesurfer"
                        row["note"] = f"Falta {license_path}. Ejecuta segmentación V9 cuando tengas licencia FreeSurfer."
                        rows.append(row)
                        continue
                    if sys.platform.startswith("win"):
                        cmd = [
                            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                            "-File", str(brainstem_ps1),
                            "-ProjectRoot", str(cfg.project_root),
                            "-T1Path", str(t1),
                            "-LicensePath", str(license_path),
                            "-SubjectId", subject_id,
                        ]
                        code = _run_subprocess(cmd, cwd=brainstem_ps1.parent, env=os.environ.copy(), log_path=log_dir / "brainstem_v9_stdout.log")
                        if code != 0:
                            row["status"] = f"fallo_brainstem_codigo_{code}"
                            rows.append(row)
                            continue
                    else:
                        # v3.21.20: la segmentación del tronco ya no depende de lanzadores .sh.
                        # Se ejecuta desde Python para que GitHub tenga un único punto de entrada.
                        try:
                            bs_result = run_brainstem_segmentation(
                                project_root=cfg.project_root,
                                t1_path=t1,
                                license_path=license_path,
                                subject_id=subject_id,
                                out_base=out_base,
                                threads=max(1, int(os.environ.get("VCE_FS_THREADS", "8"))),
                                subjects_dir=subjects_dir,
                                validator_path=brainstem_validator_linux,
                            )
                            row["brainstem_python_status"] = bs_result.status
                            row["brainstem_python_report"] = str(bs_result.report_json or "")
                        except Exception as exc:
                            row["status"] = "fallo_brainstem_python"
                            row["error"] = str(exc)
                            rows.append(row)
                            continue

                # Validación dura: mesencéfalo solo es aceptado si antes existe aparc+aseg de FreeSurfer.
                row["aparc_aseg_exists_post"] = bool(aparc_aseg.exists())
                print("", flush=True)
                print("[FREESURFER] Validación post recon-all", flush=True)
                print(f"  recon-all.done: {recon_done.exists()}", flush=True)
                print(f"  aseg.mgz: {aseg.exists()}", flush=True)
                print(f"  aparc+aseg.mgz: {aparc_aseg.exists()}", flush=True)
                if require_aparc_gate and not (recon_done.exists() and aparc_aseg.exists() and aseg.exists()):
                    row["status"] = "fallo_sin_recon_all_aparc_aseg"
                    row["note"] = "No se procesa CST: falta recon-all.done, aseg.mgz o aparc+aseg.mgz. El mesencéfalo no queda anatómicamente confiable."
                    rows.append(row)
                    continue
                if not all(p.exists() for p in required_brainstem):
                    row["status"] = "fallo_sin_mesencefalo_puente_bulbo_validados"
                    row["note"] = "No se procesa CST: faltan midbrain_T1/pons_T1/medulla_T1 validados."
                    rows.append(row)
                    continue

                env = os.environ.copy()
                env.update({
                    "VCE_PROJECT_ROOT": str(cfg.project_root),
                    "VCE_PATIENT_ID": patient,
                    "VCE_STAGE": stage,
                    "VCE_T1_PATH": str(t1),
                    "VCE_TRACTOGRAPHY_DIR": str(out_base),
                    "VCE_WHOLEBRAIN_TRK": str(wholebrain),
                    "VCE_BRAINSTEM_DIR": str(brainstem_dir),
                    "VCE_CST_OUTPUT_DIR": str(cst_out),
                    "VCE_CST_RESCATE_MESENCEFALO_ENABLE": os.environ.get("VCE_CST_RESCATE_MESENCEFALO_ENABLE", "1"),
                    "VCE_CST_DILATACION_TRONCO_MM": os.environ.get("VCE_CST_DILATACION_TRONCO_MM", "4.0"),
                    "VCE_CST_DILATACION_CORTEZA_MM": os.environ.get("VCE_CST_DILATACION_CORTEZA_MM", "8.0"),
                    "VCE_CST_FRACCION_IPSILATERAL_MINIMA": os.environ.get("VCE_CST_FRACCION_IPSILATERAL_MINIMA", "0.60"),
                })
                if cfg.force or not (cst_out / "cst_ventral_medial_reporte.json").exists():
                    print("", flush=True)
                    print("[CST] Reconstruyendo vía corticoespinal usando mesencéfalo validado como waypoint...", flush=True)
                    code = _run_subprocess([sys.executable, str(cst_script)], cwd=module_root, env=env, log_path=log_dir / "cst_v4_stdout.log")
                    if code != 0:
                        row["status"] = f"fallo_cst_codigo_{code}"
                        rows.append(row)
                        continue

                report = cst_out / "cst_ventral_medial_reporte.json"
                final_cst_trk = out_base / "via_cortico_espinal_completa.trk"
                final_cst_nii = out_base / "via_cortico_espinal_completa.nii"
                final_slicer_trk = out_base / "via_cortico_espinal_completa_slicer.trk"
                final_slicer_vtk = out_base / "via_cortico_espinal_completa_slicer.vtk"
                row["via_cortico_espinal_completa_trk"] = str(final_cst_trk)
                row["via_cortico_espinal_completa_nii"] = str(final_cst_nii)
                row["via_cortico_espinal_completa_slicer_trk"] = str(final_slicer_trk)
                row["via_cortico_espinal_completa_slicer_vtk"] = str(final_slicer_vtk)
                row["via_cortico_espinal_completa_trk_exists"] = bool(final_cst_trk.exists())
                row["via_cortico_espinal_completa_nii_exists"] = bool(final_cst_nii.exists())
                row["via_cortico_espinal_completa_slicer_trk_exists"] = bool(final_slicer_trk.exists())
                row["via_cortico_espinal_completa_slicer_vtk_exists"] = bool(final_slicer_vtk.exists())
                row["status"] = "ok" if report.exists() and final_cst_trk.exists() and final_cst_nii.exists() and final_slicer_trk.exists() and final_slicer_vtk.exists() else ("ok_sin_salida_slicer" if report.exists() and final_cst_trk.exists() and final_cst_nii.exists() else ("ok_sin_salida_final_fusionada" if report.exists() else "finalizado_sin_reporte_json"))
                row["report_json"] = str(report)
                rows.append(row)
            except Exception as exc:
                log_exception(log_dir / "cst_tronco_exception.txt", exc)
                row["status"] = "error"
                row["error"] = str(exc)
                rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = cfg.results_root() / "resumen_modulo_cst_tronco.csv"
    save_dataframe(df, out_csv)
    return df
