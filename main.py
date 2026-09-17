"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     ORQUESTADOR GENERAL DE LA SUITE VCE                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: main.py
Versión: v3.21.24

Descripción
-----------
Punto de entrada único para ejecutar secciones por paciente, etapa o corrida
completa.


-----------------------------------------
Integra módulos heterogéneos bajo un flujo reproducible por paciente y etapa.
La lógica de ejecución conserva la relación entre espacios de imagen mediante
transformaciones afines homogéneas 4x4, selección de secciones y propagación
ordenada de productos: señal -> imagen -> morfometría -> tractografía ->
reporte.
El modelo computacional se formula como un grafo acíclico de dependencias:
D_i = f_i(D_{i-1}, parámetros), donde cada sección consume datos validados y
escribe resultados trazables sin modificar los datos crudos.

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

###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  1. IMPORTACIONES BÁSICAS Y DEFINICIONES DE SECCIONES
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

VERSION = "3.21.24"
DEFAULT_PROJECT_ROOT = Path("/home/humath/Escritorio")

SECTION_ALIASES = {
    "todo": "todo",
    "biomecanica": "biomecanica",
    "emg": "biomecanica",
    "dinamometria": "biomecanica",
    "tomografia": "tomografia",
    "tac": "tomografia",
    "tomografia_avanzada": "tomografia_avanzada",
    "mapas": "mapas",
    "resonancias": "resonancias",
    "mri": "resonancias",
    "diagnostico": "diagnostico",
    "derivados": "derivados",
    "preparar_derivados": "preparar_derivados",
    "preparar_derivados_reales": "preparar_derivados_reales",
    "morfometria_interna": "morfometria_interna",
    "corteza_fix": "corteza_fix",
    "fs_mni_motor": "fs_mni_motor",
    "freesurfer": "fs_mni_motor",
    "cst": "cst_tronco",
    "cst_tronco": "cst_tronco",
    "tractografia": "cst_tronco",
    "tractografia_rgb": "cst_tronco",
    "tractografia_cst": "cst_tronco",
    "tractografia_cst_rgb": "cst_tronco",
    "cst_rgb": "cst_tronco",
    "rgb_cst": "cst_tronco",
    "via_corticoespinal": "cst_tronco",
    "via_corticoespinal_rgb": "cst_tronco",
    "via_cortico_espinal": "cst_tronco",
    "via_cortico_espinal_rgb": "cst_tronco",
    "zonas": "zonas_correlacion",
    "19_zonas": "zonas_correlacion",
    "regiones": "zonas_correlacion",
    "zonas_correlacion": "zonas_correlacion",
    "correlacion_zonas": "zonas_correlacion",
    "estructura_funcion": "estructura_funcion",
    "correlaciones": "correlaciones",
    "comparacion": "comparacion",
    "consolidado": "consolidado",
}

DEFAULT_SECTIONS = ("todo",)


SECTION_ORDER = (
    "todo",
    "diagnostico",
    "biomecanica",
    "tomografia",
    "tomografia_avanzada",
    "mapas",
    "resonancias",
    "preparar_derivados",
    "preparar_derivados_reales",
    "morfometria_interna",
    "corteza_fix",
    "derivados",
    "fs_mni_motor",
    "cst_tronco",
    "zonas_correlacion",
    "estructura_funcion",
    "correlaciones",
    "comparacion",
    "consolidado",
)

SECTION_CATALOG = {
    "todo": {
        "titulo": "Pipeline completo",
        "descripcion": "Ejecuta todas las secciones disponibles en el orden fisiológico y computacional del proyecto.",
        "modelo": "Orquestación secuencial D_i = f_i(D_{i-1}, parámetros), manteniendo trazabilidad paciente/etapa.",
        "alias": "todo",
    },
    "diagnostico": {
        "titulo": "Diagnóstico de resonancias y datos disponibles",
        "descripcion": "Inspecciona carpetas, NIfTI, DICOM, bval/bvec y disponibilidad anatómica antes de procesar.",
        "modelo": "Control de consistencia geométrica y dimensional: forma del volumen, número de gradientes y coherencia de metadatos.",
        "alias": "diagnostico",
    },
    "biomecanica": {
        "titulo": "Biomecánica, EMG y dinamometría",
        "descripcion": "Procesa señales musculares y cinéticas para métricas temporales, espectrales y de simetría.",
        "modelo": "RMS, normalización, FFT, correlación cruzada, autocorrelación e interpolación multi-rate.",
        "alias": "biomecanica, emg, dinamometria",
    },
    "tomografia": {
        "titulo": "Tomografía TAC base",
        "descripcion": "Segmenta tejido muscular/adiposo y genera métricas morfológicas desde imágenes TAC.",
        "modelo": "Unidades Hounsfield, umbralización tisular, morfología matemática y cálculo de área/volumen por voxel spacing.",
        "alias": "tomografia, tac",
    },
    "tomografia_avanzada": {
        "titulo": "Tomografía avanzada",
        "descripcion": "Ejecuta análisis complementario de miembro completo, cortes de control y métricas refinadas.",
        "modelo": "Selección anatómica de cortes, detección de regiones bilaterales y cuantificación volumétrica por integración discreta.",
        "alias": "tomografia_avanzada",
    },
    "mapas": {
        "titulo": "Mapas funcionales y derivados de imagen",
        "descripcion": "Procesa mapas derivados y productos intermedios de neuroimagen.",
        "modelo": "Remuestreo espacial, máscaras binarias, operaciones voxel a voxel y preservación de afines NIfTI.",
        "alias": "mapas",
    },
    "resonancias": {
        "titulo": "Resonancias estructurales y funcionales",
        "descripcion": "Procesa resonancias disponibles por paciente y etapa para análisis morfológico/funcional.",
        "modelo": "Alineación de volúmenes, normalización de intensidades y comparación en espacio anatómico nativo.",
        "alias": "resonancias, mri",
    },
    "preparar_derivados": {
        "titulo": "Plantilla para derivados externos",
        "descripcion": "Crea carpetas y plantillas para anexar derivados externos del proyecto.",
        "modelo": "Organización reproducible de entradas derivadas sin modificar datos crudos.",
        "alias": "preparar_derivados",
    },
    "preparar_derivados_reales": {
        "titulo": "Workspace de derivados reales",
        "descripcion": "Prepara estructura de trabajo para generar derivados reales con trazabilidad.",
        "modelo": "Separación de insumos, derivados y reportes mediante rutas determinísticas por paciente/etapa.",
        "alias": "preparar_derivados_reales",
    },
    "morfometria_interna": {
        "titulo": "Morfometría interna",
        "descripcion": "Integra métricas volumétricas y morfológicas calculadas dentro de la suite.",
        "modelo": "Conteo voxelizado: volumen = número de voxeles × tamaño de voxel; índices de asimetría y diferencia longitudinal.",
        "alias": "morfometria_interna",
    },
    "corteza_fix": {
        "titulo": "Corrección de ROI cortical",
        "descripcion": "Ajusta regiones corticales motoras a la geometría del T1/rAnatomico.",
        "modelo": "Máscaras corticales, shell cerebral, transformaciones afines y búsqueda geométrica de superposición anatómica.",
        "alias": "corteza_fix",
    },
    "derivados": {
        "titulo": "Integración de morfometría externa",
        "descripcion": "Integra resultados externos al consolidado del paciente.",
        "modelo": "Unificación tabular de métricas por paciente, etapa, hemisferio, estructura y unidad física.",
        "alias": "derivados",
    },
    "fs_mni_motor": {
        "titulo": "FreeSurfer, MNI y corteza motora",
        "descripcion": "Usa FreeSurfer y atlas anatómicos para corteza motora y etiquetas estructurales.",
        "modelo": "recon-all, aparc+aseg, aseg, atlas MNI, transformaciones directas/inversas y remuestreo nearest-neighbor para etiquetas.",
        "alias": "fs_mni_motor, freesurfer",
    },
    "cst_tronco": {
        "titulo": "Tractografía CST, vía corticoespinal RGB y tronco encefálico",
        "descripcion": "Reconstruye la vía corticoespinal bilateral, delimita mesencéfalo/puente/bulbo y genera TRK, VTK, NIfTI RGB, máscara y densidad.",
        "modelo": "DWI con reconstrucción por streamlines, filtrado anatómico por waypoints corticales y tronco encefálico, densidad voxelizada y color RGB por orientación local de la fibra.",
        "alias": "cst_tronco, cst, tractografia, tractografia_rgb, tractografia_cst_rgb, cst_rgb, via_corticoespinal_rgb",
    },
    "zonas_correlacion": {
        "titulo": "19 zonas anatómicas, correlación local y Excel",
        "descripcion": "Delimita 19 regiones sobre T1, calcula correlación local Antes/Después y genera NIfTI/Excel.",
        "modelo": "FreeSurfer aparc+aseg/aseg, remuestreo nearest-neighbor, volumen = voxeles × spacing, registro rígido, z-score robusto y correlación local 2D Pearson/NCC en tres planos.",
        "alias": "zonas_correlacion, zonas, 19_zonas, regiones, correlacion_zonas",
    },
    "estructura_funcion": {
        "titulo": "Acople estructura-función",
        "descripcion": "Relaciona mediciones estructurales con resultados funcionales/biomecánicos.",
        "modelo": "Normalización de variables, emparejamiento longitudinal y métricas de asociación entre dominios.",
        "alias": "estructura_funcion",
    },
    "correlaciones": {
        "titulo": "Correlaciones de volumen e imagen",
        "descripcion": "Calcula correlaciones y mapas comparativos entre volúmenes antes/después.",
        "modelo": "Correlación de Pearson/Spearman según datos, comparación voxel-wise y métricas de similitud espacial.",
        "alias": "correlaciones",
    },
    "comparacion": {
        "titulo": "Comparación Antes vs Después",
        "descripcion": "Genera diferencias longitudinales por paciente y etapa.",
        "modelo": "Delta = Después - Antes, porcentaje de cambio, índices de asimetría y tablas longitudinales.",
        "alias": "comparacion",
    },
    "consolidado": {
        "titulo": "Consolidado final por paciente",
        "descripcion": "Une métricas, reportes y salidas principales en tablas finales.",
        "modelo": "Agregación tabular reproducible con claves paciente/etapa/sección/métrica/unidad.",
        "alias": "consolidado",
    },
}


def print_section_catalog() -> None:
    """Imprime el catálogo profesional de secciones disponibles para ejecución."""
    print("\n" + "=" * 100)
    print("SECCIONES DISPONIBLES DEL PIPELINE")
    print("=" * 100)
    for i, section in enumerate(SECTION_ORDER, start=1):
        info = SECTION_CATALOG[section]
        print(f"{i:02d}. {section}")
        print(f"    Nombre:  {info['titulo']}")
        print(f"    Uso:     --sections {section}")
        print(f"    Alias:   {info['alias']}")
        print(f"    Modelo:  {info['modelo']}")
        print(f"    Salida:  {info['descripcion']}")
        print("-" * 100)
    print("Ejemplo CST RGB: python main.py run --project-root /home/humath/Escritorio --patients 3 --stages Antes --sections tractografia_cst_rgb")
    print("Nota: tractografia_cst_rgb, cst_rgb y via_corticoespinal_rgb ejecutan internamente cst_tronco.")
    print("=" * 100 + "\n")


def print_terminal_examples() -> None:
    """Imprime comandos listos para copiar en la terminal integrada de Visual Studio Code."""
    root = "/home/humath/Escritorio"
    repo = "/home/humath/Escritorio/plataforma_neurobiomecanica_multimodal-main"
    print("\n" + "=" * 100)
    print("COMANDOS RÁPIDOS PARA TERMINAL DE VISUAL STUDIO CODE")
    print("=" * 100)
    print(f"cd {repo}")
    print("source .venv/bin/activate")
    print("export FREESURFER_HOME=/usr/local/freesurfer/8.2.0")
    print('source "$FREESURFER_HOME/SetUpFreeSurfer.sh"')
    print("export FS_LICENSE=/home/humath/Escritorio/license.txt")
    print("python main.py doctor --project-root /home/humath/Escritorio")
    print("python main.py --list-sections")
    print("python main.py")
    print("\nEjecución recomendada paciente 3 Antes: tractografía CST + vía corticoespinal RGB:")
    print(f"python main.py run --project-root {root} --patients 3 --stages Antes --sections tractografia_cst_rgb --registro-com-corregir --registro-com-ejes z --no-suspend")
    print("\nComando equivalente con nombre técnico interno:")
    print(f"python main.py run --project-root {root} --patients 3 --stages Antes --sections cst_tronco --registro-com-corregir --registro-com-ejes z --no-suspend")
    print("\n19 zonas + correlación Antes/Después para paciente 3:")
    print(f"python main.py run --project-root {root} --patients 3 --stages Antes Despues --sections zonas_correlacion --no-suspend")
    print("\nCST RGB + 19 zonas + correlación para paciente 3:")
    print(f"python main.py run --project-root {root} --patients 3 --stages Antes Despues --sections tractografia_cst_rgb zonas_correlacion --registro-com-corregir --registro-com-ejes z --no-suspend")
    print("\nEjecución completa de todos los pacientes con CST RGB + 19 zonas:")
    print(f"python main.py run --project-root {root} --all-patients --stages Antes Despues --sections tractografia_cst_rgb zonas_correlacion --registro-com-corregir --registro-com-ejes z --no-suspend")
    print("=" * 100 + "\n")


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  2. NORMALIZACIÓN DE PACIENTES, ETAPAS Y SECCIONES
# ══════════════════════════════════════════════════════════════════════════════


def normalize_patient_token(token: str) -> str:
    """Convierte entradas como '3', 'paciente3' o 'paciente 7' a carpeta estándar."""
    t = str(token or "").strip().strip(",;")
    if not t:
        return ""
    low = t.lower().replace("_", " ")
    if low in {"y", "e", "and", "or", "o"}:
        return ""
    if low == "sano":
        return "sano"
    m = re.search(r"(?:paciente\s*)?(\d+)", low)
    if m:
        return f"paciente {int(m.group(1))}"
    if not low.startswith("paciente"):
        return f"paciente {t}"
    return " ".join(t.split())


def normalize_patients(values: Iterable[str] | str | None) -> tuple[str, ...]:
    """Normaliza listas separadas por espacios, comas o punto y coma."""
    raw = " ".join(str(v) for v in values) if isinstance(values, (list, tuple)) else str(values or "")
    raw = re.sub(r"paciente\s*(\d+)", r"\1", raw, flags=re.IGNORECASE)
    tokens = [x for x in re.split(r"[,;\s]+", raw) if x.strip()]
    out: list[str] = []
    for tok in tokens:
        patient = normalize_patient_token(tok)
        if patient and patient != "sano" and patient not in out:
            out.append(patient)
    return tuple(out)


def normalize_sections(values: Iterable[str] | str | None) -> tuple[str, ...]:
    """Convierte nombres cortos de sección a los nombres internos del pipeline."""
    raw = " ".join(str(v) for v in values) if isinstance(values, (list, tuple)) else str(values or "")
    tokens = [x.strip().lower() for x in re.split(r"[,;\s]+", raw) if x.strip()]
    if not tokens:
        return DEFAULT_SECTIONS
    out: list[str] = []
    for token in tokens:
        if token not in SECTION_ALIASES:
            raise SystemExit(f"Sección no reconocida: {token}. Usa --list-sections para ver opciones.")
        section = SECTION_ALIASES[token]
        if section not in out:
            out.append(section)
    return tuple(out)


def prompt_list(prompt: str, default: str) -> str:
    """Lectura interactiva segura para terminal o ejecución redireccionada."""
    try:
        value = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        value = ""
    return value or default


def discover_patients(data_root: Path) -> tuple[str, ...]:
    """Detecta carpetas paciente N dentro del directorio de datos."""
    if not data_root.exists():
        return ()
    found = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and re.match(r"paciente\s*\d+", p.name, flags=re.IGNORECASE):
            found.append(normalize_patient_token(p.name))
    return tuple(dict.fromkeys(found))


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  3. CONFIGURACIÓN DEL PIPELINE Y CONTROL DE ENERGÍA
# ══════════════════════════════════════════════════════════════════════════════


def build_config(args):
    """Crea PipelineConfig con importación diferida para que `setup` no requiera paquetes instalados."""
    from src.config import PipelineConfig

    project_root = Path(args.project_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    results_root = Path(args.results_root).expanduser().resolve() if args.results_root else None

    if getattr(args, "interactive", False):
        print_section_catalog()
        patients = normalize_patients(prompt_list("Pacientes a correr, ejemplo 3,6,7,9", "3"))
        stages = tuple(x.strip() for x in re.split(r"[,;\s]+", prompt_list("Etapas", "Antes")) if x.strip())
        sections = normalize_sections(prompt_list("Secciones, ejemplo tractografia_cst_rgb, zonas_correlacion, cst_tronco, tomografia, fs_mni_motor o todo", "tractografia_cst_rgb"))
    elif getattr(args, "all_patients", False):
        patients = discover_patients(data_root or (project_root / "datos"))
        stages = tuple(args.stages)
        sections = normalize_sections(args.sections)
    else:
        patients = normalize_patients(args.patients)
        stages = tuple(args.stages)
        sections = normalize_sections(args.sections)

    if not patients:
        raise SystemExit("No se detectaron pacientes. Usa --patients 3 6 7 o revisa la carpeta de datos.")

    # Variables FreeSurfer y CST que antes estaban en scripts .sh. Ahora quedan explícitas aquí.
    if args.fs_home:
        os.environ["FREESURFER_HOME"] = str(Path(args.fs_home).expanduser())
    if args.fs_license:
        os.environ["FS_LICENSE"] = str(Path(args.fs_license).expanduser())
    if args.fs_subjects_dir:
        os.environ["VCE_FREESURFER_SUBJECTS_DIR"] = str(Path(args.fs_subjects_dir).expanduser())
    os.environ["VCE_FS_THREADS"] = str(args.fs_threads)

    if args.registro_com_corregir:
        os.environ["VCE_REGISTRO_COM_CORREGIR"] = "1"
        os.environ["VCE_REGISTRO_COM_EJES"] = args.registro_com_ejes

    os.environ.setdefault("VCE_REQUIRE_APARC_ASEG_BEFORE_MIDBRAIN", "1")
    os.environ.setdefault("VCE_CST_RESCATE_MESENCEFALO_ENABLE", "1")
    os.environ.setdefault("VCE_CST_DILATACION_CORTEZA_MM", "8.0")
    os.environ.setdefault("VCE_CST_DILATACION_TRONCO_MM", "4.0")
    os.environ.setdefault("VCE_CST_FRACCION_IPSILATERAL_MINIMA", "0.60")

    cfg = PipelineConfig(
        project_root=project_root,
        data_root_override=data_root,
        results_root_override=results_root,
        patients=patients,
        control_name=args.control,
        stages=stages,
        resume=not args.no_resume,
        force=args.force,
        skip_stuck=args.skip_stuck,
        gpu=args.gpu,
        gpu_device=args.gpu_device,
        gpu_min_elements=args.gpu_min_elements,
    )
    return cfg, sections


def maybe_reexec_with_systemd_inhibit(argv: list[str]) -> None:
    """Evita suspensión durante corridas largas sin depender de `.sh`."""
    if os.environ.get("VCE_SYSTEMD_INHIBITED") == "1":
        return
    exe = shutil.which("systemd-inhibit")
    if not exe:
        print("[ENERGÍA] systemd-inhibit no está disponible. Continúo sin bloqueo de suspensión.", flush=True)
        return
    env = os.environ.copy()
    env["VCE_SYSTEMD_INHIBITED"] = "1"
    cmd = [
        exe,
        "--what=sleep:shutdown:idle",
        "--who=VCE Suite",
        "--why=Procesamiento biomédico largo FreeSurfer/tractografía",
        sys.executable,
        str(Path(__file__).resolve()),
        *[x for x in argv if x != "--no-suspend"],
    ]
    print("[ENERGÍA] Reejecutando con systemd-inhibit para evitar suspensión.", flush=True)
    os.execvpe(exe, cmd, env)


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  4. LIMPIEZA CONTROLADA DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════


def find_tractography_dirs(results_root: Path) -> list[Path]:
    """Encuentra variantes de carpetas de tractografía propia para limpiar."""
    patterns = [
        "tractografia_propia",
        "tractografias_propias",
        "tractografia propia",
        "tractografias propias",
    ]
    out: list[Path] = []
    if not results_root.exists():
        return out
    for p in results_root.rglob("*"):
        if not p.is_dir():
            continue
        name = p.name.lower()
        if name in patterns or name.startswith("tractografia_propia_backup"):
            out.append(p)
    return sorted(out, key=lambda x: len(str(x)), reverse=True)


def clean_tractography(results_root: Path, *, yes: bool) -> int:
    """Borra carpetas viejas de tractografía solo si el usuario confirma con --yes."""
    targets = find_tractography_dirs(results_root)
    print("=" * 90)
    print("LIMPIEZA DE TRACTOGRAFÍAS PROPIAS")
    print(f"Raíz resultados: {results_root}")
    print(f"Carpetas encontradas: {len(targets)}")
    print("=" * 90)
    for path in targets:
        print(path)
    if not yes:
        print("\nModo vista previa: no se borró nada. Para borrar usa: python main.py clean --target tractografia --yes")
        return 0
    for path in targets:
        try:
            if path.is_symlink():
                path.unlink()
            else:
                shutil.rmtree(path)
            print(f"BORRADO: {path}")
        except Exception as exc:
            print(f"NO SE PUDO BORRAR {path}: {exc}")
    return 0


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  5. EJECUCIÓN DE SECCIONES DEL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def print_run_header(cfg, sections: tuple[str, ...], gpu_status: dict) -> None:
    """Muestra una cabecera profesional y trazable de la corrida."""
    print("=" * 100)
    print(f"SUITE INTEGRADA VCE v{VERSION}")
    print(f"Proyecto:   {cfg.project_root}")
    print(f"Datos:      {cfg.data_root()}")
    print(f"Resultados: {cfg.results_root()}")
    print(f"Pacientes:  {', '.join(cfg.patients)}")
    print(f"Etapas:     {', '.join(cfg.stages)}")
    print(f"Secciones:  {', '.join(sections)}")
    print(f"Resume:     {'sí' if cfg.resume else 'no'} | Force: {'sí' if cfg.force else 'no'} | Skip stuck: {'sí' if cfg.skip_stuck else 'no'}")
    if gpu_status.get("enabled"):
        print(f"GPU:        activa · {gpu_status.get('device_name')} · backend={gpu_status.get('backend')}")
    else:
        print(f"GPU:        no activa · {gpu_status.get('reason', 'CPU')}")
    print("=" * 100)


def run_pipeline_sections(cfg, sections: tuple[str, ...]) -> int:
    """Ejecuta únicamente las secciones solicitadas, conservando el flujo validado."""
    from src.paths import ensure_result_tree
    from src.biomechanics import run_biomechanics
    from src.tomography import run_tomography
    from src.neuroimage import run_maps, run_resonances, run_volume_correlations, run_resonance_diagnostic
    from src.structure_function import run_structure_function_coupling
    from src.external_morphometry import run_external_morphometry_integration, prepare_derivatives_workspace
    from src.internal_morphometry import run_internal_morphometry_integration
    from src.real_derivatives import run_real_derivatives_preparation
    from src.longitudinal import run_before_after_comparisons
    from src.reports import build_global_report, write_readme_results
    from src.gpu import configure_gpu
    from src.consolidation import run_patient_consolidation
    from src.tomography_advanced_runner import run_advanced_tomography
    from src.cortical_roi_runner import run_cortical_roi_corrections
    from src.freesurfer_mni_motor import run_freesurfer_mni_motor
    from src.cst_tronco_runner import run_cst_tronco_module
    from src.zonas_correlacion_runner import run_zonas_correlacion_module

    gpu_status = configure_gpu(cfg)
    cfg.results_root().mkdir(parents=True, exist_ok=True)
    for patient in list(cfg.patients) + [cfg.control_name]:
        ensure_result_tree(cfg.results_root(), patient, cfg.stages, cfg.tests)

    requested = set(sections)
    run_all = "todo" in requested
    print_run_header(cfg, sections, gpu_status)

    if "preparar_derivados" in requested:
        workspace = prepare_derivatives_workspace(cfg)
        print(f"Plantilla de derivados externos creada en: {workspace}")
        if not run_all and requested == {"preparar_derivados"}:
            return 0

    if "preparar_derivados_reales" in requested:
        workspace = run_real_derivatives_preparation(cfg)
        print(f"Workspace para generar derivados reales creado en: {workspace}")
        if not run_all and requested == {"preparar_derivados_reales"}:
            return 0

    if "diagnostico" in requested:
        run_resonance_diagnostic(cfg)
        if not run_all and requested == {"diagnostico"}:
            print(f"Diagnóstico guardado en: {cfg.results_root() / '_diagnostico_resonancias_v2'}")
            return 0

    if run_all or "biomecanica" in requested:
        run_biomechanics(cfg)
    if run_all or "tomografia" in requested:
        run_tomography(cfg)
        run_advanced_tomography(cfg)
    if "tomografia_avanzada" in requested:
        run_advanced_tomography(cfg)
    if run_all or "mapas" in requested:
        run_maps(cfg)
    if run_all or "resonancias" in requested:
        run_resonances(cfg)
    if run_all or "morfometria_interna" in requested:
        run_internal_morphometry_integration(cfg)
    if run_all or "corteza_fix" in requested:
        run_cortical_roi_corrections(cfg)
    if run_all or "derivados" in requested:
        run_external_morphometry_integration(cfg)
    if run_all or "fs_mni_motor" in requested:
        run_freesurfer_mni_motor(cfg)
    if run_all or "cst_tronco" in requested:
        run_cst_tronco_module(cfg)
    if run_all or "zonas_correlacion" in requested:
        run_zonas_correlacion_module(cfg)
    if run_all or "estructura_funcion" in requested:
        run_structure_function_coupling(cfg)
    if run_all or "correlaciones" in requested:
        run_volume_correlations(cfg)
    if run_all or "comparacion" in requested:
        run_before_after_comparisons(cfg)
    if run_all or "consolidado" in requested:
        run_patient_consolidation(cfg)

    report = build_global_report(cfg)
    readme = write_readme_results(cfg)
    print("\n" + "=" * 100)
    print("FINALIZADO")
    print(f"Resultados en: {cfg.results_root()}")
    if report:
        print(f"Reporte global: {report}")
    print(f"Guía de resultados: {readme}")
    print("=" * 100)
    return 0


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  6. SETUP, DIAGNÓSTICO Y CLI
# ══════════════════════════════════════════════════════════════════════════════


def run_setup(args) -> int:
    """Crea/actualiza .venv e instala dependencias Python sin usar `.sh`."""
    root = Path(args.project_root).expanduser().resolve()
    venv = root / ".venv"
    req = Path(__file__).resolve().parent / "requirements.txt"
    print("=" * 90)
    print("SETUP LINUX PYTHON")
    print(f"Proyecto: {root}")
    print(f"Entorno:  {venv}")
    print(f"Reqs:     {req}")
    print("=" * 90)
    if not venv.exists():
        subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
    py = venv / "bin" / "python"
    subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    subprocess.check_call([str(py), "-m", "pip", "install", "-r", str(req)])
    print("\nSetup finalizado. Para ejecutar usa:")
    print(f"{py} {Path(__file__).resolve()} run --patients 3 --stages Antes --sections tractografia_cst_rgb --force")
    return 0


def run_doctor(args) -> int:
    """Diagnóstico rápido de entorno y estructura sin procesar imágenes."""
    project_root = Path(args.project_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else project_root / "datos"
    results_root = Path(args.results_root).expanduser().resolve() if args.results_root else project_root / "resultados"
    print("=" * 90)
    print("DOCTOR VCE")
    print(f"Python:     {sys.executable}")
    print(f"Proyecto:   {project_root} -> {project_root.exists()}")
    print(f"Datos:      {data_root} -> {data_root.exists()}")
    print(f"Resultados: {results_root} -> {results_root.exists()}")
    print(f"FreeSurfer: {os.environ.get('FREESURFER_HOME', 'no definido')}")
    print(f"FS_LICENSE: {os.environ.get('FS_LICENSE', str(project_root / 'license.txt'))}")
    print("Pacientes detectados:", ", ".join(discover_patients(data_root)) or "ninguno")
    print("=" * 90)
    return 0


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos compartidos para `run`."""
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT), help="Raíz del proyecto. Linux recomendado: /home/humath/Escritorio")
    parser.add_argument("--data-root", default=None, help="Carpeta exacta de datos. Por defecto: <project-root>/datos")
    parser.add_argument("--results-root", default=None, help="Carpeta exacta de resultados. Por defecto: <project-root>/resultados")
    parser.add_argument("--patients", nargs="*", default=["3"], help="Pacientes: --patients 3 6 7 9")
    parser.add_argument("--all-patients", action="store_true", help="Detecta y corre todos los pacientes encontrados en datos/")
    parser.add_argument("--stages", nargs="*", default=["Antes"], help="Etapas: --stages Antes Despues")
    parser.add_argument("--sections", "--only", nargs="*", default=list(DEFAULT_SECTIONS), help="Secciones: tractografia_cst_rgb, cst_tronco, zonas_correlacion, tomografia, resonancias, todo, etc.")
    parser.add_argument("--interactive", action="store_true", help="Pregunta pacientes, etapas y secciones en consola.")
    parser.add_argument("--control", default="sano", help="Nombre de carpeta del control sano.")
    parser.add_argument("--force", action="store_true", help="Reprocesa aunque existan salidas/checkpoints.")
    parser.add_argument("--no-resume", action="store_true", help="Desactiva reanudación por checkpoints.")
    parser.add_argument("--skip-stuck", action="store_true", help="Salta tareas previas marcadas como running/failed.")
    parser.add_argument("--gpu", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-min-elements", type=int, default=250_000)
    parser.add_argument("--fs-home", default=os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer/8.2.0"))
    parser.add_argument("--fs-license", default=os.environ.get("FS_LICENSE", "/home/humath/Escritorio/license.txt"))
    parser.add_argument("--fs-subjects-dir", default=os.environ.get("VCE_FREESURFER_SUBJECTS_DIR", str(Path.home() / "freesurfer_subjects")))
    parser.add_argument("--fs-threads", type=int, default=int(os.environ.get("VCE_FS_THREADS", "8")))
    parser.add_argument("--registro-com-corregir", action="store_true", help="Activa corrección COM del registro b0→T1.")
    parser.add_argument("--registro-com-ejes", choices=["z", "xyz"], default="z", help="Ejes de corrección COM.")
    parser.add_argument("--clean-tractografia", action="store_true", help="Borra tractografías previas antes de correr.")
    parser.add_argument("--no-suspend", action="store_true", help="Usa systemd-inhibit para evitar suspensión del equipo.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Suite Integrada VCE v{VERSION}: punto de entrada único para Linux/GitHub.")
    parser.add_argument("--list-sections", action="store_true", help="Muestra secciones disponibles y sale.")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Ejecuta una o varias secciones por paciente/etapa.")
    add_run_arguments(p_run)

    p_setup = sub.add_parser("setup", help="Crea .venv e instala requirements.txt sin scripts .sh.")
    p_setup.add_argument("--project-root", default=str(Path.cwd()))

    p_clean = sub.add_parser("clean", help="Limpieza controlada de resultados.")
    p_clean.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    p_clean.add_argument("--results-root", default=None)
    p_clean.add_argument("--target", choices=["tractografia"], default="tractografia")
    p_clean.add_argument("--yes", action="store_true", help="Confirma borrado real. Sin --yes solo muestra vista previa.")

    p_doctor = sub.add_parser("doctor", help="Verifica estructura y entorno.")
    p_doctor.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    p_doctor.add_argument("--data-root", default=None)
    p_doctor.add_argument("--results-root", default=None)

    sub.add_parser("examples", help="Imprime comandos rápidos para la terminal integrada de Visual Studio Code.")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Compatibilidad: si el usuario usa el estilo viejo `python main.py --only ...`, asumimos `run`.
    # Se respeta `--help` para mostrar la ayuda general del programa.
    if argv and argv[0].startswith("--") and argv[0] not in {"--list-sections", "--help", "-h"}:
        argv = ["run", *argv]
    if not argv:
        argv = ["run", "--interactive"]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_sections:
        print_section_catalog()
        return 0

    if args.command == "examples":
        print_terminal_examples()
        return 0

    if args.command == "setup":
        return run_setup(args)
    if args.command == "doctor":
        return run_doctor(args)
    if args.command == "clean":
        root = Path(args.results_root).expanduser().resolve() if args.results_root else Path(args.project_root).expanduser().resolve() / "resultados"
        return clean_tractography(root, yes=args.yes)

    if args.command in {None, "run"}:
        if getattr(args, "no_suspend", False):
            maybe_reexec_with_systemd_inhibit(argv)
        cfg, sections = build_config(args)
        if getattr(args, "clean_tractografia", False):
            clean_tractography(cfg.results_root(), yes=True)
        return run_pipeline_sections(cfg, sections)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
