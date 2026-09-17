"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 MESENCÉFALO, PUENTE Y BULBO DESDE FREESURFER                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/brainstem_freesurfer.py
Versión: v3.21.21

Descripción
-----------
Ejecuta FreeSurfer BrainstemSubstructures y transforma etiquetas al T1 nativo.

Fundamento
-----------------------------------------
Segmenta subestructuras del tronco encefálico usando etiquetas anatómicas de
FreeSurfer. El modelo espacial conserva la correspondencia voxel-mundo con
affines NIfTI/MGZ: x_RAS = A [i,j,k,1]^T. Las máscaras se obtienen por
selección
de etiquetas discretas: mesencéfalo=173, puente=174, bulbo=175 y SCP=178. Para
llevarlas al T1 del paciente se usa remuestreo con vecino más cercano,
preservando
la naturaleza categórica de las etiquetas.

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
#  1. IMPORTACIONES Y UTILIDADES GENERALES
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class BrainstemSegmentationResult:
    """Resultado trazable de la segmentación anatómica del tronco."""

    status: str
    subject_id: str
    subjects_dir: Path
    aparc_aseg: Path
    aseg: Path
    recon_done: Path
    raw_output_dir: Path
    validated_output_dir: Path
    segmentation_mgz: Path | None = None
    report_json: Path | None = None


def step(message: str) -> None:
    """Imprime una etapa con marca temporal para seguimiento en consola."""
    print("", flush=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _run(cmd: list[str], *, env: dict[str, str], cwd: Path | None = None) -> int:
    """Ejecuta un subproceso mostrando stdout/stderr en tiempo real."""
    print("", flush=True)
    print("─" * 90, flush=True)
    print("COMANDO:", " ".join(map(str, cmd)), flush=True)
    if cwd:
        print("DIRECTORIO:", cwd, flush=True)
    print("─" * 90, flush=True)
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
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
    proc.wait()
    print(f"\nSubproceso terminado con código {proc.returncode}", flush=True)
    return int(proc.returncode or 0)


def _which(command: str, env: dict[str, str]) -> str | None:
    """Busca un binario usando el PATH del entorno recibido."""
    path = shutil.which(command, path=env.get("PATH"))
    return path


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  2. ENTORNO FREESURFER
# ══════════════════════════════════════════════════════════════════════════════


def find_freesurfer_home() -> Path | None:
    """Encuentra FREESURFER_HOME priorizando la instalación del usuario."""
    candidates = []
    if os.environ.get("FREESURFER_HOME"):
        candidates.append(Path(os.environ["FREESURFER_HOME"]))
    candidates.extend([
        Path("/usr/local/freesurfer/8.2.0"),
        Path("/usr/local/freesurfer"),
    ])
    try:
        candidates.extend(sorted(Path("/usr/local").glob("freesurfer*/**/SetUpFreeSurfer.sh")))
    except Exception:
        pass
    for c in candidates:
        home = c.parent if c.name == "SetUpFreeSurfer.sh" else c
        if (home / "SetUpFreeSurfer.sh").exists() and (home / "bin").exists():
            return home
    return None


def build_freesurfer_env(
    *,
    project_root: Path,
    license_path: Path,
    subjects_dir: Path,
    threads: int,
) -> dict[str, str]:
    """Construye un entorno robusto para ejecutar comandos FreeSurfer.

    En Linux, `source SetUpFreeSurfer.sh` modifica muchas variables de entorno.
    Para evitar depender de `.sh`, aquí se prepara lo esencial desde Python y se
    antepone `FREESURFER_HOME/bin` al PATH. Esto mantiene el flujo ejecutable con
    `python main.py`.
    """
    fs_home = find_freesurfer_home()
    if fs_home is None:
        raise RuntimeError("No se encontró FreeSurfer. Define FREESURFER_HOME o instala FreeSurfer.")
    env = os.environ.copy()
    env["FREESURFER_HOME"] = str(fs_home)
    env["FS_LICENSE"] = str(license_path)
    env["SUBJECTS_DIR"] = str(subjects_dir)
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(max(1, int(threads)))
    env["OMP_NUM_THREADS"] = str(max(1, int(threads)))
    env["PATH"] = str(fs_home / "bin") + os.pathsep + env.get("PATH", "")
    # Variables habituales que SetUpFreeSurfer define y que algunos binarios esperan.
    env.setdefault("FSFAST_HOME", str(fs_home / "fsfast"))
    env.setdefault("FREESURFER", str(fs_home))
    env.setdefault("LOCAL_DIR", str(fs_home / "local"))
    env.setdefault("FUNCTIONALS_DIR", str(fs_home / "sessions"))
    env.setdefault("MNI_DIR", str(fs_home / "mni"))
    return env


def assert_freesurfer_commands(env: dict[str, str]) -> None:
    """Valida comandos mínimos antes de iniciar procesos largos."""
    required = ["recon-all", "mri_vol2vol", "mri_binarize", "mri_info"]
    missing = [cmd for cmd in required if _which(cmd, env) is None]
    if missing:
        raise RuntimeError(f"Faltan comandos de FreeSurfer en PATH: {', '.join(missing)}")


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  3. RECON-ALL Y VALIDACIÓN APARC+ASEG
# ══════════════════════════════════════════════════════════════════════════════


def ensure_recon_all(
    *,
    t1_path: Path,
    subject_id: str,
    subjects_dir: Path,
    stage_root: Path,
    env: dict[str, str],
    threads: int,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Crea o reanuda recon-all hasta obtener aparc+aseg.mgz.

    Se copia el rT1/rAnatomico a una ruta sin espacios para reducir fallos de
    herramientas internas y se exige `aparc+aseg.mgz` antes de continuar.
    """
    local_t1 = stage_root / subject_id / "rT1.nii"
    local_t1.parent.mkdir(parents=True, exist_ok=True)
    if not local_t1.exists() or not _same_file_bytes(t1_path, local_t1):
        print(f"Copiando rT1/rAnatomico a ruta Linux sin espacios: {local_t1}", flush=True)
        shutil.copy2(t1_path, local_t1)

    mri_dir = subjects_dir / subject_id / "mri"
    recon_done = subjects_dir / subject_id / "scripts" / "recon-all.done"
    orig_001 = mri_dir / "orig" / "001.mgz"
    aparc_aseg = mri_dir / "aparc+aseg.mgz"
    aseg = mri_dir / "aseg.mgz"
    rawavg = mri_dir / "rawavg.mgz"
    norm = mri_dir / "norm.mgz"

    step("[CHECK] Verificando recon-all antes del mesencéfalo")
    for label, path in [
        ("recon-all.done", recon_done),
        ("aparc+aseg.mgz", aparc_aseg),
        ("aseg.mgz", aseg),
        ("rawavg.mgz", rawavg),
        ("norm.mgz", norm),
    ]:
        print(f"  {label}: {path} -> {path.exists()}", flush=True)

    if not (recon_done.exists() and aparc_aseg.exists() and aseg.exists() and rawavg.exists()):
        print("recon-all no está completo. Se ejecutará/reanudará para crear aparc+aseg.mgz.", flush=True)
        if orig_001.exists():
            cmd = ["recon-all", "-s", subject_id, "-sd", str(subjects_dir), "-all", "-openmp", str(threads)]
        else:
            cmd = ["recon-all", "-i", str(local_t1), "-s", subject_id, "-sd", str(subjects_dir), "-all", "-openmp", str(threads)]
        code = _run(cmd, env=env)
        if code != 0:
            raise RuntimeError(f"recon-all terminó con código {code}")

    step("[CHECK] Validación obligatoria post recon-all")
    missing = []
    for path in [recon_done, aparc_aseg, aseg, rawavg, norm]:
        if path.exists() and path.stat().st_size > 0:
            print(f"OK: {path}", flush=True)
        else:
            print(f"ERROR: falta archivo obligatorio: {path}", flush=True)
            missing.append(path)
    if missing:
        raise RuntimeError("No se procesa mesencéfalo porque falta recon-all/aparc+aseg/aseg/rawavg/norm.")

    code = _run(["mri_info", str(aparc_aseg)], env=env)
    if code != 0:
        raise RuntimeError("mri_info no pudo leer aparc+aseg.mgz")
    return local_t1, mri_dir, recon_done, aparc_aseg, aseg, rawavg


def _same_file_bytes(a: Path, b: Path) -> bool:
    """Comparación conservadora para decidir si se debe actualizar el T1 local."""
    try:
        return a.stat().st_size == b.stat().st_size and a.read_bytes() == b.read_bytes()
    except Exception:
        return False


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  4. BRAINSTEMSUBSTRUCTURES Y EXPORTACIÓN A rT1
# ══════════════════════════════════════════════════════════════════════════════


def _find_brainstem_segmentation(mri_dir: Path) -> Path | None:
    """Busca la segmentación brainstemSsLabels más reciente del sujeto."""
    for pattern in ["brainstemSsLabels*.FSvoxelSpace.mgz", "brainstemSsLabels*.mgz"]:
        matches = sorted(mri_dir.glob(pattern))
        if matches:
            return matches[-1]
    return None


def ensure_brainstem_substructures(
    *,
    subject_id: str,
    subjects_dir: Path,
    mri_dir: Path,
    env: dict[str, str],
    threads: int,
) -> Path:
    """Ejecuta segment_subregions o segmentBS.sh hasta crear brainstemSsLabels."""
    step("Segmentando mesencéfalo, puente, bulbo y SCP desde FreeSurfer BrainstemSubstructures")
    seg = _find_brainstem_segmentation(mri_dir)
    if seg:
        print(f"Segmentación existente encontrada: {seg}", flush=True)
        return seg

    if _which("segment_subregions", env):
        print("Intentando segment_subregions brainstem...", flush=True)
        _run([
            "segment_subregions", "brainstem", "--cross", subject_id,
            "--sd", str(subjects_dir), "--threads", str(threads), "--out-dir", str(mri_dir),
        ], env=env)
        seg = _find_brainstem_segmentation(mri_dir)
        if seg:
            return seg

    if _which("segmentBS.sh", env):
        print("Intentando fallback segmentBS.sh...", flush=True)
        _run(["segmentBS.sh", subject_id, str(subjects_dir)], env=env)
        seg = _find_brainstem_segmentation(mri_dir)
        if seg:
            return seg
    else:
        print("ADVERTENCIA: segmentBS.sh no está en PATH.", flush=True)

    raise RuntimeError(
        "No apareció brainstemSsLabels*.mgz. Revisa Matlab Runtime MCRv97/R2019b de FreeSurfer."
    )


def export_brainstem_to_t1(
    *,
    seg_mgz: Path,
    rawavg: Path,
    local_t1: Path,
    original_t1: Path,
    raw_output_dir: Path,
    validated_output_dir: Path,
    validator_path: Path,
    env: dict[str, str],
) -> Path:
    """Convierte etiquetas FreeSurfer a NIfTI en espacio del rT1/rAnatomico."""
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    validated_output_dir.mkdir(parents=True, exist_ok=True)

    seg_rawavg = raw_output_dir / "brainstem_substructures_rawavg.nii.gz"
    seg_t1 = raw_output_dir / "brainstem_substructures_T1.nii.gz"

    _run(["mri_vol2vol", "--mov", str(seg_mgz), "--targ", str(rawavg), "--regheader", "--o", str(seg_rawavg), "--interp", "nearest", "--no-save-reg"], env=env)
    _run(["mri_vol2vol", "--mov", str(seg_rawavg), "--targ", str(local_t1), "--regheader", "--o", str(seg_t1), "--interp", "nearest", "--no-save-reg"], env=env)

    label_targets = {
        "midbrain_T1.nii.gz": ["173"],
        "pons_T1.nii.gz": ["174"],
        "medulla_T1.nii.gz": ["175"],
        "scp_T1.nii.gz": ["178"],
        "brainstem_complete_T1.nii.gz": ["173", "174", "175", "178"],
    }
    for filename, labels in label_targets.items():
        _run(["mri_binarize", "--i", str(seg_t1), "--match", *labels, "--o", str(raw_output_dir / filename)], env=env)

    provenance = raw_output_dir / "procedencia_freesurfer.txt"
    provenance.write_text(
        "\n".join([
            "FreeSurfer BrainstemSubstructures",
            f"FREESURFER_HOME={env.get('FREESURFER_HOME')}",
            f"SUBJECTS_DIR={env.get('SUBJECTS_DIR')}",
            f"FS_LICENSE={env.get('FS_LICENSE')}",
            f"T1 original={original_t1}",
            f"T1 local={local_t1}",
            f"Segmentación original={seg_mgz}",
            "Etiquetas: 173=mesencéfalo, 174=puente, 175=bulbo, 178=SCP",
        ]) + "\n",
        encoding="utf-8",
    )

    step("Validando máscaras y creando labelmap ITK-SNAP")
    code = _run([
        sys.executable, str(validator_path),
        "--t1-path", str(original_t1),
        "--segmentation-path", str(seg_t1),
        "--output-dir", str(validated_output_dir),
    ], env=env)
    if code != 0:
        raise RuntimeError(f"Validador de brainstem terminó con código {code}")
    return validated_output_dir / "brainstem_substructures_report.json"


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  5. FUNCIÓN PÚBLICA DEL MÓDULO
# ══════════════════════════════════════════════════════════════════════════════


def run_brainstem_segmentation(
    *,
    project_root: Path,
    t1_path: Path,
    license_path: Path,
    subject_id: str,
    out_base: Path,
    threads: int = 8,
    subjects_dir: Path | None = None,
    validator_path: Path | None = None,
) -> BrainstemSegmentationResult:
    """Ejecuta el flujo completo recon-all → brainstem → máscaras T1."""
    project_root = Path(project_root)
    t1_path = Path(t1_path)
    license_path = Path(license_path)
    out_base = Path(out_base)
    subjects_dir = Path(subjects_dir or os.environ.get("VCE_FREESURFER_SUBJECTS_DIR", str(Path.home() / "freesurfer_subjects")))
    stage_root = Path.home() / "freesurfer_stage"
    raw_output_dir = out_base / "brainstem_substructures"
    validated_output_dir = out_base / "brainstem_substructures_validated"

    if validator_path is None:
        validator_path = Path(__file__).resolve().parents[1] / "modulos" / "modulo_CST_tronco_integracion_completa" / "02_segmentacion_tronco" / "FINAL_LINUX" / "validar_brainstem_substructures_T1_linux.py"
    validator_path = Path(validator_path)

    if not t1_path.exists():
        raise FileNotFoundError(f"No existe rT1/rAnatomico: {t1_path}")
    if not license_path.exists():
        raise FileNotFoundError(f"Falta licencia FreeSurfer: {license_path}")
    if not validator_path.exists():
        raise FileNotFoundError(f"Falta validador de brainstem: {validator_path}")

    raw_output_dir.mkdir(parents=True, exist_ok=True)
    validated_output_dir.mkdir(parents=True, exist_ok=True)
    subjects_dir.mkdir(parents=True, exist_ok=True)

    env = build_freesurfer_env(project_root=project_root, license_path=license_path, subjects_dir=subjects_dir, threads=threads)
    assert_freesurfer_commands(env)

    print("=" * 90, flush=True)
    print("FREE SURFER + BRAINSTEM SUBSTRUCTURES", flush=True)
    print(f"FREESURFER_HOME={env.get('FREESURFER_HOME')}", flush=True)
    print(f"SUBJECTS_DIR={subjects_dir}", flush=True)
    print(f"FS_LICENSE={license_path}", flush=True)
    print(f"T1 original={t1_path}", flush=True)
    print(f"Sujeto={subject_id}", flush=True)
    print(f"Salida raw={raw_output_dir}", flush=True)
    print(f"Salida validada={validated_output_dir}", flush=True)
    print("=" * 90, flush=True)

    local_t1, mri_dir, recon_done, aparc_aseg, aseg, rawavg = ensure_recon_all(
        t1_path=t1_path,
        subject_id=subject_id,
        subjects_dir=subjects_dir,
        stage_root=stage_root,
        env=env,
        threads=threads,
    )
    seg = ensure_brainstem_substructures(subject_id=subject_id, subjects_dir=subjects_dir, mri_dir=mri_dir, env=env, threads=threads)
    report_json = export_brainstem_to_t1(
        seg_mgz=seg,
        rawavg=rawavg,
        local_t1=local_t1,
        original_t1=t1_path,
        raw_output_dir=raw_output_dir,
        validated_output_dir=validated_output_dir,
        validator_path=validator_path,
        env=env,
    )

    return BrainstemSegmentationResult(
        status="ok",
        subject_id=subject_id,
        subjects_dir=subjects_dir,
        aparc_aseg=aparc_aseg,
        aseg=aseg,
        recon_done=recon_done,
        raw_output_dir=raw_output_dir,
        validated_output_dir=validated_output_dir,
        segmentation_mgz=seg,
        report_json=report_json,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segmentación FreeSurfer del tronco cerebral en Linux, sin scripts .sh.")
    parser.add_argument("--project-root", default="/home/humath/Escritorio")
    parser.add_argument("--t1-path", required=True)
    parser.add_argument("--license-path", default="/home/humath/Escritorio/license.txt")
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--out-base", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--subjects-dir", default=None)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    run_brainstem_segmentation(
        project_root=Path(args.project_root),
        t1_path=Path(args.t1_path),
        license_path=Path(args.license_path),
        subject_id=args.subject_id,
        out_base=Path(args.out_base),
        threads=args.threads,
        subjects_dir=Path(args.subjects_dir) if args.subjects_dir else None,
    )
    print("Segmentación del tronco terminada correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
