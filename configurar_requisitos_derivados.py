"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CONFIGURACIÓN DE REQUISITOS DERIVADOS                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: configurar_requisitos_derivados.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Configura dependencias necesarias para derivar métricas y visualizaciones. Su
función es garantizar que bibliotecas numéricas, lectura DICOM/NIfTI y módulos
de
visualización estén disponibles antes de ejecutar operaciones de señal, imagen
y
tractografía.

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
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

DEFAULT_PROJECT = Path(r"D:\EAFIT\01-2026\proyecto")
FREESURFER_REG = "https://surfer.nmr.mgh.harvard.edu/registration.html"
DOCKER_WIN = "https://docs.docker.com/desktop/setup/install/windows-install/"
FMRIPREP_DOCS = "https://fmriprep.org/en/stable/usage.html"


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        value = ""
    return value or default


def yes(prompt: str, default: bool = True) -> bool:
    d = "s" if default else "n"
    ans = ask(prompt + " (s/n)", d).lower()
    return ans in {"s", "si", "sí", "y", "yes", "1", "true"}


def run(cmd: list[str], check: bool = False) -> int:
    print("\n$ " + " ".join(cmd))
    try:
        return subprocess.run(cmd, check=check).returncode
    except FileNotFoundError:
        return 127


def docker_ok() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(["docker", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=25)
        return r.returncode == 0
    except Exception:
        return False


def is_valid_license(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        txt = path.read_text(errors="ignore").strip()
    except Exception:
        return False
    if len(txt) < 20:
        return False
    bad = ["pega", "placeholder", "aqui", "aquí", "freesurfer license aqui"]
    return not any(b in txt.lower() for b in bad)


def main() -> int:
    print("=" * 90)
    print("CONFIGURAR REQUISITOS PARA DERIVADOS REALES: FreeSurfer / fMRIPrep")
    print("=" * 90)
    project = Path(ask("Ruta del proyecto", str(DEFAULT_PROJECT)))
    project.mkdir(parents=True, exist_ok=True)
    license_path = project / "license.txt"
    out = project / "resultados" / "derivados_externos"
    out.mkdir(parents=True, exist_ok=True)

    guide = out / "PASOS_REQUISITOS_DERIVADOS_REALES.txt"
    guide.write_text(f"""
REQUISITOS PARA GENERAR DERIVADOS REALES

1) Docker Desktop para Windows
   {DOCKER_WIN}

2) Licencia FreeSurfer gratuita
   {FREESURFER_REG}
   Guarda el archivo recibido exactamente como:
   {license_path}

3) fMRIPrep se ejecuta como imagen Docker nipreps/fmriprep.
   También necesita la licencia FreeSurfer.
   {FMRIPREP_DOCS}

4) Después de instalar Docker y poner license.txt, vuelve a abrir:
   configurar_requisitos_derivados.bat

5) Luego corre:
   run_todo_interactivo.bat
""".strip(), encoding="utf-8")

    print(f"\nGuía creada en: {guide}")

    if not docker_ok():
        print("\nDocker Desktop NO está listo.")
        if yes("¿Abrir la página oficial de Docker Desktop?", True):
            webbrowser.open(DOCKER_WIN)
        if os.name == "nt" and yes("¿Intentar instalar Docker Desktop con winget? Requiere permisos y reinicio/WSL2", False):
            code = run(["winget", "install", "-e", "--id", "Docker.DockerDesktop"])
            print(f"winget terminó con código: {code}")
            print("Después de instalar, abre Docker Desktop, espera que diga Running, y vuelve a correr este script.")
        else:
            print("Instala Docker Desktop manualmente, ábrelo una vez, y vuelve a correr este script.")
    else:
        print("\nDocker Desktop: OK")

    if not is_valid_license(license_path):
        print(f"\nNo encontré una licencia válida en: {license_path}")
        print("Esa licencia NO se inventa ni la genera Python; FreeSurfer la entrega gratis al registrarte.")
        if yes("¿Abrir formulario oficial de FreeSurfer para pedir license.txt?", True):
            webbrowser.open(FREESURFER_REG)
        marker = project / "PONER_LICENSE_TXT_AQUI.txt"
        marker.write_text(f"Cuando FreeSurfer te entregue el archivo, guárdalo como:\n{license_path}\n", encoding="utf-8")
        print(f"Dejé recordatorio en: {marker}")
    else:
        print(f"\nlicense.txt: OK ({license_path})")

    if docker_ok() and is_valid_license(license_path):
        print("\nYa están los dos requisitos fuertes: Docker + license.txt")
        if yes("¿Descargar/probar imagen Docker de FreeSurfer ahora?", True):
            run(["docker", "pull", "freesurfer/freesurfer:7.4.1"])
        if yes("¿Descargar/probar imagen Docker de fMRIPrep ahora?", True):
            run(["docker", "pull", "nipreps/fmriprep:latest"])
        print("\nListo. Ahora puedes correr run_todo_interactivo.bat y aceptar derivados reales.")
    else:
        print("\nAún falta Docker o license.txt. La suite puede seguir con morfometría interna, pero FreeSurfer/fMRIPrep reales necesitan esos requisitos.")

    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
