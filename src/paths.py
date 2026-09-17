"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                RESOLUCIÓN DE RUTAS CLÍNICAS Y EXPERIMENTALES                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/paths.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.


-----------------------------------------
Resuelve rutas manteniendo una convención paciente/etapa/sección. El modelo
computacional separa datos crudos D, resultados R y derivados intermedios para
preservar reproducibilidad. Las búsquedas recursivas priorizan nombres
clínicos
conocidos como rT1, rAnatomico, DWI y ResultadosFuncional.

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

import re
from pathlib import Path
from typing import Iterable, Optional


_STAGE_ALIASES = {
    "antes": ("Antes", "antes", "PRE", "Pre", "PRE-QX", "pre-qx"),
    "despues": ("Despues", "Después", "despues", "después", "POST", "Post", "POST-QX", "post-qx"),
}


def norm_key(text: str) -> str:
    return (
        str(text)
        .lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
        .strip()
    )


def stage_key(stage: str) -> str:
    k = norm_key(stage)
    if k in {"despues", "post", "post-qx"}:
        return "despues"
    return "antes"


def resolve_stage_dir(subject_dir: Path, stage: str, fallback_to_subject: bool = False) -> Optional[Path]:
    if not subject_dir.exists():
        return None
    aliases = _STAGE_ALIASES[stage_key(stage)]
    for alias in aliases:
        candidate = subject_dir / alias
        if candidate.exists():
            return candidate
    # búsqueda tolerante solo un nivel
    target = stage_key(stage)
    for child in subject_dir.iterdir():
        if child.is_dir() and stage_key(child.name) == target:
            return child
    return subject_dir if fallback_to_subject else None


def first_existing_dir(base: Path, names: Iterable[str]) -> Optional[Path]:
    if not base or not base.exists():
        return None
    for name in names:
        p = base / name
        if p.exists() and p.is_dir():
            return p
    wanted = {norm_key(n) for n in names}
    for child in base.iterdir():
        if child.is_dir() and norm_key(child.name) in wanted:
            return child
    return None


def find_file_by_side_and_test(folder: Path, side: str, test: int, suffix: str = ".csv") -> Optional[Path]:
    """Busca derecha_1.csv / izquierda_1.csv con tolerancia a mayúsculas."""
    if not folder or not folder.exists():
        return None
    side_key = "derecha" if norm_key(side).startswith(("der", "right")) else "izquierda"
    patterns = [
        f"{side_key}_{test}{suffix}",
        f"{side_key} {test}{suffix}",
        f"{side_key}-{test}{suffix}",
        f"{side_key}{test}{suffix}",
    ]
    lower_to_path = {p.name.lower(): p for p in folder.glob(f"*{suffix}")}
    for pat in patterns:
        if pat.lower() in lower_to_path:
            return lower_to_path[pat.lower()]
    rx = re.compile(rf"{side_key}.*\b{test}\b.*{re.escape(suffix)}$", re.I)
    for p in folder.glob(f"*{suffix}"):
        if rx.search(p.name):
            return p
    return None


def ensure_result_tree(results_root: Path, patient: str, stages: Iterable[str], tests: Iterable[int]) -> None:
    for stage in stages:
        base = results_root / patient / stage
        for name in ["emg", "dinamometria", "tomografia", "mapas", "resonancias", "correlaciones", "reportes"]:
            (base / name).mkdir(parents=True, exist_ok=True)
        for test in tests:
            (base / "emg" / f"prueba_{test}").mkdir(parents=True, exist_ok=True)
            (base / "dinamometria" / f"prueba_{test}").mkdir(parents=True, exist_ok=True)
    (results_root / patient / "comparacion" / "antes_vs_despues").mkdir(parents=True, exist_ok=True)
