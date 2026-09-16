"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 ENTRADA, SALIDA Y TRAZABILIDAD DE RESULTADOS                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/io_utils.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Centraliza escritura segura, logs y serialización. Aunque no implementa un
modelo
físico, sostiene la trazabilidad matemática del pipeline: cada matriz, tabla,
NIfTI o tractograma queda asociado a ruta, tiempo y módulo de origen.

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
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: Path) -> None:
    safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(obj: Any):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    safe_mkdir(path.parent)
    if path.suffix.lower() == ".xlsx":
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def append_log(log_file: Path, msg: str) -> None:
    safe_mkdir(log_file.parent)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(str(msg).rstrip() + "\n")


def log_exception(log_file: Path, context: str, exc: BaseException) -> None:
    append_log(log_file, f"[ERROR] {context}: {exc}")
    append_log(log_file, traceback.format_exc())
