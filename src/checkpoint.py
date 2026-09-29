"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 CONTROL DE REANUDACIÓN Y ESTADO DEL PIPELINE                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/checkpoint.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Controla reanudación mediante estados discretos. El pipeline se representa
como
una secuencia de tareas con estado {pendiente, ejecutado, omitido, error}.
Esta
representación permite repetir solo secciones faltantes sin recalcular
productos
válidos ni sobrescribir datos crudos.

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

import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

from .io_utils import safe_mkdir, save_json


def _json_default(obj: Any):
    # -----------------------------------------------------------------------------
    # Entrada:
    #   - obj: parámetro de entrada usado por la función para controlar el cálculo o suministrar datos.
    #       Tipo esperado/anotado: Any.
    # Proceso:
    #   - Serializa o lee estructuras JSON para preservar metadatos, métricas y trazabilidad del
    #     procesamiento.
    # Salida:
    #   - valor calculado, estructura de resultados, tabla, ruta o None según la operación específica de
    #     la función.
    # -----------------------------------------------------------------------------
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def _stable_hash(data: Any) -> str:
    # -----------------------------------------------------------------------------
    # Entrada:
    #   - data: datos de entrada en memoria, tabla, arreglo o estructura compuesta. Tipo
    #       esperado/anotado: Any.
    # Proceso:
    #   - Ejecuta el bloque "stable hash" dentro del módulo de control de ejecución reproducible
    #     mediante huellas digitales, estados y métricas serializadas.
    # Salida:
    #   - valor calculado, estructura de resultados, tabla, ruta o None según la operación específica de
    #     la función. Tipo de retorno anotado: str.
    # -----------------------------------------------------------------------------
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def file_fingerprint(path: Path) -> dict[str, Any]:
    # -----------------------------------------------------------------------------
    # Entrada:
    #   - path: ruta de archivo o carpeta sobre la que se realiza lectura, escritura o validación. Tipo
    #       esperado/anotado: Path.
    # Proceso:
    #   - Ejecuta el bloque "file fingerprint" dentro del módulo de control de ejecución reproducible
    #     mediante huellas digitales, estados y métricas serializadas.
    # Salida:
    #   - valor calculado, estructura de resultados, tabla, ruta o None según la operación específica de
    #     la función. Tipo de retorno anotado: dict[str, Any].
    # -----------------------------------------------------------------------------
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        st = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }
    except Exception as exc:
        return {"path": str(path), "exists": True, "error": str(exc)}


def folder_fingerprint(path: Path, max_files: int = 20000) -> dict[str, Any]:
    """Firma liviana de carpeta.

    No lee píxeles ni contenido pesado; usa nombres, tamaños y mtimes.
    Sirve para saber si una serie cambió y decidir si se salta o se repite.
    """
    # -----------------------------------------------------------------------------
    # Entrada:
    #   - path: ruta de archivo o carpeta sobre la que se realiza lectura, escritura o validación. Tipo
    #       esperado/anotado: Path.
    #   - max_files: parámetro de entrada usado por la función para controlar el cálculo o suministrar
    #       datos. Tipo esperado/anotado: int.
    # Proceso:
    #   - Aplica el procedimiento descrito por la función: Firma liviana de carpeta. No lee píxeles ni
    #     contenido pesado; usa nombres, tamaños y mtimes. Sirve para saber si una serie cambió y
    #     decidir si se salta o se repite.
    # Salida:
    #   - valor calculado, estructura de resultados, tabla, ruta o None según la operación específica de
    #     la función. Tipo de retorno anotado: dict[str, Any].
    # -----------------------------------------------------------------------------
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    files = []
    count = 0
    total_size = 0
    latest_mtime = 0
    for f in sorted(path.rglob("*")):
        if not f.is_file():
            continue
        count += 1
        if count <= max_files:
            fp = file_fingerprint(f)
            files.append(fp)
        try:
            st = f.stat()
            total_size += int(st.st_size)
            latest_mtime = max(latest_mtime, int(st.st_mtime_ns))
        except Exception:
            pass
    return {
        "path": str(path),
        "exists": True,
        "count": count,
        "total_size": total_size,
        "latest_mtime_ns": latest_mtime,
        "files_sample": files,
        "truncated": count > max_files,
    }


class CheckpointManager:
    """Estado persistente para reanudar el pipeline.

    Cada unidad de trabajo queda registrada en:
      resultados/_estado_pipeline/checkpoints.json

    Si el proceso se cierra, se va la luz, se pega una serie DICOM o se detiene Python,
    al volver a ejecutar se saltan las unidades terminadas con la misma firma de entrada.
    """

    def __init__(self, cfg):
        # -----------------------------------------------------------------------------
        # Entrada:
        #   - self: instancia actual del objeto; permite acceder a estado interno, rutas, métricas o
        #       configuración acumulada.
        #   - cfg: configuración del pipeline con rutas, pacientes, etapas, opciones de ejecución y
        #       parámetros del módulo.
        # Proceso:
        #   - Ejecuta el bloque "init" dentro del módulo de control de ejecución reproducible mediante
        #     huellas digitales, estados y métricas serializadas.
        # Salida:
        #   - valor calculado, estructura de resultados, tabla, ruta o None según la operación
        #     específica de la función.
        # -----------------------------------------------------------------------------
        self.cfg = cfg
        self.root = safe_mkdir(cfg.results_root() / "_estado_pipeline")
        self.path = self.root / "checkpoints.json"
        self.log_path = self.root / "pipeline_eventos.log"
        self.state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        # -----------------------------------------------------------------------------
        # Entrada:
        #   - self: instancia actual del objeto; permite acceder a estado interno, rutas, métricas o
        #       configuración acumulada.
        # Proceso:
        #   - Ejecuta el bloque "load" dentro del módulo de control de ejecución reproducible mediante
        #     huellas digitales, estados y métricas serializadas.
        # Salida:
        #   - datos cargados en memoria, por ejemplo imagen, tabla, señal, JSON o estructura auxiliar.
        #     Tipo de retorno anotado: dict[str, Any].
        # -----------------------------------------------------------------------------
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                backup = self.path.with_suffix(".corrupt.json")
                try:
                    self.path.rename(backup)
                except Exception:
                    pass
        return {"version": 3, "tasks": {}}

    def save(self) -> None:
        # -----------------------------------------------------------------------------
        # Entrada:
        #   - self: instancia actual del objeto; permite acceder a estado interno, rutas, métricas o
        #       configuración acumulada.
        # Proceso:
        #   - Ejecuta el bloque "save" dentro del módulo de control de ejecución reproducible mediante
        #     huellas digitales, estados y métricas serializadas.
        # Salida:
        #   - archivo escrito en disco, ruta de salida, registro de log o confirmación implícita
        #     mediante ausencia de error. Tipo de retorno anotado: None.
        # -----------------------------------------------------------------------------
        safe_mkdir(self.path.parent)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        os.replace(tmp, self.path)

    def log(self, message: str) -> None:
        # -----------------------------------------------------------------------------
        # Entrada:
        #   - self: instancia actual del objeto; permite acceder a estado interno, rutas, métricas o
        #       configuración acumulada.
        #   - message: parámetro de entrada usado por la función para controlar el cálculo o suministrar
        #       datos. Tipo esperado/anotado: str.
        # Proceso:
        #   - Ejecuta el bloque "log" dentro del módulo de control de ejecución reproducible mediante
        #     huellas digitales, estados y métricas serializadas.
        # Salida:
        #   - archivo escrito en disco, ruta de salida, registro de log o confirmación implícita
        #     mediante ausencia de error. Tipo de retorno anotado: None.
        # -----------------------------------------------------------------------------
        safe_mkdir(self.log_path.parent)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {message}\n")

    def signature(self, inputs: Iterable[Path] | None = None, params: dict[str, Any] | None = None) -> str:
        # -----------------------------------------------------------------------------
        # Entrada:
        #   - self: instancia actual del objeto; permite acceder a estado interno, rutas, métricas o
        #       configuración acumulada.
        #   - inputs: parámetro de entrada usado por la función para controlar el cálculo o suministrar
        #       datos. Tipo esperado/anotado: Iterable[Path] | None.
        #   - params: parámetro de entrada usado por la función para controlar el cálculo o suministrar
        #       datos. Tipo esperado/anotado: dict[str, Any] | None.
        # Proceso:
        #   - Ejecuta el bloque "signature" dentro del módulo de control de ejecución reproducible
        #     mediante huellas digitales, estados y métricas serializadas.
        # Salida:
        #   - valor calculado, estructura de resultados, tabla, ruta o None según la operación
        #     específica de la función. Tipo de retorno anotado: str.
        # -----------------------------------------------------------------------------
        fps = []
        for p in inputs or []:
            p = Path(p)
            if p.is_dir():
                fps.append(folder_fingerprint(p))
            else:
                fps.append(file_fingerprint(p))
        return _stable_hash({"inputs": fps, "params": params or {}})

    def is_done(self, task_id: str, signature: str, outputs: Iterable[Path] | None = None) -> bool:
        # -----------------------------------------------------------------------------
        # Entrada:
        #   - self: instancia actual del objeto; permite acceder a estado interno, rutas, métricas o
        #       configuración acumulada.
        #   - task_id: parámetro de entrada usado por la función para controlar el cálculo o suministrar
        #       datos. Tipo esperado/anotado: str.
        #   - signature: parámetro de entrada usado por la función para controlar el cálculo o
        #       suministrar datos. Tipo esperado/anotado: str.
        #   - outputs: parámetro de entrada usado por la función para controlar el cálculo o suministrar
        #       datos. Tipo esperado/anotado: Iterable[Path] | None.
        # Proceso:
        #   - Ejecuta el bloque "is done" dentro del módulo de control de ejecución reproducible
        #     mediante huellas digitales, estados y métricas serializadas.
        # Salida:
        #   - valor booleano que indica si se cumple la condición evaluada. Tipo de retorno anotado:
        #     bool.
        # -----------------------------------------------------------------------------
        if getattr(self.cfg, "force", False) or not getattr(self.cfg, "resume", True):
            return False
        task = self.state.get("tasks", {}).get(task_id)
        if not task or task.get("status") != "done" or task.get("signature") != signature:
            return False
        for out in outputs or []:
            if not Path(out).exists():
                return False
        return True

    def start(self, task_id: str, signature: str, meta: dict[str, Any] | None = None) -> None:
        # -----------------------------------------------------------------------------
        # Entrada:
        #   - self: instancia actual del objeto; permite acceder a estado interno, rutas, métricas o
        #       configuración acumulada.
        #   - task_id: parámetro de entrada usado por la función para controlar el cálculo o suministrar
        #       datos. Tipo esperado/anotado: str.
        #   - signature: parámetro de entrada usado por la función para controlar el cálculo o
        #       suministrar datos. Tipo esperado/anotado: str.
        #   - meta: parámetro de entrada usado por la función para controlar el cálculo o suministrar
        #       datos. Tipo esperado/anotado: dict[str, Any] | None.
        # Proceso:
        #   - Ejecuta el bloque "start" dentro del módulo de control de ejecución reproducible mediante
        #     huellas digitales, estados y métricas serializadas.
        # Salida:
        #   - valor calculado, estructura de resultados, tabla, ruta o None según la operación
        #     específica de la función. Tipo de retorno anotado: None.
        # -----------------------------------------------------------------------------
        self.state.setdefault("tasks", {})[task_id] = {
            "status": "running",
            "signature": signature,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "meta": meta or {},
        }
        self.save()
        self.log(f"START {task_id}")

    def done(self, task_id: str, outputs: Iterable[Path] | None = None, result_summary: dict[str, Any] | None = None) -> None:
        # -----------------------------------------------------------------------------
        # Entrada:
        #   - self: instancia actual del objeto; permite acceder a estado interno, rutas, métricas o
        #       configuración acumulada.
        #   - task_id: parámetro de entrada usado por la función para controlar el cálculo o suministrar
        #       datos. Tipo esperado/anotado: str.
        #   - outputs: parámetro de entrada usado por la función para controlar el cálculo o suministrar
        #       datos. Tipo esperado/anotado: Iterable[Path] | None.
        #   - result_summary: parámetro de entrada usado por la función para controlar el cálculo o
        #       suministrar datos. Tipo esperado/anotado: dict[str, Any] | None.
        # Proceso:
        #   - Ejecuta el bloque "done" dentro del módulo de control de ejecución reproducible mediante
        #     huellas digitales, estados y métricas serializadas.
        # Salida:
        #   - valor calculado, estructura de resultados, tabla, ruta o None según la operación
        #     específica de la función. Tipo de retorno anotado: None.
        # -----------------------------------------------------------------------------
        task = self.state.setdefault("tasks", {}).setdefault(task_id, {})
        task.update({
            "status": "done",
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "outputs": [str(Path(o)) for o in (outputs or [])],
            "result_summary": result_summary or {},
        })
        self.save()
        self.log(f"DONE {task_id}")

    def fail(self, task_id: str, exc: BaseException) -> None:
        # -----------------------------------------------------------------------------
        # Entrada:
        #   - self: instancia actual del objeto; permite acceder a estado interno, rutas, métricas o
        #       configuración acumulada.
        #   - task_id: parámetro de entrada usado por la función para controlar el cálculo o suministrar
        #       datos. Tipo esperado/anotado: str.
        #   - exc: parámetro de entrada usado por la función para controlar el cálculo o suministrar
        #       datos. Tipo esperado/anotado: BaseException.
        # Proceso:
        #   - Ejecuta el bloque "fail" dentro del módulo de control de ejecución reproducible mediante
        #     huellas digitales, estados y métricas serializadas.
        # Salida:
        #   - valor calculado, estructura de resultados, tabla, ruta o None según la operación
        #     específica de la función. Tipo de retorno anotado: None.
        # -----------------------------------------------------------------------------
        task = self.state.setdefault("tasks", {}).setdefault(task_id, {})
        task.update({
            "status": "failed",
            "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        self.save()
        self.log(f"FAIL {task_id}: {exc}")

    def run(self, task_id: str, inputs: Iterable[Path] | None, outputs: Iterable[Path] | None, params: dict[str, Any], fn: Callable[[], Any]):
        # -----------------------------------------------------------------------------
        # Entrada:
        #   - self: instancia actual del objeto; permite acceder a estado interno, rutas, métricas o
        #       configuración acumulada.
        #   - task_id: parámetro de entrada usado por la función para controlar el cálculo o suministrar
        #       datos. Tipo esperado/anotado: str.
        #   - inputs: parámetro de entrada usado por la función para controlar el cálculo o suministrar
        #       datos. Tipo esperado/anotado: Iterable[Path] | None.
        #   - outputs: parámetro de entrada usado por la función para controlar el cálculo o suministrar
        #       datos. Tipo esperado/anotado: Iterable[Path] | None.
        #   - params: parámetro de entrada usado por la función para controlar el cálculo o suministrar
        #       datos. Tipo esperado/anotado: dict[str, Any].
        #   - fn: parámetro de entrada usado por la función para controlar el cálculo o suministrar
        #       datos. Tipo esperado/anotado: Callable[[], Any].
        # Proceso:
        #   - Ejecuta el bloque "run" dentro del módulo de control de ejecución reproducible mediante
        #     huellas digitales, estados y métricas serializadas.
        # Salida:
        #   - resultado del proceso, reporte, métricas, archivos generados o estructura de resumen del
        #     módulo.
        # -----------------------------------------------------------------------------
        sig = self.signature(inputs=inputs, params=params)
        outputs = list(outputs or [])
        if self.is_done(task_id, sig, outputs):
            self.log(f"SKIP {task_id}")
            return None, "skipped"
        previous = self.state.get("tasks", {}).get(task_id)
        if (
            getattr(self.cfg, "skip_stuck", False)
            and previous
            and previous.get("signature") == sig
            and previous.get("status") in {"running", "failed"}
        ):
            self.log(f"SKIP_STUCK {task_id}")
            previous["status"] = "skipped_stuck"
            previous["skipped_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.save()
            return None, "skipped_stuck"
        self.start(task_id, sig, params)
        try:
            result = fn()
            summary = {}
            if hasattr(result, "shape"):
                summary["shape"] = str(getattr(result, "shape"))
            self.done(task_id, outputs, summary)
            return result, "done"
        except Exception as exc:
            self.fail(task_id, exc)
            raise


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    # -----------------------------------------------------------------------------
    # Entrada:
    #   - path: ruta de archivo o carpeta sobre la que se realiza lectura, escritura o validación. Tipo
    #       esperado/anotado: Path.
    # Proceso:
    #   - Serializa o lee estructuras JSON para preservar metadatos, métricas y trazabilidad del
    #     procesamiento.
    # Salida:
    #   - datos cargados en memoria, por ejemplo imagen, tabla, señal, JSON o estructura auxiliar. Tipo
    #     de retorno anotado: dict[str, Any] | None.
    # -----------------------------------------------------------------------------
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_metrics_json(data: dict[str, Any], path: Path) -> Path:
    # -----------------------------------------------------------------------------
    # Entrada:
    #   - data: datos de entrada en memoria, tabla, arreglo o estructura compuesta. Tipo
    #       esperado/anotado: dict[str, Any].
    #   - path: ruta de archivo o carpeta sobre la que se realiza lectura, escritura o validación. Tipo
    #       esperado/anotado: Path.
    # Proceso:
    #   - Serializa o lee estructuras JSON para preservar metadatos, métricas y trazabilidad del
    #     procesamiento.
    # Salida:
    #   - archivo escrito en disco, ruta de salida, registro de log o confirmación implícita mediante
    #     ausencia de error. Tipo de retorno anotado: Path.
    # -----------------------------------------------------------------------------
    save_json(data, path)
    return path
