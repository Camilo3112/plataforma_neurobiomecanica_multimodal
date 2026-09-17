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

Fundamento
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
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def file_fingerprint(path: Path) -> dict[str, Any]:
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
        self.cfg = cfg
        self.root = safe_mkdir(cfg.results_root() / "_estado_pipeline")
        self.path = self.root / "checkpoints.json"
        self.log_path = self.root / "pipeline_eventos.log"
        self.state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
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
        safe_mkdir(self.path.parent)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        os.replace(tmp, self.path)

    def log(self, message: str) -> None:
        safe_mkdir(self.log_path.parent)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {message}\n")

    def signature(self, inputs: Iterable[Path] | None = None, params: dict[str, Any] | None = None) -> str:
        fps = []
        for p in inputs or []:
            p = Path(p)
            if p.is_dir():
                fps.append(folder_fingerprint(p))
            else:
                fps.append(file_fingerprint(p))
        return _stable_hash({"inputs": fps, "params": params or {}})

    def is_done(self, task_id: str, signature: str, outputs: Iterable[Path] | None = None) -> bool:
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
        self.state.setdefault("tasks", {})[task_id] = {
            "status": "running",
            "signature": signature,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "meta": meta or {},
        }
        self.save()
        self.log(f"START {task_id}")

    def done(self, task_id: str, outputs: Iterable[Path] | None = None, result_summary: dict[str, Any] | None = None) -> None:
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
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_metrics_json(data: dict[str, Any], path: Path) -> Path:
    save_json(data, path)
    return path
