"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         GESTIÓN DE RECURSOS GPU/CPU                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/gpu.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

o
-----------------------------------------
Gestiona cómputo CPU/GPU para operaciones numéricas. El modelo computacional
prioriza matrices y tensores en GPU cuando existe soporte, manteniendo
equivalente
la operación matemática respecto a CPU. Se documenta disponibilidad, memoria y
backend para reproducibilidad.

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

"""Utilidades opcionales de GPU para la suite VCE.

La suite funciona 100 % en CPU. Si CuPy está instalado y hay GPU NVIDIA/CUDA,
estas funciones aceleran operaciones pesadas de neuroimagen: normalización,
resize/interpolación, wavelet/LoG y correlaciones volumen-a-volumen.
"""

import os
import math
from dataclasses import asdict
from typing import Any, Iterable

import numpy as np

_CUPY = None
_CUPYX_NDIMAGE = None
_CUPY_ERROR: str | None = None
_CONFIGURED_DEVICE: int | None = None


def _mode_from_cfg(cfg: Any | None) -> str:
    return str(getattr(cfg, "gpu", os.environ.get("SUITE_VCE_GPU", "auto"))).lower()


def _min_elements_from_cfg(cfg: Any | None) -> int:
    try:
        return int(getattr(cfg, "gpu_min_elements", os.environ.get("SUITE_VCE_GPU_MIN_ELEMENTS", 250_000)))
    except Exception:
        return 250_000


def _validate_cupy_runtime(cp: Any) -> None:
    """Valida que CuPy no solo importe, sino que pueda compilar/ejecutar kernels.

    En Windows es frecuente que `import cupy` y `getDeviceCount()` funcionen, pero
    falle el primer kernel con "Failed to find CUDA headers". Por eso se hace una
    prueba mínima aquí antes de declarar la GPU como disponible.
    """
    x = cp.arange(16, dtype=cp.float32)
    y = cp.sqrt(x + cp.float32(1)).sum()
    _ = float(y.get())
    cp.cuda.Stream.null.synchronize()


def cupy_available() -> bool:
    global _CUPY, _CUPYX_NDIMAGE, _CUPY_ERROR
    if _CUPY is not None:
        return True
    if _CUPY_ERROR is not None:
        return False
    try:
        import cupy as cp  # type: ignore
        from cupyx.scipy import ndimage as cndimage  # type: ignore
        ndev = int(cp.cuda.runtime.getDeviceCount())
        if ndev <= 0:
            _CUPY_ERROR = "CuPy importó, pero no detectó GPU CUDA."
            return False
        _validate_cupy_runtime(cp)
        _CUPY = cp
        _CUPYX_NDIMAGE = cndimage
        return True
    except Exception as exc:  # pragma: no cover - depende del equipo del usuario
        _CUPY = None
        _CUPYX_NDIMAGE = None
        _CUPY_ERROR = str(exc)
        return False


def get_cupy():
    if not cupy_available():
        raise RuntimeError(_CUPY_ERROR or "CuPy no disponible")
    return _CUPY


def get_cupyx_ndimage():
    if not cupy_available():
        raise RuntimeError(_CUPY_ERROR or "cupyx.scipy.ndimage no disponible")
    return _CUPYX_NDIMAGE


def configure_gpu(cfg: Any | None = None) -> dict:
    """Configura la GPU si se solicitó y devuelve un resumen imprimible."""
    global _CONFIGURED_DEVICE
    mode = _mode_from_cfg(cfg)
    if mode in {"off", "false", "0", "cpu", "no"}:
        return {"enabled": False, "mode": mode, "reason": "GPU desactivada por configuración."}

    if not cupy_available():
        if mode in {"on", "true", "1", "gpu", "cupy"}:
            return {"enabled": False, "mode": mode, "reason": f"GPU solicitada, pero CuPy/CUDA no está disponible: {_CUPY_ERROR}"}
        return {"enabled": False, "mode": mode, "reason": f"CuPy/CUDA no disponible; usando CPU. Detalle: {_CUPY_ERROR}"}

    cp = get_cupy()
    try:
        requested_device = int(getattr(cfg, "gpu_device", 0)) if cfg is not None else 0
    except Exception:
        requested_device = 0
    ndev = int(cp.cuda.runtime.getDeviceCount())
    if requested_device < 0 or requested_device >= ndev:
        requested_device = 0
    try:
        cp.cuda.Device(requested_device).use()
        _CONFIGURED_DEVICE = requested_device
    except Exception:
        _CONFIGURED_DEVICE = None

    info = gpu_info()
    info.update({"enabled": True, "mode": mode})
    return info


def gpu_info() -> dict:
    if not cupy_available():
        return {"enabled": False, "reason": _CUPY_ERROR or "CuPy no disponible"}
    cp = get_cupy()
    try:
        dev_id = int(cp.cuda.Device().id)
        props = cp.cuda.runtime.getDeviceProperties(dev_id)
        name = props.get("name", b"")
        if isinstance(name, bytes):
            name = name.decode(errors="ignore")
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        return {
            "enabled": True,
            "backend": "cupy",
            "device_id": dev_id,
            "device_name": str(name),
            "gpu_mem_free_gb": round(float(free_b) / 1024**3, 3),
            "gpu_mem_total_gb": round(float(total_b) / 1024**3, 3),
        }
    except Exception as exc:
        return {"enabled": True, "backend": "cupy", "warning": str(exc)}


def should_use_gpu_for_shape(shape_or_array: Any, cfg: Any | None = None) -> bool:
    mode = _mode_from_cfg(cfg)
    if mode in {"off", "false", "0", "cpu", "no"}:
        return False
    if not cupy_available():
        return False
    try:
        shape = shape_or_array.shape if hasattr(shape_or_array, "shape") else tuple(shape_or_array)
        n = int(np.prod(shape))
    except Exception:
        n = 0
    return n >= _min_elements_from_cfg(cfg)


def asnumpy(x: Any) -> np.ndarray:
    if cupy_available():
        cp = get_cupy()
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    return np.asarray(x)


def free_gpu_memory() -> None:
    if not cupy_available():
        return
    cp = get_cupy()
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def robust_zscore_gpu(vol: np.ndarray, cfg: Any | None = None) -> np.ndarray | None:
    if not should_use_gpu_for_shape(vol, cfg):
        return None
    cp = get_cupy()
    try:
        x = cp.asarray(vol, dtype=cp.float32)
        finite = cp.isfinite(x)
        if int(cp.sum(finite).get()) == 0:
            return np.zeros_like(np.asarray(vol), dtype=np.float32)
        vals = x[finite]
        lo, hi = cp.percentile(vals, cp.asarray([1, 99], dtype=cp.float32))
        x = cp.clip(x, lo, hi)
        vals2 = x[finite]
        med = cp.median(vals2)
        mad = cp.median(cp.abs(vals2 - med))
        out = ((x - med) / (cp.float32(1.4826) * mad + cp.float32(1e-6))).astype(cp.float32)
        return cp.asnumpy(out)
    finally:
        free_gpu_memory()


def minmax_gpu(vol: np.ndarray, cfg: Any | None = None) -> np.ndarray | None:
    if not should_use_gpu_for_shape(vol, cfg):
        return None
    cp = get_cupy()
    try:
        x = cp.asarray(vol, dtype=cp.float32)
        finite = cp.isfinite(x)
        if int(cp.sum(finite).get()) == 0:
            return np.zeros_like(np.asarray(vol), dtype=np.float32)
        vals = x[finite]
        lo, hi = cp.percentile(vals, cp.asarray([1, 99], dtype=cp.float32))
        out = cp.clip((x - lo) / (hi - lo + cp.float32(1e-6)), 0, 1).astype(cp.float32)
        return cp.asnumpy(out)
    finally:
        free_gpu_memory()


def resize_to_shape_gpu(vol: np.ndarray, shape: tuple[int, ...], order: int = 1, cfg: Any | None = None) -> np.ndarray | None:
    if not should_use_gpu_for_shape(vol, cfg):
        return None
    cp = get_cupy()
    cndimage = get_cupyx_ndimage()
    try:
        x = cp.asarray(vol, dtype=cp.float32)
        factors = [shape[i] / x.shape[i] for i in range(len(shape))]
        y = cndimage.zoom(x, factors, order=order)
        return cp.asnumpy(y.astype(cp.float32))
    finally:
        free_gpu_memory()


def wavelet_energy_gpu(vol: np.ndarray, scales: Iterable[float], cfg: Any | None = None) -> np.ndarray | None:
    if not should_use_gpu_for_shape(vol, cfg):
        return None
    cp = get_cupy()
    cndimage = get_cupyx_ndimage()
    try:
        data_cpu = np.asarray(vol, dtype=np.float32)
        z_cpu = robust_zscore_gpu(data_cpu, cfg)
        if z_cpu is None:
            return None
        x = cp.asarray(z_cpu, dtype=cp.float32)
        acc = cp.zeros_like(x, dtype=cp.float32)
        count = 0
        for s in scales:
            if float(s) <= 0:
                continue
            response = cndimage.gaussian_laplace(x, sigma=float(s))
            acc += cp.abs(response).astype(cp.float32)
            count += 1
        if count == 0:
            return cp.asnumpy(acc)
        return cp.asnumpy((acc / count).astype(cp.float32))
    finally:
        free_gpu_memory()


def pearson_gpu(a: np.ndarray, b: np.ndarray, cfg: Any | None = None) -> float | None:
    if not should_use_gpu_for_shape(a, cfg):
        return None
    cp = get_cupy()
    try:
        aa = cp.asarray(a, dtype=cp.float32).ravel()
        bb = cp.asarray(b, dtype=cp.float32).ravel()
        finite = cp.isfinite(aa) & cp.isfinite(bb)
        n = int(cp.sum(finite).get())
        if n < 3:
            return math.nan
        aa = aa[finite]
        bb = bb[finite]
        aa = aa - cp.mean(aa)
        bb = bb - cp.mean(bb)
        denom = cp.sqrt(cp.sum(aa * aa) * cp.sum(bb * bb)) + cp.float32(1e-12)
        val = cp.sum(aa * bb) / denom
        return float(val.get())
    finally:
        free_gpu_memory()
