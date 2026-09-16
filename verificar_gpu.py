"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       VERIFICACIÓN DE ACELERACIÓN GPU                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: verificar_gpu.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Inspecciona disponibilidad de aceleración GPU. El resultado no cambia el
modelo
matemático del análisis, pero permite seleccionar backend de ejecución para
FFT,
matrices, filtros y procesamiento volumétrico de forma trazable.

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

from src.gpu import configure_gpu, cupy_available, gpu_info
from src.config import PipelineConfig, DEFAULT_PROJECT_ROOT

cfg = PipelineConfig(project_root=DEFAULT_PROJECT_ROOT, gpu="on", gpu_device=0, gpu_min_elements=1)
status = configure_gpu(cfg)
print("Estado GPU:")
for k, v in status.items():
    print(f"  {k}: {v}")

if status.get("enabled"):
    try:
        import cupy as cp
        x = cp.arange(1_000_000, dtype=cp.float32)
        y = cp.sqrt(x + cp.float32(1)).sum()
        cp.cuda.Stream.null.synchronize()
        print(f"Prueba CuPy OK. Resultado: {float(y.get()):.3f}")
    except Exception as exc:
        print("CuPy detectó la GPU, pero falló al ejecutar un kernel CUDA.")
        print(f"Detalle: {exc}")
        print("Solución recomendada:")
        print('  python -m pip uninstall -y cupy cupy-cuda11x cupy-cuda12x cupy-cuda13x')
        print('  python -m pip install "cupy-cuda12x[ctk]"')
else:
    print("No se activó GPU. La suite correrá en CPU.")
    print("Para NVIDIA/CUDA 12.x instala:")
    print('  python -m pip uninstall -y cupy cupy-cuda11x cupy-cuda12x cupy-cuda13x')
    print('  python -m pip install "cupy-cuda12x[ctk]"')
