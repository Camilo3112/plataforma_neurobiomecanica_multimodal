"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MOTOR DE TRANSFORMADA WAVELET CONTINUA                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: tools/cwt_engine.py
Versión: v3.21.21

Descripción
-----------
Calcula análisis tiempo-frecuencia por wavelets.

Fundamento físico-matemático implementado
-----------------------------------------
Calcula transformada wavelet continua para estudiar cambios tiempo-frecuencia.
La CWT se expresa como W_x(a,b)=1/sqrt(a) ∫ x(t) ψ*((t-b)/a) dt, donde a
controla
escala/frecuencia y b localización temporal. Esto permite observar activación
EMG
no estacionaria y cambios transitorios que una FFT global puede suavizar.

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
from dataclasses import dataclass
import numpy as np
try:
    from scipy.signal import cwt as scipy_cwt, ricker
    SCIPY_CWT_OK = True
except Exception:
    SCIPY_CWT_OK = False

@dataclass
class CWTResult:
    scales: np.ndarray
    coefficients: np.ndarray
    power: np.ndarray

def _mexican_hat_kernel(width: float, radius_factor: float = 6.0) -> np.ndarray:
    # Entrada:
    #   - width: parámetro `width` usado por la función para controlar la operación, definir
    #   datos de entrada o especificar rutas/resultados.
    #   - radius_factor: parámetro `radius_factor` usado por la función para controlar la
    #   operación, definir datos de entrada o especificar rutas/resultados.
    # Proceso:
    #   Construye un núcleo tipo sombrero mexicano/LoG a partir de una escala; este operador
    #   aproxima una segunda derivada suavizada y permite detectar transiciones, picos y
    #   cambios locales en una señal.
    # Salida:
    #   Devuelve el resultado procesado por la función, listo para alimentar el siguiente
    #   paso del pipeline o para ser exportado.
    width = max(float(width), 1e-6)
    half = max(3, int(np.ceil(radius_factor * width)))
    x = np.arange(-half, half + 1, dtype=float)
    a2 = width ** 2
    kernel = (1.0 - (x ** 2) / a2) * np.exp(-(x ** 2) / (2.0 * a2))
    norm = np.sqrt(np.sum(kernel ** 2))
    if norm > 0:
        kernel = kernel / norm
    return kernel

def cwt_1d(signal, wavelet="mexican_hat", n_scales=12, min_scale=1.0, max_scale=6.0):
    # Entrada:
    #   - signal: parámetro `signal` usado por la función para controlar la operación,
    #   definir datos de entrada o especificar rutas/resultados.
    #   - wavelet: parámetro `wavelet` usado por la función para controlar la operación,
    #   definir datos de entrada o especificar rutas/resultados.
    #   - n_scales: parámetro `n_scales` usado por la función para controlar la operación,
    #   definir datos de entrada o especificar rutas/resultados.
    #   - min_scale: parámetro `min_scale` usado por la función para controlar la operación,
    #   definir datos de entrada o especificar rutas/resultados.
    #   - max_scale: parámetro `max_scale` usado por la función para controlar la operación,
    #   definir datos de entrada o especificar rutas/resultados.
    # Proceso:
    #   Normaliza la señal, aplica convoluciones multiescala con una wavelet tipo sombrero
    #   mexicano/LoG y calcula una representación energía-escala similar a un escalograma
    #   CWT.
    # Salida:
    #   Devuelve el resultado procesado por la función, listo para alimentar el siguiente
    #   paso del pipeline o para ser exportado.
    x = np.asarray(signal, dtype=float).ravel()
    scales = np.linspace(float(min_scale), float(max_scale), int(n_scales))
    if x.size == 0:
        coefs = np.zeros((len(scales), 0), dtype=float)
        return CWTResult(scales=scales, coefficients=coefs, power=coefs)
    if SCIPY_CWT_OK:
        coefs = scipy_cwt(x, ricker, scales)
    else:
        coefs = np.vstack([np.convolve(x, _mexican_hat_kernel(sc)[::-1], mode="same") for sc in scales])
    return CWTResult(scales=scales, coefficients=coefs, power=np.abs(coefs) ** 2)
