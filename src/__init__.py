"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       PAQUETE CENTRAL DE LA SUITE VCE                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/__init__.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Inicializa el paquete y expone utilidades centrales. Este archivo no calcula
biomarcadores directamente; organiza el espacio de nombres para que los
modelos
de señal, imagen, morfometría y tractografía se importen de forma consistente.

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

__version__ = "3.21.20"
