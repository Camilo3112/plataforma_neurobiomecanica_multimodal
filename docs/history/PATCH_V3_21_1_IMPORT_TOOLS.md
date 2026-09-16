# Patch v3.21.1

Corrige `ModuleNotFoundError: No module named tools` en `tools/tomografia_avanzada_core.py`.

Causa: el script de tomografía avanzada se ejecutaba directamente desde la carpeta `tools`, por lo que Python no siempre veía `tools` como paquete desde la raíz de la suite.

Solución: import robusto:

```python
try:
    from tools.cwt_engine import cwt_1d
except ModuleNotFoundError:
    from cwt_engine import cwt_1d
```
