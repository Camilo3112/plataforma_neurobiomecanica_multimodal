# V3.21.12 - Fix CST target_mask + DWI selectiva real

Esta versión corrige dos problemas vistos en el diagnóstico global del 25-08-2026:

1. **CST fallaba con `streamlines points are outside of target_mask`**.
   Se reemplazó el filtrado de DIPY `utils.target` por un filtrado tolerante que ignora puntos fuera del FOV del T1 y evalúa solo puntos válidos dentro del volumen.

2. **Algunas carpetas DICOM mezclaban varias series**.
   Ahora, antes de llamar `dcm2niix`, el módulo agrupa archivos por `SeriesInstanceUID`, selecciona la serie DWI/DTI cruda mejor puntuada y crea una carpeta temporal con enlaces simbólicos solo de esa serie.

Variables útiles:

```bash
export VCE_DWI_FILTRAR_SERIE_UID=1      # por defecto 1
export VCE_DWI_LIMPIAR_CONVERTIDOS=1    # por defecto 1
export VCE_DWI_MAX_FOLDERS_TO_TRY=12
```

Archivo diagnóstico nuevo esperado:

```text
dwi_series_groups_dwi_candidata_001.json
```

Si después de este fix el CST queda en 0 streamlines, ya no es error de programa sino que el filtrado anatómico M1/M2 + tronco está demasiado estricto o las ROIs no cruzan las fibras. En ese caso se debe usar dilatación mayor o extracción por proximidad.
