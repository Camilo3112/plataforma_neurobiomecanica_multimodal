# Corrección v3.21.5 — QC de dos fémures / dos miembros

Esta versión agrega una verificación antes de procesar tomografía avanzada:

1. No mezcla series DICOM: agrupa por `SeriesInstanceUID` y geometría.
2. Selecciona la serie que contiene ambos miembros, verificando la presencia de **dos fémures** en múltiples cortes.
3. Si no se detectan suficientes cortes con dos fémures, el proceso se detiene para evitar medir un solo miembro y duplicarlo.
4. Ajusta los cortes medios si caen en un corte donde solo aparece un fémur.
5. Genera archivos QC:
   - `QC_DICOM_Series_Dos_Femures.csv`
   - `QC_Cortes_Dos_Femures.csv`
   - `QC_Cortes_Medios_Ajustados_Dos_Femures.csv`

## Variables de entorno opcionales

```bat
set VCE_TAC_MIN_TWO_FEMUR_SLICES=8
set VCE_TAC_MIN_FEMUR_PAIR_SEPARATION_PX=35
set VCE_TAC_MIDPOINT_TWO_FEMUR_WINDOW=90
```
