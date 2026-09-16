# v3.19 - Tomografía con tejido adiposo NIfTI y métricas completas

Esta versión asegura que cada corrida de tomografía avanzada genere archivos NIfTI específicos para tejido adiposo y métricas tabulares completas.

## Salidas obligatorias por sujeto/etapa

Carpeta:

```bash
/home/humath/Escritorio/resultados/<paciente>/<Antes|Despues>/tomografia/avanzada_tejido_adiposo_landmarks
```

Archivos NIfTI principales:

- `Composicion_Corporal_CorteMedio.nii`: labelmap del corte medio con epidermis/dermis estimadas, grasa subcutánea y músculo total.
- `TAC_Composicion_Corporal_CorteMedio.nii`: TAC con overlay de composición del corte medio.
- `Grasa_Intramuscular_Muscular.nii`: labelmap 3D de grasa intramuscular por músculo.
- `TAC_Grasa_Intramuscular_Muscular.nii`: TAC con overlay de grasa intramuscular.
- `Grasa_Subcutanea_Volumen.nii.gz`: labelmap 3D de grasa subcutánea por miembro.
- `Tejido_Adiposo_Total_Volumen.nii.gz`: labelmap 3D unificado de grasa subcutánea + intramuscular.

Archivos de métricas:

- `Metricas_Tejido_Adiposo.csv`: métricas completas de grasa/músculo en formato largo.
- `resumen_metricas_tomografia_avanzada.csv`: resumen integrado de métricas de tomografía avanzada.
- `Areas_Grasa_Intramuscular_Por_Corte.csv`: áreas de grasa intramuscular por corte.
- `manifiesto_archivos_tejido_adiposo.csv`: verificación de existencia/tamaño de salidas obligatorias.
- `Reporte_Morfometrico_Completo.txt`: reporte legible completo.

## Etiquetas de tejido adiposo total

Archivo: `Tejido_Adiposo_Total_Volumen.nii.gz`

- 0 = fondo
- 1 = grasa subcutánea del miembro derecho
- 2 = grasa subcutánea del miembro izquierdo
- 3 = grasa intramuscular del miembro derecho
- 4 = grasa intramuscular del miembro izquierdo

## Correr solo tomografía

```bash
cd /home/humath/Escritorio/suite_integrada_vce_v3_19_linux_tomografia_adiposo_completo
source .venv/bin/activate
./run_solo_tomografia_linux.sh
```

## Correr solo tomografía avanzada

```bash
cd /home/humath/Escritorio/suite_integrada_vce_v3_19_linux_tomografia_adiposo_completo
source .venv/bin/activate
./run_solo_tomografia_avanzada_linux.sh
```

## Verificar que quedó completo

```bash
find /home/humath/Escritorio/resultados -path '*avanzada_tejido_adiposo_landmarks*' \
  \( -name 'Grasa_Subcutanea_Volumen.nii.gz' \
  -o -name 'Grasa_Intramuscular_Muscular.nii' \
  -o -name 'Tejido_Adiposo_Total_Volumen.nii.gz' \
  -o -name 'Metricas_Tejido_Adiposo.csv' \
  -o -name 'manifiesto_archivos_tejido_adiposo.csv' \) -print
```

