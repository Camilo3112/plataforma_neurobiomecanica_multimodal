# FIX v3.21.7 — Tractografía propia como etapa ejecutable explícita

## Problema detectado

En la versión v3.21.6 el anexo CST/tronco quedó integrado como módulo documental y como etapa `cst_tronco`, pero el script interactivo general corría `--only todo`. Por diseño, `todo` no ejecutaba `cst_tronco` porque FreeSurfer y tractografía pueden tardar horas.

Por eso, al finalizar el procesamiento general, podía no aparecer la carpeta:

```text
resultados/paciente X/Antes/tractografia_propia
```

## Corrección realizada

La versión v3.21.7 agrega ejecución real del anexo de tractografía propia:

1. Crea siempre la estructura `tractografia_propia` por paciente y etapa cuando se ejecuta `--only cst_tronco`.
2. Si falta `wholebrain_T1.trk`, intenta generarlo desde los DICOM crudos de difusión.
3. Usa el `rT1.nii` del paciente como referencia anatómica.
4. Guarda la tractografía completa en espacio T1.
5. Luego corre segmentación de tronco y extracción/visualización CST.
6. Genera un resumen global en `resultados/resumen_modulo_cst_tronco.csv`.
7. Crea un alias opcional `tractografias_propias` apuntando a `tractografia_propia` cuando el sistema lo permite.

## Comando recomendado

```bash
./run_interactivo_tractografia_cst_linux.sh
```

O para paciente 3 y 6:

```bash
./run_tractografia_cst_pacientes_3_6_linux.sh
```

## Salidas esperadas

```text
resultados/paciente 3/Antes/tractografia_propia/
resultados/paciente 3/Despues/tractografia_propia/
resultados/paciente 6/Antes/tractografia_propia/
resultados/paciente 6/Despues/tractografia_propia/
```

Dentro deben aparecer, si el cálculo fue exitoso:

```text
wholebrain_T1.trk
wholebrain_density_T1.nii.gz
wholebrain_rgb_T1.nii.gz
FA_en_T1.nii.gz
ColorFA_en_T1.nii.gz
b0_en_T1.nii.gz
atlas_motor_rois/
brainstem_substructures_validated/
cst_visualizacion_ventral_medial/
logs_cst_tronco/
```

## Diagnóstico

Revisar:

```text
resultados/resumen_modulo_cst_tronco.csv
```

Estados importantes:

- `ok`: módulo completo ejecutado.
- `fallo_tractografia_propia_codigo_X`: falló la generación de `wholebrain_T1.trk`.
- `omitido_sin_rT1`: no encontró `rT1.nii`.
- `omitido_sin_dicom_dwi`: no encontró carpeta DICOM de difusión.
- `pendiente_license_freesurfer`: falta licencia FreeSurfer para segmentar tronco.
