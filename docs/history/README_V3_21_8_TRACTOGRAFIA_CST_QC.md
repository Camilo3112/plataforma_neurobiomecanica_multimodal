# Suite VCE v3.21.8 · Linux completa con tractografía propia + CST/tronco + QC TAC

Esta entrega consolida la suite completa e incorpora un modo robusto para correr exclusivamente el anexo de tractografía propia y CST/tronco.

## Cambios principales

1. `00_instalar_todo_linux.sh`: instala dependencias del sistema, crea `.venv`, instala librerías Python y verifica `dcm2niix`.
2. `01_run_interactivo_tractografia_sin_suspension_linux.sh`: ejecuta solo `--only cst_tronco`, bloqueando suspensión, bloqueo por inactividad y apagado de pantalla hasta finalizar.
3. Carga FreeSurfer de forma tolerante a variables no definidas, compatible con `/usr/local/freesurfer/8.2.0`.
4. Si falta `.venv`, el script de tractografía lo crea automáticamente.
5. `03_verificar_suite_linux.sh`: valida archivos críticos, sintaxis Bash y compilación Python.
6. El runner CST usa `FS_LICENSE` si está definido; por defecto usa `/home/humath/Escritorio/license.txt`.
7. Mantiene el control de calidad TAC de dos fémures y rescate de corte bilateral.

## Flujo del anexo CST

Al ejecutar `--only cst_tronco`, la suite:

1. Localiza `rT1.nii` en la etapa del paciente.
2. Crea `resultados/<paciente>/<etapa>/tractografia_propia/`.
3. Si no existe `wholebrain_T1.trk`, intenta generarlo desde DICOM DWI crudo con `dcm2niix` y DIPY.
4. Genera salidas en espacio T1: `wholebrain_T1.trk`, densidad, RGB, FA, ColorFA y b0.
5. Ejecuta segmentación de tronco con FreeSurfer cuando faltan `midbrain_T1.nii.gz`, `pons_T1.nii.gz` y `medulla_T1.nii.gz`.
6. Extrae la CST con waypoints anatómicos: corteza motora M1/M2, mesencéfalo, puente y bulbo, priorizando la región ventral/medial del tronco.
7. Exporta `.trk`, labelmaps NIfTI, densidades, visualizaciones y reporte JSON.
8. Escribe el resumen global `resultados/resumen_modulo_cst_tronco.csv`.

## Salidas esperadas

```text
resultados/paciente 6/Antes/tractografia_propia/
  wholebrain_T1.trk
  wholebrain_density_T1.nii.gz
  wholebrain_rgb_T1.nii.gz
  FA_en_T1.nii.gz
  ColorFA_en_T1.nii.gz
  b0_en_T1.nii.gz
  brainstem_substructures_validated/
    midbrain_T1.nii.gz
    pons_T1.nii.gz
    medulla_T1.nii.gz
  cst_visualizacion_ventral_medial/
    cst_ventral_medial_izquierda_T1.trk
    cst_ventral_medial_derecha_T1.trk
    cst_ventral_medial_bilateral_T1.trk
    cst_ventral_medial_colores_itksnap_T1.nii.gz
    cst_ventral_medial_reporte.json
```

