# V3.21.14 - Mesencéfalo primero para CST

Esta versión corrige el flujo anatómico de la vía corticoespinal:

1. Primero delimita subestructuras reales de tronco desde FreeSurfer BrainstemSubstructures.
2. Extrae y valida `midbrain_T1.nii.gz`, `pons_T1.nii.gz`, `medulla_T1.nii.gz` y `brainstem_substructures_labelmap_T1.nii.gz`.
3. Usa el mesencéfalo completo como waypoint principal antes de intentar reconstruir la CST.
4. Si el filtro completo corteza -> mesencéfalo -> puente -> bulbo queda vacío, activa un rescate explícito corteza -> mesencéfalo.
5. No usa tractografía del resonador; parte de `wholebrain_T1.trk` generado desde DWI.

## Archivos clave para abrir en ITK-SNAP

- `brainstem_substructures_validated/brainstem_substructures_labelmap_T1.nii.gz`
- `brainstem_substructures_validated/midbrain_T1.nii.gz`
- `cst_visualizacion_ventral_medial/cst_ventral_medial_colores_itksnap_T1.nii.gz`
- `cst_visualizacion_ventral_medial/cst_ventral_medial_bilateral_T1.trk`
- `wholebrain_T1.trk`
- `FA_en_T1.nii.gz`
- `ColorFA_en_T1.nii.gz`

## Comando recomendado

```bash
cd /home/humath/Escritorio/suite_integrada_vce_v3_21_14_linux_mesencefalo_primero_cst
./05_run_paciente_3_antes_mesencefalo_primero_cst_sin_suspension_linux.sh
```

