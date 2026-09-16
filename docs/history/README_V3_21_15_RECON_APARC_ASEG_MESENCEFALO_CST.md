# V3.21.15 — recon-all/aparc+aseg obligatorio antes del mesencéfalo

Esta versión agrega una compuerta de calidad antes de segmentar el mesencéfalo:

1. Ejecuta o reanuda `recon-all`.
2. Verifica `recon-all.done`.
3. Verifica que existan y se puedan leer:
   - `mri/aparc+aseg.mgz`
   - `mri/aseg.mgz`
   - `mri/rawavg.mgz`
   - `mri/norm.mgz`
4. Solo después ejecuta BrainstemSubstructures/segmentBS.
5. Solo después valida `midbrain_T1.nii.gz`, `pons_T1.nii.gz` y `medulla_T1.nii.gz`.
6. Solo después reconstruye CST usando mesencéfalo como waypoint principal.

## Correr paciente 3 Antes

```bash
cd /home/humath/Escritorio
unzip -o suite_integrada_vce_v3_21_16_linux_recon_aparc_aseg_mesencefalo_cst.zip
cd /home/humath/Escritorio/suite_integrada_vce_v3_21_16_linux_recon_aparc_aseg_mesencefalo_cst
chmod +x *.sh
./00_instalar_todo_linux.sh
./06_run_paciente_3_antes_recon_aparc_mesencefalo_cst_sin_suspension_linux.sh
```

## Verificación rápida

```bash
SUBJECTS_DIR=${VCE_FREESURFER_SUBJECTS_DIR:-$HOME/freesurfer_subjects}
SID="paciente3_Antes"
ls -lh "$SUBJECTS_DIR/$SID/mri/aparc+aseg.mgz" "$SUBJECTS_DIR/$SID/mri/aseg.mgz" "$SUBJECTS_DIR/$SID/scripts/recon-all.done"
mri_info "$SUBJECTS_DIR/$SID/mri/aparc+aseg.mgz" | head -40
```

Archivos esperados:

```text
resultados/paciente 3/Antes/tractografia_propia/brainstem_substructures_validated/midbrain_T1.nii.gz
resultados/paciente 3/Antes/tractografia_propia/brainstem_substructures_validated/pons_T1.nii.gz
resultados/paciente 3/Antes/tractografia_propia/brainstem_substructures_validated/medulla_T1.nii.gz
resultados/paciente 3/Antes/tractografia_propia/cst_visualizacion_ventral_medial/cst_ventral_medial_izquierda_T1.trk
resultados/paciente 3/Antes/tractografia_propia/cst_visualizacion_ventral_medial/cst_ventral_medial_derecha_T1.trk
```
