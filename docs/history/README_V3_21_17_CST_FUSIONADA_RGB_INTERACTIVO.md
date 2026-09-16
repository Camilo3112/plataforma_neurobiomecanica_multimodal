# V3.21.17 - Vía corticoespinal completa fusionada + NIfTI RGB

Esta versión agrega la salida final bilateral solicitada en la raíz de `tractografia_propia`:

- `via_cortico_espinal_completa.trk`: streamlines reales fusionadas izquierda + derecha.
- `via_cortico_espinal_completa.nii`: representación voxelizada RGB de la vía fusionada, con colores tipo tractografía.
- `via_cortico_espinal_completa_mask.nii.gz`: máscara binaria.
- `via_cortico_espinal_completa_density.nii.gz`: mapa de densidad.
- `via_cortico_espinal_completa_colores_itksnap_T1.nii.gz` y `via_cortico_espinal_completa_labels_itksnap.txt`: labelmap para ITK-SNAP.
- `via_cortico_espinal_completa_preview.png`: vista rápida.

El `.trk` conserva las fibras reales. El `.nii` es una imagen RGB voxelizada para visualización; no reemplaza la tractografía vectorial.
