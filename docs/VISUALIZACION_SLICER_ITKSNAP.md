# Visualización en 3D Slicer e ITK-SNAP

## 3D Slicer

1. Abrir `rAnatomico.nii` como volumen principal.
2. Abrir `b0_en_T1.nii.gz` o `b0_en_T1_refinado_COM.nii.gz` para revisar registro.
3. Abrir `via_cortico_espinal_completa_slicer.trk` como FiberBundle.
4. En `Tractography Display`, usar `Color by Orientation`.
5. Si el `.trk` falla, abrir `via_cortico_espinal_completa_slicer.vtk` como Model.
6. Para superposición directa en cortes, abrir `via_cortico_espinal_completa_colores_itksnap_T1.nii.gz` como labelmap/segmentation.

## ITK-SNAP

1. `File > Open Main Image`: cargar `rAnatomico.nii`.
2. `Segmentation > Open Segmentation`: cargar `via_cortico_espinal_completa_colores_itksnap_T1.nii.gz`.
3. Cargar `via_cortico_espinal_completa_labels_itksnap.txt` como label descriptions si se desea tabla de colores.

## Validación mínima

Antes de interpretar la CST, verificar que:

- `b0_en_T1` coincida con el cerebro de `rAnatomico`.
- `midbrain_T1`, `pons_T1` y `medulla_T1` caigan anatómicamente en tronco.
- La vía descienda por cápsula interna, mesencéfalo, puente y bulbo.
