# V3.21.19 - CST fusionada con salidas compatibles con 3D Slicer

Esta versión integra en el flujo principal:

1. DWI selectiva por serie.
2. Registro DWI/b0 a rT1/rAnatomico con corrección opcional de desplazamiento.
3. Recon-all obligatorio antes del mesencéfalo.
4. Verificación de `aparc+aseg.mgz` y `aseg.mgz`.
5. Delimitación de mesencéfalo, puente, bulbo y SCP.
6. Reconstrucción de vía corticoespinal usando mesencéfalo como waypoint.
7. Fusión izquierda + derecha en un solo archivo final.
8. Exportación para ITK-SNAP y 3D Slicer.

## Archivos finales en `tractografia_propia/`

- `via_cortico_espinal_completa.trk`: streamlines reales fusionadas izquierda + derecha.
- `via_cortico_espinal_completa.nii`: volumen RGB voxelizado.
- `via_cortico_espinal_completa_rgb_direccion_T1.nii.gz`: RGB por dirección.
- `via_cortico_espinal_completa_mask.nii.gz`: máscara binaria.
- `via_cortico_espinal_completa_density.nii.gz`: mapa de densidad.
- `via_cortico_espinal_completa_colores_itksnap_T1.nii.gz`: labelmap para ITK-SNAP.
- `via_cortico_espinal_completa_labels_itksnap.txt`: tabla de colores para ITK-SNAP.
- `via_cortico_espinal_completa_preview.png`: vista previa.
- `via_cortico_espinal_completa_slicer.trk`: TRK con header limpio para 3D Slicer.
- `via_cortico_espinal_completa_slicer.vtk`: respaldo para 3D Slicer como Model con RGB por orientación.

## Ver en 3D Slicer

1. Instalar extensión SlicerDMRI si se desea cargar `.trk` como FiberBundle.
2. Cargar `rAnatomico.nii` como volumen principal.
3. Cargar `b0_en_T1.nii.gz` para validar registro.
4. Cargar `midbrain_T1.nii.gz`, `pons_T1.nii.gz`, `medulla_T1.nii.gz` como labelmaps/segmentaciones.
5. Cargar `via_cortico_espinal_completa_slicer.trk`.
6. Si el `.trk` falla, cargar `via_cortico_espinal_completa_slicer.vtk` como Model.

## Colores en 3D Slicer

Para `.trk`:

- Módulo `Tractography Display`.
- Seleccionar el FiberBundle.
- Activar visibilidad 2D/3D.
- Usar `Color by Orientation`.
- Activar Tube Display si se desea más grosor.

Para `.vtk`:

- Módulo `Models`.
- Seleccionar `via_cortico_espinal_completa_slicer`.
- Activar `Visibility`.
- Activar `Scalar Visibility` si aparece disponible.
- Si no reconoce los RGB por punto, asignar color manual en Display.

## Validación importante

Antes de evaluar la CST, revisar:

- `b0_en_T1.nii.gz` debe coincidir con `rAnatomico.nii`.
- `midbrain_T1.nii.gz` debe delimitar el mesencéfalo completo.
- `pons_T1.nii.gz` debe caer en puente.
- `medulla_T1.nii.gz` debe caer en bulbo.
- La CST debe bajar por cápsula interna, pedúnculo cerebral, puente y bulbo.

Si `b0_en_T1` queda desplazado, usar:

```bash
export VCE_REGISTRO_COM_CORREGIR=1
export VCE_REGISTRO_COM_EJES=z
```

Si sigue desplazado en más de un eje:

```bash
export VCE_REGISTRO_COM_CORREGIR=1
export VCE_REGISTRO_COM_EJES=xyz
```
