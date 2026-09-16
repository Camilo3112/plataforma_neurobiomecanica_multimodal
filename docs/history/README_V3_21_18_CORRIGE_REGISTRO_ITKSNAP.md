# V3.21.19 - Corrección de desplazamiento b0->rAnatomico + guía de colores ITK-SNAP

## Qué corrige

Esta versión mantiene la salida fusionada:

- `via_cortico_espinal_completa.trk`
- `via_cortico_espinal_completa.nii`
- `via_cortico_espinal_completa_colores_itksnap_T1.nii.gz`
- `via_cortico_espinal_completa_labels_itksnap.txt`

Además agrega una corrección automática de desplazamiento residual b0->T1 por centro de masa. Esto ayuda cuando la tractografía se ve desplazada hacia superior al abrirla sobre `rAnatomico.nii` y no pasa por bulbo/mesencéfalo.

## Variables útiles

Por defecto la corrección se aplica en eje Z anatómico mundial:

```bash
export VCE_REGISTRO_COM_CORREGIR=1
export VCE_REGISTRO_COM_EJES=z
export VCE_REGISTRO_COM_MAX_MM=60
export VCE_REGISTRO_COM_MIN_MM=3
```

Si la desviación también aparece lateral o antero-posterior, usar:

```bash
export VCE_REGISTRO_COM_EJES=xyz
```

Si quieres desactivar la corrección:

```bash
export VCE_REGISTRO_COM_CORREGIR=0
```

## Archivos nuevos de control

En `tractografia_propia/`:

- `b0_en_T1.nii.gz`
- `b0_en_T1_refinado_COM.nii.gz`
- `affine_b0_to_T1.txt`
- `affine_b0_to_T1_inicial.txt`
- `affine_b0_to_T1_correccion_COM.txt`
- `registro_b0_T1_qc.json`

Validar siempre `b0_en_T1.nii.gz` sobre `rAnatomico.nii` antes de confiar en CST.

## ITK-SNAP

ITK-SNAP no renderiza `.trk` como fibras 3D tipo TrackVis/Slicer. Para ITK-SNAP usar el labelmap:

- Main image: `rAnatomico.nii` o `rT1.nii`
- Segmentation: `via_cortico_espinal_completa_colores_itksnap_T1.nii.gz`
- Label descriptions: `via_cortico_espinal_completa_labels_itksnap.txt`

Para ver streamlines reales con colores por dirección, usar 3D Slicer, TrackVis, DSI Studio o MRtrix/MRView con `via_cortico_espinal_completa.trk`.
