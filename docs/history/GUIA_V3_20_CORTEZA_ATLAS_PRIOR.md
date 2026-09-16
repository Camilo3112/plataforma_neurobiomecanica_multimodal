# V3.20 — Corrección de corteza motora con atlas-prior

Esta versión mantiene la corrección geométrica de máscara cortical por Brain00mm/capa cortical y agrega una segunda etapa de refinamiento atlas-guiado.

## Qué hace

1. Remuestrea la máscara al espacio del Brain00mm.
2. Corrige la máscara por traslación hacia la capa cortical externa.
3. Infere lado y región desde el nombre del archivo:
   - derecha / izquierda
   - primaria = M1
   - secundaria / premotora / M2 = M2
4. Genera una prior probabilística heurística M1/M2 para ese lado.
5. Intersecta o reubica la máscara sobre la capa cortical + zona motora esperada.
6. Guarda NIfTI, PNG y JSON de control de calidad.

## Salidas nuevas

En `resultados/<paciente>/<Antes|Despues>/morfometria/interna/corregidas_corticales`:

- `*_corregida_auto.nii.gz`: corrección geométrica por capa cortical.
- `*_atlas_prior_m1_izquierda.nii.gz` o similar: prior probabilística.
- `*_atlas_target_same_volume.nii.gz`: zona atlas con volumen similar a la máscara corregida.
- `*_atlas_refinada.nii.gz`: máscara final recomendada para el protocolo.
- `*_atlas_refinada_preview.png`: control visual.
- `*_atlas_refinada_reporte.json`: métricas de calidad.

## Importante

Esto es un atlas-prior heurístico, no un registro anatómico MNI/FreeSurfer completo. Si existe FreeSurfer, la suite también puede integrar `aparc+aseg.mgz`, que es más anatómico para precentral/M1.
