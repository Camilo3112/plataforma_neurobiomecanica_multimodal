# Checklist de integración

## Antes de integrar
- [ ] Existe `rT1.nii` y abre correctamente en ITK-SNAP.
- [ ] Existe `wholebrain_T1.trk` en espacio T1/RAS+ mm.
- [ ] Existen las máscaras M1 izquierda/derecha; M2 es opcional pero recomendada.
- [ ] Existe `license.txt` de FreeSurfer.
- [ ] WSL2 + Ubuntu 22.04 están funcionales.
- [ ] FreeSurfer 7.4.1 está instalado dentro de WSL.
- [ ] No se utilizan máscaras CST ni tractografía derivadas del resonador.

## Segmentación del tronco
- [ ] Ejecutar `EJECUTAR_SEGMENTACION_TRONCO_WINDOWS_V9.cmd`.
- [ ] Confirmar `brainstem_substructures_labelmap_T1.nii.gz`.
- [ ] Verificar visualmente mesencéfalo, puente, bulbo y SCP sobre `rT1.nii`.
- [ ] Confirmar que el reporte del validador indique geometría compatible.

## CST
- [ ] Ejecutar `EJECUTAR_CST_COLORES_V4_CORONAL.cmd`.
- [ ] Revisar el porcentaje de puntos del tractograma dentro del T1.
- [ ] Abrir `cst_ventral_medial_colores_itksnap_T1.nii.gz`.
- [ ] Aplicar `cst_labels_itksnap.txt`.
- [ ] Revisar axial, coronal y sagital.
- [ ] Revisar los waypoints ventrales-mediales como capas separadas.
- [ ] Leer el JSON para comprobar si cada lado es completo o parcial.

## Antes de usar en un análisis grupal
- [ ] Parametrizar paciente y momento.
- [ ] Guardar logs por ejecución.
- [ ] Guardar versión del código y parámetros.
- [ ] No sobreescribir resultados anteriores.
- [ ] Registrar si la CST llegó al bulbo o solo al puente.
- [ ] Aplicar el mismo criterio a todos los pacientes.
