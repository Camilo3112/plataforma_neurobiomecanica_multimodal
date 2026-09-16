# Documentación interna de modelos físicos y matemáticos — v3.21.21

Esta versión incorpora en cada archivo Python un encabezado técnico con el modelo físico, matemático o computacional usado por el módulo.

## Bloques documentados

- Señales EMG/dinamometría: RMS, FFT, correlación cruzada, autocorrelación, interpolación temporal y CWT.
- TAC: unidades Hounsfield, umbralización tisular, morfología matemática, área y volumen con geometría DICOM.
- Neuroimagen: matrices afines, remuestreo, máscaras, volumen voxelizado y comparación pre/post.
- FreeSurfer/MNI: transferencia de etiquetas, `aparc+aseg`, ROI cortical y espacio nativo.
- Tronco encefálico: etiquetas mesencéfalo=173, puente=174, bulbo=175 y SCP=178.
- CST: modelo DWI, streamlines, waypoints anatómicos, densidad voxelizada y color por orientación local.
- Reportes y consolidación: organización paciente-etapa-variable, deltas absolutos/relativos y trazabilidad.
