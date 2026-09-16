SEGMENTACIÓN DEL TRONCO EN LINUX — V9

Este directorio permite ejecutar FreeSurfer directamente en Linux, sin WSL.

Entradas obligatorias:
- rT1.nii del paciente/etapa.
- license.txt de FreeSurfer.
- FreeSurfer instalado y cargable.

Salida:
- brainstem_substructures/: segmentación raw de FreeSurfer convertida a NIfTI.
- brainstem_substructures_validated/: máscaras separadas y validadas en espacio rT1.

Ejemplo:
./segmentar_tronco_todo_linux.sh \
  --project-root /home/humath/Escritorio \
  --t1-path "/home/humath/Escritorio/datos/paciente 6/Antes/Resultados funcional/REFORMATEO/rT1.nii" \
  --license-path /home/humath/Escritorio/license.txt \
  --subject-id paciente6_Antes \
  --out-base "/home/humath/Escritorio/resultados/paciente 6/Antes/tractografia_propia" \
  --threads 8
