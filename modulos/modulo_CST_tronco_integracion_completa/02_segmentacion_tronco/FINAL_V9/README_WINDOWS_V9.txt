WINDOWS V9 — SOLUCIÓN DEL ERROR "pushd: Too many arguments"

DIAGNÓSTICO CONFIRMADO

El T1 original está en:

D:\EAFIT\01-2026\proyecto\datos\paciente 6\Antes\
Resultados funcional\REFORMATEO\rT1.nii

La ruta contiene espacios. FreeSurfer 7.4.1 procesa internamente el
directorio del parámetro -i con una instrucción tcsh similar a:

pushd $InVolDir

Al no proteger el directorio entre comillas, tcsh interpreta cada parte
separada por espacios como un argumento distinto y produce:

pushd: Too many arguments

SOLUCIÓN V9

La V9 no entrega a recon-all ninguna ruta de Windows con espacios.

Antes de ejecutar recon-all copia:

rT1.nii
a:
$HOME/freesurfer_stage/paciente6_Antes/rT1.nii

license.txt
a:
$HOME/freesurfer_stage/paciente6_Antes/license.txt

El procesamiento se realiza en:

$HOME/freesurfer_subjects
$HOME/freesurfer_exports/paciente6_Antes

Al finalizar, copia los NIfTI y reportes a la carpeta de resultados en D:.

La copia del T1 conserva exactamente los datos y el affine del archivo
original; solo cambia la ubicación en el sistema de archivos.

ARCHIVOS

- EJECUTAR_SEGMENTACION_TRONCO_WINDOWS_V9.cmd
- segmentar_tronco_todo_windows_V9.ps1
- validar_brainstem_substructures_T1_windows.py

EJECUCIÓN

Haz doble clic en:

EJECUTAR_SEGMENTACION_TRONCO_WINDOWS_V9.cmd

No se descargará ni reinstalará FreeSurfer.
