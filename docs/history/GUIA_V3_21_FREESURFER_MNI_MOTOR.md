# Suite VCE v3.21 — FreeSurfer/MNI motor real

Esta versión deja FreeSurfer como método principal para ubicar corteza motora.

## Qué hace

1. Prepara T1/BIDS mínimo desde los DICOM/NIfTI detectados.
2. Corre o reanuda `recon-all` si FreeSurfer está instalado y existe `license.txt`.
3. Extrae ROIs motoras desde FreeSurfer:
   - M1: `precentral`.
   - M2 aproximada: `caudalmiddlefrontal + superiorfrontal + paracentral`.
4. Genera máscaras `.nii.gz` y métricas: volumen, grosor, área, asimetría, estabilidad interpaciente y cambio Antes vs Después.
5. Inventaría transformaciones MNI si existen por fMRIPrep/ANTs.

## Instalación Linux completa

```bash
cd /home/humath/Escritorio
unzip suite_integrada_vce_v3_21_linux_freesurfer_mni_motor.zip
cd /home/humath/Escritorio/suite_integrada_vce_v3_21_linux_freesurfer_mni_motor
chmod +x *.sh
./instalar_dependencias_neuro_linux.sh
```

## FreeSurfer license.txt

Guarda la licencia en:

```bash
/home/humath/Escritorio/license.txt
```

Si FreeSurfer no está instalado, descarga el paquete Linux oficial y luego corre:

```bash
./instalar_freesurfer_nativo_linux.sh
```

Si descargaste un `.deb` en otra ruta:

```bash
FREESURFER_DEB=/ruta/freesurfer_xxx.deb ./instalar_freesurfer_nativo_linux.sh
```

## Correr FreeSurfer/MNI motor

```bash
cd /home/humath/Escritorio/suite_integrada_vce_v3_21_linux_freesurfer_mni_motor
source .venv/bin/activate
./run_freesurfer_mni_motor_linux.sh
```

Cuando pregunte pacientes:

```text
3, 6, 7, 9
```

## Solo integrar si FreeSurfer ya está calculado

```bash
python3 main.py   --project-root "/home/humath/Escritorio"   --data-root "/home/humath/Escritorio/datos"   --patients 3 6 7 9   --only derivados fs_mni_motor corteza_fix consolidado   --gpu off   --skip-stuck
```

## Salidas principales

```text
/home/humath/Escritorio/resultados/<paciente>/<Antes|Despues>/morfometria/freesurfer_mni_motor/
    metricas_corteza_motora_fs_mni.csv
    metricas_asimetria_corteza_motora.csv
    mni_transformaciones_detectadas.csv
    roi_motor_final/*.nii.gz

/home/humath/Escritorio/resultados/resumen_global_corteza_motora_freesurfer_mni.csv
/home/humath/Escritorio/resultados/qc_estabilidad_volumen_corteza_motora_freesurfer.csv
/home/humath/Escritorio/resultados/comparacion_antes_despues_volumen_corteza_motora_freesurfer.csv
```

## Nota MNI

MNI no se fuerza a ciegas. Para hacer MNI real se necesitan transformaciones MNI↔T1w, normalmente generadas por fMRIPrep/ANTs. Si existen, la suite las inventaría en `mni_transformaciones_detectadas.csv`. La medición final recomendada queda en espacio del paciente/FreeSurfer.
