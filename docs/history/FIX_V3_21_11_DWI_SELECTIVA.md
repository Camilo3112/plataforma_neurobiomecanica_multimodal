# FIX v3.21.11 — Conversión selectiva de DWI para tractografía CST

La tractografía propia no necesita convertir toda la carpeta `RESONANCIA`. Esta versión escanea cabeceras DICOM, detecta carpetas candidatas a DWI/DTI crudo, convierte solo esas candidatas y se detiene cuando obtiene una DWI 4D válida con `.bval` y `.bvec`. El T1 anatómico se busca aparte como `rAnatomico.nii`/`rT1.nii` dentro de `ResultadosFuncional`.

Archivos de diagnóstico:

```text
tractografia_propia/01_nifti_convertidos/dwi_dicom_folder_candidates.json
tractografia_propia/01_nifti_convertidos/dwi_nifti_candidates.json
```

Para forzar la carpeta DWI exacta:

```bash
export VCE_DWI_DICOM_DIR="/home/humath/Escritorio/datos/paciente 3/Antes/RESONANCIA/CARPETA_DWI"
```
