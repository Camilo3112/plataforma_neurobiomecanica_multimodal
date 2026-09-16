# Comandos para integrar 19 zonas, correlación local y ejecución desde Visual Studio Code

Este documento deja los comandos para añadir al repositorio los archivos nuevos:

- `main.py`
- `src/zonas_correlacion_runner.py`
- `.vscode/launch.json`
- `.vscode/settings.json`
- `docs/COMANDOS_ZONAS_CORRELACION_VISUAL.md`

La nueva sección del pipeline se llama:

```bash
zonas_correlacion
```

Alias aceptados:

```bash
zonas
19_zonas
regiones
correlacion_zonas
```

---

## 1. Copiar archivos en Windows antes de subir a GitHub

Ubícate en la carpeta interna del proyecto:

```bat
cd /d "E:\suite_integrada_vce_v3_21_21_linux_github_modelos_documentados\suite_integrada_vce_v3_21_21_linux_github_modelos_documentados"
```

Si los archivos quedaron en Descargas, copia así:

```bat
copy /Y "%USERPROFILE%\Downloads\main.py" "main.py"

mkdir src
copy /Y "%USERPROFILE%\Downloads\zonas_correlacion_runner.py" "src\zonas_correlacion_runner.py"

mkdir docs
copy /Y "%USERPROFILE%\Downloads\COMANDOS_ZONAS_CORRELACION_VISUAL.md" "docs\COMANDOS_ZONAS_CORRELACION_VISUAL.md"

mkdir .vscode
copy /Y "%USERPROFILE%\Downloads\launch.json" ".vscode\launch.json"
copy /Y "%USERPROFILE%\Downloads\settings.json" ".vscode\settings.json"
```

Si alguno ya existe, Windows puede mostrar que ya está creado; no es problema.

---

## 2. Subir cambios a GitHub desde CMD

```bat
cd /d "E:\suite_integrada_vce_v3_21_21_linux_github_modelos_documentados\suite_integrada_vce_v3_21_21_linux_github_modelos_documentados"

git status

git add main.py src\zonas_correlacion_runner.py docs\COMANDOS_ZONAS_CORRELACION_VISUAL.md .vscode\launch.json .vscode\settings.json

git status

git commit -m "Agrega seccion de 19 zonas y correlacion local"

git push -u origin main
```

Si GitHub rechaza el push por historial, usar:

```bat
git push -u origin main --force-with-lease
```

---

## 3. Actualizar en Linux sin borrar datos ni resultados

Abrir Visual Studio Code con el repo clonado:

```bash
code /home/humath/Escritorio/plataforma_neurobiomecanica_multimodal
```

En la terminal integrada de Visual:

```bash
cd /home/humath/Escritorio/plataforma_neurobiomecanica_multimodal

git status

git pull origin main
```

Verificar archivos:

```bash
ls main.py
ls src/zonas_correlacion_runner.py
ls .vscode/launch.json
ls docs/COMANDOS_ZONAS_CORRELACION_VISUAL.md
```

---

## 4. Preparar entorno en Linux

```bash
cd /home/humath/Escritorio/plataforma_neurobiomecanica_multimodal

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install openpyxl SimpleITK nibabel scipy pandas numpy
```

Verificar intérprete correcto:

```bash
which python
python -c "import openpyxl, SimpleITK, nibabel, scipy; print('dependencias OK')"
```

El `which python` debe mostrar una ruta dentro de `.venv`.

---

## 5. Listar secciones desde el main

```bash
python main.py --list-sections
```

Debe aparecer la nueva sección:

```text
zonas_correlacion
```

---

## 6. Correr solo delimitación de 19 zonas y correlación de paciente 3

Este comando usa Antes y Después porque la correlación necesita ambos momentos:

```bash
export FREESURFER_HOME=/usr/local/freesurfer/8.2.0
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
export FS_LICENSE=/home/humath/Escritorio/license.txt
export VCE_FREESURFER_SUBJECTS_DIR=/home/humath/freesurfer_subjects

python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes Despues \
  --sections zonas_correlacion \
  --no-suspend
```

---

## 7. Correr CST + 19 zonas + correlación para paciente 3

```bash
export FREESURFER_HOME=/usr/local/freesurfer/8.2.0
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
export FS_LICENSE=/home/humath/Escritorio/license.txt
export VCE_FREESURFER_SUBJECTS_DIR=/home/humath/freesurfer_subjects

python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes Despues \
  --sections cst_tronco zonas_correlacion \
  --registro-com-corregir \
  --registro-com-ejes z \
  --no-suspend
```

---

## 8. Correr 19 zonas y correlación para todos los pacientes

```bash
export FREESURFER_HOME=/usr/local/freesurfer/8.2.0
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
export FS_LICENSE=/home/humath/Escritorio/license.txt
export VCE_FREESURFER_SUBJECTS_DIR=/home/humath/freesurfer_subjects

python main.py run \
  --project-root /home/humath/Escritorio \
  --all-patients \
  --stages Antes Despues \
  --sections zonas_correlacion \
  --no-suspend
```

---

## 9. Correr desde el botón de Visual Studio Code

Después de copiar `.vscode/launch.json`, abrir Visual Studio Code en el repo:

```bash
code /home/humath/Escritorio/plataforma_neurobiomecanica_multimodal
```

Luego en Visual:

```text
Run and Debug
```

Escoger una opción:

```text
VCE | Menú interactivo
VCE | Listar secciones
VCE | CST paciente 3 Antes
VCE | 19 zonas + correlación paciente 3
VCE | CST + 19 zonas paciente 3
VCE | 19 zonas + correlación todos
```

Y dar clic en el botón verde de correr.

---

## 10. Verificar salidas generadas

Paciente 3:

```bash
OUT=/home/humath/Escritorio/resultados/paciente\ 3

find "$OUT" -path "*regiones_adicionales_T1*" -maxdepth 6 -type f | sort | head -80

ls -lh "$OUT/correlacion_antes_despues_19_zonas"
```

Archivos esperados:

```text
tractografia_propia/regiones_adicionales_T1/regiones_adicionales_multietiqueta_T1.nii
tractografia_propia/regiones_adicionales_T1/regiones_adicionales_labels_itksnap.txt
tractografia_propia/regiones_adicionales_T1/volumen_regiones_adicionales_por_hemisferio.csv
tractografia_propia/regiones_adicionales_T1/regiones_adicionales_reporte.json
correlacion_antes_despues_19_zonas/correlacion_local_3planos_T1.nii.gz
correlacion_antes_despues_19_zonas/baja_correlacion_score_T1.nii.gz
correlacion_antes_despues_19_zonas/baja_correlacion_mask_T1.nii.gz
correlacion_antes_despues_19_zonas/baja_correlacion_zonas_labelmap_T1.nii.gz
correlacion_antes_despues_19_zonas/baja_correlacion_zonas_labels_itksnap.txt
```

Excel global:

```bash
ls -lh /home/humath/Escritorio/resultados/resumen_volumenes_19_zonas.xlsx
ls -lh /home/humath/Escritorio/resultados/resumen_correlacion_19_zonas.xlsx
ls -lh /home/humath/Escritorio/resultados/resumen_modulo_19_zonas_correlacion.csv
```

---

## 11. Visualización en ITK-SNAP

Para ver las 19 zonas:

```text
Main Image: rAnatomico.nii
Segmentation: regiones_adicionales_multietiqueta_T1.nii
Label descriptions: regiones_adicionales_labels_itksnap.txt
```

Para ver baja correlación:

```text
Main Image: rAnatomico Antes
Segmentation: baja_correlacion_zonas_labelmap_T1.nii.gz
Label descriptions: baja_correlacion_zonas_labels_itksnap.txt
Overlay continuo: baja_correlacion_score_T1.nii.gz
```
