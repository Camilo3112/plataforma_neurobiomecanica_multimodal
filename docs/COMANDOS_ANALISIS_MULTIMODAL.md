# Comandos para actualizar análisis multimodal integrado

## Windows CMD

```bat
cd /d "E:\suite_integrada_vce_v3_21_21_linux_github_modelos_documentados\suite_integrada_vce_v3_21_21_linux_github_modelos_documentados"

set SRC=C:\Users\User\Downloads\actualizacion_analisis_multimodal_v3_21_28\actualizacion_analisis_multimodal_v3_21_28

if not exist src mkdir src
if not exist docs mkdir docs
if not exist .vscode mkdir .vscode

copy /Y "%SRC%\main.py" "main.py"
copy /Y "%SRC%\src\analisis_multimodal_runner.py" "src\analisis_multimodal_runner.py"
copy /Y "%SRC%\docs\README_ANALISIS_MULTIMODAL.md" "docs\README_ANALISIS_MULTIMODAL.md"
copy /Y "%SRC%\docs\COMANDOS_ANALISIS_MULTIMODAL.md" "docs\COMANDOS_ANALISIS_MULTIMODAL.md"
copy /Y "%SRC%\.vscode\launch.json" ".vscode\launch.json"
copy /Y "%SRC%\.vscode\settings.json" ".vscode\settings.json"

git status

git add main.py src\analisis_multimodal_runner.py docs\README_ANALISIS_MULTIMODAL.md docs\COMANDOS_ANALISIS_MULTIMODAL.md
git add -f .vscode\launch.json .vscode\settings.json

git commit -m "Agrega analisis multimodal integrado por paciente"

git push -u origin main
```

Si el push se rechaza:

```bat
git push -u origin main --force-with-lease
```

## Linux / Visual Studio Code

```bash
cd /home/humath/Escritorio/plataforma_neurobiomecanica_multimodal

git pull origin main

source .venv/bin/activate

python -m pip install openpyxl pandas numpy nibabel

python main.py --list-sections
```

## Ejecutar análisis multimodal

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --all-patients \
  --stages Antes Despues \
  --sections analisis_multimodal
```

## Ejecutar solo una sección conceptual

```bash
python main.py run --project-root /home/humath/Escritorio --all-patients --stages Antes Despues --sections features_multimodales
python main.py run --project-root /home/humath/Escritorio --all-patients --stages Antes Despues --sections tabla_maestra_multimodal
python main.py run --project-root /home/humath/Escritorio --all-patients --stages Antes Despues --sections tractometria_cst
python main.py run --project-root /home/humath/Escritorio --all-patients --stages Antes Despues --sections radiomica_zonas
python main.py run --project-root /home/humath/Escritorio --all-patients --stages Antes Despues --sections integracion_multidimensional
python main.py run --project-root /home/humath/Escritorio --all-patients --stages Antes Despues --sections reporte_paciente_multimodal
```

## Ver salidas

```bash
ls -lh /home/humath/Escritorio/resultados/analisis
ls -lh /home/humath/Escritorio/resultados/analisis/matrices_pacientes
ls -lh /home/humath/Escritorio/resultados/analisis/reportes_paciente
```
