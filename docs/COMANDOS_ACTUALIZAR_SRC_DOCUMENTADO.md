# Comandos para copiar y subir la documentación de funciones en `src/`

## 1. CMD de Windows

Ruta de destino del repositorio:

```bat
cd /d "E:\suite_integrada_vce_v3_21_21_linux_github_modelos_documentados\suite_integrada_vce_v3_21_21_linux_github_modelos_documentados"
```

Ruta de origen de esta actualización, ajusta si la descomprimiste en otra carpeta:

```bat
set SRC=C:\Users\User\Downloads\actualizacion_src_funciones_documentadas_v3_21_25\actualizacion_src_funciones_documentadas_v3_21_25
```

Copiar archivos:

```bat
if not exist src mkdir src
if not exist docs mkdir docs

xcopy /E /I /Y "%SRC%\src" "src"
copy /Y "%SRC%\docs\REPORTE_DOCUMENTACION_FUNCIONES_SRC.md" "docs\REPORTE_DOCUMENTACION_FUNCIONES_SRC.md"
copy /Y "%SRC%\COMANDOS_ACTUALIZAR_SRC_DOCUMENTADO.md" "docs\COMANDOS_ACTUALIZAR_SRC_DOCUMENTADO.md"
```

Verificar:

```bat
git status
```

Subir a GitHub:

```bat
git add src docs\REPORTE_DOCUMENTACION_FUNCIONES_SRC.md docs\COMANDOS_ACTUALIZAR_SRC_DOCUMENTADO.md

git commit -m "Documenta entradas procesos y salidas de funciones src"

git push -u origin main
```

Si el push se rechaza:

```bat
git push -u origin main --force-with-lease
```

## 2. Linux — actualizar repositorio

```bash
cd /home/humath/Escritorio/plataforma_neurobiomecanica_multimodal

git pull origin main

source .venv/bin/activate

python -m py_compile src/*.py

python main.py --list-sections
```

## 3. Probar desde Visual Studio Code

1. Abrir la carpeta del repo en Visual Studio Code.
2. Seleccionar intérprete `.venv/bin/python`.
3. Abrir `main.py`.
4. Ejecutar desde `Run and Debug` o con el botón de Python.
