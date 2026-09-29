# Comandos para actualizar documentación de funciones en tools

## 1. Copiar actualización al repositorio en Windows CMD

```bat
cd /d "E:\suite_integrada_vce_v3_21_21_linux_github_modelos_documentados\suite_integrada_vce_v3_21_21_linux_github_modelos_documentados"

set SRC=C:\Users\User\Downloads\actualizacion_tools_funciones_documentadas_v3_21_26\actualizacion_tools_funciones_documentadas_v3_21_26

if not exist tools mkdir tools
if not exist docs mkdir docs

xcopy /E /I /Y "%SRC%\tools" "tools"
copy /Y "%SRC%\docs\REPORTE_DOCUMENTACION_FUNCIONES_TOOLS.md" "docs\REPORTE_DOCUMENTACION_FUNCIONES_TOOLS.md"
copy /Y "%SRC%\COMANDOS_ACTUALIZAR_TOOLS_DOCUMENTADO.md" "docs\COMANDOS_ACTUALIZAR_TOOLS_DOCUMENTADO.md"

git status

git add tools docs\REPORTE_DOCUMENTACION_FUNCIONES_TOOLS.md docs\COMANDOS_ACTUALIZAR_TOOLS_DOCUMENTADO.md

git commit -m "Documenta entradas procesos y salidas de funciones tools"

git push -u origin main
```

Si el push se rechaza:

```bat
git push -u origin main --force-with-lease
```

## 2. Actualizar en Linux

```bash
cd /home/humath/Escritorio/plataforma_neurobiomecanica_multimodal

git pull origin main

source .venv/bin/activate

python -m py_compile tools/*.py
```

## 3. Verificar comentarios

```bash
grep -R "# Entrada:" -n tools | head -20
grep -R "# Proceso:" -n tools | head -20
grep -R "# Salida:" -n tools | head -20
```
