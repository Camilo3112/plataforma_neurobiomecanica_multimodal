# Comandos de terminal para Visual Studio Code

Guía rápida para ejecutar la plataforma neurobiomecánica multimodal desde la terminal integrada de Visual Studio Code en Linux.

---

## 1. Abrir el proyecto en Visual Studio Code

```bash
code /home/humath/Escritorio/plataforma_neurobiomecanica_multimodal-main
```

Si ya estás dentro de Visual Studio Code:

```text
File > Open Folder
/home/humath/Escritorio/plataforma_neurobiomecanica_multimodal-main
```

Después abre la terminal integrada:

```text
Terminal > New Terminal
```

---

## 2. Entrar a la carpeta del proyecto

```bash
cd /home/humath/Escritorio/plataforma_neurobiomecanica_multimodal-main
```

---

## 3. Crear entorno virtual

Solo se hace una vez. Si `.venv` ya existe, no pasa nada.

```bash
python3 -m venv .venv
```

---

## 4. Activar entorno virtual

```bash
source .venv/bin/activate
```

La terminal debe mostrar algo parecido a:

```text
(.venv) humath@equipo:~/Escritorio/plataforma_neurobiomecanica_multimodal-main$
```

Verifica que Visual esté usando el Python correcto:

```bash
which python
```

Debe salir:

```text
/home/humath/Escritorio/plataforma_neurobiomecanica_multimodal-main/.venv/bin/python
```

Si sale `/usr/bin/python3`, no estás usando el entorno virtual correcto.

---

## 5. Instalar dependencias

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install openpyxl
```

Verifica `openpyxl`:

```bash
python -c "import openpyxl; print('openpyxl OK:', openpyxl.__version__)"
```

---

## 6. Configurar FreeSurfer en la terminal

```bash
export FREESURFER_HOME=/usr/local/freesurfer/8.2.0
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
export FS_LICENSE=/home/humath/Escritorio/license.txt
export VCE_FREESURFER_SUBJECTS_DIR=/home/humath/freesurfer_subjects
```

Verifica FreeSurfer:

```bash
which recon-all
which mri_info
recon-all -version
```

Verifica licencia:

```bash
ls -lh /home/humath/Escritorio/license.txt
```

---

## 7. Diagnóstico del entorno

```bash
python main.py doctor --project-root /home/humath/Escritorio
```

---

## 8. Ver todas las secciones disponibles

```bash
python main.py --list-sections
```

También puedes imprimir comandos rápidos desde el propio `main.py`:

```bash
python main.py examples
```

---

## 9. Modo interactivo

```bash
python main.py
```

Cuando pregunte:

```text
Pacientes a correr, ejemplo 3,6,7,9 [3]:
```

Puedes escribir:

```text
3
```

Cuando pregunte:

```text
Etapas [Antes]:
```

Puedes escribir:

```text
Antes
```

Cuando pregunte:

```text
Secciones, ejemplo cst_tronco, tomografia, fs_mni_motor o todo [cst_tronco]:
```

Puedes escribir:

```text
cst_tronco
```

---

## 10. Correr paciente 3 Antes: vía corticoespinal y tronco encefálico

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections cst_tronco \
  --registro-com-corregir \
  --registro-com-ejes z \
  --no-suspend
```

---

## 11. Correr paciente 3 Antes forzando reproceso

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections cst_tronco \
  --force \
  --registro-com-corregir \
  --registro-com-ejes z \
  --no-suspend
```

---

## 12. Si el registro sigue desplazado, probar corrección XYZ

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections cst_tronco \
  --force \
  --registro-com-corregir \
  --registro-com-ejes xyz \
  --no-suspend
```

---

## 13. Limpiar tractografías viejas

Vista previa, no borra nada:

```bash
python main.py clean \
  --project-root /home/humath/Escritorio \
  --target tractografia
```

Borrado real:

```bash
python main.py clean \
  --project-root /home/humath/Escritorio \
  --target tractografia \
  --yes
```

---

## 14. Correr todos los pacientes solo con CST/tronco

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --all-patients \
  --stages Antes Despues \
  --sections cst_tronco \
  --registro-com-corregir \
  --registro-com-ejes z \
  --no-suspend
```

---

## 15. Correr todos los pacientes forzando reproceso CST/tronco

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --all-patients \
  --stages Antes Despues \
  --sections cst_tronco \
  --force \
  --registro-com-corregir \
  --registro-com-ejes z \
  --no-suspend
```

---

# Comandos por sección

## A. Diagnóstico

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections diagnostico
```

## B. Biomecánica, EMG y dinamometría

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections biomecanica
```

## C. Tomografía TAC base

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections tomografia
```

## D. Tomografía avanzada

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections tomografia_avanzada
```

## E. Mapas derivados

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections mapas
```

## F. Resonancias

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections resonancias
```

## G. Preparar derivados externos

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections preparar_derivados
```

## H. Preparar derivados reales

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections preparar_derivados_reales
```

## I. Morfometría interna

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections morfometria_interna
```

## J. Corrección de ROI cortical

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections corteza_fix
```

## K. Integración de derivados externos

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections derivados
```

## L. FreeSurfer, MNI y corteza motora

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections fs_mni_motor \
  --no-suspend
```

## M. CST, mesencéfalo, puente y bulbo

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections cst_tronco \
  --registro-com-corregir \
  --registro-com-ejes z \
  --no-suspend
```

## N. Acople estructura-función

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections estructura_funcion
```

## O. Correlaciones

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections correlaciones
```

## P. Comparación Antes vs Después

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes Despues \
  --sections comparacion
```

## Q. Consolidado final

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes Despues \
  --sections consolidado
```

## R. Todo el pipeline

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes Despues \
  --sections todo \
  --registro-com-corregir \
  --registro-com-ejes z \
  --no-suspend
```

---

# Verificar salidas principales CST

```bash
OUT="/home/humath/Escritorio/resultados/paciente 3/Antes/tractografia_propia"

ls -lh "$OUT"/via_cortico_espinal_completa*
find "$OUT/brainstem_substructures_validated" -type f | sort
cat "/home/humath/Escritorio/resultados/resumen_modulo_cst_tronco.csv"
```

Archivos esperados:

```text
via_cortico_espinal_completa.trk
via_cortico_espinal_completa.nii
via_cortico_espinal_completa_slicer.trk
via_cortico_espinal_completa_slicer.vtk
via_cortico_espinal_completa_colores_itksnap_T1.nii.gz
via_cortico_espinal_completa_mask.nii.gz
via_cortico_espinal_completa_density.nii.gz
```

---

# Subir cambios a GitHub desde Linux

Después de reemplazar `main.py` y copiar este archivo dentro de `docs/COMANDOS_TERMINAL_VISUAL.md`:

```bash
cd /home/humath/Escritorio/plataforma_neurobiomecanica_multimodal-main

git status

git add main.py docs/COMANDOS_TERMINAL_VISUAL.md

git commit -m "Mejora menu interactivo y agrega comandos de terminal"

git push -u origin main
```

Si el repositorio quedó en conflicto por un rebase anterior:

```bash
git rebase --abort

git status

git add main.py docs/COMANDOS_TERMINAL_VISUAL.md

git commit -m "Mejora menu interactivo y agrega comandos de terminal"

git push -u origin main --force-with-lease
```

---

# Subir cambios a GitHub desde Windows CMD

Si estás usando la copia de Windows:

```bat
cd /d "E:\suite_integrada_vce_v3_21_21_linux_github_modelos_documentados\suite_integrada_vce_v3_21_21_linux_github_modelos_documentados"

git status

git add main.py docs\COMANDOS_TERMINAL_VISUAL.md

git commit -m "Mejora menu interactivo y agrega comandos de terminal"

git push -u origin main
```

Si hay conflictos de rebase:

```bat
git rebase --abort

git status

git add main.py docs\COMANDOS_TERMINAL_VISUAL.md

git commit -m "Mejora menu interactivo y agrega comandos de terminal"

git push -u origin main --force-with-lease
```
