# Comandos para tractografía CST, vía corticoespinal RGB y ejecución desde Visual Studio Code

Esta actualización hace visible en el `main.py` el comando explícito para generar la tractografía y la vía corticoespinal en RGB.

## Sección recomendada

```bash
tractografia_cst_rgb
```

## Alias aceptados

```bash
cst_tronco
cst
tractografia
tractografia_rgb
tractografia_cst
tractografia_cst_rgb
cst_rgb
rgb_cst
via_corticoespinal
via_corticoespinal_rgb
via_cortico_espinal
via_cortico_espinal_rgb
```

Todos estos alias ejecutan internamente el mismo módulo validado: `cst_tronco`.

## Salidas esperadas

Dentro de:

```text
/home/humath/Escritorio/resultados/paciente X/Etapa/tractografia_propia
```

se esperan archivos como:

```text
via_cortico_espinal_completa.trk
via_cortico_espinal_completa_slicer.trk
via_cortico_espinal_completa_slicer.vtk
via_cortico_espinal_completa.nii
via_cortico_espinal_completa_rgb_direccion_T1.nii.gz
via_cortico_espinal_completa_colores_itksnap_T1.nii.gz
via_cortico_espinal_completa_mask.nii.gz
via_cortico_espinal_completa_density.nii.gz
via_cortico_espinal_completa_preview.png
```

## Probar paciente 3 Antes

```bash
cd /home/humath/Escritorio/plataforma_neurobiomecanica_multimodal
source .venv/bin/activate

export FREESURFER_HOME=/usr/local/freesurfer/8.2.0
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
export FS_LICENSE=/home/humath/Escritorio/license.txt

python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections tractografia_cst_rgb \
  --registro-com-corregir \
  --registro-com-ejes z \
  --no-suspend
```

## Probar paciente 3 Antes rehaciendo resultados

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes \
  --sections tractografia_cst_rgb \
  --force \
  --registro-com-corregir \
  --registro-com-ejes z \
  --no-suspend
```

## Correr todos los pacientes

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --all-patients \
  --stages Antes Despues \
  --sections tractografia_cst_rgb \
  --registro-com-corregir \
  --registro-com-ejes z \
  --no-suspend
```

## Correr CST RGB + 19 zonas

```bash
python main.py run \
  --project-root /home/humath/Escritorio \
  --patients 3 \
  --stages Antes Despues \
  --sections tractografia_cst_rgb zonas_correlacion \
  --registro-com-corregir \
  --registro-com-ejes z \
  --no-suspend
```

## Verificar secciones del main

```bash
python main.py --list-sections
python main.py examples
```

Debe aparecer la sección de tractografía CST RGB dentro del bloque `cst_tronco`, con los alias explícitos.
