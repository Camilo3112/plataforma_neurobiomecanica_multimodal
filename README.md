# Suite Integrada VCE v3.21.21

Procesamiento multimodal biomédico para EMG, dinamometría, tomografía/TAC, neuroimagen estructural-funcional, FreeSurfer y tractografía de vía corticoespinal.

Esta versión está preparada para GitHub y usa un único punto de entrada en Python:

```bash
python main.py run --patients 3 --stages Antes --sections cst_tronco --force --registro-com-corregir --registro-com-ejes z --no-suspend
```

## Cambios principales de v3.21.21

- Se reemplazaron los lanzadores `.sh` por `main.py`.
- La segmentación FreeSurfer del tronco cerebral ahora se ejecuta desde `src/brainstem_freesurfer.py`.
- El flujo exige `recon-all.done`, `aseg.mgz` y `aparc+aseg.mgz` antes de aceptar mesencéfalo/puente/bulbo.
- La CST final se guarda fusionada como `via_cortico_espinal_completa.trk`, `via_cortico_espinal_completa.nii`, `via_cortico_espinal_completa_slicer.trk` y `via_cortico_espinal_completa_slicer.vtk`.
- Se añadieron cabeceras y documentación técnica en los módulos Python siguiendo el estilo de secciones delimitadas usado por el proyecto.

## Estructura

```text
.
├── main.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── config/
│   └── vce_config.example.json
├── docs/
│   ├── USO_MAIN.md
│   ├── MODELOS_FISICOS_MATEMATICOS.md
│   ├── ESTRUCTURA_GITHUB.md
│   ├── VISUALIZACION_SLICER_ITKSNAP.md
│   └── history/
├── src/
│   ├── brainstem_freesurfer.py
│   ├── cst_tronco_runner.py
│   ├── tomography.py
│   ├── neuroimage.py
│   └── ...
├── tools/
└── modulos/
```

## Instalación rápida

```bash
cd /home/humath/Escritorio/suite_integrada_vce_v3_21_21_linux_github_modelos_documentados
python3 main.py setup --project-root /home/humath/Escritorio/suite_integrada_vce_v3_21_21_linux_github_modelos_documentados
source .venv/bin/activate
```

## Verificar entorno

```bash
python main.py doctor --project-root /home/humath/Escritorio
```

## Probar solo paciente 3 Antes, sección CST

```bash
python main.py run   --project-root /home/humath/Escritorio   --patients 3   --stages Antes   --sections cst_tronco   --force   --registro-com-corregir   --registro-com-ejes z   --no-suspend
```

## Correr todos los pacientes detectados

```bash
python main.py run   --project-root /home/humath/Escritorio   --all-patients   --stages Antes Despues   --sections cst_tronco   --force   --registro-com-corregir   --registro-com-ejes z   --no-suspend
```

## Limpiar tractografías previas

Vista previa:

```bash
python main.py clean --project-root /home/humath/Escritorio --target tractografia
```

Borrado real:

```bash
python main.py clean --project-root /home/humath/Escritorio --target tractografia --yes
```

## Notas metodológicas

Los resultados de tractografía deben validarse visualmente. La salida `.trk` conserva las fibras; la salida `.nii` es una representación voxelizada/coloreada para superposición y visualización.


## Actualización v3.21.21

Esta versión refuerza la documentación técnica directamente dentro de los módulos Python.
Cada archivo incluye un encabezado profesional con el fundamento físico-matemático aplicado: señales, FFT, correlación, wavelets, geometría DICOM/NIfTI, afines, morfometría, FreeSurfer, segmentación de tronco encefálico y tractografía de la vía corticoespinal.

La ejecución sigue centralizada en `main.py` para correr pacientes, etapas y secciones específicas o el procesamiento completo.
