# Linux sin Docker y sin MATLAB

Ruta del proyecto:

```bash
/home/humath/Escritorio
```

Ruta de datos:

```bash
/home/humath/Escritorio/datos
```

## Decisión de esta versión

Esta versión **no depende de MATLAB ni CAT12**.

El flujo queda así:

1. **FreeSurfer nativo en Linux** para morfometría anatómica real desde T1.
2. **FastSurfer opcional** si quieres una alternativa rápida basada en deep learning.
3. **dcm2niix + Python interno** para preparar NIfTI/BIDS mínimo.
4. **Mapas funcionales existentes** en `ResultadosFuncional` para estructura-función.
5. **Morfometría interna** como respaldo si FreeSurfer todavía no está listo.

## Qué ya no usa

- MATLAB
- SPM12
- CAT12
- Docker

## Requisitos base

```bash
sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip python3-dev build-essential \
  git unzip p7zip-full dcm2niix bc wget curl \
  libglu1-mesa libxmu6 libxmu-dev libxt6 libxext6 libxrender1 libgomp1 \
  tcsh perl
```

## Licencia FreeSurfer

FreeSurfer necesita `license.txt`.

Guárdala aquí:

```bash
/home/humath/Escritorio/license.txt
```

También puedes copiarla dentro de FreeSurfer:

```bash
sudo cp /home/humath/Escritorio/license.txt $FREESURFER_HOME/license.txt
```

## Instalar FreeSurfer nativo

Instala FreeSurfer Linux desde su página oficial. Luego debes tener algo como:

```bash
/usr/local/freesurfer/8.0.0/SetUpFreeSurfer.sh
```

o:

```bash
/usr/local/freesurfer/7.4.1/SetUpFreeSurfer.sh
```

Activa FreeSurfer manualmente para probar:

```bash
source /usr/local/freesurfer/8.0.0/SetUpFreeSurfer.sh
export FS_LICENSE=/home/humath/Escritorio/license.txt
recon-all -version
```

## Correr derivados reales sin Docker/MATLAB

```bash
cd /home/humath/Escritorio/suite_integrada_vce_v3_15_linux_sin_matlab
chmod +x *.sh
./setup_linux.sh
./setup_sin_docker_linux.sh
./run_derivados_nativos_linux.sh
```

Cuando pregunte pacientes:

```text
3, 6, 7
```

## Después correr todo

```bash
./run_todo_interactivo_linux.sh
```

## Salidas esperadas

FreeSurfer real:

```bash
/home/humath/Escritorio/resultados/derivados_externos/freesurfer
```

Integración en la suite:

```bash
/home/humath/Escritorio/resultados/<paciente>/<Antes|Despues>/morfometria
/home/humath/Escritorio/resultados/<paciente>/<Antes|Despues>/estructura_funcion
/home/humath/Escritorio/resultados/<paciente>/comparacion/antes_vs_despues
```

## Nota

Si FreeSurfer aún no está instalado o falta `license.txt`, la suite continúa con la morfometría interna, pero para el resultado anatómico más defendible debes correr `recon-all`.
