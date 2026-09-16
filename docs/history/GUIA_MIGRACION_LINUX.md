# Guía rápida Linux - suite_integrada_vce v3.13

Ruta de datos configurada:

```bash
/home/humath/Escritorio/datos
```

La carpeta raíz del proyecto queda como:

```bash
/home/humath/Escritorio
```

Por eso los resultados salen en:

```bash
/home/humath/Escritorio/resultados
```

## Instalación base

```bash
cd /home/humath/Escritorio/suite_integrada_vce_v3_13_linux
chmod +x *.sh
./setup_linux.sh
```

Si `setup_linux.sh` agrega tu usuario al grupo `docker`, cierra sesión y vuelve a entrar antes de correr Docker.

## GPU opcional NVIDIA

```bash
./instalar_gpu_linux_nvidia.sh
```

## Correr pacientes 3, 6 y 7

Interactivo:

```bash
./run_todo_interactivo_linux.sh
```

Directo:

```bash
./run_todo_pacientes_3_6_7_linux.sh
```

## Derivados reales

Primero guarda la licencia FreeSurfer aquí:

```bash
/home/humath/Escritorio/license.txt
```

Luego prepara BIDS y scripts Linux:

```bash
./run_preparar_derivados_reales_linux.sh
```

Después ejecuta los `.sh` que quedan en:

```bash
/home/humath/Escritorio/resultados/derivados_externos
```

Orden recomendado:

```bash
cd /home/humath/Escritorio/resultados/derivados_externos
./run_02_freesurfer_recon_all_docker.sh
./run_03_fmriprep_anat_only_docker.sh
./run_06_integrar_derivados_en_suite.sh
```

## Consolidado final por sujeto

Al terminar el flujo completo, revisa:

```text
/home/humath/Escritorio/resultados/paciente 3/consolidado/
/home/humath/Escritorio/resultados/paciente 6/consolidado/
/home/humath/Escritorio/resultados/paciente 7/consolidado/
/home/humath/Escritorio/resultados/sano/consolidado/
```

El archivo más útil para análisis posterior es:

```text
<sujeto>_todos_los_datos_largo.csv
```

Este archivo trae una fila por métrica y una explicación de la métrica en `metric_explanation`.

Si ya corriste los análisis y solo quieres reconstruir los CSV consolidados:

```bash
./run_consolidar_metricas_linux.sh
```
