# Corrección v3.21.6 — Rescate de cortes bilaterales e integración del módulo tronco/CST

## 1. Problema corregido en tomografía

En la versión anterior el control de calidad verificaba que la serie DICOM tuviera evidencia de dos fémures. Sin embargo, en algunos estudios el corte elegido para el análisis de composición corporal podía caer en una zona donde se veía solo un fémur, aunque el volumen DICOM completo sí tuviera ambos miembros en otros cortes. Esto podía producir un resultado no válido: el algoritmo medía un solo miembro y luego esa información podía terminar duplicada o interpretada como si existieran ambos lados.

## 2. Nueva política v3.21.6

La versión v3.21.6 ya no aborta inmediatamente cuando encuentra un corte puntual con un solo fémur. Ahora aplica una estrategia de rescate anatómico:

1. Escanea la serie DICOM completa.
2. Detecta componentes óseos compatibles con fémur en cada corte.
3. Identifica cortes donde existen dos fémures plausibles, separados espacialmente y con relación de área compatible.
4. Si el corte medio seleccionado tiene un solo fémur, busca el corte bilateral más cercano.
5. Si no hay corte bilateral dentro de la ventana local, puede usar rescate global dentro del stack.
6. Solo se detiene si no existe ningún corte con dos fémures en toda la serie seleccionada.

Esto evita que un corte parcial genere métricas duplicadas para derecho e izquierdo.

## 3. Variables de entorno nuevas

```bat
set VCE_TAC_RESCATE_GLOBAL_DOS_FEMURES=1
set VCE_TAC_PERMITIR_POCOS_CORTES_DOS_FEMURES=1
```

- `VCE_TAC_RESCATE_GLOBAL_DOS_FEMURES=1`: si no hay corte bilateral cerca del corte medio, busca el corte bilateral más cercano en todo el volumen.
- `VCE_TAC_PERMITIR_POCOS_CORTES_DOS_FEMURES=1`: si hay pocos cortes bilaterales pero existe evidencia real de ambos miembros, continúa en modo rescate.

## 4. Cambios internos

Se añadieron funciones para:

- obtener centros femorales bilaterales por corte;
- asignar un centro diferente para cada lado de imagen;
- impedir que el miembro derecho e izquierdo usen el mismo fémur;
- ajustar automáticamente cortes medios no bilaterales;
- registrar el ajuste en `QC_Cortes_Medios_Ajustados_Dos_Femures.csv`.

Cuando ocurre un rescate aparece en consola una línea similar a:

```text
QC RESCATE: Derecho Z=120 tenía un solo fémur; se usará Z=135, corte con dos fémures.
```

## 5. Integración del módulo tronco/CST

Se integró el trabajo externo `modulo_CST_tronco_integracion_completa` dentro de:

```text
modulos/modulo_CST_tronco_integracion_completa/
```

El módulo conserva sus versiones finales:

- segmentación del tronco encefálico: `02_segmentacion_tronco/FINAL_V9/`;
- visualización/selección de vía corticoespinal: `03_cst_visualizacion/FINAL_V4_CORONAL/`.

También se parametrizó el script de CST para poder recibir rutas por variables de entorno de la suite:

```text
VCE_PROJECT_ROOT
VCE_PATIENT_ID
VCE_STAGE
VCE_T1_PATH
VCE_TRACTOGRAPHY_DIR
VCE_WHOLEBRAIN_TRK
VCE_BRAINSTEM_DIR
VCE_CST_OUTPUT_DIR
```

De esta forma el módulo ya no queda limitado solamente a `paciente 6 / Antes`, aunque esa sigue siendo la configuración de desarrollo probada.

## 6. Nuevo modo de la suite

Se agregó el modo:

```bat
--only cst_tronco
```

Este modo no corre dentro de `--only todo`, porque FreeSurfer puede tardar horas. Debe ejecutarse explícitamente cuando se tengan listos:

- `rT1.nii`;
- `wholebrain_T1.trk` propio;
- máscaras motoras M1/M2;
- licencia FreeSurfer;
- WSL2/Ubuntu para segmentación de tronco si aún no existen las máscaras.

## 7. Comandos principales

Tomografía avanzada corregida para pacientes 3 y 6:

```bat
run_reprocesar_tomografia_qc_dos_femures_pacientes_3_6_windows.bat
```

Módulo tronco/CST para paciente 6 Antes:

```bat
run_cst_tronco_paciente6_antes_windows.bat
```

También puede usarse:

```bat
COMANDOS_MANUALES_WINDOWS_V3_21_6.txt
```
