# Análisis de los códigos originales integrados

Revisé los módulos que venían repetidos o dispersos en:

- `convoluciones/`
- `fmi/`
- `fMIR/`
- `mapas/`
- `tomografias y demas/`

## Lo que se conservó y se integró

### 1. EMG y dinamometría
De los códigos `codigo.py`, `EMG_DINAMOMETRIA.py` y `signal_analysis_suite.py` se conservaron las ideas principales:

- correlación cruzada 1D por FFT;
- autocorrelación;
- alineación temporal;
- comparación paciente vs sano;
- gráficos científicos por fase;
- análisis de simetría y acople electromecánico.

En v3 se agregó además:

- coherencia espectral;
- phase locking value;
- Hurst R/S;
- entropía multiescala;
- bandpower por rangos de frecuencia;
- escalogramas CWT-like y matrices `.npy`.

### 2. Wavelets
De `cwt_engine.py`, `pipeline.py`, `wavelets.py`, `singularity.py`, `hurst.py` y `wavelet_3d.py` se tomó la lógica de análisis multiescala.

En v3 se usa una implementación estable basada en filtros LoG/Gaussian Laplace para evitar fallos por versiones diferentes de SciPy y mantener el pipeline robusto en Windows.

### 3. TAC / tomografía
De `tomografia.py` se integró:

- lectura DICOM robusta;
- eliminación de duplicados espaciales;
- detección automática de rango Z;
- segmentación polar alrededor del fémur;
- búsqueda de fascia con respuesta multiescala;
- medición de volúmenes y áreas transversales;
- exportación NIfTI.

### 4. fMRI y resonancia
De `fmri_dicom_correlation.py`, `fMRI.py` y `fmri_cristian.py` se conservaron las ideas de:

- autocorrelación temporal;
- PSD;
- correlación tipo task/seed cuando existe volumen compatible;
- análisis por FFT;
- mapas y topografías.

En v3 se procesan las series reales detectadas en el manifest: T1, fMRI motor, t-maps, DTI/FA/ADC/RD/AD/TRACE, tractografía y tracto corticoespinal.

### 5. Organización y reanudación
El problema principal de los códigos originales era que había varios `main.py`, rutas fijas, duplicación de módulos y salidas dispersas.

En v3 queda:

- un solo `main.py`;
- rutas configurables por `--project-root`;
- estructura de resultados automática;
- checkpoint por tarea;
- log de eventos;
- opción `--force` para reprocesar;
- opción `--skip-stuck` para continuar si una tarea quedó pegada.

## Actualización v3.4 - Aceleración GPU opcional

Se agregó un backend opcional basado en CuPy/CUDA para descargar trabajo pesado de CPU/RAM hacia GPU en las operaciones volumétricas:

- normalización robusta de volúmenes;
- remuestreo/interpolación de imágenes para comparar paciente vs sano;
- energía wavelet/LoG multiescala en mapas funcionales y resonancias;
- correlación global volumen-a-volumen;
- correlaciones de TAC, mapas y resonancias contra sano.

La implementación se hizo de forma segura: si CuPy o CUDA no están disponibles, el código vuelve automáticamente a CPU. La segmentación anatómica por ray-casting de TAC, lectura DICOM, exportación NIfTI y generación de gráficos se mantienen en CPU porque son I/O, trazado o lógica anatómica iterativa.


## v3.5 · comparación Antes vs Después

Se agregó un módulo `src/longitudinal.py` para que el análisis no solo compare paciente vs sano, sino también la evolución del mismo paciente entre `Antes` y `Despues`. La salida queda en `resultados/<paciente>/comparacion/antes_vs_despues`.

La lógica reutiliza las rutinas de correlación volumen-volumen, corte-corte, máscaras y señales, pero cambia el sentido del análisis a **Después - Antes**. Esto aplica a mapas, tomografía, resonancias, EMG y dinamometría.


## v3.8 - Mejora basada en revisión estructura-función

Se agregó módulo `src/structure_function.py`, evitando inferir volumen cortical desde BOLD solo. El pipeline ahora cruza mapas funcionales con T1/cáscara cortical y con máscaras M1/M2 derivadas de AD + wavelet/atlas-prior. También agrega ALFF/fALFF, tSNR, DVARS y GCOR aproximado para stacks fMRI.


## v3.12
- Lanzador interactivo para seleccionar pacientes antes de iniciar.
- Acepta entradas tipo `3, 6, 7` y las normaliza a carpetas `paciente 3`, `paciente 6`, `paciente 7`.
- Flujo maestro que prepara derivados reales y corre el análisis completo.
