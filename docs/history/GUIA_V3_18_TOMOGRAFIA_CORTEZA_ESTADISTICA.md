# Suite v3.18 Linux avanzada

## Cambios principales

1. Tomografía avanzada integrada desde el código nuevo: landmarks trocantérico/tibial, corte medio, composición corporal, grasa subcutánea, grasa intramuscular y relación grasa/músculo.
2. Corrección automática de ROI cortical: remuestreo al Brain00mm, detección del cerebro, construcción de capa cortical externa, búsqueda de traslación y QC con preview PNG + JSON.
3. Notebook estadístico avanzado: normalización global, z-score, robust z-score, ratio contra sano, volumen de corteza motora como métrica de estabilidad del protocolo, relación grasa/músculo y análisis inter-paciente.

## Correr solo tomografía

```bash
cd /home/humath/Escritorio/suite_integrada_vce_v3_18_linux_avanzado
source .venv/bin/activate
./run_solo_tomografia_linux.sh
```

Esto corre la tomografía clásica y la avanzada. Si solo quieres la avanzada:

```bash
./run_solo_tomografia_avanzada_linux.sh
```

## Correr solo corrección de corteza

```bash
./run_corregir_corteza_linux.sh
```

## Correr tomografía + corrección cortical + consolidado

```bash
./run_tomografia_corteza_estadistica_linux.sh
```

## Abrir notebook estadístico avanzado

```bash
./run_jupyter_estadistica_avanzada_linux.sh
```

## Salidas nuevas

- `resultados/<paciente>/<Antes|Despues>/tomografia/avanzada_tejido_adiposo_landmarks/`
- `resultados/<paciente>/<Antes|Despues>/morfometria/interna/corregidas_corticales/`
- `resultados/analisis_estadistico_avanzado_linux/`

## Nota sobre atlas

El ajuste automático por capa cortical corrige una máscara que ya existe cuando el principal error es geométrico. El atlas es otra capa: se registra el atlas completo al cerebro del paciente y luego se extraen regiones motoras atlas-guiadas. El algoritmo de corrección no reemplaza el registro atlas-paciente; lo complementa como control geométrico local.
