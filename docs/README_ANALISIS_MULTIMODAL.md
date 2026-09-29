# Análisis multimodal integrado VCE v3.21.28

Esta actualización agrega una capa de análisis que no reemplaza los módulos existentes. Lee la carpeta `resultados/`, detecta pacientes y etapas disponibles, extrae variables numéricas ya generadas por la suite y crea una carpeta nueva:

```text
resultados/analisis/
```

## Secciones nuevas del `main.py`

```text
analisis_multimodal
features_multimodales
tabla_maestra_multimodal
tractometria_cst
radiomica_zonas
integracion_multidimensional
reporte_paciente_multimodal
```

Todas estas secciones usan el módulo:

```text
src/analisis_multimodal_runner.py
```

## Modelo de la matriz multimodal

Cada paciente se representa como una matriz:

```text
variable multimodal × etapa
```

Cada variable se define como:

```text
modalidad | región | parámetro | unidad
```

La idea física es integrar información de varios niveles del sistema neuromotor:

```text
corteza motora / zonas cerebrales
vía corticoespinal
músculo y composición grasa-músculo
activación EMG
fuerza, torque, trabajo y potencia
```

Así se puede revisar si un cambio periférico en músculo o fuerza está acompañado por cambios centrales en resonancia, zonas o tractografía CST.

## Archivos que genera

Dentro de `resultados/analisis/` genera:

```text
features_multimodales.xlsx
tabla_maestra_multimodal.xlsx
tractometria_cst.xlsx
radiomica_zonas.xlsx
integracion_multidimensional.xlsx
delthas_variables_multimodal.xlsx
resumen_analisis_multimodal.json
README_ANALISIS_MULTIMODAL_RESULTADOS.md
matrices_pacientes/
reportes_paciente/
```

## Excel de deltas solicitado

El archivo:

```text
delthas_variables_multimodal.xlsx
```

contiene cuatro hojas:

1. `entre_pacientes_antes`: paciente 3 contra pacientes 6, 7 y 9 en Antes.
2. `entre_pacientes_despues`: paciente 3 contra pacientes 6, 7 y 9 en Después.
3. `paciente3_vs_sano`: paciente 3 contra control sano, si existe en resultados.
4. `cross_vs_sano`: pacientes cross education contra control sano, si existe en resultados.

## Variables físicas que intenta recuperar

### EMG

- RMS: activación muscular efectiva.
- iEMG: carga eléctrica acumulada.
- Frecuencia mediana/media: fatiga o desplazamiento espectral.
- Energía CWT: distribución multiescala de activación.
- Coactivación y simetría: coordinación y balance entre músculos/lados.

### Dinamometría

- Pico torque: fuerza máxima.
- Trabajo total: producción mecánica acumulada.
- Potencia: rapidez de producción mecánica.
- LSI: simetría de miembros.
- Fatiga: caída de rendimiento.

### Tomografía

- Área/volumen muscular: masa estructural.
- Grasa intramuscular/subcutánea: infiltración y composición.
- Relación grasa/músculo: calidad muscular.
- HU muscular: densidad tisular.
- Textura: heterogeneidad del tejido.

### Resonancias y 19 zonas

- Volumen por zona: morfometría regional.
- Intensidad media/mediana: señal anatómica regional.
- Correlación Antes/Después: estabilidad local.
- Mapa de baja correlación: posible región de cambio.
- Entropía/textura: heterogeneidad anatómica.

### Tractografía CST

- Número de streamlines: cantidad reconstruida de fibras.
- Densidad CST: concentración de trayectorias.
- Volumen no cero de mapas CST: extensión espacial de la vía.
- RGB/orientación: dirección dominante de la trayectoria.
- Waypoints mesencéfalo-puente-bulbo: validez anatómica de la vía.

## Cómo entender la matriz

Una fila representa una variable física concreta, por ejemplo:

```text
Tractografía CST | via_cortico_espinal_completa_density | nii_media_no_cero | intensidad
Tomografía | muslo | relacion_grasa_musculo | ratio
EMG | vasto_medial_derecho | RMS | uV
```

Las columnas `Antes`, `Despues`, `delta_despues_menos_antes` y `delta_pct` permiten revisar cambio longitudinal. Si una variable existe en un paciente pero no en otro, queda como vacío para no inventar datos.

## Uso recomendado

Primero correr los módulos que producen resultados:

```bash
python main.py run --project-root /home/humath/Escritorio --patients 3 --stages Antes --sections cst_tronco
```

Luego correr el análisis multimodal:

```bash
python main.py run --project-root /home/humath/Escritorio --all-patients --stages Antes Despues --sections analisis_multimodal
```
