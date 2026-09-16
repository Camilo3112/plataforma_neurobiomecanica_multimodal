# Módulo de segmentación del tronco encefálico y visualización de la vía corticoespinal

## 1. Propósito de esta carpeta

Esta carpeta reúne el trabajo desarrollado para reconstruir y visualizar, en el espacio anatómico del `rT1.nii`, un corredor corticoespinal propio sin utilizar la tractografía ni las máscaras de vía corticoespinal generadas por el resonador.

El módulo fue desarrollado y depurado inicialmente para:

- **Paciente:** `paciente 6`
- **Momento:** `Antes`
- **Sistema principal:** Windows
- **Subsistema requerido para FreeSurfer:** WSL2 + Ubuntu 22.04
- **FreeSurfer utilizado:** 7.4.1
- **Referencia anatómica:** `rT1.nii`

El objetivo de esta carpeta no es reemplazar un pipeline completo de difusión, sino servir como un **módulo de postprocesamiento anatómico y de selección/visualización de CST** que pueda anexarse a una suite mayor.

---

# 2. Qué contiene el paquete

La estructura está organizada así:

```text
modulo_CST_tronco_integracion_completa/
│
├── README.md
├── MANIFEST.json
│
├── 01_diagnostico_datos/
│   ├── inspeccionar_datos_tractografia.py
│   ├── diagnostico.json
│   ├── series_summary.csv
│   ├── errores_lectura.csv
│   ├── reporte_resumen.txt
│   ├── atlas_motor_reporte.json
│   └── atlas_motor_candidatos.json
│
├── 02_segmentacion_tronco/
│   ├── FINAL_V9/
│   │   ├── EJECUTAR_SEGMENTACION_TRONCO_WINDOWS_V9.cmd
│   │   ├── segmentar_tronco_todo_windows_V9.ps1
│   │   ├── validar_brainstem_substructures_T1_windows.py
│   │   └── README_WINDOWS_V9.txt
│   │
│   └── soporte_WSL_DNS/
│       ├── REPARAR_DNS_WSL.cmd
│       └── REPARAR_DNS_WSL.ps1
│
├── 03_cst_visualizacion/
│   └── FINAL_V4_CORONAL/
│       ├── EJECUTAR_CST_COLORES_V4_CORONAL.cmd
│       ├── extraer_visualizar_cst_colores_V4_coronal.py
│       └── README_CST_COLORES_V4_CORONAL.txt
│
├── 04_validacion/
│   └── validar_brainstem_substructures_T1_windows.py
│
├── 05_configuracion_ejemplo/
│   ├── config_proyecto_ejemplo.json
│   ├── requirements_python.txt
│   └── CHECKLIST_INTEGRACION.md
│
├── 06_evidencia/
│   └── capturas de control visual
│
└── 99_historico/
    ├── segmentacion_tronco/
    ├── cst/
    └── tractografia_prototipos/
```

La carpeta `99_historico` conserva versiones que sirvieron durante el desarrollo, pero **no deben usarse como versión de producción**.

---

# 3. Qué versiones deben considerarse finales

## 3.1 Segmentación del tronco

La versión de trabajo que debe conservarse es:

```text
02_segmentacion_tronco/FINAL_V9/
```

Ejecutor:

```text
EJECUTAR_SEGMENTACION_TRONCO_WINDOWS_V9.cmd
```

La V9 solucionó los problemas encontrados durante el desarrollo:

1. instalación y detección de WSL;
2. conversión incorrecta de rutas Windows a WSL;
3. errores DNS de WSL;
4. descarga de FreeSurfer;
5. diferencias entre variables de PowerShell y Bash;
6. `FREESURFER_HOME` no definido;
7. `set -e` / `set -u` durante `SetUpFreeSurfer.sh`;
8. error `pushd: Too many arguments` causado por rutas Windows con espacios.

La solución final consiste en copiar temporalmente el T1 a una ruta Linux sin espacios antes de ejecutar `recon-all`.

---

## 3.2 Visualización de la CST

La versión final del filtro/visualización de CST incluida en esta carpeta es:

```text
03_cst_visualizacion/FINAL_V4_CORONAL/
```

Ejecutor:

```text
EJECUTAR_CST_COLORES_V4_CORONAL.cmd
```

Esta versión incorpora:

- separación izquierda/derecha;
- origen cortical M1 y, cuando está disponible, M2;
- waypoints de mesencéfalo, puente y bulbo;
- refinamiento **anterior/ventral** para el plano sagital;
- refinamiento **paramediano/medial** para el plano coronal;
- estimación de línea media usando el tronco encefálico;
- salida coloreada para ITK-SNAP;
- salida `.trk`;
- mapas de densidad;
- mapa RGB direccional;
- reporte JSON.

---

# 4. Entradas requeridas

El módulo necesita cinco grupos de entrada.

## 4.1 T1 de referencia

Ruta utilizada durante el desarrollo:

```text
D:\EAFIT\01-2026\proyecto\datos\paciente 6\Antes\
Resultados funcional\REFORMATEO\rT1.nii
```

Este archivo es la referencia espacial final. Todos los NIfTI de salida deben quedar alineados con él.

---

## 4.2 Tractografía whole-brain propia

Ruta utilizada:

```text
D:\EAFIT\01-2026\proyecto\resultados\paciente 6\Antes\
tractografia_propia\wholebrain_T1.trk
```

Requisito fundamental:

> Debe ser una tractografía generada por el pipeline propio y expresada en coordenadas compatibles con el T1.

En el caso probado, la comprobación espacial mostró aproximadamente **99.8 % de puntos de control dentro del volumen T1**, lo que indica compatibilidad geométrica muy alta.

El script no debe apuntar a la tractografía del resonador.

---

## 4.3 Máscaras corticales M1/M2

Se esperan dentro de:

```text
...\morfometria\interna\corregidas_corticales\
```

El código intenta localizar automáticamente:

- M1 izquierda;
- M1 derecha;
- M2 izquierda;
- M2 derecha.

M1 es necesaria. M2 puede incluirse como origen motor adicional.

---

## 4.4 Segmentación del tronco

La V9 genera las máscaras anatómicas finales mediante FreeSurfer.

Las estructuras utilizadas son:

```text
midbrain_T1.nii.gz
pons_T1.nii.gz
medulla_T1.nii.gz
scp_T1.nii.gz
brainstem_substructures_labelmap_T1.nii.gz
```

Las etiquetas FreeSurfer que se usaron durante el proceso fueron:

```text
173 = mesencéfalo
174 = puente
175 = bulbo raquídeo
178 = pedúnculo cerebeloso superior
```

La máscara consolidada se utiliza para control visual, mientras que los archivos individuales se utilizan como base para los waypoints.

---

## 4.5 Licencia de FreeSurfer

La instalación espera:

```text
D:\EAFIT\01-2026\proyecto\license.txt
```

La licencia no se incluye en este paquete.

---

# 5. Datos que NO están incluidos

Por tamaño, privacidad y reproducibilidad, esta carpeta no contiene:

- DICOM originales;
- `rT1.nii` del paciente;
- tractografía real del paciente;
- máscaras anatómicas generadas en el equipo local;
- archivo de licencia FreeSurfer;
- instalador `.deb` de FreeSurfer de 6.51 GB.

Durante el desarrollo el instalador quedó almacenado en:

```text
D:\EAFIT\01-2026\proyecto\instaladores\
freesurfer_ubuntu22-7.4.1_amd64.deb
```

No es necesario copiarlo dentro del repositorio del proyecto.

---

# 6. Dependencias

## Windows

Se utilizó Windows como sistema principal.

## WSL

La segmentación FreeSurfer se ejecuta dentro de:

```text
Ubuntu-22.04
```

Verificación:

```powershell
wsl -l -v
```

Debe aparecer `VERSION 2`.

## Python

Dependencias:

```text
numpy
scipy
nibabel
matplotlib
```

Instalación:

```bat
py -m pip install numpy scipy nibabel matplotlib
```

También se incluye:

```text
05_configuracion_ejemplo/requirements_python.txt
```

---

# 7. Flujo completo del módulo

El flujo recomendado es:

```text
DATOS DEL PACIENTE
      │
      ├── rT1.nii
      ├── M1/M2
      └── wholebrain_T1.trk
      │
      ▼
SEGMENTACIÓN DEL TRONCO
FreeSurfer BrainstemSubstructures
      │
      ├── mesencéfalo
      ├── puente
      ├── bulbo
      └── SCP
      │
      ▼
VALIDACIÓN EN ESPACIO T1
      │
      ▼
REFINAMIENTO DE WAYPOINTS
      │
      ├── anterior/ventral
      └── paramediano
      │
      ▼
FILTRADO DE STREAMLINES
      │
      ├── izquierda
      └── derecha
      │
      ▼
EXPORTACIÓN
      │
      ├── .trk
      ├── labelmap ITK-SNAP
      ├── densidad
      ├── RGB direccional
      ├── preview 3D
      └── reporte JSON
```

---

# 8. Ejecución paso a paso

## Paso 1 — Comprobar datos

Usar:

```text
01_diagnostico_datos/inspeccionar_datos_tractografia.py
```

Su función es documentar series y geometría antes de comenzar.

No modifica los datos.

---

## Paso 2 — Generar el tronco encefálico

Entrar a:

```text
02_segmentacion_tronco/FINAL_V9/
```

Ejecutar:

```text
EJECUTAR_SEGMENTACION_TRONCO_WINDOWS_V9.cmd
```

El script:

1. comprueba WSL;
2. comprueba FreeSurfer;
3. convierte las rutas Windows;
4. copia T1 y licencia a una ruta Linux sin espacios;
5. ejecuta/reanuda `recon-all`;
6. ejecuta BrainstemSubstructures;
7. convierte las estructuras al espacio T1;
8. extrae mesencéfalo, puente, bulbo y SCP;
9. copia las salidas a Windows;
10. ejecuta el validador.

---

# 9. Verificación visual del tronco

En ITK-SNAP:

## Imagen principal

```text
rT1.nii
```

## Segmentación

```text
brainstem_substructures_labelmap_T1.nii.gz
```

La segmentación debe:

- seguir los límites anatómicos del tronco;
- permanecer alineada en axial, coronal y sagital;
- distinguir mesencéfalo, puente y bulbo;
- no mostrar el rectángulo/sobredimensionamiento de las primeras aproximaciones.

Solo después de esta validación debe continuarse con la CST.

---

# 10. Extracción y visualización de la CST

Entrar a:

```text
03_cst_visualizacion/FINAL_V4_CORONAL/
```

Ejecutar:

```text
EJECUTAR_CST_COLORES_V4_CORONAL.cmd
```

La lógica general es:

```text
M1/M2
 ↓
mesencéfalo ventral
 ↓
puente ventral y paramediano
 ↓
bulbo anterior/paramediano
```

El código evalúa cada streamline en ambos sentidos porque un archivo `.trk` no garantiza que el orden de los puntos vaya de corteza a tronco.

---

# 11. Por qué se refinó el plano sagital

Las máscaras completas de mesencéfalo, puente y bulbo contienen regiones anteriores y posteriores.

Una fibra podía cumplir el criterio de “entrar al puente” pasando por una región demasiado posterior.

En ITK-SNAP, en la vista sagital:

```text
A = anterior
P = posterior
```

Por eso un aparente desplazamiento “a la derecha de la pantalla” puede equivaler a un desplazamiento posterior, no a un error izquierda/derecha.

La versión V3 añadió waypoints anteriores/ventrales.

---

# 12. Por qué se refinó el plano coronal

La primera separación izquierda/derecha utilizaba el punto medio entre los centroides de M1.

Esto puede introducir sesgo cuando las máscaras corticales tienen:

- tamaño diferente;
- forma diferente;
- centroides asimétricos.

La V4 estima la línea media utilizando directamente el centro del tronco encefálico por niveles axiales.

Luego conserva corredores paramedianos independientes para:

- mesencéfalo;
- puente;
- bulbo.

Así se corrige la **selección anatómica**, no las coordenadas.

> No se recomienda trasladar manualmente el tractograma para que “se vea centrado”.

Una traslación podría mejorar un plano y dañar la correspondencia en los otros dos.

---

# 13. Colores de salida

La convención utilizada es:

```text
Rojo    = CST izquierda
Azul    = CST derecha
Magenta = superposición voxelizada
```

El mapa RGB direccional utiliza otra convención:

```text
R = dirección izquierda-derecha
G = dirección anterior-posterior
B = dirección superior-inferior
```

No confundir el archivo de etiquetas hemisféricas con el mapa direccional.

---

# 14. Archivos principales de salida

La carpeta final de CST es:

```text
...\tractografia_propia\cst_visualizacion_ventral_medial\
```

Debe contener archivos equivalentes a:

```text
cst_ventral_medial_colores_itksnap_T1.nii.gz
cst_ventral_medial_izquierda_T1.trk
cst_ventral_medial_derecha_T1.trk
cst_ventral_medial_bilateral_T1.trk

cst_ventral_medial_densidad_izquierda_T1.nii.gz
cst_ventral_medial_densidad_derecha_T1.nii.gz
cst_ventral_medial_densidad_bilateral_T1.nii.gz

cst_ventral_medial_rgb_direccion_T1.nii.gz
cst_ventral_medial_vista_3D_colores.png
cst_ventral_medial_reporte.json
cst_labels_itksnap.txt
```

También se generan waypoints para control anatómico.

---

# 15. Interpretación de CST completa vs parcial

Durante la prueba del paciente 6 se observó:

```text
Streamlines revisadas: 51,088
Compatibilidad espacial con rT1: 99.8 %
CST completa hasta bulbo: 0 en ambos lados
```

Esto significa que el problema no era la alineación con el T1.

El tractograma `wholebrain_T1.trk` no contenía streamlines que cumplieran simultáneamente todo el recorrido hasta el bulbo.

Por eso se añadió una política explícita:

- si una fibra cumple hasta bulbo: `completa_hasta_bulbo`;
- si solo llega de forma válida hasta puente: `segmento_hasta_puente`.

Esto es importante para análisis grupales.

Nunca debe reportarse como “CST completa” un resultado cuyo JSON indique que solo llega al puente.

---

# 16. Cómo anexar el módulo a un trabajo más grande

Esta es la parte más importante para integración.

La recomendación es tratar este componente como un módulo independiente:

```text
pipeline_general/
│
├── adquisición
├── conversión
├── preprocesamiento_DWI
├── tractografía_wholebrain
├── morfometría
│
├── modulo_tronco_CST     <── ESTA CARPETA
│
├── métricas
├── correlaciones
└── reportes
```

No mezclar el código de CST con la lógica de adquisición o GUI.

El módulo debería recibir rutas y devolver rutas/resultados.

---

# 17. Contrato de entrada recomendado

En una arquitectura grande, el módulo debería recibir un objeto como:

```json
{
  "patient": "6",
  "timepoint": "Antes",
  "t1": ".../rT1.nii",
  "wholebrain_trk": ".../wholebrain_T1.trk",
  "m1_left": "...",
  "m1_right": "...",
  "m2_left": "...",
  "m2_right": "...",
  "license": ".../license.txt",
  "output_dir": "..."
}
```

Actualmente los scripts finales contienen algunas rutas configuradas para paciente 6.

Para integrar de forma limpia, el siguiente cambio arquitectónico recomendado es convertir esas constantes en:

- argumentos de línea de comandos;
- un archivo JSON;
- o parámetros de función.

No se recomienda realizar `str.replace()` sobre el archivo fuente por paciente.

---

# 18. Interfaz ideal para el proyecto mayor

Una API interna adecuada sería conceptualmente:

```python
brainstem_outputs = segment_brainstem(
    t1_path=t1,
    output_dir=brainstem_dir,
    freesurfer_license=license_path
)

cst_outputs = extract_cst(
    t1_path=t1,
    wholebrain_trk=wholebrain,
    motor_masks=motor_masks,
    brainstem_masks=brainstem_outputs,
    output_dir=cst_dir
)
```

Y devolver:

```python
{
    "left_trk": "...",
    "right_trk": "...",
    "bilateral_trk": "...",
    "labelmap": "...",
    "density": "...",
    "rgb": "...",
    "report": "..."
}
```

Esto permite que otra parte de la suite consuma resultados sin conocer los detalles internos de FreeSurfer o ITK-SNAP.

---

# 19. Parametrización para varios pacientes

Actualmente:

```text
paciente 6 / Antes
```

Para un estudio mayor se recomienda una convención:

```text
datos/
  paciente 001/
    Antes/
    Evolucion_1/
    Evolucion_2/
    Final/
```

Y en resultados:

```text
resultados/
  paciente 001/
    Antes/
      tractografia_propia/
        brainstem_substructures_validated/
        cst_visualizacion_ventral_medial/
```

El módulo debe recibir:

```text
patient_id
timepoint
```

y construir las rutas.

Nunca codificar manualmente un paciente dentro de la lógica anatómica.

---

# 20. Checkpoints

El procesamiento de FreeSurfer puede tardar horas.

Por eso el pipeline mayor debe ser **reanudadable**.

Estados sugeridos:

```text
00_input_ok.json
10_reconall_complete.json
20_brainstem_complete.json
30_brainstem_validated.json
40_cst_complete.json
50_cst_validated.json
```

Cada checkpoint debería contener:

```json
{
  "patient": "6",
  "timepoint": "Antes",
  "timestamp": "...",
  "status": "ok",
  "inputs": {},
  "outputs": {},
  "parameters": {},
  "software_versions": {}
}
```

No iniciar nuevamente `recon-all` si ya está completo.

---

# 21. Registro de logs

Para un estudio grande, guardar:

```text
logs/
  paciente_006_Antes_brainstem.log
  paciente_006_Antes_cst.log
```

Cada log debería incluir:

- fecha;
- versión del script;
- entradas;
- parámetros;
- número de streamlines;
- porcentaje de compatibilidad espacial;
- conteo CST izquierda/derecha;
- estado completa/parcial;
- errores.

---

# 22. Identificación de versiones

No usar nombres ambiguos como:

```text
resultado_final_final2.nii.gz
```

Mantener:

```text
module_version
algorithm_version
patient
timepoint
```

Por ejemplo:

```text
cst_v4_paciente006_Antes_reporte.json
```

Para publicación o tesis, congelar una sola versión antes de ejecutar la cohorte completa.

---

# 23. Validaciones automáticas recomendadas

Antes de aceptar un resultado:

## Geometría

Comprobar:

```text
shape
affine
orientation
voxel size
```

## Tractograma

Comprobar:

- porcentaje de puntos dentro del T1;
- longitud de streamlines;
- hemisferio predominante;
- intersección con waypoints.

## Máscaras

Comprobar:

- no vacías;
- dentro del T1;
- volumen razonable;
- centroide;
- continuidad anatómica.

## Salida CST

Comprobar:

- izquierda/derecha no intercambiadas;
- colores correctos;
- ausencia de saltos espaciales;
- correspondencia axial/sagital/coronal.

---

# 24. Integración con una interfaz gráfica

Si la suite principal es PySide6, Streamlit u otra GUI:

la GUI **no debería ejecutar directamente toda la lógica anatómica en el hilo principal**.

Patrón recomendado:

```text
GUI
 ↓
job manager
 ↓
subprocess
 ↓
script de tronco/CST
 ↓
JSON de estado
 ↓
GUI actualiza progreso
```

El proceso puede tardar horas, especialmente `recon-all`.

Debe ejecutarse en:

- proceso separado;
- worker;
- cola de tareas.

No bloquear la interfaz.

---

# 25. Progreso y estado

Para integrarlo a una GUI mayor, leer:

```text
recon-all.log
```

y presentar estados de alto nivel:

```text
Preparando T1
Recon-all
Segmentando tronco
Registrando al T1
Validando
Filtrando CST
Generando NIfTI
Generando reporte
Completado
```

No intentar estimar un porcentaje exacto de `recon-all` únicamente por tiempo.

---

# 26. Manejo de errores

Errores encontrados durante el desarrollo y su significado:

## `WSL_E_DISTRO_NOT_FOUND`

Ubuntu no estaba instalado.

## `Temporary failure in name resolution`

WSL no resolvía DNS.

## `$FS_DEB` o `.Replace unexpected`

Interacción incorrecta PowerShell/Bash.

## `FREESURFER_HOME: unbound variable`

El entorno FreeSurfer no estaba definido antes de hacer `source`.

## `pushd: Too many arguments`

Ruta del T1 con espacios.

La V9 evita este último copiando el archivo a una ruta interna de WSL.

---

# 27. Uso de los archivos históricos

La carpeta:

```text
99_historico/
```

sirve para trazabilidad del desarrollo.

No debe importarse desde el pipeline principal.

En particular:

- las primeras aproximaciones de tronco basadas en máscaras geométricas no deben reutilizarse;
- la aproximación Harvard-Oxford no delimitó suficientemente mesencéfalo/puente/bulbo para el objetivo;
- las versiones V3–V8 contienen correcciones parciales previas a V9.

Para producción usar únicamente las carpetas marcadas `FINAL`.

---

# 28. Papel de los scripts antiguos de tractografía

En:

```text
99_historico/tractografia_prototipos/
```

se conservan scripts usados durante la exploración.

No deben considerarse automáticamente el pipeline DWI final de un estudio clínico o académico grande.

El módulo de esta carpeta debe integrarse **después** de que el pipeline mayor haya generado un `wholebrain_T1.trk` confiable.

---

# 29. Advertencia metodológica para el DWI

La reconstrucción whole-brain usada durante el desarrollo permitió estudiar y depurar el módulo anatómico.

Sin embargo, para una cohorte definitiva se recomienda que el pipeline de difusión anterior a este módulo incorpore, de forma consistente:

- denoising;
- corrección de Gibbs;
- corrección de movimiento/eddy;
- corrección de distorsión EPI/susceptibilidad según los datos disponibles;
- rotación correcta de b-vectors;
- registro DWI ↔ T1;
- control de calidad.

Este paquete no sustituye esa etapa.

---

# 30. Regla fundamental sobre datos del resonador

La tractografía generada por el resonador puede utilizarse posteriormente como:

```text
comparación visual externa
```

pero no debe participar en:

- semillas;
- máscaras;
- waypoints;
- selección de streamlines;
- ajuste de parámetros;
- validación de las fibras propias.

Esto evita contaminación metodológica.

---

# 31. Recomendación para métricas posteriores

Una vez generado el CST, el pipeline mayor puede calcular por lado:

```text
n_streamlines
longitud_media
longitud_mediana
volumen_voxelizado
densidad
FA_media
MD_media
AD_media
RD_media
```

si las métricas DWI están correctamente registradas al mismo espacio.

Estas métricas deben llevar explícitamente:

```text
lado
paciente
momento
estado_CST
```

donde `estado_CST` diferencia completa vs parcial.

---

# 32. Integración con comparaciones longitudinales

Para:

```text
Antes
Evolución
Después
```

no comparar solamente el número bruto de streamlines.

Guardar y comparar:

- geometría de la máscara;
- volumen;
- densidad normalizada;
- métricas microestructurales;
- estado de completitud;
- parámetros de tractografía.

Los mismos parámetros deben aplicarse a todos los momentos del paciente.

---

# 33. Reproducibilidad

Por paciente guardar un JSON con:

```text
versión FreeSurfer
versión Python
versión nibabel
parámetros
rutas
hash de entradas
hash del script
fecha
```

Idealmente el pipeline mayor debe calcular SHA-256 de:

```text
rT1.nii
wholebrain_T1.trk
máscaras M1/M2
scripts
```

Esto permite saber exactamente qué produjo cada resultado.

---

# 34. Visualización en ITK-SNAP

Abrir:

## Main Image

```text
rT1.nii
```

## Segmentation

```text
cst_ventral_medial_colores_itksnap_T1.nii.gz
```

Luego importar:

```text
cst_labels_itksnap.txt
```

Revisar siempre:

```text
axial
sagital
coronal
```

No aprobar una vía únicamente porque se vea correcta en un plano.

---

# 35. Qué hacer si vuelve a verse desplazada

No modificar el affine manualmente como primera opción.

Orden de diagnóstico:

1. revisar `rT1`;
2. revisar máscaras de tronco;
3. revisar waypoints;
4. revisar `.trk`;
5. revisar affine/header;
6. determinar si el problema es realmente geometría o selección de fibras.

Si el desplazamiento aparece solo en una región anatómica específica, refinar el waypoint de esa región.

Si todo el tractograma está desplazado de forma rígida en los tres planos, entonces sí corresponde revisar el registro DWI↔T1.

---

# 36. Posible mejora futura: cápsula interna posterior

La V4 refina el tronco.

Si en pacientes futuros el trayecto se desvía principalmente entre:

```text
corteza
↓
corona radiata
↓
tálamo
```

el siguiente waypoint anatómico a incorporar debería ser la **cápsula interna posterior**.

No debe añadirse únicamente para “hacer que se vea bonito”; debe definirse mediante un atlas/segmentación reproducible.

---

# 37. Ejecución por lote

En un estudio grande:

```python
for patient in patients:
    for timepoint in timepoints:
        validate_inputs()
        run_brainstem_if_missing()
        validate_brainstem()
        run_cst()
        validate_cst()
        save_summary()
```

No lanzar varios `recon-all` simultáneamente sin controlar RAM, CPU y espacio.

---

# 38. Recomendación de arquitectura de resultados

```text
resultados/
└── paciente 006/
    └── Antes/
        └── tractografia_propia/
            ├── wholebrain_T1.trk
            ├── brainstem_substructures_validated/
            │   ├── midbrain_T1.nii.gz
            │   ├── pons_T1.nii.gz
            │   ├── medulla_T1.nii.gz
            │   └── ...
            │
            └── cst_visualizacion_ventral_medial/
                ├── *.trk
                ├── *.nii.gz
                ├── *.json
                └── *.png
```

Mantener las entradas separadas de las salidas.

---

# 39. Resumen de integración

Para anexar este módulo a un proyecto mayor:

1. conservar únicamente V9 y V4 como versiones activas;
2. convertir las rutas hardcodeadas en parámetros;
3. usar JSON de configuración o argumentos CLI;
4. ejecutar FreeSurfer como tarea externa;
5. mantener checkpoints;
6. validar automáticamente geometría;
7. validar visualmente una muestra;
8. guardar JSON de resultados;
9. distinguir CST completa de parcial;
10. no utilizar resultados del resonador para construir la propia vía.

---

# 40. Estado actual del módulo

El módulo ha sido probado en el caso de desarrollo hasta obtener:

- tronco encefálico correctamente delimitado sobre el T1;
- mesencéfalo, puente y bulbo separados;
- alineación correcta del tronco;
- tractograma con alta compatibilidad espacial con T1;
- CST coloreada izquierda/derecha;
- refinamiento sagital anterior/ventral;
- refinamiento coronal paramediano;
- exportación compatible con ITK-SNAP.

El principal límite observado en el tractograma de prueba es que las streamlines disponibles no atravesaban el bulbo, por lo que el resultado debe clasificarse como segmento corticoespinal hasta puente cuando corresponda.

---

# 41. Archivos que deben quedar bajo control de versiones

Sí incluir:

```text
.py
.ps1
.cmd
.md
.json de configuración
requirements
```

No incluir en Git:

```text
DICOM
T1 clínicos
TRK grandes
FreeSurfer subjects
*.mgz grandes
instalador .deb de 6.51 GB
license.txt
datos identificables
```

Usar `.gitignore`.

---

# 42. Ejemplo de `.gitignore`

```gitignore
# Pacientes
datos/
resultados/

# FreeSurfer
freesurfer_subjects/
*.mgz

# Tractografía y NIfTI
*.trk
*.nii
*.nii.gz

# Instaladores/licencias
*.deb
license.txt

# Python
__pycache__/
*.pyc

# Logs temporales
*.log
```

---

# 43. Punto de entrada recomendado para la suite

En una aplicación más grande, el usuario no debería tener que buscar los scripts manualmente.

La aplicación principal debería tener una acción conceptual:

```text
Procesar CST
```

que internamente:

```text
1. valida inputs
2. comprueba tronco
3. ejecuta V9 si falta
4. valida tronco
5. ejecuta CST V4
6. lee JSON
7. registra métricas
8. muestra resultado
```

La GUI puede abrir el NIfTI final o mostrar la imagen preview, pero la lógica anatómica debe permanecer desacoplada.

---

# 44. Conclusión

Este paquete debe entenderse como un módulo reutilizable de:

```text
segmentación anatómica de tronco
+
selección anatómica de CST
+
exportación para análisis/visualización
```

Su integración correcta depende de mantener tres principios:

1. **misma referencia espacial T1** para todas las salidas;
2. **criterios anatómicos reproducibles** en lugar de desplazamientos visuales manuales;
3. **separación estricta de la tractografía del resonador** frente a la reconstrucción propia.

Las carpetas `FINAL_V9` y `FINAL_V4_CORONAL` son las que deben migrarse a la rama principal del proyecto. Todo lo demás se conserva para trazabilidad y diagnóstico.
