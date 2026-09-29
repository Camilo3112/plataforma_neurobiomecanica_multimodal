---
abstract: |
  Este documento revisa técnicas adicionales de procesamiento digital de
  señales, neuroimagen computacional, fusión multimodal y análisis
  multidimensional que pueden complementar una plataforma que integra
  EMG, dinamometría, TAC, resonancia estructural, difusión, tractografía
  de la vía corticoespinal, segmentación de 19 zonas y comparación
  longitudinal antes/después. La propuesta central es añadir una capa de
  integración por paciente que no reemplace los módulos actuales, sino
  que consolide sus salidas en un espacio común de características,
  regiones, modalidades y tiempo. A partir de esta representación se
  pueden aplicar modelos de factores latentes, descomposición
  conjunta/individual, fusión por grafos, tractometría, radiomía,
  análisis tensorial y métricas de discordancia entre modalidades.
author:
- Documento de trabajo metodológico
bibliography: referencias(1).bib
nocite: "@\\*"
title: Revisión técnica para un análisis multidimensional multimodal en
  la Plataforma Neurobiomecánica VCE
---

# Planteamiento del problema

La plataforma actual ya puede producir mediciones parciales por
modalidad: actividad mioeléctrica, fuerza, morfometría, segmentaciones,
mapas de correlación, tractografía y métricas anatómicas. El siguiente
avance metodológico no consiste únicamente en extraer más variables
aisladas, sino en construir un *modelo multidimensional por paciente*
que agrupe todos los estudios disponibles y permita responder preguntas
integradas:

- ¿Qué regiones cambian de forma coherente entre neuroimagen,
  tractografía y mediciones periféricas?

- ¿Qué modalidad explica más variación en el cambio antes/después?

- ¿Existen zonas con deterioro anatómico pero compensación funcional?

- ¿La vía corticoespinal muestra cambios compatibles con la evolución de
  fuerza, EMG o masa muscular?

- ¿Qué variables parecen redundantes y cuáles aportan información
  complementaria?

El objetivo recomendado es crear una capa llamada, por ejemplo,
`integracion_multidimensional`, que reciba las salidas de los módulos ya
existentes y genere: una matriz de características, un tensor
paciente–tiempo–modalidad–región, un reporte de contribuciones por
modalidad, mapas de cambio integrado y tablas Excel interpretables.

# Representación matemática base

Para agrupar los estudios de un paciente, se propone organizar la
información en un tensor multimodal:

$$\mathcal{X}_{p,t,m,r,k},$$

donde $p$ representa el paciente, $t$ el momento de evaluación
(Antes/Después), $m$ la modalidad (EMG, fuerza, TAC, T1, DWI, CST, zonas
anatómicas), $r$ la región anatómica o canal funcional, y $k$ la
característica extraída. Para análisis tabular, este tensor también
puede desplegarse como una matriz:

$$\mathbf{X} \in \mathbb{R}^{n \times d},$$

donde cada fila representa una unidad de análisis. Dependiendo del
objetivo, una fila puede ser un paciente, un paciente-momento, una
región, una región-momento o un segmento de tracto. Las columnas
corresponden a características normalizadas.

Para comparación longitudinal se recomienda usar cambios relativos y
diferencias normalizadas:

$$\Delta x = x_{Despues} - x_{Antes},$$

$$\Delta_{\%}x = 100 \cdot \frac{x_{Despues} - x_{Antes}}{|x_{Antes}| + \epsilon},$$

y asimetrías laterales:

$$A_{LR} = 100 \cdot \frac{x_{Derecha} - x_{Izquierda}}{\frac{1}{2}(x_{Derecha}+x_{Izquierda}) + \epsilon}.$$

Estas formulaciones permiten integrar mediciones con unidades distintas,
siempre que se normalicen por modalidad antes de fusionarlas.

# Banco de características por modalidad

## EMG y señales neuromusculares

Además de RMS, iEMG, frecuencia media/mediana y correlación, se
recomiendan técnicas de análisis no estacionario y no lineal:

- **Transformada wavelet continua y coherencia wavelet:** permite
  estudiar cambios de energía tiempo–frecuencia y sincronización entre
  músculos o entre EMG y fuerza. Es útil cuando la activación muscular
  no es estacionaria o cuando los eventos tienen duración variable.

- **Wavelet packet y energía por bandas:** genera una descomposición más
  fina que la STFT y permite crear firmas espectrales por músculo, lado
  y etapa.

- **Hilbert–Huang Transform / EMD:** separa modos intrínsecos de
  oscilación y calcula frecuencia instantánea. Es útil en señales no
  lineales y no estacionarias.

- **Entropía muestral, entropía aproximada y complejidad multiescala:**
  cuantifican regularidad o variabilidad de activación neuromuscular.

- **Análisis de recurrencia:** permite identificar patrones repetitivos,
  estabilidad y transiciones en señales de movimiento o fuerza.

- **Acople EMG–fuerza:** puede modelarse mediante coherencia,
  correlación cruzada, fase relativa, retardos y modelos de regresión
  con desfase temporal.

Una característica importante sería construir un índice de acople
electromecánico:

$$C_{EMG,F}(f,t) = \frac{|S_{EMG,F}(f,t)|^2}{S_{EMG}(f,t)S_F(f,t)},$$

donde $S_{EMG,F}$ es el espectro cruzado local y $S_{EMG}$, $S_F$ son
espectros de potencia. Esta métrica puede resumirse por bandas
fisiológicamente relevantes.

## Dinamometría y cinética

En fuerza se recomienda extraer no solo picos, sino dinámica de
generación de fuerza:

- tasa de desarrollo de fuerza, pendiente máxima y tiempo al pico;

- impulso mecánico $I=\int F(t)dt$;

- simetría izquierda/derecha;

- estabilidad de la curva de fuerza mediante varianza local, entropía y
  área bajo la curva;

- acople con EMG usando retardos óptimos y coherencia tiempo–frecuencia.

## TAC, morfometría y radiomía muscular

Para TAC y tejidos musculares/adiposos se recomienda añadir radiomía y
textura. Además de volumen y área, pueden extraerse:

- distribución de unidades Hounsfield por músculo;

- percentiles de intensidad;

- textura GLCM: contraste, homogeneidad, energía y correlación;

- GLRLM/GLSZM: longitud de rachas, tamaño de zonas y heterogeneidad;

- forma 3D: elongación, compacidad, esfericidad, superficie y relación
  superficie/volumen;

- infiltración grasa intramuscular y subcutánea.

Estas variables ayudan a detectar cambios que no se observan solo con
volumen. Dos regiones con igual volumen pueden diferir en calidad
tisular, textura o infiltración grasa.

## T1, FreeSurfer y 19 zonas anatómicas

Las 19 zonas pueden convertirse en una tabla longitudinal por región:

$$\mathbf{Z}_{r} = [V_r, I_r, \sigma_r, P_{10,r}, P_{90,r}, Corr_r, \Delta V_r, \Delta I_r],$$

donde $V_r$ es volumen, $I_r$ intensidad media, $\sigma_r$ desviación de
intensidad, $P_{10}$ y $P_{90}$ percentiles, $Corr_r$ la correlación
local antes/después y $\Delta$ los cambios longitudinales.

Se recomienda agregar un índice de discordancia regional:

$$D_r = w_1 |z(\Delta V_r)| + w_2 |z(1-Corr_r)| + w_3 |z(\Delta I_r)|,$$

que ordena zonas con mayor cambio multidimensional.

## DWI, tractografía y vía corticoespinal

La vía corticoespinal no debería resumirse solamente con número total de
streamlines. Se recomienda añadir tractometría o perfil a lo largo del
tracto:

$$T(s) = [FA(s), MD(s), RD(s), AD(s), Density(s), Curvature(s)], \quad s \in [0,1],$$

donde $s$ parametriza la trayectoria del tracto desde corteza motora
hasta tronco encefálico. Esto permite ubicar si el cambio ocurre cerca
de corteza, cápsula interna, mesencéfalo, puente o bulbo.

Salidas recomendadas:

- perfil de FA/MD/densidad en 100 nodos a lo largo de la CST;

- densidad de streamlines por segmento anatómico;

- curvatura, longitud media y dispersión espacial;

- proporción de fibras que atraviesan mesencéfalo, puente y bulbo;

- índice CST integrado.

# Modelos de integración multidimensional

## PCA multibloque

La PCA multibloque separa variables por modalidad y permite identificar
ejes globales de variación sin que una modalidad domine solo por tener
más columnas. Cada bloque se normaliza y se proyecta hacia componentes
latentes compartidos. Puede usarse como primer modelo exploratorio.

## JIVE: variación conjunta e individual

JIVE descompone la información en componentes compartidos entre bloques
y componentes específicos de cada modalidad:

$$\mathbf{X}^{(m)} = \mathbf{J}^{(m)} + \mathbf{A}^{(m)} + \mathbf{E}^{(m)},$$

donde $\mathbf{J}^{(m)}$ es la variación conjunta, $\mathbf{A}^{(m)}$ la
variación individual de la modalidad y $\mathbf{E}^{(m)}$ el ruido
residual. Para VCE, esto permitiría distinguir si un cambio
antes/después es global del sistema motor o aparece solo en neuroimagen,
solo en músculo o solo en fuerza.

## Factorización latente tipo MOFA

Los modelos de factores latentes permiten integrar modalidades con
diferente número de variables y datos faltantes:

$$\mathbf{X}^{(m)} \approx \mathbf{W}^{(m)}\mathbf{Z} + \mathbf{E}^{(m)},$$

donde $\mathbf{Z}$ son factores latentes comunes del paciente y
$\mathbf{W}^{(m)}$ son cargas por modalidad. En el proyecto, un factor
podría representar recuperación motora periférica, otro integridad
corticoespinal y otro cambio morfológico muscular.

## sPLS, CCA y DIABLO

Los modelos PLS/CCA buscan variables de distintas modalidades que
covarían. Su versión dispersa permite seleccionar variables relevantes.
En VCE puede usarse para encontrar relaciones como:

- cambios de FA en CST asociados a cambios de fuerza;

- cambios de EMG asociados a volumen muscular;

- zonas cerebrales cuya correlación local se relaciona con simetría de
  fuerza.

## Similarity Network Fusion

Cuando haya varios pacientes o varias unidades de análisis, se puede
construir una red de similitud por modalidad y fusionarlas. Cada
modalidad produce una matriz $S^{(m)}$ de similitud entre pacientes,
regiones o evaluaciones. SNF genera una red integrada que resume la
estructura común entre modalidades.

## Multiple Kernel Learning

MKL permite representar cada modalidad mediante un kernel y aprender
pesos de importancia:

$$K = \sum_{m=1}^{M} \beta_m K_m, \quad \beta_m \geq 0.$$

En VCE, $K_m$ puede venir de EMG, fuerza, TAC, T1, DWI, CST o zonas. El
peso $\beta_m$ sirve como medida de contribución relativa de cada
modalidad para una tarea de clasificación o predicción.

## Análisis tensorial

Si se organiza la información como paciente $\times$ tiempo $\times$
región $\times$ modalidad, se puede aplicar descomposición CP/Tucker:

$$\mathcal{X} \approx \sum_{q=1}^{Q} \lambda_q \mathbf{a}_q \circ \mathbf{b}_q \circ \mathbf{c}_q \circ \mathbf{d}_q,$$

lo que permite detectar patrones compartidos de cambio: por ejemplo, un
factor que activa “Después”, regiones motoras, CST y fuerza.

## Grafos multimodales por paciente

Un enfoque especialmente útil para la plataforma es crear un grafo por
paciente. Los nodos serían regiones anatómicas, músculos, lados
corporales, segmentos de CST y variables clínicas. Las aristas
representarían correlación, coherencia, conectividad anatómica, acople
EMG–fuerza o proximidad anatómica.

Métricas útiles:

- centralidad de regiones;

- comunidades funcionales/anatómicas;

- nodos con mayor cambio antes/después;

- aristas que aparecen o desaparecen;

- distancia de red entre corteza motora, CST y músculo.

# Propuesta concreta para la plataforma VCE

## Nivel 1: extracción de características

Crear una sección futura:

    --sections features_multimodales

Salidas:

- `features_emg.xlsx`

- `features_dinamometria.xlsx`

- `features_tac.xlsx`

- `features_zonas_cerebrales.xlsx`

- `features_cst_tractometria.xlsx`

## Nivel 2: tabla maestra

Crear una tabla maestra:

    resultados/integracion_multidimensional/features_master.xlsx

Columnas recomendadas:

- paciente;

- etapa;

- modalidad;

- región;

- lado;

- variable;

- valor;

- valor normalizado;

- cambio absoluto;

- cambio porcentual;

- QC/fuente.

## Nivel 3: modelos de integración

Crear una sección:

    --sections integracion_multidimensional

Modelos iniciales recomendados:

1.  PCA multibloque.

2.  JIVE o aproximación conjunta/individual.

3.  MOFA-like si hay datos faltantes.

4.  sPLS/CCA para pares de modalidades.

5.  grafo multimodal paciente-específico.

## Nivel 4: reporte clínico/investigativo

Salidas recomendadas:

- `reporte_integrado_paciente.xlsx`

- `ranking_zonas_cambio.xlsx`

- `contribucion_modalidades.xlsx`

- `grafo_multimodal.graphml`

- `mapa_cambio_integrado.nii.gz`

- `dashboard_paciente.html`

# Qué técnicas son más viables según el tamaño actual de datos

Con pocos pacientes y pocas evaluaciones, no conviene empezar por
modelos profundos ni clasificadores complejos. El orden recomendado es:

1.  **Muy viable ahora:** tabla maestra, normalización, índices de
    cambio, tractometría, radiomía, PCA multibloque, correlaciones
    cruzadas y grafos descriptivos.

2.  **Viable con cuidado:** CCA/sPLS, JIVE, MOFA-like, SNF sobre
    regiones o pacientes-momento, análisis tensorial simple.

3.  **Para más adelante:** MKL supervisado, GNN, deep learning
    multimodal, modelos predictivos clínicos robustos.

# Índices integrados propuestos

## Índice de recuperación neuromotora

$$IRN = \alpha_1 z(\Delta Fuerza) + \alpha_2 z(\Delta EMG) + \alpha_3 z(\Delta CST) + \alpha_4 z(\Delta Musculo) - \alpha_5 z(Deterioro),$$

con pesos ajustables según criterio metodológico.

## Índice de discordancia central-periférica

$$IDCP = |z(\Delta Neuroimagen) - z(\Delta Periferico)|.$$

Este índice identifica pacientes o regiones donde el sistema central y
periférico no evolucionan de forma concordante.

## Índice de integridad corticoespinal segmentaria

$$I_{CST}(s)= w_1 z(FA(s)) - w_2 z(MD(s)) + w_3 z(Density(s)) - w_4 z(Curvature(s)).$$

Permite comparar segmentos específicos de la CST entre Antes y Después.

# Conclusión técnica

Sí es posible hacer un análisis multidimensional que agrupe todos los
estudios de un paciente. La recomendación principal es no empezar por un
modelo único demasiado complejo, sino construir una arquitectura
escalonada: extracción de características por modalidad, tabla maestra
longitudinal, normalización, modelos exploratorios multibloque, grafos
por paciente y reportes de contribución por modalidad. Las técnicas con
mejor relación entre potencia interpretativa y viabilidad actual son
tractometría CST, radiomía por ROI, PCA multibloque, JIVE, MOFA-like,
sPLS/CCA, SNF y grafos multimodales. Esta capa permitiría pasar de
reportes separados por estudio a un perfil integrado de recuperación
neuromotora.
