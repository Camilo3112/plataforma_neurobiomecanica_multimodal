# Metodología de Procesamiento Multimodal para Tomografía, Tractografía, Corteza Motora, EMG y Dinamometría

**Pipeline VCE:** transformaciones, fundamentos matemáticos, atlas, CST y métricas  
**Autor:** Cristian Camilo Zapata

---

## Resumen general del procesamiento

El objetivo del pipeline es transformar datos crudos provenientes de tomografía axial computarizada (TAC), resonancia magnética estructural y funcional, electromiografía (EMG) y dinamometría en métricas cuantitativas comparables entre sujetos y entre etapas temporales. El esquema general es:

$$\text{Datos crudos}
\rightarrow
\text{preprocesamiento}
\rightarrow
\text{segmentación}
\rightarrow
\text{registro espacial}
\rightarrow
\text{extracción de métricas}
\rightarrow
\text{normalización}
\rightarrow
\text{comparación estadística}.$$

El análisis se organiza por paciente y por etapa:

$$s \in \{\text{paciente 3}, \text{paciente 6}, \text{paciente 7}, \text{paciente 9}, \text{sano}\},$$

$$t \in \{\text{Antes}, \text{Después}\}.$$

La intención científica no es depender de una única variable, sino integrar información periférica, muscular, adiposa, funcional y cortical:

$$\text{Estado funcional total}
=
f(\text{músculo},\text{grasa},\text{fuerza},\text{EMG},\text{corteza motora},\text{CST}).$$

Por eso el pipeline combina:

- TAC: músculo, grasa, áreas, volúmenes y landmarks.

- Resonancia / Brain00mm: ubicación anatómica y corteza motora.

- Difusión / tractografía: reconstrucción propia de streamlines y selección anatómica de la vía corticoespinal.

- Atlas / FreeSurfer / MNI: localización anatómica guiada.

- EMG: activación muscular y comportamiento temporal-frecuencial.

- Dinamometría: desempeño mecánico.

- Estadística: Antes vs. Después, paciente vs. sano y estabilidad del protocolo.

## Procesamiento de tomografías

### Lectura de imágenes DICOM

La TAC se recibe como una serie de archivos DICOM. Cada archivo representa un corte axial o una instancia de adquisición. El primer paso consiste en leer los cortes válidos, extraer su información geométrica y ordenarlos espacialmente.

Sea $I_k(i,j)$ la imagen del corte $k$, donde $i$ representa filas y $j$ columnas. La pila completa se expresa como:

$$V(i,j,k) = I_k(i,j),$$

donde $V$ es el volumen tridimensional.

Los cortes se ordenan usando su posición espacial, normalmente el atributo DICOM `ImagePositionPatient`. Si $\mathbf{p}_k$ es la posición física del corte $k$, el orden se obtiene por la proyección:

$$z_k = \mathbf{p}_k \cdot \mathbf{n},$$

donde $\mathbf{n}$ es el vector normal del plano de corte. Los cortes se ordenan de menor a mayor $z_k$.

### Conversión a Unidades Hounsfield

La TAC no debe analizarse como una imagen de intensidad arbitraria. Debe convertirse a Unidades Hounsfield (HU), porque los tejidos tienen rangos de densidad diferentes. La conversión se realiza con:

$$HU(i,j,k)=PV(i,j,k)\cdot m + b,$$

donde:

- $PV$ es el valor crudo del píxel.

- $m$ es `RescaleSlope`.

- $b$ es `RescaleIntercept`.

Esta transformación permite distinguir de forma aproximada hueso, músculo, grasa y aire.

| Tejido       | Rango aproximado HU     |
|:-------------|:------------------------|
| Aire / fondo | $HU < -500$             |
| Grasa        | $-190 \leq HU \leq -30$ |
| Músculo      | $-29 \leq HU \leq 150$  |
| Hueso        | $HU > 300$              |

Rangos HU usados conceptualmente para segmentación

### Construcción de la matriz affine

Para que el volumen tenga ubicación espacial correcta al exportarse como NIfTI, se construye una matriz affine $A\in\mathbb{R}^{4\times4}$. Esta matriz transforma coordenadas de voxel a coordenadas físicas en milímetros:

$$\begin{bmatrix}
x\\y\\z\\1
\end{bmatrix}
=
A
\begin{bmatrix}
i\\j\\k\\1
\end{bmatrix}.$$

La forma general de la matriz es:

$$A =
\begin{bmatrix}
\Delta_i r_x & \Delta_j c_x & \Delta_k n_x & x_0 \\
\Delta_i r_y & \Delta_j c_y & \Delta_k n_y & y_0 \\
\Delta_i r_z & \Delta_j c_z & \Delta_k n_z & z_0 \\
0 & 0 & 0 & 1
\end{bmatrix},$$

donde:

- $\mathbf{r}$ es el vector de orientación de filas.

- $\mathbf{c}$ es el vector de orientación de columnas.

- $\mathbf{n}=\mathbf{r}\times\mathbf{c}$ es el vector normal al corte.

- $\Delta_i,\Delta_j$ son los tamaños de píxel.

- $\Delta_k$ es el espaciamiento entre cortes.

- $(x_0,y_0,z_0)$ es la posición física del primer corte.

El volumen de un voxel se calcula como:

$$V_{\text{voxel}} = \Delta_x\Delta_y\Delta_z.$$

Si las dimensiones están en milímetros, entonces:

$$V_{\text{voxel,cm}^3} =
\frac{\Delta_x\Delta_y\Delta_z}{1000}.$$

## Detección anatómica y landmarks en TAC

### Segmentación inicial de hueso

El hueso se detecta aplicando un umbral alto en HU:

$$B(i,j,k)=
\begin{cases}
1, & HU(i,j,k)>T_{\text{hueso}},\\
0, & \text{en otro caso}.
\end{cases}$$

Con:

$$T_{\text{hueso}}\approx 300\;HU.$$

Esta máscara permite identificar estructuras óseas principales, separar componentes conectados y seguir la trayectoria anatómica del fémur y la tibia.

### Componentes conectados

A partir de la máscara ósea $B$, se identifican componentes conectados:

$$B = \bigcup_{q=1}^{Q} C_q,$$

donde $C_q$ representa el componente conectado $q$. Para cada componente se calculan propiedades como área, centroide y extensión:

$$\mathbf{c}_q =
\frac{1}{|C_q|}
\sum_{\mathbf{x}\in C_q}\mathbf{x}.$$

El componente principal suele corresponder al hueso anatómicamente dominante del corte.

### Detección de trocánter y punto tibial anteromedial

El pipeline busca dos landmarks anatómicos:

$$\mathbf{L}_{\text{troc}}=
(x_{\text{troc}},y_{\text{troc}},z_{\text{troc}}),$$

$$\mathbf{L}_{\text{tib}}=
(x_{\text{tib}},y_{\text{tib}},z_{\text{tib}}).$$

El punto medio anatómico se calcula como:

$$\mathbf{L}_{\text{medio}}
=
\frac{\mathbf{L}_{\text{troc}}+\mathbf{L}_{\text{tib}}}{2}.$$

En coordenada longitudinal:

$$z_{\text{medio}}
=
\frac{z_{\text{troc}}+z_{\text{tib}}}{2}.$$

Este corte medio anatómico es más reproducible que escoger simplemente el corte central del archivo:

$$z_{\text{archivo}} = \frac{z_{\min}+z_{\max}}{2}.$$

La diferencia es que $z_{\text{medio}}$ depende de landmarks corporales, mientras que $z_{\text{archivo}}$ depende del rango de adquisición.

## Segmentación muscular

### Separación por lado

El volumen se divide en dos regiones laterales. Si $W$ es el ancho de la imagen:

$$R_{\text{izq\_imagen}} = \{(i,j,k): j < W/2\},$$

$$R_{\text{der\_imagen}} = \{(i,j,k): j \geq W/2\}.$$

Según la orientación radiológica, una mitad de la imagen puede corresponder al miembro derecho del paciente y la otra al izquierdo. Por eso esta correspondencia debe validarse visualmente.

### Ray casting desde el hueso

Para segmentar vasto medial y vasto lateral se utiliza una estrategia radial desde el centroide óseo. Sea $\mathbf{c}_k=(x_c,y_c)$ el centroide del hueso en el corte $k$. Un rayo en dirección angular $\theta$ se parametriza como:

$$\mathbf{r}_{\theta}(\rho)
=
\mathbf{c}_k
+
\rho
\begin{bmatrix}
\cos\theta\\
\sin\theta
\end{bmatrix},$$

donde $\rho$ es la distancia radial.

A lo largo del rayo se obtiene una señal de intensidad:

$$s_{\theta}(\rho)=HU(\mathbf{r}_{\theta}(\rho),k).$$

La segmentación busca el punto $\rho^\star$ donde ocurre una transición compatible con borde muscular o fascia:

$$\rho^\star =
\arg\max_{\rho}
\left|
\frac{d}{d\rho}s_{\theta}(\rho)
\right|.$$

Sin embargo, en la práctica el borde puede ser ruidoso; por eso se usa una transformada wavelet continua.

### Transformada wavelet continua para bordes

La transformada wavelet continua de una señal $s(t)$ se define como:

$$W_s(a,b)
=
\frac{1}{\sqrt{a}}
\int_{-\infty}^{\infty}
s(t)
\psi^\ast\left(\frac{t-b}{a}\right)
dt,$$

donde:

- $a$ es la escala.

- $b$ es la posición.

- $\psi$ es la wavelet madre.

- $^\ast$ denota conjugado complejo.

Para detección de bordes se puede usar una wavelet tipo segunda derivada de Gauss, conocida como sombrero mexicano:

$$\psi(t)=
\left(1-t^2\right)e^{-t^2/2}.$$

El borde se estima como el punto con mayor respuesta multiescala:

$$\rho^\star =
\arg\max_{\rho}
\sum_{a\in\mathcal{A}}
\left|
W_{s_\theta}(a,\rho)
\right|.$$

Para evitar falsos bordes cerca del hueso, se aplica una zona ciega:

$$\rho > \rho_{\text{ciega}}.$$

Y para evitar que el rayo se aleje anatómicamente demasiado:

$$\rho < \rho_{\max}.$$

### Construcción de la máscara muscular

Una vez detectados los puntos de borde para múltiples ángulos $\theta_1,\ldots,\theta_N$, se construye una región poligonal:

$$P_k =
\operatorname{poly}
\left\{
\mathbf{r}_{\theta_n}(\rho_n^\star)
\right\}_{n=1}^{N}.$$

La máscara muscular del corte queda:

$$M(i,j,k)=
\begin{cases}
1, & (i,j)\in P_k \;\text{y}\; HU(i,j,k)\in [T_{\text{muscle,min}},T_{\text{muscle,max}}],\\
0, & \text{en otro caso}.
\end{cases}$$

Con un rango conceptual:

$$T_{\text{muscle,min}}\approx -29,
\qquad
T_{\text{muscle,max}}\approx 150.$$

## Segmentación de tejido adiposo

### Grasa subcutánea

La grasa subcutánea se define como tejido graso localizado entre el contorno externo del miembro y la fascia profunda. Matemáticamente, se busca una región $S$ tal que:

$$S = R_{\text{subcutánea}}\cap R_{\text{grasa}},$$

donde:

$$R_{\text{grasa}} =
\{(i,j,k): -190 \leq HU(i,j,k)\leq -30\}.$$

El área de grasa subcutánea en un corte $k$ es:

$$A_{\text{GS}}(k)
=
N_{\text{GS}}(k)\Delta_x\Delta_y,$$

donde $N_{\text{GS}}(k)$ es el número de píxeles clasificados como grasa subcutánea en el corte $k$.

Convertida a centímetros cuadrados:

$$A_{\text{GS,cm}^2}(k)
=
\frac{N_{\text{GS}}(k)\Delta_x\Delta_y}{100}.$$

### Volumen de grasa subcutánea

El volumen total se obtiene sumando sobre cortes:

$$V_{\text{GS}}
=
\sum_{k=1}^{K}
N_{\text{GS}}(k)
\Delta_x\Delta_y\Delta_z.$$

En centímetros cúbicos:

$$V_{\text{GS,cm}^3}
=
\frac{1}{1000}
\sum_{k=1}^{K}
N_{\text{GS}}(k)
\Delta_x\Delta_y\Delta_z.$$

### Grasa intramuscular

La grasa intramuscular se calcula dentro de la máscara muscular $M$. La máscara de grasa intramuscular es:

$$G_{\text{IM}}(i,j,k)=
\begin{cases}
1, & M(i,j,k)=1 \;\text{y}\; -190\leq HU(i,j,k)\leq -30,\\
0, & \text{en otro caso}.
\end{cases}$$

El volumen es:

$$V_{\text{GIM,cm}^3}
=
\frac{1}{1000}
\sum_{i,j,k}
G_{\text{IM}}(i,j,k)
\Delta_x\Delta_y\Delta_z.$$

### Volumen muscular

La máscara muscular completa se denota como $M(i,j,k)$. El volumen muscular es:

$$V_{\text{músculo,cm}^3}
=
\frac{1}{1000}
\sum_{i,j,k}
M(i,j,k)
\Delta_x\Delta_y\Delta_z.$$

### Relaciones grasa/músculo

La relación grasa intramuscular/músculo es:

$$R_{\text{GIM/M}}
=
\frac{V_{\text{GIM}}}{V_{\text{músculo}}+\varepsilon},$$

donde $\varepsilon$ evita división por cero.

La grasa total se define como:

$$V_{\text{GT}}
=
V_{\text{GS}} + V_{\text{GIM}}.$$

Y la relación grasa total/músculo:

$$R_{\text{GT/M}}
=
\frac{V_{\text{GT}}}{V_{\text{músculo}}+\varepsilon}.$$

La fracción grasa porcentual se calcula como:

$$F_{\text{grasa}}(\%)=
100
\frac{V_{\text{GT}}}{V_{\text{GT}}+V_{\text{músculo}}+\varepsilon}.$$

## Exportación de máscaras NIfTI y labelmaps

Las segmentaciones se exportan como archivos NIfTI para permitir revisión visual y análisis posterior. Una máscara discreta se representa como:

$$L(i,j,k)\in \{0,1,2,3,4\}.$$

Para tejido adiposo total se propuso el siguiente labelmap:

$$L_{\text{adiposo}}(i,j,k)=
\begin{cases}
0, & \text{fondo},\\
1, & \text{grasa subcutánea derecha},\\
2, & \text{grasa subcutánea izquierda},\\
3, & \text{grasa intramuscular derecha},\\
4, & \text{grasa intramuscular izquierda}.
\end{cases}$$

Estos mapas permiten inspección en visores como 3D Slicer, ITK-SNAP o Freeview. Además, guardan la relación espacial mediante la matriz affine $A$.

## Procesamiento de resonancia y corrección cortical

### Problema de alineación

Las máscaras corticales iniciales pueden quedar desalineadas con respecto al volumen anatómico del paciente. El problema se expresa como:

$$\Omega_{\text{ROI}} \not\subset \Omega_{\text{corteza}},$$

donde $\Omega_{\text{ROI}}$ es la región de interés de corteza motora y $\Omega_{\text{corteza}}$ es la capa cortical del cerebro del paciente.

El objetivo es encontrar una transformación $T$ tal que:

$$T(\Omega_{\text{ROI}})\approx \Omega_{\text{corteza motora}}.$$

### Remuestreo al espacio de referencia

Sea $I_B$ la imagen anatómica de referencia, por ejemplo `Brain00mm.nii`, y sea $R$ la máscara de corteza motora original. Cada una tiene su propia grilla y affine:

$$A_B,\qquad A_R.$$

Para comparar voxel a voxel, la máscara $R$ se remuestrea al espacio de $I_B$. La relación entre coordenadas es:

$$\mathbf{x}_B =
A_B^{-1}A_R\mathbf{x}_R.$$

El remuestreo de máscaras se hace con vecino más cercano:

$$R_B(\mathbf{x}_B)
=
R\left(
\operatorname{NN}
\left(A_R^{-1}A_B\mathbf{x}_B\right)
\right),$$

donde $\operatorname{NN}$ representa nearest neighbor. No se usa interpolación lineal porque una máscara tiene etiquetas discretas.

### Detección de máscara cerebral

A partir de la imagen anatómica $I_B$, se estima una máscara cerebral:

$$B(\mathbf{x}) =
\begin{cases}
1, & I_B(\mathbf{x}) > T_{\text{brain}},\\
0, & \text{en otro caso}.
\end{cases}$$

El umbral $T_{\text{brain}}$ puede obtenerse de manera adaptativa usando estadísticos del fondo o métodos como Otsu. Luego se conserva el componente conectado dominante:

$$B^\star = \arg\max_{C_q} |C_q|.$$

Después se aplican operaciones morfológicas:

$$B_{\text{clean}}
=
\operatorname{FillHoles}
\left(
\operatorname{Close}(B^\star)
\right).$$

### Capa cortical externa

La corteza se aproxima como una cáscara externa del cerebro. Para ello se calcula la transformada de distancia euclidiana dentro de la máscara cerebral:

$$D(\mathbf{x})
=
\min_{\mathbf{y}\in \partial B}
\|\mathbf{x}-\mathbf{y}\|_2.$$

La capa cortical se define como:

$$C_{\text{shell}}(\mathbf{x})
=
\begin{cases}
1, & B(\mathbf{x})=1 \;\text{y}\; D(\mathbf{x})\leq d_{\text{shell}},\\
0, & \text{en otro caso}.
\end{cases}$$

Con:

$$d_{\text{shell}}\approx 5\text{ mm}.$$

El fundamento es que una ROI cortical debe caer sobre una capa externa del cerebro, no en regiones profundas ni fuera del volumen cerebral.

### Corrección por traslación

La máscara remuestreada puede corregirse buscando una traslación:

$$\mathbf{t}=
(t_x,t_y,t_z).$$

La transformación aplicada a cada punto de la ROI es:

$$T_{\mathbf{t}}(\mathbf{x})=\mathbf{x}+\mathbf{t}.$$

Si se permite escala, la forma general es:

$$T_{\mathbf{t},s}(\mathbf{x})
=
s(\mathbf{x}-\mathbf{c})+\mathbf{c}+\mathbf{t},$$

donde:

- $s$ es el factor de escala.

- $\mathbf{c}$ es el centroide de la ROI.

- $\mathbf{t}$ es la traslación.

Por seguridad metodológica, el factor de escala debe usarse con cautela, porque una reducción artificial de la máscara podría mejorar el score sin mejorar la validez anatómica.

### Función de puntuación de la corrección

Para cada traslación candidata se calcula un score. Sea $\Omega_T=T(\Omega_{\text{ROI}})$. Se definen:

$$f_{\text{inside}}
=
\frac{|\Omega_T\cap B|}{|\Omega_T|},$$

$$f_{\text{shell}}
=
\frac{|\Omega_T\cap C_{\text{shell}}|}{|\Omega_T|},$$

$$f_{\text{outside}}
=
1-f_{\text{inside}}.$$

También se puede definir una cercanía suave a la corteza mediante el mapa de distancia:

$$f_{\text{close}}
=
\frac{1}{|\Omega_T|}
\sum_{\mathbf{x}\in\Omega_T}
\exp\left(-\frac{D_C(\mathbf{x})}{\sigma}\right),$$

donde $D_C(\mathbf{x})$ es la distancia de $\mathbf{x}$ a la capa cortical.

La penalización por magnitud de movimiento puede ser:

$$P_{\text{trans}}
=
\frac{\|\mathbf{t}\|_2}{t_{\max}}.$$

La penalización por hemisferio se define como:

$$P_{\text{hemi}} =
\begin{cases}
0, & h(T(\Omega_{\text{ROI}}))=h(\Omega_{\text{ROI}}),\\
1, & h(T(\Omega_{\text{ROI}}))\neq h(\Omega_{\text{ROI}}),
\end{cases}$$

donde $h(\cdot)$ representa el hemisferio dominante de la ROI.

El score global puede escribirse como:

$$S(T)
=
w_1 f_{\text{close}}
+
w_2 f_{\text{inside}}
+
w_3 f_{\text{shell}}
-
w_4 f_{\text{outside}}
-
w_5 P_{\text{trans}}
-
w_6 P_{\text{hemi}}.$$

La transformación óptima es:

$$T^\star =
\arg\max_T S(T).$$

La corrección se busca en dos niveles:

$$T^\star_{\text{grueso}}
=
\arg\max_{T\in\mathcal{T}_{\text{gruesa}}} S(T),$$

$$T^\star_{\text{fino}}
=
\arg\max_{T\in\mathcal{N}(T^\star_{\text{grueso}})} S(T).$$

## Atlas cerebral

### Definición de atlas

Un atlas cerebral es un mapa anatómico que asigna etiquetas a regiones del cerebro. Formalmente, un atlas es una función discreta:

$$A_{\text{atlas}}(\mathbf{x}) = \ell,$$

donde $\ell$ es una etiqueta anatómica. Por ejemplo:

$$\ell=
\begin{cases}
\text{precentral},\\
\text{postcentral},\\
\text{superior frontal},\\
\text{paracentral},\\
\text{otras regiones}.
\end{cases}$$

La utilidad del atlas es que permite definir regiones con significado anatómico, no solo regiones geométricas.

### Atlas-prior heurístico

El atlas-prior heurístico no es un atlas clínico completo. Es una restricción anatómica aproximada que combina:

$$\text{hemisferio correcto}
+
\text{capa cortical}
+
\text{zona espacial compatible con corteza motora}.$$

La ROI refinada puede expresarse como:

$$\Omega_{\text{ROI,refinada}}
=
\Omega_{\text{ROI,corregida}}
\cap
C_{\text{shell}}
\cap
P_{\text{motor}},$$

donde $P_{\text{motor}}$ es el prior espacial motor.

Ventajas:

- Rápido.

- No requiere instalación pesada.

- Útil como respaldo cuando no existe FreeSurfer.

Limitaciones:

- No identifica surcos ni giros reales del paciente.

- No reemplaza un registro anatómico atlas-paciente.

- Requiere validación visual.

### FreeSurfer como atlas anatómico nativo

FreeSurfer reconstruye la anatomía cortical del paciente a partir de una resonancia T1. El flujo conceptual es:

$$T1_{\text{paciente}}
\rightarrow
\text{recon-all}
\rightarrow
\text{superficies corticales}
\rightarrow
\text{parcelación anatómica}
\rightarrow
\text{regiones motoras}.$$

La ventaja principal es que la medición se realiza en el espacio anatómico del paciente, no en una plantilla promedio. Sea $A_{\text{FS}}$ el atlas FreeSurfer en espacio nativo:

$$A_{\text{FS}}:\Omega_{\text{paciente}}\rightarrow \mathcal{L}_{\text{FS}}.$$

Para M1 se usa la región precentral:

$$\Omega_{\text{M1}} =
\{\mathbf{x}\in\Omega_{\text{paciente}}:
A_{\text{FS}}(\mathbf{x})=\text{precentral}\}.$$

Para M2 se define una región compuesta:

$$\Omega_{\text{M2}}
=
\Omega_{\text{caudal middle frontal}}
\cup
\Omega_{\text{superior frontal}}
\cup
\Omega_{\text{paracentral}},$$

o una combinación equivalente según el atlas disponible.

### Atlas MNI

MNI es una plantilla poblacional estándar. El registro paciente–MNI se expresa como:

$$\phi:\Omega_{\text{paciente}}\rightarrow\Omega_{\text{MNI}}.$$

Si una ROI está definida en MNI, para llevarla al paciente se usa la transformación inversa:

$$\phi^{-1}:\Omega_{\text{MNI}}\rightarrow\Omega_{\text{paciente}}.$$

Por tanto:

$$\Omega_{\text{MNI}\rightarrow\text{paciente}}
=
\phi^{-1}(\Omega_{\text{atlas MNI}}).$$

La medición final recomendada se hace en espacio nativo del paciente:

$$V_{\text{ROI}}
=
\sum_{\mathbf{x}\in\Omega_{\text{paciente}}}
\mathbb{1}_{\Omega_{\text{ROI}}}(\mathbf{x})
V_{\text{voxel}}.$$

Esto evita que la deformación a MNI afecte directamente el volumen.

### Tipos de transformaciones de registro

#### Transformación rígida

La transformación rígida conserva distancias y volúmenes:

$$\mathbf{x}' = R\mathbf{x}+\mathbf{t},$$

donde $R$ es una matriz de rotación y $\mathbf{t}$ una traslación.

$$R^TR=I,\qquad \det(R)=1.$$

#### Transformación afín

La transformación afín permite escala, cizalla, rotación y traslación:

$$\mathbf{x}' = M\mathbf{x}+\mathbf{t},$$

donde $M\in\mathbb{R}^{3\times3}$.

En coordenadas homogéneas:

$$\begin{bmatrix}
\mathbf{x}'\\1
\end{bmatrix}
=
\begin{bmatrix}
M & \mathbf{t}\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
\mathbf{x}\\1
\end{bmatrix}.$$

#### Transformación no lineal

Una deformación no lineal se expresa como:

$$\mathbf{x}' = \phi(\mathbf{x}) = \mathbf{x}+\mathbf{u}(\mathbf{x}),$$

donde $\mathbf{u}(\mathbf{x})$ es un campo de desplazamiento.

Este tipo de transformación es útil para normalización a MNI, pero debe interpretarse con cuidado cuando se analizan volúmenes.

## Integración anatómica y funcional de la corteza motora

La ROI final de corteza motora debe combinar información anatómica y, si está disponible, funcional:

$$\Omega_{\text{motor final}}
=
\Omega_{\text{atlas}}
\cap
C_{\text{shell}}
\cap
\Omega_{\text{funcional}}.$$

Si no existe mapa funcional, puede usarse:

$$\Omega_{\text{motor final}}
=
\Omega_{\text{atlas}}
\cap
C_{\text{shell}}.$$

La señal BOLD de fMRI no mide volumen directamente. Por eso se recomienda:

$$\text{fMRI}
\rightarrow
\text{mapa de activación}
\rightarrow
\text{registro al T1}
\rightarrow
\text{extracción dentro de ROI anatómica}.$$

La métrica funcional dentro de la ROI puede ser:

$$\bar{B}_{\text{ROI}}
=
\frac{1}{|\Omega_{\text{ROI}}|}
\sum_{\mathbf{x}\in\Omega_{\text{ROI}}}
B(\mathbf{x}),$$

donde $B(\mathbf{x})$ representa intensidad funcional, estadístico de activación o valor derivado.

## Métricas de corteza motora

### Volumen

El volumen de una ROI cortical es:

$$V_{\text{ROI,cm}^3}
=
\frac{1}{1000}
\sum_{\mathbf{x}}
\mathbb{1}_{\Omega_{\text{ROI}}}(\mathbf{x})
\Delta_x\Delta_y\Delta_z.$$

Para M1:

$$V_{\text{M1,L}},\qquad V_{\text{M1,R}}.$$

Para M2:

$$V_{\text{M2,L}},\qquad V_{\text{M2,R}}.$$

### Asimetría interhemisférica

La asimetría porcentual se calcula como:

$$A_{\text{M1}}(\%)
=
100
\frac{V_{\text{M1,R}}-V_{\text{M1,L}}}
{\frac{1}{2}(V_{\text{M1,R}}+V_{\text{M1,L}})+\varepsilon}.$$

De forma análoga:

$$A_{\text{M2}}(\%)
=
100
\frac{V_{\text{M2,R}}-V_{\text{M2,L}}}
{\frac{1}{2}(V_{\text{M2,R}}+V_{\text{M2,L}})+\varepsilon}.$$

### Estabilidad Antes–Después

Para una métrica cortical $m$:

$$\Delta m = m_{\text{Después}}-m_{\text{Antes}},$$

$$\Delta m_{\%}
=
100\frac{m_{\text{Después}}-m_{\text{Antes}}}
{m_{\text{Antes}}+\varepsilon}.$$

Cambios extremadamente grandes en volumen cortical pueden indicar error de segmentación, registro o calidad de imagen.

## Procesamiento de difusión, tractografía y vía corticoespinal

### Propósito del módulo de tractografía

El módulo de tractografía se incorporó para reconstruir y visualizar un corredor corticoespinal propio dentro del espacio anatómico del paciente, usando como referencia principal el volumen estructural `rT1.nii`. El objetivo es anexar al pipeline VCE una representación de la conectividad motora descendente, especialmente de la vía corticoespinal o CST (*corticospinal tract*), sin depender de las máscaras ni de la tractografía final generada automáticamente por el resonador.

El flujo implementado se organiza en cinco etapas:

1.  diagnóstico de la serie de difusión y verificación de gradientes;

2.  conversión de DICOM de difusión a NIfTI con archivos `bval` y `bvec`;

3.  reconstrucción de tractografía de cerebro completo;

4.  adaptación de regiones motoras M1/M2 y tronco encefálico al espacio del `rT1`;

5.  filtrado anatómico de streamlines para obtener CST izquierda, CST derecha, mapas de densidad y volúmenes coloreados para revisión en ITK-SNAP.

La estrategia final no desplaza globalmente las streamlines para forzar una apariencia visual, sino que vuelve a filtrar la tractografía mediante waypoints anatómicos más específicos: origen cortical motor, mesencéfalo, puente y bulbo. En la depuración del plano sagital se priorizó la porción anterior/ventral del tronco encefálico; en el plano coronal se añadió un criterio paramediano/medial calculado desde el centro del tronco para reducir desplazamientos laterales aparentes.

### Modelo físico de difusión en resonancia magnética

La tractografía se basa en imágenes de difusión. El fundamento físico es que las moléculas de agua no se desplazan igual en todas las direcciones dentro de la sustancia blanca. En un medio libre, la difusión tiende a ser isotrópica; en fascículos axonales, las membranas, mielina y organización microestructural restringen más el movimiento perpendicular que el movimiento paralelo al eje de las fibras.

La señal de difusión puede aproximarse mediante la ecuación de Stejskal–Tanner:

$$S(b,\mathbf{g})
=
S_0\exp\left(-b\,\mathbf{g}^{T}D\mathbf{g}\right),$$

donde:

- $S(b,\mathbf{g})$ es la señal medida con ponderación de difusión;

- $S_0$ es la señal sin ponderación de difusión o imagen $b=0$;

- $b$ es el factor de ponderación de difusión;

- $\mathbf{g}$ es la dirección del gradiente de difusión;

- $D$ es el tensor de difusión $3\times3$.

El factor $b$ resume la intensidad y duración de los gradientes de difusión:

$$b = \gamma^2G^2\delta^2\left(\Delta-\frac{\delta}{3}\right),$$

donde $\gamma$ es la razón giromagnética, $G$ es la amplitud del gradiente, $\delta$ es la duración del pulso y $\Delta$ es el tiempo entre pulsos de gradiente.

### Tensor de difusión y direcciones principales

En el modelo DTI, el tensor de difusión se expresa como una matriz simétrica positiva:

$$D=
\begin{bmatrix}
D_{xx} & D_{xy} & D_{xz}\\
D_{xy} & D_{yy} & D_{yz}\\
D_{xz} & D_{yz} & D_{zz}
\end{bmatrix}.$$

Su descomposición espectral es:

$$D = V\Lambda V^T,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\lambda_2,\lambda_3),$$

con:

$$\lambda_1\geq \lambda_2\geq \lambda_3.$$

La dirección principal de difusión se aproxima con el autovector asociado a $\lambda_1$:

$$\mathbf{v}_1 = V_{:,1}.$$

A partir de los autovalores se calculan métricas de difusión:

$$AD = \lambda_1,$$

$$RD = \frac{\lambda_2+\lambda_3}{2},$$

$$MD = \frac{\lambda_1+\lambda_2+\lambda_3}{3},$$

$$FA =
\sqrt{\frac{3}{2}}
\frac{
\sqrt{(\lambda_1-MD)^2+(\lambda_2-MD)^2+(\lambda_3-MD)^2}
}
{
\sqrt{\lambda_1^2+\lambda_2^2+\lambda_3^2}
}.$$

La anisotropía fraccional $FA$ se usa como indicador de direccionalidad de la difusión. Valores más altos sugieren una dirección dominante más clara, aunque no deben interpretarse por sí solos como integridad anatómica sin revisar adquisición, registro, ruido y cruces de fibras.

### Reconstrucción de streamlines

La tractografía reconstruye curvas tridimensionales que siguen un campo local de orientación. Cada streamline se modela como una curva:

$$\gamma(s)=
\begin{bmatrix}
x(s)\\y(s)\\z(s)
\end{bmatrix},$$

cuya evolución está gobernada por:

$$\frac{d\gamma(s)}{ds}=\mathbf{u}(\gamma(s)),$$

donde $\mathbf{u}(\gamma(s))$ es la orientación local estimada a partir del tensor, de la dirección principal o de una función de distribución de orientaciones cuando se usa un modelo más avanzado.

En forma discreta, el avance de una streamline puede representarse como:

$$\gamma_{n+1}=\gamma_n+h\,\mathbf{u}(\gamma_n),$$

donde $h$ es el tamaño de paso. Para evitar saltos anatómicamente incoherentes, se controla el ángulo entre pasos consecutivos:

$$\theta_n =
\cos^{-1}
\left(
\frac{
\mathbf{u}_n\cdot\mathbf{u}_{n-1}
}
{
\|\mathbf{u}_n\|\,\|\mathbf{u}_{n-1}\|
}
\right).$$

La propagación se detiene si se cumple alguna condición de parada:

$$FA < FA_{\min},
\qquad
\theta_n>\theta_{\max},
\qquad
\gamma_n\notin \Omega_{\text{cerebro}},
\qquad
L(\gamma)>L_{\max}.$$

Aquí $\Omega_{\text{cerebro}}$ representa la máscara cerebral válida y $L(\gamma)$ es la longitud acumulada:

$$L(\gamma)=
\sum_{n=1}^{N-1}
\|\gamma_{n+1}-\gamma_n\|_2.$$

### Registro de difusión al espacio anatómico T1

Para que las streamlines puedan visualizarse junto al `rT1.nii`, se transforman desde el espacio de difusión al espacio anatómico. Si $\mathbf{x}_{DWI}$ es un punto de una streamline en el espacio de difusión, su posición en T1 se expresa como:

$$\mathbf{x}_{T1}=T_{DWI\rightarrow T1}(\mathbf{x}_{DWI}),$$

donde $T_{DWI\rightarrow T1}$ puede ser una transformación rígida, afín o compuesta, según las salidas disponibles del preprocesamiento. En coordenadas homogéneas:

$$\begin{bmatrix}
\mathbf{x}_{T1}\\1
\end{bmatrix}
=
A_{DWI\rightarrow T1}
\begin{bmatrix}
\mathbf{x}_{DWI}\\1
\end{bmatrix}.$$

Para los labelmaps anatómicos, el remuestreo se realiza con interpolación de vecino más cercano, porque las etiquetas no son intensidades continuas:

$$R_{T1}(\mathbf{x})=
R_{DWI}\left(T_{T1\rightarrow DWI}(\mathbf{x})\right).$$

El resultado final debe conservar la geometría del `rT1`: misma matriz affine, mismo tamaño de voxel y misma orientación espacial. Esto permite abrir simultáneamente `rT1.nii`, mapas de densidad, mapas RGB y máscaras de CST en ITK-SNAP sin desalineación espacial.

### Definición anatómica de la vía corticoespinal

La selección de la vía corticoespinal se planteó como un problema de filtrado de streamlines mediante regiones de inclusión. Sea $\Gamma$ el conjunto de streamlines de cerebro completo:

$$\Gamma=\{\gamma_1,\gamma_2,\ldots,\gamma_N\}.$$

Sea $I(\gamma,R)$ una función indicadora que vale 1 si la streamline $\gamma$ intersecta la región anatómica $R$:

$$I(\gamma,R)=
\begin{cases}
1, & \exists s: \gamma(s)\in R,\\
0, & \text{en otro caso}.
\end{cases}$$

Para cada hemisferio se definieron regiones motoras y waypoints del tronco encefálico:

$$S_L = M1_L\cup M2_L,
\qquad
S_R = M1_R\cup M2_R,$$

$$W_L = W_{\text{mes},L}\cap W_{\text{puente},L}\cap W_{\text{bulbo},L},$$

$$W_R = W_{\text{mes},R}\cap W_{\text{puente},R}\cap W_{\text{bulbo},R}.$$

De forma operativa, la CST izquierda se define como:

$$CST_L=
\left\{
\gamma\in\Gamma:
I(\gamma,S_L)=1
\land
I(\gamma,W_{\text{mes},L})=1
\land
I(\gamma,W_{\text{puente},L})=1
\land
I(\gamma,W_{\text{bulbo},L})=1
\land
I(\gamma,E)=0
\right\},$$

y de forma análoga:

$$CST_R=
\left\{
\gamma\in\Gamma:
I(\gamma,S_R)=1
\land
I(\gamma,W_{\text{mes},R})=1
\land
I(\gamma,W_{\text{puente},R})=1
\land
I(\gamma,W_{\text{bulbo},R})=1
\land
I(\gamma,E)=0
\right\}.$$

$E$ representa una máscara de exclusión para streamlines anatómicamente incompatibles. En caso de que una vía completa hasta bulbo no cumpla todos los waypoints, el pipeline puede conservar un resultado diagnóstico parcial hasta el nivel anatómico más confiable, por ejemplo puente, pero el reporte debe marcarlo como parcial y no como CST completa.

### Corrección ventral y paramediana en tronco encefálico

Durante la validación visual se observó que la vía podía verse desplazada en plano sagital o coronal si los waypoints del tronco eran demasiado amplios. Por esta razón se implementó un refinamiento anatómico del corredor dentro del tronco encefálico.

Sea $B_z$ la sección axial del tronco en el nivel $z$. Su centroide se estima como:

$$\mathbf{c}_B(z)=
\frac{1}{|B_z|}
\sum_{\mathbf{x}\in B_z}
\mathbf{x}.$$

La línea media se aproxima por la coordenada mediolateral del centroide:

$$m(z)=c_{B,x}(z).$$

Para cada hemisferio se conserva un corredor paramediano:

$$P_L(z)=\{\mathbf{x}\in B_z: x<m(z),\; |x-m(z)|<\alpha r_x(z)\},$$

$$P_R(z)=\{\mathbf{x}\in B_z: x>m(z),\; |x-m(z)|<\alpha r_x(z)\},$$

donde $r_x(z)$ es una medida del radio mediolateral del tronco en ese corte y $\alpha$ controla qué tan cerca de la línea media se conserva el corredor.

El criterio ventral se expresa de forma general como:

$$V(z)=\{\mathbf{x}\in B_z: a(\mathbf{x})\leq q_{\beta}(a(B_z))\},$$

donde $a(\mathbf{x})$ representa la coordenada anteroposterior en el sistema anatómico usado y $q_{\beta}$ es un cuantil que conserva la porción anterior/ventral. Con esto, los waypoints finales del tronco se calculan como:

$$W_{\text{nivel},h}^{\star}=B_{\text{nivel}}\cap P_h\cap V,
\qquad
h\in\{L,R\}.$$

Este refinamiento no modifica las coordenadas de las streamlines; únicamente selecciona mejor qué streamlines son compatibles con la trayectoria esperada de la CST en mesencéfalo, puente y bulbo.

### Mapas de densidad y visualización RGB

Además del archivo `.trk`, la tractografía se voxeliza para generar mapas NIfTI de densidad. Si $v$ es un voxel y $\Gamma_C$ un conjunto de streamlines filtradas, la densidad se calcula como:

$$\rho(v)=
\sum_{\gamma\in\Gamma_C}
\sum_{n=1}^{N_{\gamma}}
\mathbf{1}\left(\gamma_n\in v\right).$$

También se genera una visualización RGB direccional. Si $\mathbf{d}(v)$ es la dirección media local de los segmentos que atraviesan el voxel, el color puede expresarse como:

$$RGB(v)=
\left(
\frac{|d_x(v)|}{\|\mathbf{d}(v)\|},
\frac{|d_y(v)|}{\|\mathbf{d}(v)\|},
\frac{|d_z(v)|}{\|\mathbf{d}(v)\|}
\right).$$

Para visualización específica de CST en ITK-SNAP se genera un volumen de etiquetas o colores:

$$C(v)=
\begin{cases}
1, & v\in CST_L,\\
2, & v\in CST_R,\\
3, & v\in CST_L\cap CST_R,\\
0, & \text{fondo}.
\end{cases}$$

En la implementación visual, estos valores se representan de forma conceptual como rojo para CST izquierda, azul para CST derecha y magenta para superposición.

### Métricas derivadas de tractografía

Para cada conjunto de streamlines se pueden calcular métricas de volumen, densidad, longitud y asimetría:

$$N_{CST,h}=|CST_h|,
\qquad h\in\{L,R\}.$$

$$\overline{L}_{CST,h}=\frac{1}{N_{CST,h}}\sum_{\gamma\in CST_h}L(\gamma).$$

$$V_{CST,h}=N_{vox}(\rho_h>0)\cdot V_{voxel}.$$

$$D_{CST,h}=\frac{1}{N_{vox}(\rho_h>0)}
\sum_{v:\rho_h(v)>0}\rho_h(v).$$

$$A_{CST}=100\cdot
\frac{M_R-M_L}{(M_R+M_L)/2},$$

donde $M_h$ puede representar número de streamlines, volumen voxelizado, longitud media o densidad media. La interpretación del signo depende de la métrica seleccionada y de la convención izquierda/derecha.

### Control de calidad específico de tractografía

El control de calidad de este módulo incluye:

- verificación de que la serie DWI tenga volúmenes $b=0$ y direcciones de difusión válidas;

- revisión de archivos `bval`/`bvec`;

- confirmación de que `FA_en_T1.nii.gz`, `ColorFA_en_T1.nii.gz`, mapas de densidad y `wholebrain_T1.trk` estén alineados con `rT1.nii`;

- validación visual de M1, M2, tronco encefálico y waypoints por hemisferio;

- revisión de número de streamlines retenidas por hemisferio;

- advertencia explícita cuando solo existe una vía parcial hasta puente o mesencéfalo;

- registro en JSON de rutas, parámetros y advertencias.

La regla metodológica principal es que la tractografía se considera un modelo geométrico de conectividad probable, no una visualización directa de axones individuales. Por eso los resultados se reportan junto con criterios de inclusión, filtros anatómicos y advertencias de validación visual.

## Procesamiento de EMG

### Señal cruda

Sea $x[n]$ una señal EMG discreta. Primero se eliminan valores inválidos, se organiza por músculo, lado, prueba y fase, y se aplican filtros según la calidad de la adquisición.

### RMS

La amplitud efectiva se calcula mediante RMS:

$$RMS =
\sqrt{
\frac{1}{N}
\sum_{n=1}^{N}
x[n]^2
}.$$

### iEMG

La activación acumulada se calcula como integral de la señal rectificada:

$$iEMG =
\sum_{n=1}^{N}
|x[n]|\Delta t.$$

### Frecuencia dominante

La transformada discreta de Fourier es:

$$X[k]=
\sum_{n=0}^{N-1}
x[n]e^{-j2\pi kn/N}.$$

La frecuencia dominante es:

$$f_{\text{dom}}
=
\arg\max_f |X(f)|^2.$$

### Correlación cruzada

Para dos señales $x[n]$ y $y[n]$, la correlación cruzada es:

$$R_{xy}[\tau]
=
\sum_n x[n]y[n+\tau].$$

El desfase dominante se estima como:

$$\tau^\star =
\arg\max_\tau R_{xy}[\tau].$$

### Coherencia

La coherencia entre dos señales se calcula como:

$$C_{xy}(f)
=
\frac{|P_{xy}(f)|^2}
{P_{xx}(f)P_{yy}(f)},$$

donde $P_{xy}$ es la densidad espectral cruzada y $P_{xx},P_{yy}$ las densidades espectrales de potencia.

### PLV

El phase locking value mide sincronización de fase:

$$PLV =
\left|
\frac{1}{N}
\sum_{n=1}^{N}
e^{j(\phi_x[n]-\phi_y[n])}
\right|.$$

## Procesamiento de dinamometría

Sea $F(t)$ la curva de fuerza.

### Fuerza máxima

$$F_{\max}=\max_t F(t).$$

### Fuerza media

$$\bar{F}=
\frac{1}{T}
\int_0^T F(t)\,dt.$$

En señal discreta:

$$\bar{F}
=
\frac{1}{N}
\sum_{n=1}^{N}
F[n].$$

### Área bajo la curva

$$AUC_F =
\int_0^T F(t)\,dt
\approx
\sum_{n=1}^{N}
F[n]\Delta t.$$

### Asimetría de fuerza

$$A_F(\%)
=
100
\frac{F_R-F_L}
{\frac{1}{2}(F_R+F_L)+\varepsilon}.$$

## Consolidación de métricas

Cada módulo produce métricas en archivos CSV. El consolidador organiza la información en formato largo:

$$D =
\{s,t,q,m,v,u\},$$

donde:

- $s$: sujeto.

- $t$: etapa.

- $q$: modalidad.

- $m$: nombre de métrica.

- $v$: valor.

- $u$: unidad o escala.

Ejemplo:

$$(\text{paciente 3},\text{Antes},\text{TAC},
R_{\text{GT/M}},0.18,\text{adimensional}).$$

## Normalización estadística

### Z-score

Para una métrica $x$, el z-score se calcula como:

$$z =
\frac{x-\mu}{\sigma+\varepsilon}.$$

Interpretación:

- $z=0$: valor promedio.

- $z>0$: por encima del promedio.

- $z<0$: por debajo del promedio.

### Robust z-score

Cuando hay pocos sujetos o valores extremos, se usa una normalización robusta:

$$z_{\text{robusto}}
=
\frac{x-\operatorname{mediana}(x)}
{\operatorname{MAD}(x)+\varepsilon},$$

donde:

$$MAD =
\operatorname{mediana}
\left(
|x_i-\operatorname{mediana}(x)|
\right).$$

También puede usarse IQR:

$$z_{\text{IQR}}
=
\frac{x-\operatorname{mediana}(x)}
{IQR+\varepsilon}.$$

### Ratio contra sano

Para comparar un paciente con el sano:

$$R_{\text{sano}}
=
\frac{x_{\text{paciente}}}
{x_{\text{sano}}+\varepsilon}.$$

$$R_{\text{sano}}=1
\quad\Rightarrow\quad
\text{valor igual al sano}.$$

### Distancia al sano

$$d_{\text{Antes}}
=
|x_{\text{Antes}}-x_{\text{sano}}|,$$

$$d_{\text{Después}}
=
|x_{\text{Después}}-x_{\text{sano}}|.$$

La mejora absoluta hacia el sano es:

$$M_{\text{abs}}
=
d_{\text{Antes}}-d_{\text{Después}}.$$

### Índice de recuperación hacia sano

$$I_{\text{rec}}
=
\frac{d_{\text{Antes}}-d_{\text{Después}}}
{d_{\text{Antes}}+\varepsilon}.$$

Interpretación:

$$I_{\text{rec}}>0
\Rightarrow
\text{el sujeto se acerca al sano},$$

$$I_{\text{rec}}=0
\Rightarrow
\text{sin cambio},$$

$$I_{\text{rec}}<0
\Rightarrow
\text{el sujeto se aleja del sano}.$$

## Comparación Antes vs. Después

Para cada métrica:

$$\Delta x =
x_{\text{Después}}-x_{\text{Antes}},$$

$$\Delta x_{\%}
=
100
\frac{x_{\text{Después}}-x_{\text{Antes}}}
{x_{\text{Antes}}+\varepsilon}.$$

Si existen varias observaciones pareadas, puede usarse una prueba t pareada:

$$t =
\frac{\bar{d}}
{s_d/\sqrt{n}},$$

donde $d_i=x_{i,\text{Después}}-x_{i,\text{Antes}}$.

Si no se asume normalidad, puede usarse una prueba no paramétrica como Wilcoxon o Mann–Whitney, dependiendo de la estructura de los datos.

### Tamaño del efecto

El tamaño del efecto de Hedges puede aproximarse como:

$$g =
J
\frac{\bar{x}_2-\bar{x}_1}
{s_p},$$

donde:

$$s_p =
\sqrt{
\frac{(n_1-1)s_1^2+(n_2-1)s_2^2}
{n_1+n_2-2}
},$$

y:

$$J =
1-\frac{3}{4(n_1+n_2)-9}.$$

Cliff’s delta se define como:

$$\delta =
\frac{
\#(x_i>y_j)-\#(x_i<y_j)
}
{n_x n_y}.$$

## Modelos matemáticos ampliados y explicación de ecuaciones

Esta sección amplía el fundamento interno de los modelos usados por el pipeline. La idea es dejar claro qué representa cada transformación, por qué se aplica y cómo se interpreta cada ecuación en el contexto de TAC, resonancia, atlas, EMG, dinamometría y estadística.

### Notación unificada del pipeline

Cada paciente se representa por un índice $s$, cada etapa por un índice $t$ y cada modalidad por un índice $q$:

$$s \in \mathcal{S},
\qquad
t \in \{\text{Antes},\text{Después}\},
\qquad
q \in \{\text{TAC},\text{RM},\text{EMG},\text{Dinamometría}\}.$$

Una imagen volumétrica se modela como una función discreta:

$$I:\Omega_v \subset \mathbb{Z}^3 \rightarrow \mathbb{R},$$

donde $\Omega_v$ es el dominio de voxeles y $I(i,j,k)$ es la intensidad del voxel $(i,j,k)$. En TAC, $I(i,j,k)$ representa unidades Hounsfield; en RM puede representar intensidad anatómica; y en mapas funcionales puede representar activación, estadístico o valor de señal.

Una máscara binaria se define como:

$$M:\Omega_v \rightarrow \{0,1\}.$$

Si $M(\mathbf{x})=1$, el voxel $\mathbf{x}$ pertenece a la región de interés. Si $M(\mathbf{x})=0$, el voxel queda fuera de la región.

Una máscara multiclase o labelmap se define como:

$$L:\Omega_v \rightarrow \{0,1,2,\ldots,K\},$$

donde cada número representa una clase anatómica o tisular diferente.

### Modelo físico de intensidad en TAC

La TAC se basa en la atenuación de rayos X. De forma simplificada, la intensidad detectada por el escáner se relaciona con la atenuación del tejido. La unidad clínica estandarizada es la unidad Hounsfield:

$$HU =
1000
\frac{\mu-\mu_{\text{agua}}}
{\mu_{\text{agua}}-\mu_{\text{aire}}},$$

donde:

- $\mu$ es el coeficiente de atenuación del tejido.

- $\mu_{\text{agua}}$ es la atenuación del agua.

- $\mu_{\text{aire}}$ es la atenuación del aire.

En los archivos DICOM, el valor almacenado no siempre está directamente en HU. Por eso se aplica la conversión:

$$HU(\mathbf{x}) =
m\cdot PV(\mathbf{x}) + b,$$

donde:

- $PV(\mathbf{x})$ es el valor crudo almacenado en el DICOM.

- $m$ es `RescaleSlope`.

- $b$ es `RescaleIntercept`.

También se puede modelar la imagen como una observación con ruido:

$$I_{\text{obs}}(\mathbf{x})
=
I_{\text{real}}(\mathbf{x})
+
\eta(\mathbf{x}),$$

donde $\eta(\mathbf{x})$ representa ruido de adquisición, artefactos o variaciones de reconstrucción. Por eso los umbrales HU no deben interpretarse como fronteras perfectas, sino como criterios aproximados guiados por fisiología y radiodensidad.

### Modelo de umbralización tisular

La clasificación básica de tejido se formula con funciones indicadoras. Para una clase $c$, definida por un rango de intensidades $[a_c,b_c]$, la máscara se calcula como:

$$M_c(\mathbf{x})
=
\mathbb{1}
\left(
a_c \leq I(\mathbf{x}) \leq b_c
\right).$$

Para grasa:

$$M_{\text{grasa}}(\mathbf{x})
=
\mathbb{1}
\left(
-190 \leq HU(\mathbf{x}) \leq -30
\right).$$

Para músculo:

$$M_{\text{músculo}}(\mathbf{x})
=
\mathbb{1}
\left(
-29 \leq HU(\mathbf{x}) \leq 150
\right).$$

Para hueso:

$$M_{\text{hueso}}(\mathbf{x})
=
\mathbb{1}
\left(
HU(\mathbf{x}) > 300
\right).$$

La función indicadora se define como:

$$\mathbb{1}(\text{condición})
=
\begin{cases}
1, & \text{si la condición es verdadera},\\
0, & \text{si la condición es falsa}.
\end{cases}$$

Este modelo es simple, interpretable y útil en TAC porque la escala HU tiene significado físico. Sin embargo, requiere filtros anatómicos adicionales para evitar confundir aire, fondo, ruido o tejido externo con la región real de interés.

### Modelo geométrico voxel–mundo físico

Una imagen médica tiene dos sistemas de coordenadas:

$$\text{coordenadas de voxel: } \mathbf{v}=(i,j,k)^T,$$

$$\text{coordenadas físicas: } \mathbf{r}=(x,y,z)^T.$$

La relación se modela mediante una matriz affine:

$$\tilde{\mathbf{r}}
=
A\tilde{\mathbf{v}},$$

donde:

$$\tilde{\mathbf{r}}=
\begin{bmatrix}
x\\y\\z\\1
\end{bmatrix},
\qquad
\tilde{\mathbf{v}}=
\begin{bmatrix}
i\\j\\k\\1
\end{bmatrix}.$$

La matriz affine tiene la forma:

$$A=
\begin{bmatrix}
a_{11} & a_{12} & a_{13} & x_0\\
a_{21} & a_{22} & a_{23} & y_0\\
a_{31} & a_{32} & a_{33} & z_0\\
0 & 0 & 0 & 1
\end{bmatrix}.$$

Los primeros tres vectores columna codifican orientación y tamaño del voxel; la última columna codifica el origen físico.

La matriz inversa permite pasar de coordenadas físicas a coordenadas de voxel:

$$\tilde{\mathbf{v}}
=
A^{-1}
\tilde{\mathbf{r}}.$$

Esto es fundamental para alinear máscaras, mapas funcionales y atlas.

### Modelo de remuestreo

Cuando una máscara o imagen $I_A$ está en una grilla y se quiere llevar a la grilla de otra imagen $I_B$, se usa la composición de affines:

$$\tilde{\mathbf{v}}_A
=
A_A^{-1}
A_B
\tilde{\mathbf{v}}_B.$$

La imagen remuestreada queda:

$$I_{A\rightarrow B}(\mathbf{v}_B)
=
I_A
\left(
A_A^{-1}A_B\tilde{\mathbf{v}}_B
\right).$$

Como las coordenadas resultantes pueden no ser enteras, se requiere interpolación. Para imágenes continuas se puede usar interpolación lineal:

$$I(\mathbf{x})
=
\sum_{\mathbf{n}\in\mathcal{N}(\mathbf{x})}
w_{\mathbf{n}} I(\mathbf{n}),$$

donde $w_{\mathbf{n}}$ son pesos de interpolación.

Para máscaras o etiquetas se usa vecino más cercano:

$$L_{A\rightarrow B}(\mathbf{v}_B)
=
L_A
\left(
\operatorname{round}
\left(
A_A^{-1}A_B\tilde{\mathbf{v}}_B
\right)
\right).$$

La razón es que las etiquetas son discretas. Una interpolación lineal produciría valores artificiales como $0.3$, $1.7$ o $2.4$, que no corresponden a ninguna clase anatómica real.

### Modelo morfológico para limpieza de máscaras

Después de umbralizar, las máscaras pueden contener huecos, puntos aislados o bordes irregulares. Para corregir esto se usan operaciones morfológicas.

La erosión de una máscara $M$ por un elemento estructurante $B$ se define como:

$$M \ominus B
=
\{\mathbf{x}: B_{\mathbf{x}}\subseteq M\}.$$

La dilatación se define como:

$$M \oplus B
=
\{\mathbf{x}: B_{\mathbf{x}}\cap M \neq \emptyset\}.$$

La apertura elimina objetos pequeños:

$$M \circ B =
(M \ominus B)\oplus B.$$

El cierre rellena huecos o conecta regiones cercanas:

$$M \bullet B =
(M \oplus B)\ominus B.$$

En el pipeline, estas operaciones ayudan a estabilizar máscaras de hueso, cerebro, músculo y grasa.

### Modelo de componentes conectados

Una máscara binaria puede contener varias regiones separadas. Se define una relación de conectividad $\sim$ entre voxeles vecinos. Dos voxeles pertenecen al mismo componente si existe un camino conectado entre ellos.

La máscara puede escribirse como unión disjunta:

$$M =
\bigcup_{q=1}^{Q} C_q,
\qquad
C_q \cap C_r = \emptyset
\quad
\text{si } q\neq r.$$

El componente principal se selecciona como:

$$C^\star =
\arg\max_{C_q}
|C_q|.$$

Esto se usa, por ejemplo, para conservar el cerebro principal o el componente óseo dominante y descartar ruido.

### Modelo anatómico de landmarks

El objetivo de los landmarks es evitar mediciones arbitrarias. En vez de usar el corte medio del archivo, se usan referencias anatómicas.

Sea $\mathcal{B}_k$ el conjunto de voxeles óseos del corte $k$. Su centroide se calcula como:

$$\mathbf{c}_k =
\frac{1}{|\mathcal{B}_k|}
\sum_{\mathbf{x}\in \mathcal{B}_k}
\mathbf{x}.$$

La trayectoria ósea longitudinal puede modelarse como una curva discreta:

$$\Gamma =
\{\mathbf{c}_k\}_{k=k_{\min}}^{k_{\max}}.$$

El punto trocantérico se identifica como una región proximal compatible con la morfología del fémur:

$$\mathbf{L}_{\text{troc}}
=
\arg\max_{\mathbf{c}_k\in\Gamma}
Q_{\text{troc}}(\mathbf{c}_k),$$

donde $Q_{\text{troc}}$ es una función de calidad anatómica que puede incluir posición proximal, tamaño de componente, excentricidad y continuidad.

El punto tibial anteromedial se identifica como:

$$\mathbf{L}_{\text{tib}}
=
\arg\max_{\mathbf{c}_k\in\Gamma}
Q_{\text{tib}}(\mathbf{c}_k).$$

El corte medio anatómico es:

$$\mathbf{L}_{\text{medio}}
=
\frac{1}{2}
\left(
\mathbf{L}_{\text{troc}}+\mathbf{L}_{\text{tib}}
\right).$$

La ventaja de este modelo es que el análisis se referencia a una longitud anatómica, no a la longitud variable del estudio DICOM.

### Modelo de ray casting muscular

Para cada corte se usa el hueso como referencia interna. Desde el centroide del hueso $\mathbf{c}$ se lanzan rayos:

$$\mathbf{r}_{\theta}(\rho)
=
\mathbf{c}
+
\rho
\mathbf{u}_{\theta},$$

donde:

$$\mathbf{u}_{\theta}
=
\begin{bmatrix}
\cos\theta\\
\sin\theta
\end{bmatrix}.$$

La señal de intensidad sobre el rayo es:

$$s_{\theta}(\rho)
=
I(\mathbf{r}_{\theta}(\rho)).$$

Se busca el borde muscular como una transición fuerte de intensidad, pero restringida por anatomía:

$$\rho_{\min}
<
\rho^\star
<
\rho_{\max}.$$

El borde por derivada simple sería:

$$\rho^\star =
\arg\max_{\rho}
\left|
\frac{ds_{\theta}}{d\rho}
\right|.$$

Pero para mayor robustez frente a ruido, se usa CWT.

### Modelo wavelet CWT para detección de borde

La transformada wavelet continua se define como:

$$W_s(a,b)
=
\frac{1}{\sqrt{a}}
\int
s(t)
\psi^\ast
\left(
\frac{t-b}{a}
\right)
dt.$$

En forma discreta:

$$W_s(a,b)
=
\frac{1}{\sqrt{a}}
\sum_{n=1}^{N}
s[n]
\psi
\left(
\frac{n-b}{a}
\right).$$

El parámetro $a$ controla la escala. Escalas pequeñas detectan cambios finos; escalas grandes detectan transiciones más amplias. La respuesta multiescala se resume como:

$$E(b)
=
\sum_{a\in\mathcal{A}}
|W_s(a,b)|.$$

El borde estimado es:

$$b^\star
=
\arg\max_{b\in[b_{\min},b_{\max}]}
E(b).$$

La confianza del borde puede definirse como:

$$C_{\text{edge}}
=
\frac{E(b^\star)}
{\frac{1}{N}\sum_{b}E(b)+\varepsilon}.$$

Si $C_{\text{edge}}$ es bajo, el rayo puede descartarse como no confiable.

### Modelo de área, perímetro y espesor subcutáneo

Si una región binaria $M$ contiene $N_M$ píxeles en un corte, su área es:

$$A_M =
N_M\Delta_x\Delta_y.$$

En centímetros cuadrados:

$$A_{M,\text{cm}^2}
=
\frac{N_M\Delta_x\Delta_y}{100}.$$

Si se mide espesor subcutáneo por rayos, el espesor en el rayo $n$ puede definirse como:

$$e_n =
\rho_{\text{fascia},n}
-
\rho_{\text{piel},n}.$$

El espesor medio es:

$$\bar{e}
=
\frac{1}{N_{\text{válidos}}}
\sum_{n=1}^{N_{\text{válidos}}}
e_n.$$

La fracción de rayos válidos es:

$$f_{\text{rayos}}
=
\frac{N_{\text{válidos}}}{N_{\text{total}}}.$$

El perímetro puede aproximarse a partir del contorno $\partial M$:

$$P(M)
\approx
\sum_{r=1}^{R-1}
\|\mathbf{p}_{r+1}-\mathbf{p}_{r}\|_2.$$

### Modelo de volumen y propagación de error

El volumen de una máscara 3D es:

$$V_M =
N_M\Delta_x\Delta_y\Delta_z.$$

En centímetros cúbicos:

$$V_{M,\text{cm}^3}
=
\frac{N_M\Delta_x\Delta_y\Delta_z}{1000}.$$

Si existe incertidumbre en el número de voxeles o en el espaciamiento, el error relativo aproximado es:

$$\frac{\delta V}{V}
\approx
\frac{\delta N}{N}
+
\frac{\delta \Delta_x}{\Delta_x}
+
\frac{\delta \Delta_y}{\Delta_y}
+
\frac{\delta \Delta_z}{\Delta_z}.$$

Para una razón:

$$R =
\frac{A}{B+\varepsilon},$$

la propagación de error aproximada es:

$$\frac{\delta R}{R}
\approx
\sqrt{
\left(\frac{\delta A}{A}\right)^2
+
\left(\frac{\delta B}{B}\right)^2
}.$$

Esto es importante para interpretar razones como grasa/músculo. Un pequeño error en una máscara puede amplificarse si el denominador es pequeño.

### Modelo de corrección cortical por optimización

La ROI cortical original se denota como $\Omega_R$. Se busca una transformación $T$ que ubique la ROI sobre la corteza del paciente:

$$\Omega_T = T(\Omega_R).$$

La transformación más segura inicialmente es una traslación:

$$T_{\mathbf{t}}(\mathbf{x})=
\mathbf{x}+\mathbf{t}.$$

Si se permite escala:

$$T_{\mathbf{t},s}(\mathbf{x})
=
s(\mathbf{x}-\mathbf{c})
+
\mathbf{c}
+
\mathbf{t}.$$

La máscara cerebral es $B$, la capa cortical es $C$ y el hemisferio esperado es $H$. Entonces la función objetivo se puede expresar como:

$$S(T)
=
w_c f_{\text{corteza}}(T)
+
w_b f_{\text{cerebro}}(T)
+
w_h f_{\text{hemisferio}}(T)
-
w_t P_{\text{movimiento}}(T)
-
w_o P_{\text{fuera}}(T).$$

Donde:

$$f_{\text{corteza}}(T)
=
\frac{|\Omega_T\cap C|}{|\Omega_T|},$$

$$f_{\text{cerebro}}(T)
=
\frac{|\Omega_T\cap B|}{|\Omega_T|},$$

$$P_{\text{fuera}}(T)
=
1-
f_{\text{cerebro}}(T),$$

$$P_{\text{movimiento}}(T)
=
\frac{\|\mathbf{t}\|_2}{t_{\max}}.$$

La transformación óptima es:

$$T^\star =
\arg\max_T S(T).$$

El modelo combina una restricción anatómica con una penalización de movimiento excesivo. Esto evita que la máscara se mueva demasiado solo para mejorar superficialmente el solapamiento.

### Métricas de solapamiento espacial

Para comparar dos máscaras $A$ y $B$, se usa Dice:

$$Dice(A,B)
=
\frac{2|A\cap B|}
{|A|+|B|}.$$

El índice de Jaccard es:

$$J(A,B)
=
\frac{|A\cap B|}
{|A\cup B|}.$$

La distancia entre centroides es:

$$d_c(A,B)
=
\|\mathbf{c}_A-\mathbf{c}_B\|_2,$$

donde:

$$\mathbf{c}_A=
\frac{1}{|A|}
\sum_{\mathbf{x}\in A}
\mathbf{x}.$$

También puede usarse la distancia de Hausdorff:

$$d_H(A,B)
=
\max
\left\{
\sup_{\mathbf{a}\in A}\inf_{\mathbf{b}\in B}
\|\mathbf{a}-\mathbf{b}\|,
\;
\sup_{\mathbf{b}\in B}\inf_{\mathbf{a}\in A}
\|\mathbf{b}-\mathbf{a}\|
\right\}.$$

Dice y Jaccard miden solapamiento; la distancia de centroides mide desplazamiento global; Hausdorff mide error máximo de borde.

### Modelo de atlas determinístico

Un atlas determinístico asigna una etiqueta anatómica única a cada punto:

$$A(\mathbf{x})=\ell,
\qquad
\ell\in\mathcal{L}.$$

La ROI para una etiqueta $\ell_0$ se obtiene como:

$$\Omega_{\ell_0}
=
\{\mathbf{x}:A(\mathbf{x})=\ell_0\}.$$

Para corteza motora primaria:

$$\Omega_{\text{M1}}
=
\{\mathbf{x}:A(\mathbf{x})=\text{precentral}\}.$$

Para corteza motora secundaria o premotora compuesta:

$$\Omega_{\text{M2}}
=
\bigcup_{\ell\in\mathcal{L}_{\text{M2}}}
\{\mathbf{x}:A(\mathbf{x})=\ell\}.$$

Donde:

$$\mathcal{L}_{\text{M2}}
=
\{\text{caudal middle frontal},
\text{superior frontal},
\text{paracentral},
\ldots\}.$$

### Modelo de atlas probabilístico

Un atlas probabilístico no asigna una sola etiqueta, sino una probabilidad por región:

$$P(\ell|\mathbf{x}).$$

La etiqueta más probable es:

$$\ell^\star(\mathbf{x})
=
\arg\max_{\ell}
P(\ell|\mathbf{x}).$$

Una ROI probabilística se define por umbral:

$$\Omega_{\ell,\tau}
=
\{
\mathbf{x}:P(\ell|\mathbf{x})\geq \tau
\}.$$

Por ejemplo, para una región motora:

$$\Omega_{\text{motor},\tau}
=
\{
\mathbf{x}:P(\text{motor}|\mathbf{x})\geq 0.5
\}.$$

Este modelo permite incorporar incertidumbre anatómica, especialmente cuando se usan plantillas poblacionales.

### Modelo MNI–paciente

Sea $\phi$ la transformación que lleva el paciente al espacio MNI:

$$\phi:
\Omega_{\text{paciente}}
\rightarrow
\Omega_{\text{MNI}}.$$

Si se tiene un atlas en MNI, la ROI en espacio nativo del paciente se obtiene con la inversa:

$$\Omega_{\text{paciente}}
=
\phi^{-1}
(
\Omega_{\text{MNI}}
).$$

A nivel de coordenadas:

$$\mathbf{x}_{\text{MNI}}
=
\phi(\mathbf{x}_{\text{paciente}}),$$

$$\mathbf{x}_{\text{paciente}}
=
\phi^{-1}(\mathbf{x}_{\text{MNI}}).$$

Si $\phi$ es afín:

$$\mathbf{x}_{\text{MNI}}
=
M\mathbf{x}_{\text{paciente}}
+
\mathbf{t}.$$

Entonces:

$$\mathbf{x}_{\text{paciente}}
=
M^{-1}
(
\mathbf{x}_{\text{MNI}}-\mathbf{t}
).$$

Si $\phi$ es no lineal:

$$\phi(\mathbf{x})
=
\mathbf{x}
+
\mathbf{u}(\mathbf{x}),$$

donde $\mathbf{u}(\mathbf{x})$ es un campo de desplazamiento. En este caso la inversa se calcula numéricamente.

El principio metodológico recomendado es:

$$\text{usar MNI para localizar}
\quad
\text{pero medir en espacio nativo}.$$

### Modelo FreeSurfer

FreeSurfer estima superficies corticales. Conceptualmente, reconstruye dos superficies principales:

$$S_{\text{white}}
=
\text{interfaz sustancia blanca--gris},$$

$$S_{\text{pial}}
=
\text{interfaz sustancia gris--LCR}.$$

El grosor cortical en un punto $p$ puede aproximarse como:

$$T(p)
=
\|S_{\text{pial}}(p)-S_{\text{white}}(p)\|_2.$$

El volumen cortical de una región puede aproximarse como:

$$V_{\text{cortical}}
\approx
\sum_{f\in\mathcal{F}_{ROI}}
A_f \bar{T}_f,$$

donde:

- $A_f$ es el área de una cara o elemento de superficie.

- $\bar{T}_f$ es el grosor medio asociado a esa cara.

Si se trabaja con máscara volumétrica derivada de FreeSurfer:

$$V_{\text{ROI}}
=
N_{\text{ROI}}V_{\text{voxel}}.$$

La ventaja de FreeSurfer es que la parcelación se ajusta a la anatomía cortical del paciente, especialmente a giros y surcos.

### Modelo de intersección anatómico–funcional

Cuando existe mapa funcional, la ROI final puede restringirse así:

$$\Omega_{\text{final}}
=
\Omega_{\text{atlas}}
\cap
\Omega_{\text{corteza}}
\cap
\Omega_{\text{funcional}}.$$

El mapa funcional puede provenir de un contraste estadístico. Si $Z(\mathbf{x})$ es un mapa de activación, se define:

$$\Omega_{\text{funcional}}
=
\{\mathbf{x}:Z(\mathbf{x})\geq Z_{\text{thr}}\}.$$

También puede usarse una ROI ponderada:

$$\bar{Z}_{\text{ROI}}
=
\frac{
\sum_{\mathbf{x}\in\Omega_{\text{ROI}}}
Z(\mathbf{x})
}
{
|\Omega_{\text{ROI}}|
}.$$

O ponderada por probabilidad de atlas:

$$\bar{Z}_{\text{ponderado}}
=
\frac{
\sum_{\mathbf{x}}
P_{\text{motor}}(\mathbf{x})Z(\mathbf{x})
}
{
\sum_{\mathbf{x}}
P_{\text{motor}}(\mathbf{x})+\varepsilon
}.$$

### Modelo GLM para fMRI

En fMRI, una forma estándar de modelar la señal es el modelo lineal general:

$$\mathbf{y}
=
X\boldsymbol{\beta}
+
\boldsymbol{\varepsilon}.$$

Donde:

- $\mathbf{y}$ es la serie temporal BOLD de un voxel.

- $X$ es la matriz de diseño experimental.

- $\boldsymbol{\beta}$ son los coeficientes del modelo.

- $\boldsymbol{\varepsilon}$ es el error.

La estimación por mínimos cuadrados es:

$$\hat{\boldsymbol{\beta}}
=
(X^TX)^{-1}X^T\mathbf{y}.$$

Un contraste motor se define con un vector $\mathbf{c}$:

$$\theta =
\mathbf{c}^T\hat{\boldsymbol{\beta}}.$$

El estadístico t puede expresarse como:

$$t =
\frac{
\mathbf{c}^T\hat{\boldsymbol{\beta}}
}
{
\sqrt{
\hat{\sigma}^2
\mathbf{c}^T
(X^TX)^{-1}
\mathbf{c}
}
}.$$

Este mapa estadístico puede intersectarse con la ROI anatómica para analizar activación motora dentro de M1 o M2.

### Modelo de filtrado EMG

La señal EMG cruda $x[n]$ puede contener ruido de baja frecuencia, interferencia eléctrica y artefactos. Por eso se puede modelar un filtrado:

$$x_f[n]
=
(h*x)[n]
=
\sum_{m=-\infty}^{\infty}
h[m]x[n-m].$$

Para un filtro pasa banda ideal:

$$H(f)=
\begin{cases}
1, & f_{\min}\leq |f|\leq f_{\max},\\
0, & \text{en otro caso}.
\end{cases}$$

La señal rectificada es:

$$x_r[n]=|x_f[n]|.$$

La envolvente se puede obtener con suavizado:

$$e[n]
=
\frac{1}{W}
\sum_{m=0}^{W-1}
x_r[n-m].$$

O con RMS móvil:

$$RMS[n]
=
\sqrt{
\frac{1}{W}
\sum_{m=0}^{W-1}
x_f[n-m]^2
}.$$

### Modelo temporal–frecuencial EMG

La transformada de Fourier describe contenido frecuencial global:

$$X(f)=
\sum_{n=0}^{N-1}
x[n]e^{-j2\pi fn/N}.$$

La potencia espectral es:

$$P(f)=|X(f)|^2.$$

La frecuencia mediana $f_{\text{med}}$ cumple:

$$\sum_{f=0}^{f_{\text{med}}}P(f)
=
\frac{1}{2}
\sum_{f=0}^{f_{\max}}P(f).$$

Para señales no estacionarias, puede usarse STFT:

$$X(m,\omega)
=
\sum_{n}
x[n]w[n-m]e^{-j\omega n},$$

o CWT:

$$W_x(a,b)
=
\frac{1}{\sqrt{a}}
\sum_n
x[n]\psi
\left(
\frac{n-b}{a}
\right).$$

Estas representaciones permiten ver cambios de frecuencia durante fases de contracción.

### Modelo de sincronización neuromuscular

Para dos señales $x[n]$ y $y[n]$, la coherencia mide acoplamiento en frecuencia:

$$C_{xy}(f)
=
\frac{|P_{xy}(f)|^2}
{P_{xx}(f)P_{yy}(f)}.$$

Sus valores están entre 0 y 1:

$$0\leq C_{xy}(f)\leq 1.$$

El PLV mide sincronía de fase:

$$PLV =
\left|
\frac{1}{N}
\sum_{n=1}^{N}
e^{j(\phi_x[n]-\phi_y[n])}
\right|.$$

Si $PLV\approx 1$, las fases están fuertemente sincronizadas. Si $PLV\approx 0$, no hay sincronía consistente.

### Modelo dinámico de fuerza

La dinamometría entrega una curva de fuerza $F(t)$. Además de fuerza máxima y media, se calcula la tasa de desarrollo de fuerza:

$$RFD(t)
=
\frac{dF(t)}{dt}.$$

En forma discreta:

$$RFD[n]
=
\frac{F[n]-F[n-1]}{\Delta t}.$$

El máximo desarrollo de fuerza es:

$$RFD_{\max}
=
\max_n RFD[n].$$

La fuerza normalizada al peso corporal puede escribirse como:

$$F_{\text{norm}}
=
\frac{F_{\max}}{m_{\text{corporal}}}.$$

Si se normaliza contra el sano:

$$F_{\text{rel-sano}}
=
\frac{F_{\text{paciente}}}{F_{\text{sano}}+\varepsilon}.$$

### Modelo estadístico longitudinal

Para cada métrica $m$, sujeto $s$ y etapa $t$, se tiene:

$$y_{s,t,m}.$$

El cambio individual se define como:

$$\Delta y_{s,m}
=
y_{s,\text{Después},m}
-
y_{s,\text{Antes},m}.$$

El cambio porcentual:

$$\Delta y_{s,m}^{\%}
=
100
\frac{
y_{s,\text{Después},m}
-
y_{s,\text{Antes},m}
}
{
y_{s,\text{Antes},m}+\varepsilon
}.$$

Un modelo lineal simple para evaluar efecto de etapa es:

$$y_{s,t,m}
=
\beta_0
+
\beta_1\text{Etapa}_t
+
\varepsilon_{s,t}.$$

Si se quiere incluir variabilidad entre sujetos, se puede usar un modelo mixto:

$$y_{s,t,m}
=
\beta_0
+
\beta_1\text{Etapa}_t
+
u_s
+
\varepsilon_{s,t},$$

donde $u_s$ es un efecto aleatorio del sujeto:

$$u_s\sim \mathcal{N}(0,\sigma_u^2),
\qquad
\varepsilon_{s,t}\sim \mathcal{N}(0,\sigma^2).$$

Este modelo reconoce que cada paciente puede tener un nivel basal diferente.

### Modelo de distancia al sano

El sano se usa como referencia funcional o anatómica. Para una métrica $m$:

$$d_{s,t,m}
=
|y_{s,t,m}-y_{\text{sano},t,m}|.$$

El cambio de distancia al sano es:

$$\Delta d_{s,m}
=
d_{s,\text{Antes},m}
-
d_{s,\text{Después},m}.$$

Si $\Delta d_{s,m}>0$, el paciente se acercó al sano. Si $\Delta d_{s,m}<0$, se alejó.

El índice de recuperación es:

$$IR_{s,m}
=
\frac{
d_{s,\text{Antes},m}
-
d_{s,\text{Después},m}
}
{
d_{s,\text{Antes},m}
+
\varepsilon
}.$$

Este índice tiene una interpretación directa:

$$IR=1
\Rightarrow
\text{recuperación completa hacia el sano},$$

$$IR=0
\Rightarrow
\text{sin cambio en distancia},$$

$$IR<0
\Rightarrow
\text{empeoramiento relativo}.$$

### Modelo de confiabilidad y estabilidad

Para evaluar estabilidad de una métrica entre pacientes se usa el coeficiente de variación:

$$CV(\%)
=
100
\frac{\sigma}{\mu+\varepsilon}.$$

En corteza motora, un $CV$ bajo puede sugerir estabilidad del protocolo de localización, mientras que un $CV$ muy alto puede indicar errores de segmentación, registro o variación anatómica no controlada.

La confiabilidad también puede aproximarse con ICC. Para mediciones repetidas:

$$ICC
=
\frac{\sigma_s^2}
{\sigma_s^2+\sigma_e^2},$$

donde:

- $\sigma_s^2$ es la varianza entre sujetos.

- $\sigma_e^2$ es la varianza de error.

Un $ICC$ cercano a 1 indica alta confiabilidad.

### Modelo de decisión integrada

Finalmente, una interpretación multimodal puede formularse como un vector de características:

$$\mathbf{z}_{s,t}
=
[
V_{\text{músculo}},
V_{\text{grasa}},
R_{\text{GT/M}},
F_{\max},
RMS,
iEMG,
PLV,
V_{\text{M1}},
A_{\text{M1}},
\ldots
]^T.$$

El cambio global puede medirse como distancia entre vectores:

$$D_s
=
\|
\mathbf{z}_{s,\text{Después}}
-
\mathbf{z}_{s,\text{Antes}}
\|_2.$$

La aproximación al sano puede medirse como:

$$D_{\text{sano},s,t}
=
\|
\mathbf{z}_{s,t}
-
\mathbf{z}_{\text{sano},t}
\|_2.$$

Y la recuperación vectorial como:

$$IR^{\text{multi}}_s
=
\frac{
D_{\text{sano},s,\text{Antes}}
-
D_{\text{sano},s,\text{Después}}
}
{
D_{\text{sano},s,\text{Antes}}+\varepsilon
}.$$

Este modelo resume la recuperación global integrando estructura, composición corporal, señal muscular, fuerza y corteza motora.

## Control de calidad

El pipeline no solo produce números. También genera elementos de verificación:

- Máscaras NIfTI de músculo, grasa y corteza.

- Overlays anatómicos.

- PNG de previsualización.

- CSV de métricas.

- Reportes JSON de corrección cortical.

- Archivos `.trk`, mapas de densidad, mapas RGB y labelmaps coloreados de CST.

- Waypoints de mesencéfalo, puente y bulbo para validación de tractografía.

- Manifiestos de archivos generados.

Para la corrección cortical, las métricas de calidad principales son:

$$f_{\text{inside}},
\qquad
f_{\text{shell}},
\qquad
\|\mathbf{t}\|,
\qquad
S(T^\star).$$

Para tractografía:

$$N_{CST,L},\qquad N_{CST,R},\qquad V_{CST,L},\qquad V_{CST,R},\qquad A_{CST}.$$

Para TAC:

$$N_{\text{cortes válidos}},
\qquad
\text{fracción de cortes válidos},
\qquad
\text{áreas por corte},
\qquad
\text{volúmenes finales}.$$

## Interpretación integrada

El análisis final combina métricas periféricas y centrales. Por ejemplo:

$$\text{Mejora periférica}
=
\downarrow R_{\text{GT/M}}
+
\uparrow V_{\text{músculo}}
+
\uparrow F_{\max}.$$

Una lectura neurofuncional puede integrar:

$$\text{Recuperación neuromuscular}
=
f(
RMS,\; iEMG,\; C_{xy},\; PLV,\; F_{\max},\; V_{\text{M1}},\; A_{\text{M1}}
).$$

La relación entre fuerza, músculo y activación puede analizarse con correlación:

$$r_{xy}
=
\frac{
\sum_i (x_i-\bar{x})(y_i-\bar{y})
}
{
\sqrt{\sum_i(x_i-\bar{x})^2}
\sqrt{\sum_i(y_i-\bar{y})^2}
}.$$

Donde, por ejemplo:

$$x_i=V_{\text{músculo},i},
\qquad
y_i=F_{\max,i}.$$

## Flujo completo del pipeline

    DATOS CRUDOS
    |
    |-- TAC DICOM
    |   |-- lectura y ordenamiento de cortes
    |   |-- conversion a HU
    |   |-- construccion de affine
    |   |-- segmentacion osea
    |   |-- deteccion de trocanter y punto tibial
    |   |-- corte medio anatomico
    |   |-- segmentacion muscular por ray casting + CWT
    |   |-- grasa subcutanea
    |   |-- grasa intramuscular
    |   |-- tejido adiposo total
    |   |-- exportacion NIfTI / CSV / PNG
    |
    |-- RM / Brain00mm / mapas funcionales
    |   |-- remuestreo de mascaras
    |   |-- deteccion de cerebro
    |   |-- capa cortical externa
    |   |-- correccion por traslacion
    |   |-- atlas-prior
    |   |-- FreeSurfer / MNI si existen
    |   |-- ROI M1 / M2
    |   |-- volumen y asimetria
    |
    |-- Difusion / tractografia / CST
    |   |-- diagnostico de serie DWI
    |   |-- conversion DICOM a NIfTI + bval/bvec
    |   |-- modelo de difusion y FA
    |   |-- tractografia wholebrain propia
    |   |-- registro a rT1
    |   |-- waypoints M1/M2 + mesencefalo + puente + bulbo
    |   |-- filtrado CST izquierda/derecha
    |   |-- densidad, RGB, .trk y labelmap coloreado
    |
    |-- EMG
    |   |-- limpieza de senales
    |   |-- RMS / iEMG / frecuencia
    |   |-- coherencia / PLV / correlacion cruzada
    |
    |-- Dinamometria
    |   |-- fuerza maxima
    |   |-- fuerza media
    |   |-- AUC
    |   |-- asimetria
    |
    |-- Estadistica
        |-- consolidacion
        |-- z-score
        |-- robust z-score
        |-- ratio vs sano
        |-- distancia al sano
        |-- indice de recuperacion
        |-- comparacion Antes vs Despues

## Limitaciones metodológicas

- Los rangos HU son aproximaciones y dependen de calidad de adquisición.

- El atlas-prior heurístico no reemplaza FreeSurfer ni un registro anatómico completo.

- MNI es útil para estandarización, pero la medición final de volumen debe preferirse en espacio nativo.

- La señal BOLD no mide volumen directamente; debe integrarse con anatomía estructural.

- La tractografía estima trayectorias probables a partir de difusión; no representa axones individuales y depende de calidad DWI, gradientes, registro, modelo local y criterios de filtrado.

- Con pocos pacientes, los tamaños de efecto y la dirección del cambio pueden ser más informativos que el p-value.

## Conclusión

El pipeline transforma datos médicos y biomecánicos crudos en métricas cuantificables y comparables. La TAC permite caracterizar músculo y grasa; la resonancia y los atlas permiten localizar corteza motora; EMG y dinamometría describen activación y desempeño mecánico; y la estadística integra todas las modalidades para evaluar cambios Antes–Después y aproximación al sano.

La formulación matemática del pipeline garantiza trazabilidad en cada transformación: conversión a HU, construcción affine, remuestreo, registro, segmentación, cálculo volumétrico, normalización y comparación estadística. La integración con FreeSurfer/MNI fortalece la ubicación anatómica de M1/M2 y mejora la validez del análisis de corteza motora. La incorporación de tractografía propia y filtrado CST añade un componente de conectividad estructural, permitiendo estudiar no solo la morfología cortical, muscular y adiposa, sino también la trayectoria probable de la vía motora descendente en relación con el tronco encefálico.
