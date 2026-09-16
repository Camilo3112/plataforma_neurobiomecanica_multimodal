# Modelos físicos y matemáticos documentados en la suite

## 1. Señales EMG y dinamometría

La EMG se trata como señal temporal discreta. Se remueve la media para analizar el componente alternante, se alinean tasas de muestreo diferentes por interpolación temporal y se calculan métricas de acople temporal y espectral. La correlación cruzada estima similitud en función del retardo. La FFT permite evaluar contenido frecuencial y reducir el costo computacional de correlaciones largas.

## 2. Tomografía/TAC

El modelo físico se basa en unidades Hounsfield. La intensidad del voxel aproxima atenuación radiológica relativa y permite separar rangos compatibles con aire, grasa, músculo y hueso. El procesamiento conserva geometría DICOM, espaciado, orientación y tamaño de voxel. Se usan operaciones morfológicas para limpiar máscaras y análisis de componentes conectados para separar estructuras.

## 3. Neuroimagen estructural y funcional

Las imágenes NIfTI se procesan respetando matriz afín, orientación y espaciado. Las máscaras discretas se remuestrean con vecino más cercano. Los mapas continuos pueden usar interpolación lineal. La comparación longitudinal usa delta = después - antes y métricas normalizadas cuando corresponde.

## 4. FreeSurfer y atlas motor

FreeSurfer `recon-all` genera segmentaciones estructurales y parcelación cortical. `aparc+aseg.mgz` funciona como evidencia de reconstrucción anatómica completa. Para corteza motora, M1 se aproxima con región precentral y M2 como compuesto anatómico premotor/SMA cuando el atlas lo permite.

## 5. Tractografía por DWI

La difusión se modela con gradientes b=0/b>0. La suite usa tensor de difusión como base y CSD cuando la adquisición y la estimación de respuesta lo permiten. La tractografía local sigue direcciones principales hasta que se cumple un criterio de parada basado en FA u otra métrica de coherencia. Las fibras CST se filtran por waypoints anatómicos: corteza motora, mesencéfalo, puente y bulbo.

## 6. Registro DWI → T1

La alineación entre b0 y rT1 usa registro afín/rigido con información mutua y corrección opcional por centro de masa. Si `b0_en_T1` está desplazado, la tractografía transformada también lo estará; por eso la validación visual del registro es obligatoria.

## 7. Salidas de visualización

- `.trk`: conserva streamlines reales.
- `.nii`: representación voxelizada RGB o escalar para superponer en visores.
- `.vtk`: respaldo compatible con 3D Slicer como modelo.
- labelmaps: máscaras discretas con tablas de color para ITK-SNAP/Slicer.
