CST COLOREADA V4 — AJUSTE SAGITAL Y CORONAL

QUÉ CORRIGE

La V3 corrigió el desplazamiento posterior visible en el plano sagital,
seleccionando la porción anterior/ventral del tronco.

La V4 añade dos controles para el plano coronal:

1. La línea media ya no se calcula únicamente con los centroides de M1.
   Se estima directamente desde el centro del tronco encefálico en cada
   nivel axial. Esto reduce errores por asimetría de las máscaras M1.

2. Dentro de cada mitad del tronco conserva un corredor paramediano:
   - mesencéfalo ventral y medial;
   - puente ventral y medial;
   - bulbo anterior y paramediano.

No traslada ni modifica las coordenadas de las streamlines. Vuelve a
filtrar la tractografía mediante waypoints anatómicos más específicos.

EJECUCIÓN

EJECUTAR_CST_COLORES_V4_CORONAL.cmd

SALIDA

D:\EAFIT\01-2026\proyecto\resultados\paciente 6\Antes\
tractografia_propia\cst_visualizacion_ventral_medial

ARCHIVO PRINCIPAL PARA ITK-SNAP

cst_ventral_medial_colores_itksnap_T1.nii.gz

COLORES

- rojo: CST izquierda;
- azul: CST derecha;
- magenta: superposición.

CONTROL

La carpeta también contiene seis waypoints por hemisferio. Ábrelos sobre
rT1 para verificar que estén en la zona ventral y paramediana.

IMPORTANTE

Si la desviación coronal persiste principalmente por encima del
mesencéfalo, entre el tálamo y la corteza, la corrección anatómica siguiente
debe ser añadir un waypoint de la cápsula interna posterior. No se debe
aplicar una traslación global porque desalinearía los demás planos.
