# Suite VCE v3.21.13 — rescate anatómico CST

Esta versión corrige el caso donde la tractografía wholebrain sí se genera, pero no aparecen las vías corticoespinales (`cst_m1_left_T1.trk`, `cst_m1_right_T1.trk`).

## Cambios

1. Conserva la búsqueda selectiva de DWI/DTI cruda por `SeriesInstanceUID`.
2. Conserva la búsqueda recursiva de `rT1.nii` y `rAnatomico.nii`.
3. El filtro CST ya no exige cruce voxel exacto M1/M2 + tronco si queda vacío.
4. Si el filtro estricto queda en cero, activa rescate anatómico por proximidad en mm a M1/M2 y tronco.
5. Copia las salidas CST a la carpeta `cst_visualizacion_ventral_medial/` para encontrarlas fácil.

## Variables útiles

```bash
export VCE_CST_RESCUE_ENABLE=1
export VCE_CST_RESCUE_MOTOR_DISTANCE_MM=18
export VCE_CST_RESCUE_BRAINSTEM_DISTANCE_MM=20
export VCE_CST_RESCUE_TOP_N=2500
export VCE_MOTOR_SEED_DILATION_MM=10
export VCE_BRAINSTEM_DILATION_MM=8
export VCE_MIN_CST_LENGTH_MM=35
export VCE_MIN_DESCENDING_Z_RANGE_MM=22
export VCE_MIN_IPSILATERAL_FRACTION=0.45
```

Estas salidas son anatómicamente aproximadas y deben validarse visualmente sobre T1/FA/ColorFA.
