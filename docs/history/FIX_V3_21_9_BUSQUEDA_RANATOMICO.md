# FIX v3.21.9 — búsqueda robusta de rAnatomico/rT1 para tractografía CST

## Problema
El módulo CST/tronco buscaba principalmente `rT1.nii` en `ResultadosFuncional/REFORMATEO`, pero en los datos reales el archivo anatómico reformateado puede llamarse `rAnatomico.nii` y estar en rutas como:

- `datos/paciente 3/Antes/ResultadosFuncional/REFORMATEO/rAnatomico.nii`
- `datos/paciente 6/Despues/ResultadosFuncional/Resultados/Resultados/rAnatomico.nii`

## Corrección
`src/cst_tronco_runner.py` ahora busca:

1. rutas explícitas con `rAnatomico.nii`, `rAnatomico.nii.gz`, `rT1.nii` y `rT1.nii.gz`;
2. rutas bajo `ResultadosFuncional`, `Resultados funcional`, `RESONANCIA` y `Resonancia`;
3. búsqueda recursiva de archivos NIfTI anatómicos/T1;
4. exclusión de mapas funcionales, máscaras, DWI, FA, ADC, BOLD y otros derivados no anatómicos.

El resumen global agrega `t1_path` y `t1_candidates_checked` para diagnosticar qué encontró.

## Modo de ejecución

```bash
./01_run_interactivo_tractografia_sin_suspension_linux.sh
```

O directo:

```bash
python3 main.py \
  --project-root "/home/humath/Escritorio" \
  --data-root "/home/humath/Escritorio/datos" \
  --patients 3 6 \
  --stages Antes Despues \
  --only cst_tronco \
  --gpu off \
  --force \
  --no-resume \
  --skip-stuck
```
