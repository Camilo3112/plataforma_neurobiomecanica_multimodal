# V3.21.16 - Crear aparc+aseg si falta y mostrar progreso en consola

Esta versión corrige el flujo de ejecución para que el proceso no avance a mesencéfalo ni CST sin verificar primero FreeSurfer.

## Secuencia obligatoria

1. Busca `rT1.nii` o `rAnatomico.nii`.
2. Genera `wholebrain_T1.trk` si hace falta.
3. Verifica:
   - `recon-all.done`
   - `mri/aseg.mgz`
   - `mri/aparc+aseg.mgz`
   - `mri/rawavg.mgz`
   - `mri/norm.mgz`
4. Si falta `aparc+aseg.mgz`, ejecuta o reanuda `recon-all -all`.
5. Solo después segmenta mesencéfalo, puente, bulbo y SCP.
6. Solo después reconstruye CST usando el mesencéfalo como waypoint.

## Progreso visible

Los subprocesos ya no quedan silenciosos en un `.log`. Ahora se muestran en pantalla y también se guardan en:

```text
resultados/paciente 3/Antes/tractografia_propia/logs_cst_tronco/
logs_tractografia/
```

## Comando recomendado

```bash
cd /home/humath/Escritorio/suite_integrada_vce_v3_21_16_linux_crea_aparc_muestra_progreso
chmod +x *.sh
./00_instalar_todo_linux.sh
./07_run_paciente_3_antes_crea_aparc_muestra_progreso_sin_suspension_linux.sh
```
