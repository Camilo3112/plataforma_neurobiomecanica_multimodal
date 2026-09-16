# Suite VCE v3.21.2 Windows — tomografía + corteza motora

## Ruta esperada

Suite:

```bat
D:\EAFIT\01-2026\proyecto\suite_integrada_vce_v3_21_2_windows_tomografia_corteza_motor_interactivo
```

Datos:

```bat
D:\EAFIT\01-2026\proyecto\datos
```

Resultados:

```bat
D:\EAFIT\01-2026\proyecto\resultados
```

## Instalar dependencias

```bat
cd /d "D:\EAFIT\01-2026\proyecto\suite_integrada_vce_v3_21_2_windows_tomografia_corteza_motor_interactivo"
setup_windows.bat
```

## Correr interactivo

```bat
run_interactivo_tomografia_corteza_windows.bat
```

Cuando pregunte pacientes, escribir:

```text
3, 6
```

## Correr directo pacientes 3 y 6

```bat
run_pacientes_3_6_tomografia_corteza_windows.bat
```

## Energía

Los scripts ejecutan `powercfg` para evitar suspensión y apagado de pantalla. Al terminar, si quieres restaurar tiempos aproximados:

```bat
restaurar_energia_windows.bat
```

## FreeSurfer / MNI en Windows

FreeSurfer completo no corre de forma nativa en Windows clásico. Para FreeSurfer real se recomienda Linux/WSL2/Docker.
En Windows, la suite puede:
- correr tomografía clásica y avanzada;
- correr corrección cortical atlas-prior;
- integrar salidas FreeSurfer/MNI si ya existen en resultados.

Para reconstrucción FreeSurfer clínica completa, usar la versión Linux.
