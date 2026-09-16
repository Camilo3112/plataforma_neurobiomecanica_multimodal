# Fix v3.21.4 — tomografía miembro completo

Este paquete corrige el error donde la segmentación de composición corporal cortaba el muslo por la mitad.

## Qué estaba pasando

El algoritmo anterior separaba los miembros usando la mitad izquierda/derecha de la imagen. En cortes donde un miembro cruza la línea media o el otro miembro no está completo en el campo visual, esa regla puede partir el muslo y calcular áreas falsas.

## Qué cambia

Ahora la composición corporal se calcula así:

1. Se detecta la máscara corporal del corte completo, sin partir por hemicampo.
2. Se busca el fémur más cercano al landmark del miembro.
3. Se selecciona el componente anatómico completo alrededor de ese fémur.
4. Si los muslos están tocándose, se separan por cercanía a centros femorales, no por la mitad de la imagen.
5. Los rayos ya no se detienen en la línea media de la imagen, sino en el borde del campo completo.
6. Si el contorno toca el borde de la imagen, el reporte deja una advertencia de posible truncamiento del campo de visión.

## Cómo correr solo paciente 3 y 6 en Windows

Ejecuta:

```bat
run_reprocesar_tomografia_miembro_completo_pacientes_3_6_windows.bat
```

Ese script corre:

```bat
python main.py --project-root "D:\EAFIT\01-2026\proyecto" --data-root "D:\EAFIT\01-2026\proyecto\datos" --patients 3 6 --only tomografia_avanzada --gpu auto --force --no-resume
python main.py --project-root "D:\EAFIT\01-2026\proyecto" --data-root "D:\EAFIT\01-2026\proyecto\datos" --patients 3 6 --only consolidado --gpu off --force --no-resume
```

## Si también quieres recalcular tomografía básica

Ejecuta:

```bat
run_reprocesar_tomografia_completa_3_6_windows.bat
```

## Qué revisar

Abre los PNG en:

```text
D:\EAFIT\01-2026\proyecto\resultados\paciente 3\Antes\tomografia\avanzada_tejido_adiposo_landmarks\PNG_Composicion
D:\EAFIT\01-2026\proyecto\resultados\paciente 3\Despues\tomografia\avanzada_tejido_adiposo_landmarks\PNG_Composicion
D:\EAFIT\01-2026\proyecto\resultados\paciente 6\Antes\tomografia\avanzada_tejido_adiposo_landmarks\PNG_Composicion
D:\EAFIT\01-2026\proyecto\resultados\paciente 6\Despues\tomografia\avanzada_tejido_adiposo_landmarks\PNG_Composicion
```

El contorno cyan debe rodear todo el miembro visible, no formar una línea vertical en la mitad del muslo.
