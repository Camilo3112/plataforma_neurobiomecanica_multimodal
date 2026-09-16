# Uso de `main.py`

## 1. Modo interactivo

```bash
python main.py
```

El programa pregunta pacientes, etapas y secciones.

## 2. Ejecutar una sección puntual

```bash
python main.py run --patients 3 --stages Antes --sections cst_tronco --force
```

## 3. Ejecutar varias secciones

```bash
python main.py run --patients 3 6 --stages Antes Despues --sections tomografia resonancias cst_tronco consolidado
```

## 4. Ejecutar todo el flujo

```bash
python main.py run --patients 3 6 7 9 --stages Antes Despues --sections todo --force
```

## 5. Corrección de registro b0 → T1

Primero probar eje superior/inferior:

```bash
python main.py run --patients 3 --stages Antes --sections cst_tronco --registro-com-corregir --registro-com-ejes z
```

Si todavía se ve desplazado en otros ejes:

```bash
python main.py run --patients 3 --stages Antes --sections cst_tronco --registro-com-corregir --registro-com-ejes xyz
```

## 6. Evitar suspensión

```bash
python main.py run --patients 3 --stages Antes --sections cst_tronco --no-suspend
```

En Linux usa `systemd-inhibit` si está disponible.
