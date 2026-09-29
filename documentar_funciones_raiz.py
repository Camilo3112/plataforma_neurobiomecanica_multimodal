#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       D O C U M E N T A C I Ó N   D E   F U N C I O N E S   R A Í Z        ║
║        Entrada · Proceso · Salida para main.py y configuración derivada      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Herramienta local para insertar comentarios técnicos dentro de funciones Python.
El objetivo es documentar cada función con tres bloques: Entrada, Proceso y
Salida, sin alterar la lógica ejecutable del programa.

Uso recomendado desde la raíz del repositorio:

    python documentar_funciones_raiz.py --project-root . --files main.py configurar_requisitos_derivados.py

La herramienta crea copias .bak antes de modificar cada archivo y genera un
reporte en docs/REPORTE_DOCUMENTACION_FUNCIONES_RAIZ.md.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  1. UTILIDADES DE LECTURA Y ESCRITURA
# ══════════════════════════════════════════════════════════════════════════════


def read_text_safe(path: Path) -> str:
    """Lee un archivo Python usando codificaciones comunes."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="replace")


def write_text_safe(path: Path, text: str) -> None:
    """Escribe texto en UTF-8 conservando saltos de línea tipo Unix."""
    path.write_text(text, encoding="utf-8", newline="\n")


def annotation_to_text(node: ast.AST | None) -> str:
    """Convierte una anotación de tipo del AST en texto legible."""
    if node is None:
        return "sin tipo explícito"
    try:
        return ast.unparse(node)
    except Exception:
        return "tipo no recuperable"


def function_has_value_return(node: ast.AST) -> bool:
    """Indica si la función contiene al menos un return con valor explícito."""
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and child.value is not None:
            return True
    return False


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  2. INFERENCIA DE ENTRADAS, PROCESO Y SALIDA
# ══════════════════════════════════════════════════════════════════════════════


def iter_function_arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[Tuple[str, str, str]]:
    """Extrae nombre, tipo y categoría de los argumentos de una función."""
    args: List[Tuple[str, str, str]] = []

    for arg in list(node.args.posonlyargs) + list(node.args.args):
        args.append((arg.arg, annotation_to_text(arg.annotation), "posicional"))

    if node.args.vararg is not None:
        args.append(("*" + node.args.vararg.arg, annotation_to_text(node.args.vararg.annotation), "variable"))

    for arg in node.args.kwonlyargs:
        args.append((arg.arg, annotation_to_text(arg.annotation), "keyword-only"))

    if node.args.kwarg is not None:
        args.append(("**" + node.args.kwarg.arg, annotation_to_text(node.args.kwarg.annotation), "diccionario"))

    return args


def describe_argument(name: str, type_text: str, kind: str) -> str:
    """Construye una descripción técnica corta para un argumento."""
    clean = name.lstrip("*")
    lowered = clean.lower()

    if clean == "self":
        detail = "instancia del objeto; permite acceder al estado interno, rutas, configuración o métodos auxiliares."
    elif clean == "cls":
        detail = "clase que invoca el método; se usa en constructores alternativos o utilidades de clase."
    elif name.startswith("**"):
        detail = "diccionario de argumentos nombrados adicionales; permite extender la configuración sin cambiar la firma principal."
    elif name.startswith("*"):
        detail = "colección variable de argumentos; permite recibir una cantidad flexible de elementos."
    elif any(token in lowered for token in ("path", "ruta", "file", "archivo", "dir", "folder")):
        detail = "ruta o archivo de entrada/salida; debe apuntar a un recurso válido dentro del proyecto o del sistema."
    elif any(token in lowered for token in ("cfg", "config", "settings", "opts", "args")):
        detail = "configuración de ejecución; agrupa parámetros, rutas, banderas y opciones del pipeline."
    elif any(token in lowered for token in ("patient", "paciente", "subject", "sujeto")):
        detail = "identificador del paciente o sujeto; permite localizar datos, resultados y carpetas asociadas."
    elif any(token in lowered for token in ("stage", "etapa", "timepoint", "antes", "despues")):
        detail = "etapa temporal del estudio; normalmente Antes o Despues para comparación longitudinal."
    elif any(token in lowered for token in ("img", "image", "nii", "volume", "vol", "mask", "roi")):
        detail = "imagen, volumen, máscara o región de interés; conserva información espacial para análisis voxel/ROI."
    elif any(token in lowered for token in ("signal", "senal", "y", "x", "emg", "force", "torque")):
        detail = "señal numérica de entrada; usualmente arreglo temporal, vector de muestras o variable biomecánica."
    elif any(token in lowered for token in ("scale", "window", "sigma", "kernel")):
        detail = "parámetro de escala, ventana o suavizado; controla resolución temporal, espacial o multiescala."
    elif any(token in lowered for token in ("df", "table", "data", "rows", "records")):
        detail = "tabla o estructura de datos; contiene mediciones, características o registros para procesar/exportar."
    else:
        detail = "parámetro de entrada usado por la rutina; debe cumplir el tipo y formato esperado por la lógica interna."

    return f"#   - {name}: {detail} Tipo declarado: {type_text}. Modo: {kind}."


def infer_process_text(name: str) -> List[str]:
    """Infiere una explicación del proceso a partir del nombre funcional."""
    n = name.lower()

    if any(t in n for t in ("cwt", "wavelet", "scalogram")):
        return [
            "#   - Normaliza la señal, aplica análisis multiescala tipo wavelet/LoG y estima energía local por escala.",
            "#   - El fundamento es observar la misma señal con diferentes niveles de resolución para detectar patrones finos y amplios.",
        ]
    if any(t in n for t in ("fft", "spectrum", "frecuencia", "frequency", "psd")):
        return [
            "#   - Transforma la señal al dominio frecuencial y estima componentes espectrales o potencia por banda.",
            "#   - El fundamento es la descomposición de Fourier: representar la señal como suma de oscilaciones sinusoidales.",
        ]
    if any(t in n for t in ("corr", "correlation", "coherence", "similarity")):
        return [
            "#   - Compara dos señales, regiones o mapas mediante una métrica de similitud o correlación.",
            "#   - El fundamento es cuantificar covariación normalizada para detectar patrones compartidos o discordantes.",
        ]
    if any(t in n for t in ("register", "align", "affine", "resample", "transform")):
        return [
            "#   - Alinea datos entre espacios mediante transformaciones geométricas, afines o remuestreo.",
            "#   - El fundamento es preservar correspondencia anatómica/espacial para comparar mapas, máscaras o tractos.",
        ]
    if any(t in n for t in ("segment", "mask", "roi", "label", "brainstem", "cst", "tract")):
        return [
            "#   - Delimita regiones anatómicas, máscaras o tractos usando etiquetas, umbrales, geometría o referencias espaciales.",
            "#   - El fundamento es reducir el análisis a regiones de interés con significado anatómico o funcional.",
        ]
    if any(t in n for t in ("load", "read", "find", "discover", "search")):
        return [
            "#   - Localiza, lee o valida recursos de entrada antes de enviarlos al pipeline principal.",
            "#   - El fundamento es asegurar trazabilidad y consistencia de rutas, archivos y datos antes del análisis.",
        ]
    if any(t in n for t in ("save", "write", "export", "report", "excel", "json", "csv")):
        return [
            "#   - Organiza resultados y los escribe en formatos persistentes como Excel, CSV, JSON, NIfTI o reportes.",
            "#   - El fundamento es conservar trazabilidad, reproducibilidad y revisión externa de las métricas calculadas.",
        ]
    if any(t in n for t in ("run", "execute", "main", "pipeline", "orquest", "dispatch")):
        return [
            "#   - Coordina la ejecución de una o varias etapas del pipeline, conectando configuración, datos y módulos especializados.",
            "#   - El fundamento es mantener un flujo reproducible donde cada sección genera salidas verificables para análisis posterior.",
        ]
    if any(t in n for t in ("clean", "delete", "remove", "reset")):
        return [
            "#   - Elimina, limpia o reinicia salidas seleccionadas para evitar mezclar resultados previos con una nueva ejecución.",
            "#   - El fundamento es mantener independencia entre corridas y controlar artefactos residuales.",
        ]
    if any(t in n for t in ("parse", "arg", "cli", "menu", "option")):
        return [
            "#   - Interpreta argumentos de consola o selección interactiva y los transforma en parámetros internos del programa.",
            "#   - El fundamento es separar la interfaz de usuario de la lógica técnica del pipeline.",
        ]

    return [
        f"#   - Ejecuta la rutina `{name}` aplicando validaciones, transformaciones y operaciones definidas por su bloque interno.",
        "#   - El fundamento depende del módulo donde se ubica: coordinación del pipeline, procesamiento numérico, lectura/escritura o análisis multimodal.",
    ]


def infer_output_text(node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
    """Genera la descripción de salida a partir de returns y anotación."""
    ret_ann = annotation_to_text(node.returns)
    has_return_value = function_has_value_return(node)

    if node.returns is not None and ret_ann not in ("None", "sin tipo explícito"):
        return [
            f"#   - retorna un objeto de tipo declarado `{ret_ann}`; contiene el resultado calculado, ruta generada, estado o estructura procesada.",
            "#   - La salida debe interpretarse según el contexto del módulo y suele alimentar etapas posteriores del pipeline.",
        ]
    if has_return_value:
        return [
            "#   - retorna un valor calculado por la función; puede ser una ruta, tabla, arreglo, diccionario, métrica o código de estado.",
            "#   - La forma exacta de la salida depende de las ramas internas y de los datos disponibles durante la ejecución.",
        ]
    return [
        "#   - no retorna un valor principal explícito; su efecto se refleja en archivos generados, cambios de estado, impresión de reportes o coordinación del flujo.",
        "#   - Si la función encuentra errores, puede detener la ejecución o propagar excepciones según la lógica interna.",
    ]


def build_comment_block(node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
    """Construye el bloque Entrada/Proceso/Salida para una función."""
    args = iter_function_arguments(node)
    lines: List[str] = []

    lines.append("# Entrada:")
    if args:
        for name, type_text, kind in args:
            lines.append(describe_argument(name, type_text, kind))
    else:
        lines.append("#   - no recibe argumentos directos; utiliza constantes, estado global controlado o recursos definidos en su contexto.")

    lines.append("# Proceso:")
    lines.extend(infer_process_text(node.name))

    lines.append("# Salida:")
    lines.extend(infer_output_text(node))

    return lines


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  3. INSERCIÓN SEGURA DE COMENTARIOS
# ══════════════════════════════════════════════════════════════════════════════


def already_documented(lines: Sequence[str], insert_at: int) -> bool:
    """Evita duplicar comentarios si la función ya tiene Entrada/Proceso/Salida."""
    window = "\n".join(lines[insert_at : min(len(lines), insert_at + 14)])
    return "# Entrada:" in window and "# Proceso:" in window and "# Salida:" in window


def document_python_file(path: Path) -> Tuple[int, int]:
    """Documenta todas las funciones de un archivo Python y retorna estadísticas."""
    original = read_text_safe(path)
    tree = ast.parse(original, filename=str(path))
    lines = original.splitlines()

    functions: List[ast.FunctionDef | ast.AsyncFunctionDef] = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    functions.sort(key=lambda n: n.lineno, reverse=True)

    inserted = 0
    skipped = 0

    for node in functions:
        if not node.body:
            skipped += 1
            continue

        first_body_line = node.body[0].lineno - 1
        if already_documented(lines, first_body_line):
            skipped += 1
            continue

        indent = ""
        if 0 <= first_body_line < len(lines):
            indent = lines[first_body_line][: len(lines[first_body_line]) - len(lines[first_body_line].lstrip())]

        block = [indent + line if line else "" for line in build_comment_block(node)]
        lines[first_body_line:first_body_line] = block + [""]
        inserted += 1

    if inserted:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        write_text_safe(path, "\n".join(lines) + "\n")

    # Verificación: debe seguir compilando.
    ast.parse(read_text_safe(path), filename=str(path))
    return inserted, skipped


def write_report(project_root: Path, results: List[Tuple[Path, int, int]]) -> Path:
    """Escribe un reporte Markdown con el resumen de documentación."""
    docs = project_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    report = docs / "REPORTE_DOCUMENTACION_FUNCIONES_RAIZ.md"
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_inserted = sum(r[1] for r in results)
    total_skipped = sum(r[2] for r in results)

    body = [
        "# Reporte de documentación de funciones raíz",
        "",
        f"Fecha de ejecución: {now}",
        "",
        "## Resumen",
        "",
        f"- Funciones documentadas: {total_inserted}",
        f"- Funciones omitidas por estar ya documentadas o sin cuerpo: {total_skipped}",
        "- Archivos procesados: " + str(len(results)),
        "- Tipo de cambio: comentarios internos Entrada / Proceso / Salida",
        "- Lógica ejecutable: sin modificación intencional",
        "",
        "## Archivos",
        "",
    ]

    for path, inserted, skipped in results:
        body.append(f"- `{path}`: documentadas {inserted}, omitidas {skipped}")

    body.append("")
    write_text_safe(report, "\n".join(body))
    return report


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  4. INTERFAZ DE CONSOLA
# ══════════════════════════════════════════════════════════════════════════════


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Define argumentos de consola para documentar archivos raíz."""
    parser = argparse.ArgumentParser(
        description="Agrega comentarios Entrada/Proceso/Salida a funciones de archivos Python raíz."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Raíz del repositorio donde están main.py y configurar_requisitos_derivados.py.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["main.py", "configurar_requisitos_derivados.py"],
        help="Archivos Python relativos al proyecto que se van a documentar.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta el documentador sobre los archivos seleccionados."""
    args = parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()

    if not root.exists():
        print(f"ERROR: no existe project-root: {root}")
        return 2

    results: List[Tuple[Path, int, int]] = []

    for rel in args.files:
        path = (root / rel).resolve()
        if not path.exists():
            print(f"ADVERTENCIA: no existe, se omite: {path}")
            continue
        if path.suffix.lower() != ".py":
            print(f"ADVERTENCIA: no es .py, se omite: {path}")
            continue

        print(f"Documentando: {path}")
        inserted, skipped = document_python_file(path)
        print(f"  funciones documentadas: {inserted} | omitidas: {skipped}")
        results.append((path, inserted, skipped))

    if not results:
        print("ERROR: no se procesó ningún archivo.")
        return 1

    report = write_report(root, results)
    print(f"Reporte: {report}")
    print("Proceso finalizado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
