"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             A N Á L I S I S   M U L T I M O D A L   V C E                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Módulo: src/analisis_multimodal_runner.py
Versión: v3.21.28

Descripción
-----------
Este módulo construye una capa cuantitativa integradora para el proyecto VCE.
La idea central es transformar resultados ya existentes en una tabla maestra
multimodal y en matrices por paciente, donde cada variable biomédica quede
indexada por paciente, etapa, modalidad, región, parámetro, unidad y fuente.

Modelo físico-matemático general
--------------------------------
El sistema representa la información como un tensor clínico-computacional:

    X[paciente, etapa, modalidad, región, parámetro]

A partir de este tensor se obtienen diferencias longitudinales:

    Δ = Después - Antes
    Δ% = 100 · (Después - Antes) / |Antes|

Comparaciones entre pacientes:

    Δ(P_i, P_3) = X(P_i) - X(P_3)

y medidas integradas por dominio mediante normalización robusta o z-score:

    z = (x - μ) / σ

Cada salida conserva la fuente del dato para que el resultado pueda auditarse.
El módulo no modifica datos crudos; únicamente lee productos existentes en la
carpeta de resultados y escribe una carpeta consolidada en resultados/analisis.
"""

###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  0. IMPORTACIONES
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except Exception:  # pragma: no cover - nibabel puede faltar en instalaciones mínimas.
    nib = None

warnings.filterwarnings("ignore")


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  1. CONSTANTES DE DOMINIO Y VARIABLES BIOMÉDICAS PRIORIZADAS
# ══════════════════════════════════════════════════════════════════════════════

KNOWN_PROTOCOLS = {
    "paciente 3": "normal",
    "paciente 6": "cross_education",
    "paciente 7": "cross_education",
    "paciente 9": "cross_education",
}

CONTROL_PATTERNS = ("sano", "control", "healthy", "normativo")

MODALITY_KEYWORDS = {
    "emg": "EMG",
    "electromiografia": "EMG",
    "dinamometria": "Dinamometría",
    "dinamometría": "Dinamometría",
    "fuerza": "Dinamometría",
    "torque": "Dinamometría",
    "tomografia": "Tomografía",
    "tomografía": "Tomografía",
    "tac": "Tomografía",
    "ct": "Tomografía",
    "grasa": "Tomografía",
    "adiposo": "Tomografía",
    "resonancia": "Resonancia",
    "mri": "Resonancia",
    "t1": "Resonancia",
    "ranatomico": "Resonancia",
    "zona": "Zonas",
    "zonas": "Zonas",
    "region": "Zonas",
    "regiones": "Zonas",
    "cst": "Tractografía CST",
    "cortico": "Tractografía CST",
    "tractografia": "Tractografía CST",
    "tractografía": "Tractografía CST",
    "streamline": "Tractografía CST",
    "freesurfer": "FreeSurfer",
    "aparc": "FreeSurfer",
    "aseg": "FreeSurfer",
}

IMPORTANT_VARIABLES = [
    ("EMG", "rms", "Amplitud efectiva de activación muscular."),
    ("EMG", "iemg", "Carga eléctrica acumulada durante la tarea."),
    ("EMG", "frecuencia_mediana", "Indicador espectral sensible a fatiga muscular."),
    ("EMG", "cwt", "Energía multiescala de activación muscular."),
    ("Dinamometría", "pico_torque", "Capacidad máxima de producción de fuerza."),
    ("Dinamometría", "trabajo_total", "Producción mecánica acumulada."),
    ("Dinamometría", "potencia", "Producción mecánica por unidad de tiempo."),
    ("Tomografía", "relacion_grasa_musculo", "Calidad estructural músculo-grasa."),
    ("Tomografía", "hu_muscular", "Densidad tisular del músculo."),
    ("Resonancia", "volumen", "Morfometría anatómica regional."),
    ("Zonas", "correlacion", "Estabilidad local Antes/Después."),
    ("Tractografía CST", "densidad", "Concentración de streamlines por región."),
    ("Tractografía CST", "fa", "Organización direccional de difusión."),
    ("Tractografía CST", "streamlines", "Continuidad reconstruida de la vía."),
]


@dataclass
class FeatureRecord:
    """Registro plano de una característica multimodal."""

    paciente: str
    protocolo: str
    etapa: str
    modalidad: str
    region: str
    parametro: str
    valor: float
    unidad: str
    fuente: str
    archivo: str


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  2. UTILIDADES DE TEXTO, NÚMEROS Y CLASIFICACIÓN BIOMÉDICA
# ══════════════════════════════════════════════════════════════════════════════


def _safe_float(value: Any) -> float | None:
    # Entrada:
    #   - value: objeto escalar proveniente de JSON, CSV, Excel, metadatos o cálculos numéricos.
    # Proceso:
    #   - Intenta convertir el valor a número real, elimina símbolos frecuentes y descarta NaN o infinitos.
    #   - Este filtro evita que textos clínicos, rutas o etiquetas anatómicas entren como biomarcadores.
    # Salida:
    #   - float válido si la conversión es posible; None si el valor no representa una medición numérica confiable.
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
    else:
        text = str(value).strip().replace(",", ".")
        text = re.sub(r"[%\s]+$", "", text)
        try:
            out = float(text)
        except Exception:
            return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _slug(text: Any) -> str:
    # Entrada:
    #   - text: nombre de paciente, región, archivo, variable o modalidad.
    # Proceso:
    #   - Normaliza acentos mínimos, espacios y caracteres especiales para producir claves comparables.
    #   - Esto permite cruzar variables equivalentes aunque provengan de archivos con nombres distintos.
    # Salida:
    #   - cadena estable en minúsculas, apta para llaves de matrices, columnas y archivos.
    text = str(text or "").strip().lower()
    rep = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n", "ü": "u"}
    for a, b in rep.items():
        text = text.replace(a, b)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "sin_nombre"


def _infer_protocol(patient: str) -> str:
    # Entrada:
    #   - patient: nombre normalizado del paciente o del control.
    # Proceso:
    #   - Asigna protocolo experimental conocido: paciente 3 como normal y pacientes 6, 7 y 9 como cross education.
    #   - Si el nombre sugiere control sano, clasifica como control_sano.
    # Salida:
    #   - etiqueta de protocolo: normal, cross_education, control_sano o desconocido.
    low = str(patient).lower()
    if any(x in low for x in CONTROL_PATTERNS):
        return "control_sano"
    return KNOWN_PROTOCOLS.get(low, "desconocido")


def _infer_modality(path: Path, extra_text: str = "") -> str:
    # Entrada:
    #   - path: ruta del archivo fuente.
    #   - extra_text: texto adicional tomado de columnas, hojas o claves del archivo.
    # Proceso:
    #   - Busca palabras clave asociadas a EMG, dinamometría, TAC, resonancia, zonas, FreeSurfer y tractografía.
    #   - La clasificación agrupa variables heterogéneas en dominios físicos comparables.
    # Salida:
    #   - nombre de modalidad biomédica o "Otros" si no se reconoce el dominio.
    text = f"{path.as_posix()} {extra_text}".lower()
    for key, modality in MODALITY_KEYWORDS.items():
        if key in text:
            return modality
    return "Otros"


def _infer_region(path: Path, row: dict[str, Any] | None = None) -> str:
    # Entrada:
    #   - path: ruta del archivo que contiene la medición.
    #   - row: fila tabular opcional con campos anatómicos o laterales.
    # Proceso:
    #   - Prioriza columnas explícitas como región, zona, estructura, músculo, lado o hemisferio.
    #   - Si no existen, deriva una región aproximada desde el nombre del archivo.
    # Salida:
    #   - etiqueta regional que permite construir matrices por paciente y comparar el mismo territorio anatómico.
    row = row or {}
    for key in ("region", "región", "zona", "estructura", "musculo", "músculo", "roi", "lado", "hemisferio"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    name = path.stem.replace(".nii", "")
    name = re.sub(r"^(resumen_|tabla_|metricas_|metrics_)", "", name, flags=re.IGNORECASE)
    return name[:90] or "global"


def _feature_key(row: pd.Series | dict[str, Any]) -> str:
    # Entrada:
    #   - row: registro de feature con modalidad, región, parámetro y unidad.
    # Proceso:
    #   - Combina los identificadores que definen una variable comparable entre pacientes o etapas.
    #   - Excluye paciente y etapa para que la misma variable pueda alinearse longitudinalmente.
    # Salida:
    #   - clave canónica modalidad|region|parametro|unidad.
    get = row.get if isinstance(row, dict) else row.get
    return "|".join([
        _slug(get("modalidad", "")),
        _slug(get("region", "")),
        _slug(get("parametro", "")),
        _slug(get("unidad", "")),
    ])


def _append_feature(records: list[FeatureRecord], patient: str, stage: str, modality: str, region: str,
                    parameter: str, value: Any, unit: str, source: str, file_path: Path) -> None:
    # Entrada:
    #   - records: lista acumuladora de biomarcadores.
    #   - patient, stage, modality, region, parameter, value, unit, source, file_path: metadatos de la medición.
    # Proceso:
    #   - Convierte el valor a número, descarta mediciones inválidas y añade protocolo experimental.
    #   - Mantiene la ruta relativa/original para auditoría y trazabilidad del dato.
    # Salida:
    #   - no retorna valor; agrega un FeatureRecord válido a records.
    numeric = _safe_float(value)
    if numeric is None:
        return
    records.append(FeatureRecord(
        paciente=patient,
        protocolo=_infer_protocol(patient),
        etapa=stage,
        modalidad=modality or "Otros",
        region=str(region or "global"),
        parametro=str(parameter or "valor"),
        valor=numeric,
        unidad=str(unit or ""),
        fuente=str(source or ""),
        archivo=str(file_path),
    ))


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  3. LECTURA DE RESULTADOS EXISTENTES
# ══════════════════════════════════════════════════════════════════════════════


def _flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    # Entrada:
    #   - obj: objeto JSON anidado compuesto por diccionarios, listas y escalares.
    #   - prefix: ruta acumulada de claves internas.
    # Proceso:
    #   - Recorre recursivamente el JSON y convierte cada hoja en una clave plana.
    #   - Esta operación transforma reportes complejos de módulos en variables comparables.
    # Salida:
    #   - diccionario plano clave -> valor, donde las claves preservan la jerarquía original.
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_json(value, new_key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            new_key = f"{prefix}.{i}" if prefix else str(i)
            out.update(_flatten_json(value, new_key))
    else:
        out[prefix] = obj
    return out


def _read_json_features(file_path: Path, patient: str, stage: str, records: list[FeatureRecord]) -> None:
    # Entrada:
    #   - file_path: archivo JSON producido por algún módulo de la suite.
    #   - patient, stage: contexto del archivo dentro de resultados.
    #   - records: lista acumuladora de características.
    # Proceso:
    #   - Lee el JSON, aplana sus claves y extrae únicamente hojas numéricas.
    #   - Clasifica la modalidad usando ruta y claves internas para conservar significado físico.
    # Salida:
    #   - no retorna valor; añade registros numéricos a records.
    try:
        data = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return
    flat = _flatten_json(data)
    for key, value in flat.items():
        modality = _infer_modality(file_path, key)
        region = _infer_region(file_path)
        _append_feature(records, patient, stage, modality, region, key, value, "", "json", file_path)


def _read_table_features(file_path: Path, patient: str, stage: str, records: list[FeatureRecord]) -> None:
    # Entrada:
    #   - file_path: tabla CSV, TSV o Excel ubicada en resultados.
    #   - patient, stage: identificadores derivados de la estructura de carpetas.
    #   - records: lista acumuladora de variables.
    # Proceso:
    #   - Carga una o varias hojas, detecta columnas numéricas y conserva metadatos como región, zona o unidad.
    #   - Cada columna numérica se convierte en una variable biomédica independiente.
    # Salida:
    #   - no retorna valor; agrega features tabulares a records.
    try:
        if file_path.suffix.lower() in {".xlsx", ".xls"}:
            sheets = pd.read_excel(file_path, sheet_name=None)
        elif file_path.suffix.lower() == ".tsv":
            sheets = {"tabla": pd.read_csv(file_path, sep="\t")}
        else:
            try:
                sheets = {"tabla": pd.read_csv(file_path)}
            except Exception:
                sheets = {"tabla": pd.read_csv(file_path, sep=";")}
    except Exception:
        return

    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        lower_cols = {c.lower(): c for c in df.columns}
        unit_col = next((lower_cols[c] for c in lower_cols if c in {"unidad", "unit", "units"}), None)
        parameter_col = next((lower_cols[c] for c in lower_cols if c in {"parametro", "parámetro", "variable", "metric", "metrica", "métrica"}), None)
        value_col = next((lower_cols[c] for c in lower_cols if c in {"valor", "value", "mean", "media"}), None)

        if parameter_col and value_col:
            for _, row in df.iterrows():
                row_dict = {str(k).lower(): v for k, v in row.to_dict().items()}
                modality = _infer_modality(file_path, str(row.to_dict()))
                region = _infer_region(file_path, row_dict)
                unit = row.get(unit_col, "") if unit_col else ""
                _append_feature(records, patient, stage, modality, region, row.get(parameter_col), row.get(value_col), unit, f"excel:{sheet_name}", file_path)
            continue

        for col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().sum() == 0:
                continue
            for idx, value in numeric.dropna().items():
                row_dict = {str(k).lower(): v for k, v in df.loc[idx].to_dict().items()}
                modality = _infer_modality(file_path, f"{sheet_name} {col} {row_dict}")
                region = _infer_region(file_path, row_dict)
                unit = row_dict.get("unidad", row_dict.get("unit", ""))
                parameter = col if not parameter_col else row_dict.get(parameter_col.lower(), col)
                _append_feature(records, patient, stage, modality, region, parameter, value, unit, f"tabla:{sheet_name}", file_path)


def _read_nifti_features(file_path: Path, patient: str, stage: str, records: list[FeatureRecord]) -> None:
    # Entrada:
    #   - file_path: volumen NIfTI derivado de máscara, densidad, mapa RGB, labelmap o correlación.
    #   - patient, stage: contexto de paciente y momento temporal.
    #   - records: lista acumuladora de biomarcadores.
    # Proceso:
    #   - Carga el volumen, calcula estadísticas de intensidad y volumen no nulo cuando la imagen es interpretable.
    #   - Las métricas resumen permiten comparar mapas volumétricos sin abrir manualmente cada imagen.
    # Salida:
    #   - no retorna valor; añade estadísticas NIfTI a records.
    if nib is None:
        return
    name = file_path.name.lower()
    allow = ("cst" in name or "cortico" in name or "zona" in name or "region" in name or "mask" in name
             or "density" in name or "correl" in name or "labelmap" in name or "baja" in name)
    if not allow:
        return
    try:
        img = nib.load(str(file_path))
        data = np.asarray(img.get_fdata(dtype=np.float32))
    except Exception:
        return
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    modality = _infer_modality(file_path)
    region = _infer_region(file_path)
    voxel_volume = abs(float(np.linalg.det(img.affine[:3, :3]))) if img.affine is not None else 1.0
    nonzero = finite[np.abs(finite) > 0]
    _append_feature(records, patient, stage, modality, region, "nii_media", float(np.mean(finite)), "intensidad", "nifti", file_path)
    _append_feature(records, patient, stage, modality, region, "nii_desviacion", float(np.std(finite)), "intensidad", "nifti", file_path)
    _append_feature(records, patient, stage, modality, region, "nii_voxeles_no_cero", int(nonzero.size), "voxeles", "nifti", file_path)
    _append_feature(records, patient, stage, modality, region, "nii_volumen_no_cero_mm3", float(nonzero.size * voxel_volume), "mm3", "nifti", file_path)
    if nonzero.size:
        _append_feature(records, patient, stage, modality, region, "nii_media_no_cero", float(np.mean(nonzero)), "intensidad", "nifti", file_path)
        _append_feature(records, patient, stage, modality, region, "nii_p95_no_cero", float(np.percentile(nonzero, 95)), "intensidad", "nifti", file_path)


def _discover_result_scopes(results_root: Path, cfg: Any) -> list[tuple[str, str, Path]]:
    # Entrada:
    #   - results_root: carpeta general de resultados.
    #   - cfg: configuración del pipeline con pacientes, control y etapas.
    # Proceso:
    #   - Busca carpetas paciente/etapa existentes, incluyendo control sano si está presente.
    #   - Esta detección permite ejecutar el análisis aunque algunos pacientes no tengan todas las modalidades.
    # Salida:
    #   - lista de tuplas (paciente, etapa, ruta_de_etapa) que serán escaneadas.
    scopes: list[tuple[str, str, Path]] = []
    if not results_root.exists():
        return scopes
    known_patients = list(getattr(cfg, "patients", []) or [])
    control = getattr(cfg, "control_name", "sano") or "sano"
    known_patients.append(control)

    folders = [p for p in results_root.iterdir() if p.is_dir()]
    candidates = []
    for p in folders:
        low = p.name.lower()
        if re.match(r"paciente\s*\d+", low) or any(x in low for x in CONTROL_PATTERNS) or p.name in known_patients:
            candidates.append(p)
    for patient_dir in sorted(candidates):
        patient = patient_dir.name
        stage_dirs = [d for d in patient_dir.iterdir() if d.is_dir() and d.name.lower() in {"antes", "despues", "después", "post", "pre", "sano"}]
        if stage_dirs:
            for stage_dir in sorted(stage_dirs):
                stage = "Despues" if stage_dir.name.lower() in {"despues", "después", "post"} else ("Antes" if stage_dir.name.lower() in {"antes", "pre"} else stage_dir.name)
                scopes.append((patient, stage, stage_dir))
        else:
            stage = "Sano" if any(x in patient.lower() for x in CONTROL_PATTERNS) else "Global"
            scopes.append((patient, stage, patient_dir))
    return scopes


###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  4. CONSTRUCCIÓN DE FEATURES Y MATRICES MULTIDIMENSIONALES
# ══════════════════════════════════════════════════════════════════════════════


def scan_features_from_results(cfg: Any) -> pd.DataFrame:
    # Entrada:
    #   - cfg: configuración general de la suite con ruta de resultados y pacientes esperados.
    # Proceso:
    #   - Recorre resultados por paciente/etapa, lee JSON, CSV, Excel y NIfTI derivados.
    #   - Cada medición numérica se transforma en una fila estandarizada de la tabla maestra multimodal.
    # Salida:
    #   - DataFrame largo con paciente, protocolo, etapa, modalidad, región, parámetro, valor, unidad y fuente.
    results_root = Path(cfg.results_root())
    records: list[FeatureRecord] = []
    for patient, stage, scope in _discover_result_scopes(results_root, cfg):
        for file_path in sorted(scope.rglob("*")):
            if not file_path.is_file():
                continue
            name = file_path.name.lower()
            if any(skip in file_path.as_posix().lower() for skip in (".venv", "__pycache__", "freesurfer_subjects")):
                continue
            if name.endswith(".json"):
                _read_json_features(file_path, patient, stage, records)
            elif name.endswith((".csv", ".tsv", ".xlsx", ".xls")):
                _read_table_features(file_path, patient, stage, records)
            elif name.endswith(".nii") or name.endswith(".nii.gz"):
                _read_nifti_features(file_path, patient, stage, records)
    df = pd.DataFrame([r.__dict__ for r in records])
    if df.empty:
        return pd.DataFrame(columns=["paciente", "protocolo", "etapa", "modalidad", "region", "parametro", "valor", "unidad", "fuente", "archivo", "feature_key"])
    df["feature_key"] = df.apply(_feature_key, axis=1)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["valor"])
    return df


def build_master_table(features: pd.DataFrame) -> pd.DataFrame:
    # Entrada:
    #   - features: tabla larga de biomarcadores extraídos de resultados.
    # Proceso:
    #   - Agrupa duplicados por paciente, etapa y variable canónica usando promedio numérico.
    #   - Esta tabla se convierte en la base para deltas, matrices por paciente y reportes.
    # Salida:
    #   - DataFrame maestro con una fila por paciente-etapa-variable.
    if features.empty:
        return features.copy()
    group_cols = ["paciente", "protocolo", "etapa", "modalidad", "region", "parametro", "unidad", "feature_key"]
    agg = features.groupby(group_cols, dropna=False).agg(
        valor=("valor", "mean"),
        n_observaciones=("valor", "size"),
        fuente=("fuente", lambda x: "; ".join(sorted(set(map(str, x)))[:5])),
        archivo=("archivo", lambda x: "; ".join(sorted(set(map(str, x)))[:3])),
    ).reset_index()
    return agg.sort_values(["paciente", "etapa", "modalidad", "region", "parametro"])


def build_patient_matrices(master: pd.DataFrame, analysis_dir: Path) -> dict[str, pd.DataFrame]:
    # Entrada:
    #   - master: tabla maestra multimodal.
    #   - analysis_dir: carpeta resultados/analisis donde se escribirán matrices.
    # Proceso:
    #   - Para cada paciente crea una matriz variable × etapa, con columnas Antes, Después y delta longitudinal.
    #   - La matriz resume el perfil multimodal del paciente y facilita comparar cambios centrales y periféricos.
    # Salida:
    #   - diccionario paciente -> matriz; también escribe Excel y CSV por paciente.
    matrices_dir = analysis_dir / "matrices_pacientes"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    if master.empty:
        return out
    for patient, dfp in master.groupby("paciente"):
        pivot = dfp.pivot_table(index=["feature_key", "modalidad", "region", "parametro", "unidad"], columns="etapa", values="valor", aggfunc="mean").reset_index()
        if "Antes" in pivot.columns and "Despues" in pivot.columns:
            pivot["delta_despues_menos_antes"] = pivot["Despues"] - pivot["Antes"]
            pivot["delta_pct"] = np.where(np.abs(pivot["Antes"]) > 1e-12, 100.0 * pivot["delta_despues_menos_antes"] / np.abs(pivot["Antes"]), np.nan)
        summary = dfp.groupby(["modalidad", "etapa"]).agg(n_variables=("feature_key", "nunique"), valor_promedio=("valor", "mean")).reset_index()
        safe_name = _slug(patient)
        xlsx_path = matrices_dir / f"matriz_multidimensional_{safe_name}.xlsx"
        csv_path = matrices_dir / f"matriz_multidimensional_{safe_name}.csv"
        with pd.ExcelWriter(xlsx_path) as writer:
            pivot.to_excel(writer, sheet_name="matriz", index=False)
            dfp.to_excel(writer, sheet_name="tabla_larga", index=False)
            summary.to_excel(writer, sheet_name="resumen_modalidad", index=False)
        pivot.to_csv(csv_path, index=False)
        out[patient] = pivot
    return out


def compute_pairwise_deltas(master: pd.DataFrame, analysis_dir: Path, reference_patient: str = "paciente 3") -> Path:
    # Entrada:
    #   - master: tabla maestra con variables comparables.
    #   - analysis_dir: carpeta donde se guardará el Excel de deltas.
    #   - reference_patient: paciente base del protocolo normal, por defecto paciente 3.
    # Proceso:
    #   - Calcula diferencias del paciente 3 frente a pacientes cross education en Antes y Después.
    #   - Calcula también comparaciones frente a control sano si existe una carpeta sano/control con variables compatibles.
    # Salida:
    #   - ruta del Excel delthas_variables_multimodal.xlsx con cuatro hojas solicitadas.
    out_path = analysis_dir / "delthas_variables_multimodal.xlsx"
    if master.empty:
        with pd.ExcelWriter(out_path) as writer:
            pd.DataFrame({"mensaje": ["No se encontraron variables numéricas para calcular deltas."]}).to_excel(writer, sheet_name="resumen", index=False)
        return out_path

    pivot = master.pivot_table(index=["paciente", "protocolo", "etapa", "feature_key", "modalidad", "region", "parametro", "unidad"], values="valor", aggfunc="mean").reset_index()
    controls = [p for p in pivot["paciente"].dropna().unique() if any(x in str(p).lower() for x in CONTROL_PATTERNS)]
    cross_patients = [p for p in pivot["paciente"].dropna().unique() if str(p).lower() != reference_patient.lower() and p not in controls]

    def compare(stage: str, left_patients: Iterable[str], right_patient: str | None = reference_patient) -> pd.DataFrame:
        rows = []
        right = pivot[pivot["etapa"].astype(str).str.lower() == stage.lower()]
        if right_patient is not None:
            right = right[right["paciente"].astype(str).str.lower() == right_patient.lower()]
        for left_patient in left_patients:
            left = pivot[(pivot["paciente"] == left_patient) & (pivot["etapa"].astype(str).str.lower() == stage.lower())]
            if right.empty or left.empty:
                continue
            merged = left.merge(right, on=["feature_key", "modalidad", "region", "parametro", "unidad"], suffixes=("_comparado", "_referencia"))
            for _, row in merged.iterrows():
                ref = row["valor_referencia"]
                val = row["valor_comparado"]
                rows.append({
                    "etapa": stage,
                    "paciente_comparado": left_patient,
                    "protocolo_comparado": row.get("protocolo_comparado", ""),
                    "paciente_referencia": right_patient or "control_sano",
                    "modalidad": row["modalidad"],
                    "region": row["region"],
                    "parametro": row["parametro"],
                    "unidad": row["unidad"],
                    "valor_comparado": val,
                    "valor_referencia": ref,
                    "delta_abs": val - ref,
                    "delta_pct": 100.0 * (val - ref) / abs(ref) if abs(ref) > 1e-12 else np.nan,
                    "feature_key": row["feature_key"],
                })
        return pd.DataFrame(rows)

    antes = compare("Antes", cross_patients, reference_patient)
    despues = compare("Despues", cross_patients, reference_patient)

    if controls:
        control_patient = controls[0]
        # Se compara contra la etapa disponible del control; si hay varias, se toma la primera por variable.
        control_df = pivot[pivot["paciente"] == control_patient].copy()
        if not control_df.empty:
            control_df = control_df.sort_values("etapa").drop_duplicates(["feature_key", "modalidad", "region", "parametro", "unidad"])
            ref3 = pivot[pivot["paciente"].astype(str).str.lower() == reference_patient.lower()]
            p3_sano = ref3.merge(control_df, on=["feature_key", "modalidad", "region", "parametro", "unidad"], suffixes=("_paciente", "_sano"))
            p3_sano["delta_abs"] = p3_sano["valor_paciente"] - p3_sano["valor_sano"]
            p3_sano["delta_pct"] = np.where(np.abs(p3_sano["valor_sano"]) > 1e-12, 100.0 * p3_sano["delta_abs"] / np.abs(p3_sano["valor_sano"]), np.nan)
            cross_sano = pivot[pivot["paciente"].isin(cross_patients)].merge(control_df, on=["feature_key", "modalidad", "region", "parametro", "unidad"], suffixes=("_paciente", "_sano"))
            cross_sano["delta_abs"] = cross_sano["valor_paciente"] - cross_sano["valor_sano"]
            cross_sano["delta_pct"] = np.where(np.abs(cross_sano["valor_sano"]) > 1e-12, 100.0 * cross_sano["delta_abs"] / np.abs(cross_sano["valor_sano"]), np.nan)
        else:
            p3_sano = pd.DataFrame({"mensaje": ["No se encontraron variables del control sano."]})
            cross_sano = pd.DataFrame({"mensaje": ["No se encontraron variables del control sano."]})
    else:
        p3_sano = pd.DataFrame({"mensaje": ["No hay carpeta o paciente control sano detectado en resultados."]})
        cross_sano = pd.DataFrame({"mensaje": ["No hay carpeta o paciente control sano detectado en resultados."]})

    with pd.ExcelWriter(out_path) as writer:
        (antes if not antes.empty else pd.DataFrame({"mensaje": ["Sin variables comunes para paciente 3 vs cross education en Antes."]})).to_excel(writer, sheet_name="entre_pacientes_antes", index=False)
        (despues if not despues.empty else pd.DataFrame({"mensaje": ["Sin variables comunes para paciente 3 vs cross education en Después."]})).to_excel(writer, sheet_name="entre_pacientes_despues", index=False)
        p3_sano.to_excel(writer, sheet_name="paciente3_vs_sano", index=False)
        cross_sano.to_excel(writer, sheet_name="cross_vs_sano", index=False)
    return out_path


def build_domain_outputs(master: pd.DataFrame, analysis_dir: Path) -> dict[str, Path]:
    # Entrada:
    #   - master: tabla maestra con variables multimodales.
    #   - analysis_dir: carpeta final de análisis.
    # Proceso:
    #   - Separa productos especializados: features, tabla maestra, tractometría CST, radiómica de zonas e integración.
    #   - Agrega z-scores por variable para ubicar pacientes/etapas sobre una escala comparable.
    # Salida:
    #   - diccionario nombre_producto -> ruta escrita en resultados/analisis.
    paths: dict[str, Path] = {}
    analysis_dir.mkdir(parents=True, exist_ok=True)

    features_path = analysis_dir / "features_multimodales.xlsx"
    master_path = analysis_dir / "tabla_maestra_multimodal.xlsx"
    tract_path = analysis_dir / "tractometria_cst.xlsx"
    radio_path = analysis_dir / "radiomica_zonas.xlsx"
    integ_path = analysis_dir / "integracion_multidimensional.xlsx"

    with pd.ExcelWriter(features_path) as writer:
        master.to_excel(writer, sheet_name="features", index=False)
        if not master.empty:
            master.groupby(["modalidad", "parametro"]).agg(n=("valor", "size"), media=("valor", "mean"), sd=("valor", "std")).reset_index().to_excel(writer, sheet_name="resumen_variables", index=False)
            pd.DataFrame(IMPORTANT_VARIABLES, columns=["modalidad", "variable", "importancia_fisica"]).to_excel(writer, sheet_name="variables_prioritarias", index=False)
    paths["features_multimodales"] = features_path

    with pd.ExcelWriter(master_path) as writer:
        master.to_excel(writer, sheet_name="tabla_larga", index=False)
        if not master.empty:
            wide = master.pivot_table(index=["feature_key", "modalidad", "region", "parametro", "unidad"], columns=["paciente", "etapa"], values="valor", aggfunc="mean")
            wide.reset_index().to_excel(writer, sheet_name="matriz_variables", index=False)
    paths["tabla_maestra_multimodal"] = master_path

    cst = master[master["modalidad"].astype(str).str.contains("Tractografía", case=False, na=False)].copy() if not master.empty else master.copy()
    with pd.ExcelWriter(tract_path) as writer:
        cst.to_excel(writer, sheet_name="tractometria", index=False)
        if not cst.empty:
            cst.groupby(["paciente", "etapa", "parametro"]).agg(valor=("valor", "mean")).reset_index().to_excel(writer, sheet_name="resumen_cst", index=False)
    paths["tractometria_cst"] = tract_path

    zonas = master[master["modalidad"].astype(str).str.contains("Zonas|Resonancia|Tomografía", case=False, na=False)].copy() if not master.empty else master.copy()
    with pd.ExcelWriter(radio_path) as writer:
        zonas.to_excel(writer, sheet_name="radiomica_zonas", index=False)
        if not zonas.empty:
            zonas.groupby(["paciente", "etapa", "modalidad", "region"]).agg(n_variables=("feature_key", "nunique"), media=("valor", "mean"), sd=("valor", "std")).reset_index().to_excel(writer, sheet_name="resumen_region", index=False)
    paths["radiomica_zonas"] = radio_path

    integ = master.copy()
    if not integ.empty:
        integ["z_score_variable"] = integ.groupby("feature_key")["valor"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else np.nan))
        resumen = integ.groupby(["paciente", "protocolo", "etapa", "modalidad"]).agg(
            n_variables=("feature_key", "nunique"),
            z_promedio=("z_score_variable", "mean"),
            valor_promedio=("valor", "mean"),
        ).reset_index()
    else:
        resumen = pd.DataFrame()
    with pd.ExcelWriter(integ_path) as writer:
        integ.to_excel(writer, sheet_name="variables_zscore", index=False)
        resumen.to_excel(writer, sheet_name="indice_por_modalidad", index=False)
    paths["integracion_multidimensional"] = integ_path
    return paths


def write_patient_reports(master: pd.DataFrame, matrices: dict[str, pd.DataFrame], analysis_dir: Path) -> list[Path]:
    # Entrada:
    #   - master: tabla maestra multimodal.
    #   - matrices: matrices por paciente generadas previamente.
    #   - analysis_dir: carpeta donde se escriben reportes.
    # Proceso:
    #   - Genera un Excel por paciente con resumen de variables, matriz multimodal y ranking de cambios.
    #   - El reporte facilita revisar un paciente completo sin abrir cada módulo por separado.
    # Salida:
    #   - lista de rutas de reportes generados.
    reports_dir = analysis_dir / "reportes_paciente"
    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for patient, matrix in matrices.items():
        dfp = master[master["paciente"] == patient].copy()
        path = reports_dir / f"reporte_paciente_multimodal_{_slug(patient)}.xlsx"
        ranking = matrix.copy()
        if "delta_pct" in ranking.columns:
            ranking = ranking.reindex(ranking["delta_pct"].abs().sort_values(ascending=False).index)
        with pd.ExcelWriter(path) as writer:
            dfp.to_excel(writer, sheet_name="features_paciente", index=False)
            matrix.to_excel(writer, sheet_name="matriz_multimodal", index=False)
            ranking.head(200).to_excel(writer, sheet_name="ranking_cambios", index=False)
            if not dfp.empty:
                dfp.groupby(["etapa", "modalidad"]).agg(n_variables=("feature_key", "nunique"), valor_promedio=("valor", "mean")).reset_index().to_excel(writer, sheet_name="resumen_modalidad", index=False)
        outputs.append(path)
    return outputs


def write_readme_analysis(analysis_dir: Path) -> Path:
    # Entrada:
    #   - analysis_dir: carpeta resultados/analisis creada por el módulo.
    # Proceso:
    #   - Escribe una guía breve en Markdown para interpretar productos, variables, deltas y matrices.
    #   - La explicación traduce las tablas técnicas a lectura física y clínica del sistema neuromotor.
    # Salida:
    #   - ruta del README generado dentro de resultados/analisis.
    path = analysis_dir / "README_ANALISIS_MULTIMODAL_RESULTADOS.md"
    text = """# Análisis multimodal VCE

Esta carpeta contiene una capa integradora construida a partir de los resultados existentes del proyecto.

## Matriz multidimensional por paciente

Cada paciente se representa como una matriz:

```text
variable multimodal × etapa
```

Cada variable se define por:

```text
modalidad | región | parámetro | unidad
```

La matriz permite comparar el perfil del paciente entre Antes y Después y luego cruzarlo con otros pacientes.

## Archivos principales

- `features_multimodales.xlsx`: variables numéricas detectadas en resultados.
- `tabla_maestra_multimodal.xlsx`: tabla larga y matriz general paciente-etapa-variable.
- `tractometria_cst.xlsx`: variables filtradas de vía corticoespinal y tractografía.
- `radiomica_zonas.xlsx`: variables de regiones, zonas, resonancia y tomografía.
- `integracion_multidimensional.xlsx`: z-scores e índices por modalidad.
- `delthas_variables_multimodal.xlsx`: comparaciones solicitadas entre paciente 3, cross education y control sano.
- `matrices_pacientes/`: matriz individual por paciente.
- `reportes_paciente/`: reporte Excel por paciente.

## Interpretación física

- EMG resume activación muscular y costo eléctrico.
- Dinamometría resume producción mecánica de fuerza, trabajo y potencia.
- Tomografía resume composición músculo-grasa y calidad tisular.
- Resonancia y zonas resumen morfometría, textura, intensidad y estabilidad anatómica.
- Tractografía CST resume continuidad, densidad y orientación de la vía corticoespinal.
- La integración multidimensional permite evaluar concordancias y discordancias entre sistema nervioso central, vía descendente, músculo y fuerza.

## Deltas

- `delta_abs = valor_comparado - valor_referencia`
- `delta_pct = 100 * delta_abs / |valor_referencia|`

En las hojas de comparación, el paciente 3 actúa como referencia del protocolo normal y los pacientes 6, 7 y 9 representan cross education.
"""
    path.write_text(text, encoding="utf-8")
    return path


def run_multimodal_analysis_module(cfg: Any, subsections: Iterable[str] | None = None) -> dict[str, Any]:
    # Entrada:
    #   - cfg: configuración del pipeline con project_root, results_root, pacientes y control.
    #   - subsections: nombres solicitados desde main, por ejemplo features_multimodales o tractometria_cst.
    # Proceso:
    #   - Escanea resultados, construye tabla maestra, matrices por paciente, deltas, tractometría, radiómica e integración.
    #   - Escribe todos los productos en resultados/analisis para facilitar comparación entre pacientes.
    # Salida:
    #   - diccionario con rutas principales generadas y conteos de variables/pacientes.
    analysis_dir = Path(cfg.results_root()) / "analisis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("ANÁLISIS MULTIMODAL INTEGRADO")
    print(f"Salida: {analysis_dir}")
    print("=" * 100)

    features = scan_features_from_results(cfg)
    master = build_master_table(features)
    matrices = build_patient_matrices(master, analysis_dir)
    domain_paths = build_domain_outputs(master, analysis_dir)
    deltas_path = compute_pairwise_deltas(master, analysis_dir)
    reports = write_patient_reports(master, matrices, analysis_dir)
    readme = write_readme_analysis(analysis_dir)

    summary_path = analysis_dir / "resumen_analisis_multimodal.json"
    summary = {
        "n_features_raw": int(len(features)),
        "n_features_master": int(len(master)),
        "n_pacientes": int(master["paciente"].nunique()) if not master.empty else 0,
        "n_variables": int(master["feature_key"].nunique()) if not master.empty else 0,
        "subsections": list(subsections or []),
        "archivos": {k: str(v) for k, v in domain_paths.items()},
        "deltas": str(deltas_path),
        "readme": str(readme),
        "reportes_paciente": [str(p) for p in reports],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Features crudas:  {summary['n_features_raw']}")
    print(f"Features maestras:{summary['n_features_master']}")
    print(f"Pacientes:        {summary['n_pacientes']}")
    print(f"Variables:        {summary['n_variables']}")
    print(f"Deltas:           {deltas_path}")
    print(f"Resumen:          {summary_path}")
    print("=" * 100 + "\n")
    return summary
