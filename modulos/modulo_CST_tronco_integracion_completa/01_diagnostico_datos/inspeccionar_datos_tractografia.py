"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       INSPECCIONAR DATOS TRACTOGRAFIA                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: modulos/modulo_CST_tronco_integracion_completa/01_diagnostico_datos/inspeccionar_datos_tractografia.py
Versión: v3.21.21

Descripción
-----------
Inspecciona datos de difusión y anatomía para verificar compatibilidad con
tractografía.

Fundamento físico-matemático implementado
-----------------------------------------
Diagnostica series DICOM/NIfTI para tractografía. Evalúa geometría, número de
volúmenes, b-values, direcciones no nulas y consistencia entre DWI, bvec y
bval.
El criterio matemático mínimo es contar direcciones de difusión válidas g_i
con
b_i>0 y verificar correspondencia entre S(x,y,z,n) y el gradiente asociado.

Trazabilidad de resultados
--------------------------
Los datos crudos permanecen separados de los derivados. Las salidas se escriben
con nombre de paciente, etapa, módulo y espacio de referencia para facilitar
revisión, comparación longitudinal y reproducción de la corrida.
"""

###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  0. IMPORTACIONES, CONFIGURACIÓN Y FUNCIONES DEL MÓDULO
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import pydicom
from pydicom import dcmread


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

DICOMDIR_PATH = Path(
    r"D:\EAFIT\01-2026\proyecto\datos\paciente 6\Antes\RESONANCIA\DICOMDIR"
)

T1_PATH = Path(
    r"D:\EAFIT\01-2026\proyecto\datos\paciente 6\Antes"
    r"\Resultados funcional\REFORMATEO\rT1.nii"
)

OUTPUT_DIR = Path(
    r"D:\EAFIT\01-2026\proyecto\resultados\paciente 6\Antes"
    r"\diagnostico_previo_tractografia"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

# No se guarda PatientName. El PatientID se convierte en un hash corto.
REDACT_IDENTIFIERS = True

# Máximo de errores detallados que se conservan en el JSON.
MAX_ERRORS_IN_JSON = 200

# Tolerancia para agrupar b-values cercanos, por ejemplo 995 y 1000.
B_VALUE_CLUSTER_TOLERANCE = 50.0

# Extensiones que no se intentan leer como DICOM.
SKIP_EXTENSIONS = {
    ".json", ".csv", ".txt", ".xml", ".html", ".htm",
    ".nii", ".gz", ".png", ".jpg", ".jpeg", ".bmp",
    ".pdf", ".zip", ".rar", ".7z",
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def json_safe(value: Any) -> Any:
    """Convierte valores pydicom/numpy a tipos serializables en JSON."""
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]

    try:
        return [json_safe(x) for x in value]
    except TypeError:
        return str(value)


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return "\\".join(str(x) for x in value)
    return str(value)


def as_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
        return float(value)
    except (TypeError, ValueError):
        return None


def as_float_list(value: Any, expected_length: int | None = None) -> list[float] | None:
    if value is None:
        return None

    try:
        if isinstance(value, str):
            parts = value.replace(",", "\\").split("\\")
        else:
            parts = list(value)

        result = [float(x) for x in parts]

        if expected_length is not None and len(result) != expected_length:
            return None

        return result
    except (TypeError, ValueError):
        return None


def short_hash(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:12]


def cluster_numeric_values(
    values: Iterable[float],
    tolerance: float,
) -> list[float]:
    """Agrupa valores próximos y devuelve la mediana de cada grupo."""
    cleaned = sorted(
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    )

    if not cleaned:
        return []

    groups: list[list[float]] = [[cleaned[0]]]

    for value in cleaned[1:]:
        if abs(value - np.median(groups[-1])) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])

    return [float(np.median(group)) for group in groups]


def canonical_gradient(gradient: Iterable[float] | None) -> tuple[float, float, float] | None:
    """
    Normaliza un gradiente y hace equivalentes g y -g,
    porque ambos representan el mismo eje de difusión.
    """
    if gradient is None:
        return None

    try:
        vector = np.asarray(list(gradient), dtype=float)
    except (TypeError, ValueError):
        return None

    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None

    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return None

    vector = vector / norm

    for component in vector:
        if abs(component) > 1e-6:
            if component < 0:
                vector = -vector
            break

    return tuple(float(x) for x in np.round(vector, 4))


def unique_image_types(records: list[dict[str, Any]]) -> list[str]:
    values: set[str] = set()
    for record in records:
        image_type = record.get("image_type", "")
        if image_type:
            for part in image_type.replace("\\", "|").split("|"):
                part = part.strip()
                if part:
                    values.add(part)
    return sorted(values)


def compute_slice_coordinate(
    image_position: list[float] | None,
    image_orientation: list[float] | None,
) -> float | None:
    if image_position is None or image_orientation is None:
        return None

    if len(image_position) != 3 or len(image_orientation) != 6:
        return None

    try:
        row = np.asarray(image_orientation[:3], dtype=float)
        col = np.asarray(image_orientation[3:], dtype=float)
        normal = np.cross(row, col)
        norm = np.linalg.norm(normal)

        if norm < 1e-8:
            return None

        normal /= norm
        position = np.asarray(image_position, dtype=float)
        return float(np.dot(position, normal))
    except (TypeError, ValueError):
        return None


# ============================================================
# LECTURA DE DICOMDIR
# ============================================================

def inspect_dicomdir_index(dicomdir_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "provided_path": str(dicomdir_path),
        "exists": dicomdir_path.exists(),
        "is_file": dicomdir_path.is_file(),
        "is_directory": dicomdir_path.is_dir(),
        "referenced_file_count": 0,
        "existing_referenced_file_count": 0,
        "missing_referenced_file_count": 0,
        "missing_referenced_files": [],
        "error": None,
    }

    if not dicomdir_path.exists() or not dicomdir_path.is_file():
        return result

    try:
        ds = dcmread(
            str(dicomdir_path),
            stop_before_pixels=True,
            force=True,
        )

        referenced_paths: list[Path] = []
        sequence = getattr(ds, "DirectoryRecordSequence", [])

        for record in sequence:
            referenced = getattr(record, "ReferencedFileID", None)
            if referenced is None:
                continue

            if isinstance(referenced, str):
                parts = referenced.replace("/", "\\").split("\\")
            else:
                try:
                    parts = [str(part) for part in referenced]
                except TypeError:
                    parts = [str(referenced)]

            referenced_path = dicomdir_path.parent.joinpath(*parts)
            referenced_paths.append(referenced_path)

        missing = [path for path in referenced_paths if not path.exists()]

        result.update(
            {
                "referenced_file_count": len(referenced_paths),
                "existing_referenced_file_count": len(referenced_paths) - len(missing),
                "missing_referenced_file_count": len(missing),
                "missing_referenced_files": [
                    str(path) for path in missing[:100]
                ],
            }
        )

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


# ============================================================
# EXTRACCIÓN DE METADATOS DE DIFUSIÓN
# ============================================================

def measurement_from_dataset(container: Any) -> dict[str, Any] | None:
    """
    Extrae b-value y orientación de gradiente de un Dataset o de
    un elemento de Functional Groups.
    """
    candidates: list[Any] = []

    mr_diffusion_sequence = getattr(container, "MRDiffusionSequence", None)
    if mr_diffusion_sequence:
        candidates.extend(list(mr_diffusion_sequence))

    candidates.append(container)

    for candidate in candidates:
        b_value = as_float(getattr(candidate, "DiffusionBValue", None))
        gradient = as_float_list(
            getattr(candidate, "DiffusionGradientOrientation", None),
            expected_length=3,
        )
        directionality = as_text(
            getattr(candidate, "DiffusionDirectionality", None)
        )

        # En algunos Enhanced MR la dirección puede estar en una secuencia.
        gradient_sequence = getattr(
            candidate,
            "DiffusionGradientDirectionSequence",
            None,
        )

        if gradient is None and gradient_sequence:
            for item in gradient_sequence:
                gradient = as_float_list(
                    getattr(item, "DiffusionGradientOrientation", None),
                    expected_length=3,
                )
                if gradient is not None:
                    break

        if b_value is not None or gradient is not None or directionality:
            return {
                "b_value": b_value,
                "gradient": gradient,
                "directionality": directionality,
                "source": "standard_dicom",
            }

    return None


def extract_diffusion_measurements(ds: Any) -> list[dict[str, Any]]:
    """
    Extrae mediciones de difusión estándar y, como respaldo,
    algunos tags privados habituales de Siemens.
    """
    measurements: list[dict[str, Any]] = []

    # 1. Enhanced DICOM: información por frame.
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if per_frame:
        for frame_index, frame_group in enumerate(per_frame):
            measurement = measurement_from_dataset(frame_group)
            if measurement is not None:
                measurement["frame_index"] = frame_index
                measurements.append(measurement)

        if measurements:
            return measurements

    # 2. Información compartida entre frames.
    shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if shared:
        for shared_group in shared:
            measurement = measurement_from_dataset(shared_group)
            if measurement is not None:
                measurement["frame_index"] = None
                measurements.append(measurement)

        if measurements:
            return measurements

    # 3. Tags estándar en el nivel principal.
    direct = measurement_from_dataset(ds)
    if direct is not None:
        direct["frame_index"] = None
        return [direct]

    # 4. Respaldo para algunos DICOM clásicos Siemens.
    private_b = None
    private_gradient = None

    if (0x0019, 0x100C) in ds:
        private_b = as_float(ds[(0x0019, 0x100C)].value)

    if (0x0019, 0x100E) in ds:
        private_gradient = as_float_list(
            ds[(0x0019, 0x100E)].value,
            expected_length=3,
        )

    if private_b is not None or private_gradient is not None:
        return [
            {
                "b_value": private_b,
                "gradient": private_gradient,
                "directionality": "",
                "source": "siemens_private_fallback",
                "frame_index": None,
            }
        ]

    return []


# ============================================================
# ESCANEO DE ARCHIVOS DICOM
# ============================================================

def determine_scan_root(path: Path) -> Path:
    if path.is_dir():
        return path
    return path.parent


def candidate_files(root: Path, dicomdir_path: Path) -> list[Path]:
    files: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.resolve() == dicomdir_path.resolve():
            continue

        suffix = path.suffix.lower()

        if suffix in SKIP_EXTENSIONS:
            continue

        files.append(path)

    return files


def read_dicom_header(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        ds = dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    # Evita aceptar archivos arbitrarios que force=True haya interpretado.
    if not any(
        hasattr(ds, attribute)
        for attribute in (
            "SOPClassUID",
            "SeriesInstanceUID",
            "StudyInstanceUID",
            "Modality",
            "Rows",
            "Columns",
        )
    ):
        return None, "No contiene identificadores DICOM reconocibles."

    modality = as_text(getattr(ds, "Modality", None)).upper()

    if modality == "DIR":
        return None, None

    image_position = as_float_list(
        getattr(ds, "ImagePositionPatient", None),
        expected_length=3,
    )

    image_orientation = as_float_list(
        getattr(ds, "ImageOrientationPatient", None),
        expected_length=6,
    )

    measurements = extract_diffusion_measurements(ds)

    b_values = [
        measurement["b_value"]
        for measurement in measurements
        if measurement.get("b_value") is not None
    ]

    gradients = [
        measurement["gradient"]
        for measurement in measurements
        if measurement.get("gradient") is not None
    ]

    patient_id = as_text(getattr(ds, "PatientID", None))

    record = {
        "path": str(path),
        "relative_name": path.name,
        "study_uid": as_text(getattr(ds, "StudyInstanceUID", None)),
        "series_uid": as_text(getattr(ds, "SeriesInstanceUID", None)),
        "sop_instance_uid": as_text(getattr(ds, "SOPInstanceUID", None)),
        "series_number": as_text(getattr(ds, "SeriesNumber", None)),
        "instance_number": as_text(getattr(ds, "InstanceNumber", None)),
        "acquisition_number": as_text(getattr(ds, "AcquisitionNumber", None)),
        "temporal_position": as_text(
            getattr(ds, "TemporalPositionIdentifier", None)
        ),
        "modality": modality,
        "series_description": as_text(
            getattr(ds, "SeriesDescription", None)
        ),
        "protocol_name": as_text(getattr(ds, "ProtocolName", None)),
        "sequence_name": as_text(getattr(ds, "SequenceName", None)),
        "image_type": as_text(getattr(ds, "ImageType", None)),
        "scanning_sequence": as_text(
            getattr(ds, "ScanningSequence", None)
        ),
        "sequence_variant": as_text(getattr(ds, "SequenceVariant", None)),
        "mr_acquisition_type": as_text(
            getattr(ds, "MRAcquisitionType", None)
        ),
        "manufacturer": as_text(getattr(ds, "Manufacturer", None)),
        "manufacturer_model": as_text(
            getattr(ds, "ManufacturerModelName", None)
        ),
        "magnetic_field_strength": as_float(
            getattr(ds, "MagneticFieldStrength", None)
        ),
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "columns": int(getattr(ds, "Columns", 0) or 0),
        "number_of_frames": int(getattr(ds, "NumberOfFrames", 1) or 1),
        "pixel_spacing": as_float_list(
            getattr(ds, "PixelSpacing", None),
            expected_length=2,
        ),
        "slice_thickness": as_float(
            getattr(ds, "SliceThickness", None)
        ),
        "spacing_between_slices": as_float(
            getattr(ds, "SpacingBetweenSlices", None)
        ),
        "image_position_patient": image_position,
        "image_orientation_patient": image_orientation,
        "slice_coordinate": compute_slice_coordinate(
            image_position,
            image_orientation,
        ),
        "echo_time": as_float(getattr(ds, "EchoTime", None)),
        "repetition_time": as_float(
            getattr(ds, "RepetitionTime", None)
        ),
        "flip_angle": as_float(getattr(ds, "FlipAngle", None)),
        "phase_encoding_direction": as_text(
            getattr(ds, "InPlanePhaseEncodingDirection", None)
        ),
        "b_values": b_values,
        "gradients": gradients,
        "diffusion_measurements": measurements,
        "patient_id_hash": short_hash(patient_id) if REDACT_IDENTIFIERS else patient_id,
    }

    return record, None


def scan_dicoms(
    root: Path,
    dicomdir_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    files = candidate_files(root, dicomdir_path)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    print(f"\nArchivos candidatos encontrados: {len(files)}")
    print("Leyendo únicamente encabezados; no se cargan los píxeles...")

    for index, path in enumerate(files, start=1):
        record, error = read_dicom_header(path)

        if record is not None:
            records.append(record)
        elif error:
            errors.append(
                {
                    "path": str(path),
                    "error": error,
                }
            )

        if index % 250 == 0 or index == len(files):
            print(
                f"  Procesados: {index}/{len(files)} | "
                f"DICOM válidos: {len(records)} | "
                f"No leídos: {len(errors)}"
            )

    scan_stats = {
        "scan_root": str(root),
        "candidate_file_count": len(files),
        "valid_dicom_instance_count": len(records),
        "unreadable_or_non_dicom_count": len(errors),
    }

    return records, errors, scan_stats


# ============================================================
# RESUMEN POR SERIE
# ============================================================

def classify_series(summary: dict[str, Any]) -> tuple[str, int, list[str]]:
    text = " ".join(
        [
            summary.get("series_description", ""),
            summary.get("protocol_name", ""),
            summary.get("sequence_name", ""),
            summary.get("image_type", ""),
        ]
    ).upper()

    diffusion_keywords = (
        "DWI", "DTI", "DIFF", "DIFFUSION", "DIFUSION",
        "EP2D_DIFF", "HARDI", "HARDI",
    )

    derived_keywords = (
        "ADC", "TRACE", "FA ", "FA_", "FRACTIONAL",
        "COLOR MAP", "COLFA", "TENSOR", "EXPONENTIAL",
    )

    t1_keywords = (
        "T1", "MPRAGE", "BRAVO", "SPGR", "FSPGR", "TFE",
    )

    has_diffusion_keyword = any(keyword in text for keyword in diffusion_keywords)
    has_derived_keyword = any(keyword in text for keyword in derived_keywords)
    is_derived_image = "DERIVED" in text
    has_nonzero_b = summary.get("nonzero_b_value_count", 0) > 0
    direction_count = summary.get("unique_nonzero_gradient_direction_count", 0)

    score = 0
    reasons: list[str] = []

    if has_nonzero_b:
        score += 6
        reasons.append("contiene b-values mayores que cero")

    if direction_count >= 6:
        score += 5
        reasons.append(
            f"se detectaron {direction_count} ejes de gradiente no nulos"
        )
    elif direction_count > 0:
        score += 2
        reasons.append(
            f"solo se detectaron {direction_count} ejes de gradiente"
        )

    if has_diffusion_keyword:
        score += 3
        reasons.append("el nombre/protocolo contiene términos de difusión")

    if summary.get("b0_measurement_count", 0) > 0:
        score += 2
        reasons.append("se detectaron mediciones b≈0")

    if has_derived_keyword or is_derived_image:
        score -= 8
        reasons.append("parece una imagen derivada, no la adquisición cruda")

    if (has_derived_keyword or is_derived_image) and (
        has_diffusion_keyword or has_nonzero_b
    ):
        return "MAPA_DERIVADO_DIFUSION", score, reasons

    if has_nonzero_b and direction_count >= 6:
        return "DWI_CRUDA_PROBABLE", score, reasons

    if has_nonzero_b or has_diffusion_keyword:
        return "CANDIDATA_DIFUSION_REVISAR", score, reasons

    if any(keyword in text for keyword in t1_keywords):
        return "ANATOMICA_T1_PROBABLE", score, reasons

    return "OTRA_SERIE", score, reasons


def summarize_series(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        series_uid = record.get("series_uid")
        if not series_uid:
            series_uid = (
                f"NO_UID::{record.get('series_number')}::"
                f"{record.get('series_description')}"
            )
        grouped[series_uid].append(record)

    summaries: list[dict[str, Any]] = []

    for series_uid, items in grouped.items():
        first = items[0]

        all_b_values: list[float] = []
        all_gradients: list[list[float]] = []
        measurement_sources: Counter[str] = Counter()

        for item in items:
            all_b_values.extend(item.get("b_values", []))
            all_gradients.extend(item.get("gradients", []))

            for measurement in item.get("diffusion_measurements", []):
                source = measurement.get("source")
                if source:
                    measurement_sources[source] += 1

        clustered_b_values = cluster_numeric_values(
            all_b_values,
            tolerance=B_VALUE_CLUSTER_TOLERANCE,
        )

        canonical_gradients = {
            gradient
            for gradient in (
                canonical_gradient(value) for value in all_gradients
            )
            if gradient is not None
        }

        nonzero_gradients = {
            gradient
            for gradient in canonical_gradients
            if np.linalg.norm(np.asarray(gradient)) > 0.5
        }

        b0_count = sum(
            1
            for value in all_b_values
            if value is not None and abs(float(value)) <= B_VALUE_CLUSTER_TOLERANCE
        )

        nonzero_b_count = sum(
            1
            for value in all_b_values
            if value is not None and float(value) > B_VALUE_CLUSTER_TOLERANCE
        )

        slice_coordinates = sorted(
            {
                round(float(item["slice_coordinate"]), 3)
                for item in items
                if item.get("slice_coordinate") is not None
            }
        )

        unique_slice_count = len(slice_coordinates)

        total_frames = sum(
            max(1, int(item.get("number_of_frames", 1)))
            for item in items
        )

        estimated_volumes_by_slices = None
        if unique_slice_count > 0:
            estimated_volumes_by_slices = round(
                total_frames / unique_slice_count,
                2,
            )

        acquisition_numbers = sorted(
            {
                item.get("acquisition_number")
                for item in items
                if item.get("acquisition_number")
            }
        )

        temporal_positions = sorted(
            {
                item.get("temporal_position")
                for item in items
                if item.get("temporal_position")
            }
        )

        patient_hashes = sorted(
            {
                item.get("patient_id_hash")
                for item in items
                if item.get("patient_id_hash")
            }
        )

        summary: dict[str, Any] = {
            "series_uid": series_uid,
            "series_number": first.get("series_number", ""),
            "series_description": first.get("series_description", ""),
            "protocol_name": first.get("protocol_name", ""),
            "sequence_name": first.get("sequence_name", ""),
            "modality": first.get("modality", ""),
            "manufacturer": first.get("manufacturer", ""),
            "manufacturer_model": first.get("manufacturer_model", ""),
            "magnetic_field_strength": first.get(
                "magnetic_field_strength"
            ),
            "mr_acquisition_type": first.get(
                "mr_acquisition_type",
                "",
            ),
            "image_type": " | ".join(unique_image_types(items)),
            "instance_file_count": len(items),
            "total_frame_count": total_frames,
            "rows": first.get("rows", 0),
            "columns": first.get("columns", 0),
            "pixel_spacing": first.get("pixel_spacing"),
            "slice_thickness": first.get("slice_thickness"),
            "spacing_between_slices": first.get(
                "spacing_between_slices"
            ),
            "echo_time": first.get("echo_time"),
            "repetition_time": first.get("repetition_time"),
            "flip_angle": first.get("flip_angle"),
            "phase_encoding_direction": first.get(
                "phase_encoding_direction",
                "",
            ),
            "image_orientation_patient": first.get(
                "image_orientation_patient"
            ),
            "unique_slice_position_count": unique_slice_count,
            "estimated_volume_count_from_slices": estimated_volumes_by_slices,
            "unique_acquisition_number_count": len(acquisition_numbers),
            "unique_temporal_position_count": len(temporal_positions),
            "raw_b_value_measurement_count": len(all_b_values),
            "clustered_b_values": [
                round(value, 1) for value in clustered_b_values
            ],
            "b0_measurement_count": b0_count,
            "nonzero_b_value_count": nonzero_b_count,
            "unique_gradient_axis_count": len(canonical_gradients),
            "unique_nonzero_gradient_direction_count": len(
                nonzero_gradients
            ),
            "diffusion_metadata_sources": dict(measurement_sources),
            "patient_id_hashes": patient_hashes,
            "sample_files": [
                item["path"] for item in items[:5]
            ],
        }

        classification, score, reasons = classify_series(summary)
        summary["classification"] = classification
        summary["candidate_score"] = score
        summary["classification_reasons"] = reasons

        summaries.append(summary)

    priority = {
        "DWI_CRUDA_PROBABLE": 0,
        "CANDIDATA_DIFUSION_REVISAR": 1,
        "MAPA_DERIVADO_DIFUSION": 2,
        "ANATOMICA_T1_PROBABLE": 3,
        "OTRA_SERIE": 4,
    }

    summaries.sort(
        key=lambda item: (
            priority.get(item["classification"], 99),
            -int(item["candidate_score"]),
            str(item.get("series_number", "")),
        )
    )

    return summaries


# ============================================================
# INSPECCIÓN DE NIFTI T1
# ============================================================

def nifti_bbox(mask: np.ndarray) -> dict[str, Any] | None:
    coordinates = np.argwhere(mask)

    if coordinates.size == 0:
        return None

    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)

    return {
        "min_voxel": minimum.tolist(),
        "max_voxel": maximum.tolist(),
        "size_voxels": (maximum - minimum + 1).tolist(),
    }


def save_t1_preview(img: nib.spatialimages.SpatialImage, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canonical = nib.as_closest_canonical(img)
    data = np.asarray(canonical.dataobj, dtype=np.float32)

    if data.ndim > 3:
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(
            f"No se puede generar vista previa de una imagen {data.ndim}D."
        )

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("La imagen no contiene valores finitos.")

    low, high = np.percentile(finite, [1, 99])

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))

    x = data.shape[0] // 2
    y = data.shape[1] // 2
    z = data.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(
        np.rot90(data[x, :, :]),
        cmap="gray",
        vmin=low,
        vmax=high,
    )
    axes[0].set_title("Sagital central")
    axes[0].axis("off")

    axes[1].imshow(
        np.rot90(data[:, y, :]),
        cmap="gray",
        vmin=low,
        vmax=high,
    )
    axes[1].set_title("Coronal central")
    axes[1].axis("off")

    axes[2].imshow(
        np.rot90(data[:, :, z]),
        cmap="gray",
        vmin=low,
        vmax=high,
    )
    axes[2].set_title("Axial central")
    axes[2].axis("off")

    fig.suptitle(
        f"Vista previa rT1 canónica | shape={data.shape} | "
        f"orientación={''.join(nib.aff2axcodes(canonical.affine))}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def inspect_nifti(path: Path, output_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "error": None,
    }

    if not path.exists():
        result["error"] = "El archivo NIfTI no existe."
        return result

    try:
        img = nib.load(str(path))
        header = img.header
        affine = np.asarray(img.affine, dtype=float)
        shape = tuple(int(value) for value in img.shape)
        voxel_sizes = np.asarray(
            nib.affines.voxel_sizes(affine),
            dtype=float,
        )

        data = np.asarray(img.dataobj)
        finite_mask = np.isfinite(data)
        finite_values = data[finite_mask]

        if finite_values.size:
            percentiles = np.percentile(
                finite_values,
                [0, 1, 5, 50, 95, 99, 100],
            )
            stats = {
                "finite_voxel_count": int(finite_values.size),
                "nonfinite_voxel_count": int(data.size - finite_values.size),
                "minimum": float(percentiles[0]),
                "p01": float(percentiles[1]),
                "p05": float(percentiles[2]),
                "median": float(percentiles[3]),
                "p95": float(percentiles[4]),
                "p99": float(percentiles[5]),
                "maximum": float(percentiles[6]),
                "mean": float(np.mean(finite_values)),
                "standard_deviation": float(np.std(finite_values)),
                "nonzero_fraction": float(
                    np.count_nonzero(finite_values) / finite_values.size
                ),
            }
        else:
            stats = {
                "finite_voxel_count": 0,
                "nonfinite_voxel_count": int(data.size),
            }

        qform, qform_code = img.get_qform(coded=True)
        sform, sform_code = img.get_sform(coded=True)

        try:
            obliquity_radians = nib.affines.obliquity(affine)
            obliquity_degrees = np.degrees(obliquity_radians).tolist()
        except Exception:
            obliquity_degrees = None

        nonzero_mask = finite_mask & (data != 0)

        result.update(
            {
                "image_class": type(img).__name__,
                "shape": list(shape),
                "ndim": len(shape),
                "dtype_on_disk": str(header.get_data_dtype()),
                "voxel_sizes_mm": voxel_sizes.tolist(),
                "orientation_axcodes": list(nib.aff2axcodes(affine)),
                "affine": affine.tolist(),
                "affine_determinant_3x3": float(
                    np.linalg.det(affine[:3, :3])
                ),
                "qform_code": int(qform_code or 0),
                "sform_code": int(sform_code or 0),
                "qform": None if qform is None else np.asarray(qform).tolist(),
                "sform": None if sform is None else np.asarray(sform).tolist(),
                "xyzt_units": list(header.get_xyzt_units()),
                "zooms": list(header.get_zooms()),
                "obliquity_degrees_per_axis": obliquity_degrees,
                "intensity_statistics": stats,
                "nonzero_bounding_box": nifti_bbox(nonzero_mask),
                "is_3d": len(shape) == 3,
                "has_valid_spatial_affine": bool(
                    np.all(np.isfinite(affine))
                    and abs(np.linalg.det(affine[:3, :3])) > 1e-8
                ),
                "preview_path": str(
                    output_dir / "vista_previa_rT1.png"
                ),
            }
        )

        save_t1_preview(
            img,
            output_dir / "vista_previa_rT1.png",
        )

    except Exception as exc:
        result["error"] = (
            f"{type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}"
        )

    return result


# ============================================================
# EVALUACIÓN GENERAL
# ============================================================

def build_findings(
    series_summaries: list[dict[str, Any]],
    t1_info: dict[str, Any],
    dicomdir_info: dict[str, Any],
) -> list[str]:
    findings: list[str] = []

    raw_dwi = [
        item
        for item in series_summaries
        if item["classification"] == "DWI_CRUDA_PROBABLE"
    ]

    diffusion_candidates = [
        item
        for item in series_summaries
        if item["classification"] == "CANDIDATA_DIFUSION_REVISAR"
    ]

    derived = [
        item
        for item in series_summaries
        if item["classification"] == "MAPA_DERIVADO_DIFUSION"
    ]

    if raw_dwi:
        findings.append(
            f"Se encontraron {len(raw_dwi)} serie(s) con metadatos "
            "compatibles con una adquisición DWI cruda probable."
        )
    elif diffusion_candidates:
        findings.append(
            "No se confirmó todavía una DWI cruda, pero existen "
            f"{len(diffusion_candidates)} serie(s) de difusión que requieren revisión."
        )
    else:
        findings.append(
            "No se detectó una serie DWI cruda a partir de los encabezados."
        )

    if derived:
        findings.append(
            f"Se encontraron {len(derived)} mapa(s) derivado(s) de difusión. "
            "ADC, FA o TRACE no sustituyen la adquisición DWI original."
        )

    if t1_info.get("error"):
        findings.append(
            f"No se pudo validar rT1.nii: {t1_info['error'].splitlines()[0]}"
        )
    else:
        findings.append(
            "rT1.nii pudo abrirse y se documentaron dimensiones, "
            "resolución, orientación, affine y estadísticos de intensidad."
        )

        if not t1_info.get("is_3d"):
            findings.append(
                "Advertencia: rT1.nii no es estrictamente tridimensional."
            )

        if not t1_info.get("has_valid_spatial_affine"):
            findings.append(
                "Advertencia: el affine espacial de rT1.nii parece inválido."
            )

    if dicomdir_info.get("missing_referenced_file_count", 0) > 0:
        findings.append(
            "El índice DICOMDIR referencia archivos que no están presentes "
            "en la ubicación esperada."
        )

    findings.append(
        "Este diagnóstico no evalúa artefactos, movimiento, distorsión EPI, "
        "eddy currents ni calidad visual de la DWI; eso corresponde al paso siguiente."
    )

    return findings


# ============================================================
# ESCRITURA DE RESULTADOS
# ============================================================

CSV_COLUMNS = [
    "classification",
    "candidate_score",
    "series_number",
    "series_description",
    "protocol_name",
    "sequence_name",
    "modality",
    "manufacturer",
    "manufacturer_model",
    "magnetic_field_strength",
    "mr_acquisition_type",
    "image_type",
    "instance_file_count",
    "total_frame_count",
    "rows",
    "columns",
    "pixel_spacing",
    "slice_thickness",
    "spacing_between_slices",
    "unique_slice_position_count",
    "estimated_volume_count_from_slices",
    "raw_b_value_measurement_count",
    "clustered_b_values",
    "b0_measurement_count",
    "nonzero_b_value_count",
    "unique_nonzero_gradient_direction_count",
    "phase_encoding_direction",
    "diffusion_metadata_sources",
    "classification_reasons",
    "series_uid",
]


def write_series_csv(
    path: Path,
    summaries: list[dict[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for summary in summaries:
            row = {}
            for column in CSV_COLUMNS:
                value = summary.get(column)

                if isinstance(value, (list, dict, tuple)):
                    value = json.dumps(
                        json_safe(value),
                        ensure_ascii=False,
                    )

                row[column] = value

            writer.writerow(row)


def write_errors_csv(
    path: Path,
    errors: list[dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "error"])
        writer.writeheader()
        writer.writerows(errors)


def format_value(value: Any) -> str:
    if value is None:
        return "No disponible"

    if isinstance(value, (list, tuple, dict)):
        return json.dumps(
            json_safe(value),
            ensure_ascii=False,
        )

    return str(value)


def write_text_report(
    path: Path,
    scan_stats: dict[str, Any],
    dicomdir_info: dict[str, Any],
    series_summaries: list[dict[str, Any]],
    t1_info: dict[str, Any],
    findings: list[str],
) -> None:
    lines: list[str] = []

    lines.append("=" * 90)
    lines.append("DIAGNÓSTICO PREVIO PARA RM DE DIFUSIÓN Y TRACTOGRAFÍA")
    lines.append("=" * 90)
    lines.append(
        f"Fecha de ejecución: {datetime.now().isoformat(timespec='seconds')}"
    )
    lines.append("")
    lines.append(
        "Este análisis NO convierte DICOM, NO ajusta tensores y NO genera tractografía."
    )
    lines.append("")

    lines.append("1. RUTAS")
    lines.append("-" * 90)
    lines.append(f"DICOMDIR: {DICOMDIR_PATH}")
    lines.append(f"Raíz escaneada: {scan_stats.get('scan_root')}")
    lines.append(f"T1: {T1_PATH}")
    lines.append(f"Salida: {OUTPUT_DIR}")
    lines.append("")

    lines.append("2. ESTADO DEL DICOMDIR")
    lines.append("-" * 90)
    for key in (
        "exists",
        "is_file",
        "is_directory",
        "referenced_file_count",
        "existing_referenced_file_count",
        "missing_referenced_file_count",
        "error",
    ):
        lines.append(f"{key}: {format_value(dicomdir_info.get(key))}")
    lines.append("")

    lines.append("3. ESCANEO DICOM")
    lines.append("-" * 90)
    for key, value in scan_stats.items():
        lines.append(f"{key}: {format_value(value)}")
    lines.append(f"series_detectadas: {len(series_summaries)}")
    lines.append("")

    lines.append("4. HALLAZGOS GENERALES")
    lines.append("-" * 90)
    for finding in findings:
        lines.append(f"- {finding}")
    lines.append("")

    lines.append("5. SERIES ORDENADAS POR RELEVANCIA")
    lines.append("-" * 90)

    if not series_summaries:
        lines.append("No se encontraron series DICOM legibles.")
    else:
        for index, summary in enumerate(series_summaries, start=1):
            lines.append("")
            lines.append(
                f"[{index}] {summary['classification']} | "
                f"puntaje={summary['candidate_score']}"
            )
            lines.append(
                f"Serie {summary.get('series_number', '')}: "
                f"{summary.get('series_description') or '(sin descripción)'}"
            )
            lines.append(
                f"Protocolo: {summary.get('protocol_name') or '(sin nombre)'}"
            )
            lines.append(
                f"Secuencia: {summary.get('sequence_name') or '(sin nombre)'}"
            )
            lines.append(
                f"Fabricante/modelo: "
                f"{summary.get('manufacturer') or 'N/D'} / "
                f"{summary.get('manufacturer_model') or 'N/D'}"
            )
            lines.append(
                f"Archivos/frames: "
                f"{summary.get('instance_file_count')} / "
                f"{summary.get('total_frame_count')}"
            )
            lines.append(
                f"Matriz: {summary.get('rows')} x {summary.get('columns')} | "
                f"PixelSpacing: {format_value(summary.get('pixel_spacing'))} | "
                f"SliceThickness: {format_value(summary.get('slice_thickness'))}"
            )
            lines.append(
                f"Posiciones de corte: "
                f"{summary.get('unique_slice_position_count')} | "
                f"Volúmenes estimados por cortes: "
                f"{format_value(summary.get('estimated_volume_count_from_slices'))}"
            )
            lines.append(
                f"b-values agrupados: "
                f"{format_value(summary.get('clustered_b_values'))}"
            )
            lines.append(
                f"Mediciones b≈0: {summary.get('b0_measurement_count')} | "
                f"Mediciones b>0: {summary.get('nonzero_b_value_count')}"
            )
            lines.append(
                f"Ejes de gradiente no nulos: "
                f"{summary.get('unique_nonzero_gradient_direction_count')}"
            )
            lines.append(
                f"Fuente de metadatos de difusión: "
                f"{format_value(summary.get('diffusion_metadata_sources'))}"
            )
            lines.append(
                f"ImageType: {summary.get('image_type') or 'N/D'}"
            )
            lines.append(
                "Motivos de clasificación: "
                + "; ".join(summary.get("classification_reasons", []))
            )
            lines.append(
                f"SeriesInstanceUID: {summary.get('series_uid')}"
            )

    lines.append("")
    lines.append("6. INSPECCIÓN DE rT1.nii")
    lines.append("-" * 90)

    t1_keys = (
        "exists",
        "error",
        "image_class",
        "shape",
        "ndim",
        "dtype_on_disk",
        "voxel_sizes_mm",
        "orientation_axcodes",
        "affine",
        "affine_determinant_3x3",
        "qform_code",
        "sform_code",
        "xyzt_units",
        "zooms",
        "obliquity_degrees_per_axis",
        "intensity_statistics",
        "nonzero_bounding_box",
        "is_3d",
        "has_valid_spatial_affine",
        "preview_path",
    )

    for key in t1_keys:
        lines.append(f"{key}: {format_value(t1_info.get(key))}")

    lines.append("")
    lines.append("7. ARCHIVOS QUE DEBES COMPARTIR PARA LA SIGUIENTE REVISIÓN")
    lines.append("-" * 90)
    lines.append("- reporte_resumen.txt")
    lines.append("- series_summary.csv")
    lines.append("- diagnostico.json")
    lines.append("- vista_previa_rT1.png")
    lines.append("")
    lines.append(
        "No es necesario compartir todavía los DICOM originales ni datos identificables."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> int:
    print("=" * 75)
    print("INSPECCIÓN PREVIA DE DATOS DE DIFUSIÓN Y T1")
    print("=" * 75)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DICOMDIR_PATH.exists():
        print(f"\nERROR: no existe la ruta DICOMDIR:\n{DICOMDIR_PATH}")
        return 1

    if not T1_PATH.exists():
        print(
            f"\nADVERTENCIA: no existe rT1.nii en:\n{T1_PATH}\n"
            "Se continuará con el escaneo DICOM."
        )

    scan_root = determine_scan_root(DICOMDIR_PATH)

    print(f"\nRaíz DICOM que se va a revisar:\n{scan_root}")
    print(f"\nArchivo T1:\n{T1_PATH}")
    print(f"\nCarpeta de salida:\n{OUTPUT_DIR}")

    dicomdir_info = inspect_dicomdir_index(DICOMDIR_PATH)

    records, errors, scan_stats = scan_dicoms(
        root=scan_root,
        dicomdir_path=DICOMDIR_PATH,
    )

    series_summaries = summarize_series(records)

    print("\nInspeccionando rT1.nii...")
    t1_info = inspect_nifti(T1_PATH, OUTPUT_DIR)

    findings = build_findings(
        series_summaries,
        t1_info,
        dicomdir_info,
    )

    diagnostic = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "software": {
            "python": sys.version,
            "pydicom": getattr(pydicom, "__version__", "desconocida"),
            "nibabel": getattr(nib, "__version__", "desconocida"),
            "numpy": np.__version__,
        },
        "input_paths": {
            "dicomdir": str(DICOMDIR_PATH),
            "scan_root": str(scan_root),
            "t1": str(T1_PATH),
            "output_dir": str(OUTPUT_DIR),
        },
        "dicomdir_index": dicomdir_info,
        "scan_statistics": scan_stats,
        "series_count": len(series_summaries),
        "series": series_summaries,
        "t1_nifti": t1_info,
        "findings": findings,
        "read_errors": errors[:MAX_ERRORS_IN_JSON],
    }

    json_path = OUTPUT_DIR / "diagnostico.json"
    csv_path = OUTPUT_DIR / "series_summary.csv"
    errors_path = OUTPUT_DIR / "errores_lectura.csv"
    report_path = OUTPUT_DIR / "reporte_resumen.txt"

    json_path.write_text(
        json.dumps(
            json_safe(diagnostic),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_series_csv(csv_path, series_summaries)
    write_errors_csv(errors_path, errors)

    write_text_report(
        report_path,
        scan_stats,
        dicomdir_info,
        series_summaries,
        t1_info,
        findings,
    )

    print("\n" + "=" * 75)
    print("INSPECCIÓN FINALIZADA")
    print("=" * 75)

    for finding in findings:
        print(f"- {finding}")

    print("\nResultados:")
    print(f"  {report_path}")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {OUTPUT_DIR / 'vista_previa_rT1.png'}")
    print(f"  {errors_path}")

    print(
        "\nComparte reporte_resumen.txt, series_summary.csv, "
        "diagnostico.json y vista_previa_rT1.png."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
