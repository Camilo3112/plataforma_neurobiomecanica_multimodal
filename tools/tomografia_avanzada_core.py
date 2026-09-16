"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        NÚCLEO AVANZADO DE TOMOGRAFÍA                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: tools/tomografia_avanzada_core.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Implementa segmentación avanzada de miembro inferior sobre cortes TAC. El
modelo
combina umbralización HU, conectividad de componentes, momentos geométricos,
separación bilateral por centroides y cuantificación de composición corporal.
La
medición de área/volumen utiliza la geometría física del DICOM, no conteos de
pixeles sin escala.

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

import os
import csv
import traceback
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    center_of_mass,
    gaussian_filter1d,
    label as scipy_label,
    map_coordinates,
)
from scipy.signal import find_peaks
from skimage.draw import polygon
from skimage.measure import find_contours
from skimage.segmentation import find_boundaries

# Motor de wavelets para analizar señales 1D.
try:
    from tools.cwt_engine import cwt_1d
except ModuleNotFoundError:
    # Cuando este archivo se ejecuta directamente como script, sys.path apunta
    # a la carpeta tools/, no a la raíz de la suite. En ese caso se importa local.
    from cwt_engine import cwt_1d


# ==============================================================================
# CONFIGURACIÓN GENERAL
# ==============================================================================
INPUT_DIR = os.environ.get("VCE_TAC_INPUT_DIR", "/home/humath/Escritorio/datos/paciente 3/Antes/TAC")
OUTPUT_DIR = os.environ.get("VCE_TAC_OUTPUT_DIR", "/home/humath/Escritorio/resultados/paciente 3/Antes/tomografia/avanzada")

# Convención heredada del código original:
# - mitad izquierda de la matriz = miembro derecho del paciente
# - mitad derecha de la matriz   = miembro izquierdo del paciente
# Debe verificarse visualmente en 3D Slicer según la orientación del estudio.
MUSCLES_CONFIG = {
    "Vasto_Lateral_Der": {"roi": "left", "angles": (80, 220)},
    "Vasto_Medial_Der": {"roi": "left", "angles": (-40, 80)},
    "Vasto_Lateral_Izq": {"roi": "right", "angles": (-40, 100)},
    "Vasto_Medial_Izq": {"roi": "right", "angles": (80, 220)},
}

SIDE_CONFIG = {
    "left": {
        "patient_side": "Derecho",
        "muscles": ["Vasto_Lateral_Der", "Vasto_Medial_Der"],
        "label_values": {
            "line": 1,
            "trochanter": 2,
            "tibia": 3,
            "midpoint": 4,
        },
    },
    "right": {
        "patient_side": "Izquierdo",
        "muscles": ["Vasto_Lateral_Izq", "Vasto_Medial_Izq"],
        "label_values": {
            "line": 5,
            "trochanter": 6,
            "tibia": 7,
            "midpoint": 8,
        },
    },
}

# Etiquetas del labelmap independiente de composición corporal.
# La mitad izquierda de la matriz corresponde al miembro derecho del paciente.
TISSUE_LABELS = {
    "left": {
        "epidermis": 1,
        "dermis": 2,
        "subcutaneous_fat": 3,
        "total_muscle": 4,
    },
    "right": {
        "epidermis": 5,
        "dermis": 6,
        "subcutaneous_fat": 7,
        "total_muscle": 8,
    },
}

NUM_RAYS = 60
BLIND_ZONE_MM = 15.0
MAX_MUSCLE_THICKNESS_MM = 65.0

# Parámetros Wavelet (Sombrero Mexicano).
CWT_MIN_SCALE = 1.0
CWT_MAX_SCALE = 6.0
CWT_N_SCALES = 12

# Detección ósea y landmarks.
BONE_HU_THRESHOLD = 300
MIN_BONE_COMPONENT_AREA_PX = 40
PROXIMAL_SEARCH_FRACTION = 0.35
DISTAL_SEARCH_FRACTION = 0.35
TRACK_MAX_CENTROID_SHIFT_PX = 55.0
MUSCLE_MIDPOINT_MARGIN_SLICES = 2
LANDMARK_RADIUS_PX = 4
MIDPOINT_RADIUS_PX = 5
LINE_RADIUS_PX = 1

# Análisis transversal de piel, grasa subcutánea y músculo total.
# En TAC clínico la epidermis y la dermis no suelen resolverse como capas
# histológicas independientes. El algoritmo estima la superficie cutánea externa
# y el límite interno dérmico mediante CWT y restricciones de espesor plausibles.
TISSUE_NUM_RAYS = 180
TISSUE_BODY_HU_THRESHOLD = -500.0
SUBCUTANEOUS_FAT_HU_MIN = -190.0
SUBCUTANEOUS_FAT_HU_MAX = -30.0
TOTAL_MUSCLE_HU_MIN = -29.0
TOTAL_MUSCLE_HU_MAX = 150.0
TISSUE_CWT_MIN_SCALE = 1.0
TISSUE_CWT_MAX_SCALE = 10.0
TISSUE_CWT_N_SCALES = 18
TISSUE_RAY_BLIND_ZONE_MM = 18.0
EPIDERMIS_DERMIS_MIN_DEPTH_MM = 0.8
EPIDERMIS_DERMIS_MAX_DEPTH_MM = 4.0
EPIDERMIS_DERMIS_FALLBACK_DEPTH_MM = 2.0
MIN_SUBCUTANEOUS_LAYER_MM = 2.0
OUTER_BOUNDARY_REFINEMENT_MM = 5.0
FASCIA_PRE_WINDOW_MM = 5.0
FASCIA_POST_WINDOW_MM = 5.0
MIN_TISSUE_COMPONENT_AREA_PX = 12
MIN_VALID_FASCIA_RAY_FRACTION = 0.45
TISSUE_BOUNDARY_SMOOTH_SIGMA = 2.0

# Salidas.
MUSCLE_TRACER_FILENAME = "Muslos_Cuatro_Tracer_AutoZ.nii"
LANDMARK_LABEL_FILENAME = "Landmarks_Linea_CorteMedio.nii"
LANDMARK_OVERLAY_FILENAME = "TAC_Landmarks_Linea_CorteMedio.nii"
TISSUE_LABEL_FILENAME = "Composicion_Corporal_CorteMedio.nii"
TISSUE_OVERLAY_FILENAME = "TAC_Composicion_Corporal_CorteMedio.nii"
REPORT_FILENAME = "Reporte_Morfometrico_Completo.txt"
LABELS_FILENAME = "Etiquetas_Landmarks.txt"
TISSUE_LABELS_FILENAME = "Etiquetas_Composicion_Corporal.txt"
TISSUE_PNG_PREFIX = "Composicion_Corporal_CorteMedio"
TISSUE_PNG_COMBINED_FILENAME = "Composicion_Corporal_CorteMedio_Resumen.png"
PNG_OUTPUT_SUBDIR = "PNG_Composicion"
INTRAMUSCULAR_FAT_LABEL_FILENAME = "Grasa_Intramuscular_Muscular.nii"
INTRAMUSCULAR_FAT_OVERLAY_FILENAME = "TAC_Grasa_Intramuscular_Muscular.nii"
INTRAMUSCULAR_FAT_LABELS_FILENAME = "Etiquetas_Grasa_Intramuscular.txt"
INTRAMUSCULAR_FAT_LABELS = {
    "Vasto_Lateral_Der": 1,
    "Vasto_Medial_Der": 2,
    "Vasto_Lateral_Izq": 3,
    "Vasto_Medial_Izq": 4,
}

MIN_INTRAMUSCULAR_FAT_COMPONENT_AREA_PX = 3
INTRAMUSCULAR_FAT_AREA_CSV_FILENAME = "Areas_Grasa_Intramuscular_Por_Corte.csv"

# Salidas adicionales solicitadas para tejido adiposo en NIfTI y métricas tabulares.
# Estos labelmaps son 3D y se guardan en el mismo espacio físico del TAC.
SUBCUTANEOUS_FAT_LABEL_FILENAME = "Grasa_Subcutanea_Volumen.nii.gz"
TOTAL_ADIPOSE_LABEL_FILENAME = "Tejido_Adiposo_Total_Volumen.nii.gz"
ADIPOSE_LABELS_FILENAME = "Etiquetas_Tejido_Adiposo_Total.txt"
ADIPOSE_METRICS_CSV_FILENAME = "Metricas_Tejido_Adiposo.csv"


# ==============================================================================
# UTILIDADES GENERALES
# ==============================================================================
def get_side_roi_mask(shape_2d: Tuple[int, int], roi_side: str) -> np.ndarray:
    """Crea la máscara de la mitad izquierda o derecha de la matriz."""
    h, w = shape_2d
    mask = np.zeros((h, w), dtype=bool)
    if roi_side == "left":
        mask[:, : w // 2] = True
    elif roi_side == "right":
        mask[:, w // 2 :] = True
    else:
        raise ValueError(f"ROI lateral no válido: {roi_side}")
    return mask


def round_point(point: Optional[Tuple[float, float, float]]) -> Optional[Tuple[int, int, int]]:
    if point is None:
        return None
    return tuple(int(round(v)) for v in point)


def point_inside_volume(point: Tuple[int, int, int], shape: Tuple[int, int, int]) -> bool:
    y, x, z = point
    return 0 <= y < shape[0] and 0 <= x < shape[1] and 0 <= z < shape[2]


def voxel_to_world(affine: np.ndarray, point_yxz: Optional[Tuple[float, float, float]]):
    """Convierte índices del volumen (fila, columna, corte) a coordenadas físicas."""
    if point_yxz is None:
        return None
    homogeneous = np.array([point_yxz[0], point_yxz[1], point_yxz[2], 1.0], dtype=float)
    world = affine @ homogeneous
    return tuple(float(v) for v in world[:3])


def format_point(point) -> str:
    if point is None:
        return "No detectado"
    return "(" + ", ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in point) + ")"


# ==============================================================================
# FASE 1: LECTURA DICOM ROBUSTA Y CONSTRUCCIÓN DEL AFFINE
# ==============================================================================
def _slice_projection(dcm, normal_vector: np.ndarray) -> float:
    position = np.asarray(dcm.ImagePositionPatient, dtype=float)
    return float(np.dot(position, normal_vector))


def build_dicom_affine(first_slice, slice_spacing: float, slice_direction_sign: float = 1.0) -> np.ndarray:
    """
    Construye un affine para un volumen almacenado como [fila, columna, corte].

    PixelSpacing[0] corresponde al paso entre filas y PixelSpacing[1] al paso
    entre columnas. ImageOrientationPatient contiene las direcciones de fila y
    columna del DICOM en coordenadas del paciente.
    """
    pixel_spacing = np.asarray(first_slice.PixelSpacing, dtype=float)
    position = np.asarray(first_slice.ImagePositionPatient, dtype=float)

    if hasattr(first_slice, "ImageOrientationPatient"):
        orientation = np.asarray(first_slice.ImageOrientationPatient, dtype=float)
        row_direction = orientation[:3]
        column_direction = orientation[3:]
        slice_direction = np.cross(row_direction, column_direction)

        affine = np.eye(4, dtype=float)
        # Eje 0 del ndarray: fila -> dirección de columna DICOM.
        affine[:3, 0] = column_direction * pixel_spacing[0]
        # Eje 1 del ndarray: columna -> dirección de fila DICOM.
        affine[:3, 1] = row_direction * pixel_spacing[1]
        affine[:3, 2] = slice_direction * slice_spacing * slice_direction_sign
        affine[:3, 3] = position
        return affine

    # Fallback compatible con el script original.
    affine = np.diag([pixel_spacing[0], pixel_spacing[1], slice_spacing, 1.0])
    affine[:3, 3] = position
    return affine



# ==============================================================================
# QC DICOM v3.21.5: VERIFICACIÓN DE DOS FÉMURES / DOS MIEMBROS
# ==============================================================================
# Motivo: si una serie DICOM contiene un solo miembro o se mezclan varias series,
# el algoritmo puede terminar calculando un miembro y duplicándolo para ambos lados.
# Esta verificación selecciona la serie que realmente contiene ambos fémures.
# v3.21.6 cambia la política: si un corte candidato tiene un solo fémur,
# NO se detiene inmediatamente; primero busca dentro de la misma serie el corte
# anatómico más cercano con dos fémures y lo usa para evitar duplicaciones.
MIN_TWO_FEMUR_SLICES = int(os.environ.get("VCE_TAC_MIN_TWO_FEMUR_SLICES", "8"))
MIN_FEMUR_PAIR_SEPARATION_PX = float(os.environ.get("VCE_TAC_MIN_FEMUR_PAIR_SEPARATION_PX", "35"))
MAX_FEMUR_PAIR_AREA_RATIO = float(os.environ.get("VCE_TAC_MAX_FEMUR_PAIR_AREA_RATIO", "6.0"))
MIDPOINT_TWO_FEMUR_SEARCH_WINDOW = int(os.environ.get("VCE_TAC_MIDPOINT_TWO_FEMUR_WINDOW", "90"))
# Si el corte medio no tiene dos fémures y no hay reemplazo dentro de la ventana,
# buscar en todo el stack el corte bilateral más cercano. Esto responde al caso en
# que la lectura inicial cae en un corte parcial aunque el DICOM sí contiene ambos miembros.
ALLOW_GLOBAL_TWO_FEMUR_RESCUE = os.environ.get("VCE_TAC_RESCATE_GLOBAL_DOS_FEMURES", "1").strip().lower() not in {"0", "false", "no"}
# Si el stack tiene algunos cortes bilaterales pero menos que el mínimo estricto,
# continuar en modo rescate en vez de abortar toda la ejecución.
ALLOW_LOW_COUNT_TWO_FEMUR_RESCUE = os.environ.get("VCE_TAC_PERMITIR_POCOS_CORTES_DOS_FEMURES", "1").strip().lower() not in {"0", "false", "no"}
DICOM_SERIES_QC_FILENAME = "QC_DICOM_Series_Dos_Femures.csv"
SLICE_QC_FILENAME = "QC_Cortes_Dos_Femures.csv"
MIDPOINT_QC_FILENAME = "QC_Cortes_Medios_Ajustados_Dos_Femures.csv"


def _safe_dicom_key_value(value):
    try:
        if isinstance(value, (list, tuple)):
            return "_".join(str(round(float(v), 6)) for v in value)
        return str(value)
    except Exception:
        return str(value)


def _dicom_group_key(dcm):
    """Agrupa por serie y geometría para no mezclar series DICOM distintas."""
    uid = getattr(dcm, "SeriesInstanceUID", "SIN_UID")
    rows = int(getattr(dcm, "Rows", 0))
    cols = int(getattr(dcm, "Columns", 0))
    spacing = tuple(round(float(v), 6) for v in getattr(dcm, "PixelSpacing", [1.0, 1.0]))
    orientation = tuple(round(float(v), 6) for v in getattr(dcm, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]))
    desc = str(getattr(dcm, "SeriesDescription", ""))[:80]
    return (str(uid), rows, cols, spacing, orientation, desc)


def _sort_unique_dicom_slices(raw_slices):
    if not raw_slices:
        raise ValueError("Grupo DICOM vacío.")
    first = raw_slices[0]
    if hasattr(first, "ImageOrientationPatient"):
        orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
    else:
        normal = np.array([0.0, 0.0, 1.0])

    unique_slices = {}
    for dcm in raw_slices:
        if not hasattr(dcm, "ImagePositionPatient"):
            continue
        projection = round(_slice_projection(dcm, normal), 4)
        if projection not in unique_slices:
            unique_slices[projection] = dcm

    sorted_items = sorted(unique_slices.items(), key=lambda item: item[0])
    projections = np.asarray([item[0] for item in sorted_items], dtype=float)
    slices = [item[1] for item in sorted_items]
    if not slices:
        raise ValueError("Grupo DICOM sin cortes únicos válidos.")

    if len(slices) > 1:
        spacing_diffs = np.diff(projections)
        real_z_spacing = float(np.median(np.abs(spacing_diffs)))
        direction_sign = float(np.sign(np.median(spacing_diffs))) or 1.0
    else:
        real_z_spacing = float(getattr(first, "SliceThickness", 1.0))
        direction_sign = 1.0

    return slices, real_z_spacing, direction_sign


def _build_volume_from_sorted_slices(slices, real_z_spacing, direction_sign):
    first = slices[0]
    pixel_spacing = [float(v) for v in first.PixelSpacing]
    image_3d = np.stack([dcm.pixel_array for dcm in slices], axis=-1)
    intercept = float(getattr(first, "RescaleIntercept", 0.0))
    slope = float(getattr(first, "RescaleSlope", 1.0))
    image_hu = image_3d.astype(np.float32) * slope + intercept
    affine = build_dicom_affine(first, real_z_spacing, direction_sign)
    return image_hu, pixel_spacing, affine


def _femur_components_full_field_qc(image_slice_hu: np.ndarray) -> List[Dict]:
    """Componentes óseos candidatos a fémur en todo el campo de visión."""
    bone_mask = image_slice_hu > BONE_HU_THRESHOLD
    bone_mask = binary_opening(bone_mask, iterations=1)
    labels, num = scipy_label(bone_mask)
    components = []
    for component_id in range(1, num + 1):
        component_mask = labels == component_id
        area = int(np.sum(component_mask))
        if area < MIN_BONE_COMPONENT_AREA_PX:
            continue
        ys, xs = np.where(component_mask)
        cy, cx = center_of_mass(component_mask)
        components.append({
            "area": area,
            "centroid": (float(cy), float(cx)),
            "bbox": (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())),
        })
    components.sort(key=lambda c: c["area"], reverse=True)
    return components


def _select_two_femur_pair(components: List[Dict], shape_2d: Tuple[int, int]):
    """Determina si hay dos fémures plausibles en el corte."""
    if len(components) < 2:
        return None
    candidates = components[: min(8, len(components))]
    best = None
    h, w = shape_2d
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            c1, c2 = candidates[i], candidates[j]
            a1, a2 = float(c1["area"]), float(c2["area"])
            if min(a1, a2) <= 0:
                continue
            ratio = max(a1, a2) / max(min(a1, a2), 1.0)
            if ratio > MAX_FEMUR_PAIR_AREA_RATIO:
                continue
            y1, x1 = c1["centroid"]
            y2, x2 = c2["centroid"]
            sep = float(np.hypot(y1 - y2, x1 - x2))
            if sep < MIN_FEMUR_PAIR_SEPARATION_PX:
                continue
            # En muslos bilaterales los centros suelen estar separados sobre todo en X.
            x_sep = abs(x1 - x2)
            if x_sep < 0.08 * float(w):
                continue
            # Penaliza pares con separación vertical exagerada o áreas demasiado dispares.
            y_penalty = abs(y1 - y2) / max(float(h), 1.0)
            score = sep + 0.015 * min(a1, a2) - 12.0 * y_penalty - 5.0 * np.log(ratio)
            item = (score, sep, ratio, c1, c2)
            if best is None or item[0] > best[0]:
                best = item
    return best


def get_bilateral_femur_centroids_for_slice(image_slice_hu: np.ndarray):
    """Devuelve centros femorales separados por lado de imagen.

    Convención del módulo: la mitad izquierda de la matriz representa el miembro
    derecho del paciente (roi_side="left") y la mitad derecha representa el
    miembro izquierdo (roi_side="right"). Esta función evita que ambos lados
    tomen el mismo fémur cuando el punto anatómico esperado quedó mal estimado.
    """
    components = _femur_components_full_field_qc(image_slice_hu)
    pair = _select_two_femur_pair(components, image_slice_hu.shape[:2])
    if pair is None:
        return None
    _score, sep, ratio, c1, c2 = pair
    ordered = sorted([c1, c2], key=lambda c: float(c["centroid"][1]))
    return {
        "left": ordered[0]["centroid"],
        "right": ordered[1]["centroid"],
        "separation_px": float(sep),
        "area_ratio": float(ratio),
    }


def slice_has_two_femurs(image_slice_hu: np.ndarray) -> bool:
    return get_bilateral_femur_centroids_for_slice(image_slice_hu) is not None


def scan_volume_two_femur_qc(image_hu: np.ndarray, z_values: Optional[List[int]] = None) -> List[Dict]:
    """Escanea cortes y marca cuáles contienen dos fémures plausibles."""
    records = []
    total_slices = image_hu.shape[2]
    if z_values is None:
        z_values = list(range(total_slices))
    for z in z_values:
        if z < 0 or z >= total_slices:
            continue
        comps = _femur_components_full_field_qc(image_hu[:, :, z])
        pair = _select_two_femur_pair(comps, image_hu.shape[:2])
        record = {
            "z": int(z),
            "n_bone_components": int(len(comps)),
            "has_two_femurs": bool(pair is not None),
            "pair_separation_px": "",
            "pair_area_ratio": "",
            "femur1_y": "", "femur1_x": "", "femur1_area_px": "",
            "femur2_y": "", "femur2_x": "", "femur2_area_px": "",
        }
        if pair is not None:
            _score, sep, ratio, c1, c2 = pair
            record.update({
                "pair_separation_px": float(sep),
                "pair_area_ratio": float(ratio),
                "femur1_y": float(c1["centroid"][0]),
                "femur1_x": float(c1["centroid"][1]),
                "femur1_area_px": int(c1["area"]),
                "femur2_y": float(c2["centroid"][0]),
                "femur2_x": float(c2["centroid"][1]),
                "femur2_area_px": int(c2["area"]),
            })
        records.append(record)
    return records


def _two_femur_score_for_volume(image_hu: np.ndarray) -> Dict:
    # Muestreo rápido para seleccionar serie: suficiente para detectar si contiene ambos miembros.
    n = image_hu.shape[2]
    if n <= 0:
        return {"good_slices": 0, "sampled_slices": 0, "best_z": None}
    step = max(1, n // 120)
    z_values = list(range(0, n, step))
    records = scan_volume_two_femur_qc(image_hu, z_values=z_values)
    good = [r for r in records if r["has_two_femurs"]]
    best_z = None
    if good:
        best_z = int(max(good, key=lambda r: float(r.get("pair_separation_px") or 0))["z"])
    return {
        "good_slices": int(len(good)),
        "sampled_slices": int(len(records)),
        "best_z": best_z,
        "good_fraction": float(len(good) / max(len(records), 1)),
    }


def _write_csv_rows(path: str, rows: List[Dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_series_qc_csv(rows: List[Dict]):
    path = os.path.join(OUTPUT_DIR, DICOM_SERIES_QC_FILENAME)
    fields = [
        "selected", "series_uid", "series_description", "rows", "columns", "n_raw_slices",
        "n_unique_slices", "good_two_femur_slices_sampled", "sampled_slices", "good_fraction",
        "best_two_femur_z", "pixel_spacing", "reason",
    ]
    _write_csv_rows(path, rows, fields)
    print(f"-> QC series DICOM dos fémures: {path}")


def _write_slice_qc_csv(records: List[Dict]):
    path = os.path.join(OUTPUT_DIR, SLICE_QC_FILENAME)
    fields = [
        "z", "n_bone_components", "has_two_femurs", "pair_separation_px", "pair_area_ratio",
        "femur1_y", "femur1_x", "femur1_area_px", "femur2_y", "femur2_x", "femur2_area_px",
    ]
    _write_csv_rows(path, records, fields)
    print(f"-> QC cortes dos fémures: {path}")


def validate_loaded_volume_has_two_femurs(image_hu: np.ndarray) -> List[Dict]:
    """QC de stack cargado.

    v3.21.6: no aborta cuando hay pocos cortes bilaterales si existe al menos
    uno; entra en modo rescate y las etapas posteriores reemplazan cualquier
    corte medio con un solo fémur por el corte bilateral más cercano.
    """
    records = scan_volume_two_femur_qc(image_hu)
    good = [r for r in records if r["has_two_femurs"]]
    _write_slice_qc_csv(records)
    if len(good) == 0:
        raise ValueError(
            "QC DICOM falló: no se detectó ningún corte con DOS fémures. "
            "El pipeline no puede garantizar ambos miembros y no procesará este stack. "
            f"Revise {SLICE_QC_FILENAME} y {DICOM_SERIES_QC_FILENAME} en la carpeta de salida."
        )
    if len(good) < MIN_TWO_FEMUR_SLICES:
        msg = (
            "ADVERTENCIA QC: se detectaron pocos cortes con DOS fémures "
            f"({len(good)} de mínimo recomendado {MIN_TWO_FEMUR_SLICES}). "
            "Se continúa en MODO RESCATE: cualquier corte de análisis con un solo fémur "
            "se reemplazará por el corte bilateral más cercano."
        )
        if ALLOW_LOW_COUNT_TWO_FEMUR_RESCUE:
            print(msg)
            return records
        raise ValueError(msg + f" Revise {SLICE_QC_FILENAME}.")
    print(f"QC DICOM OK: {len(good)} cortes con dos fémures detectados.")
    return records


def _qc_record_for_z(qc_records: List[Dict], z: Optional[int]):
    if z is None:
        return None
    zi = int(z)
    for r in qc_records:
        if int(r["z"]) == zi:
            return r
    return None


def _nearest_two_femur_slice(qc_records: List[Dict], z: int, window: int = MIDPOINT_TWO_FEMUR_SEARCH_WINDOW):
    good_z = [int(r["z"]) for r in qc_records if r["has_two_femurs"]]
    if not good_z:
        return None
    candidates = [zz for zz in good_z if abs(zz - int(z)) <= int(window)]
    if candidates:
        return int(min(candidates, key=lambda zz: abs(zz - int(z))))
    if ALLOW_GLOBAL_TWO_FEMUR_RESCUE:
        # Rescate global: si no hay corte bilateral dentro de la ventana, usar el
        # corte bilateral más cercano de todo el stack antes que anular el miembro.
        return int(min(good_z, key=lambda zz: abs(zz - int(z))))
    return None


def enforce_midpoint_slices_have_two_femurs(image_hu: np.ndarray, landmarks: Dict, qc_records: List[Dict]) -> Dict:
    """Evita procesar cortes medios donde solo aparece un fémur.

    Si un corte medio no tiene dos fémures, se ajusta al corte más cercano con dos fémures.
    Si no existe dentro de la ventana y el rescate global está activo, busca en todo
    el stack. Solo si no hay ningún corte bilateral se anula ese lado.
    """
    rows = []
    for roi_side, info in landmarks.items():
        z_original = info.get("midpoint_slice")
        z_final = z_original
        action = "sin_midpoint"
        ok_original = False
        if z_original is not None:
            rec = _qc_record_for_z(qc_records, int(z_original))
            ok_original = bool(rec and rec.get("has_two_femurs"))
            if ok_original:
                action = "ok"
            else:
                nearest = _nearest_two_femur_slice(qc_records, int(z_original))
                if nearest is not None:
                    z_final = int(nearest)
                    info["midpoint_slice_original"] = int(z_original)
                    info["midpoint_slice"] = int(z_final)
                    info["midpoint_qc_adjusted"] = True
                    if abs(int(z_final) - int(z_original)) <= int(MIDPOINT_TWO_FEMUR_SEARCH_WINDOW):
                        action = "ajustado_a_corte_cercano_con_dos_femures"
                    else:
                        action = "ajustado_por_rescate_global_a_corte_con_dos_femures"
                    print(
                        f"QC RESCATE: {info.get('patient_side', roi_side)} Z={z_original} tenía un solo fémur; "
                        f"se usará Z={z_final}, corte con dos fémures."
                    )
                else:
                    info["midpoint_slice_original"] = int(z_original)
                    info["midpoint_slice"] = None
                    info["midpoint_qc_adjusted"] = False
                    action = "anulado_sin_corte_cercano_con_dos_femures"
                    print(
                        f"ADVERTENCIA QC: {info.get('patient_side', roi_side)} tenía corte medio Z={z_original} "
                        "sin dos fémures y no se encontró reemplazo cercano. Se omite ese lado."
                    )
        rows.append({
            "roi_side": roi_side,
            "patient_side": info.get("patient_side", roi_side),
            "z_original": z_original if z_original is not None else "",
            "z_final": info.get("midpoint_slice") if info.get("midpoint_slice") is not None else "",
            "original_has_two_femurs": ok_original,
            "action": action,
        })
    path = os.path.join(OUTPUT_DIR, MIDPOINT_QC_FILENAME)
    _write_csv_rows(path, rows, ["roi_side", "patient_side", "z_original", "z_final", "original_has_two_femurs", "action"])
    print(f"-> QC cortes medios ajustados: {path}")
    return landmarks

def load_dicom_series(directory: str):
    print(f"Explorando DICOM en: {directory} ...")
    raw_slices = []

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.upper() == "DICOMDIR":
                continue
            path = os.path.join(root, filename)
            try:
                dcm = pydicom.dcmread(path, force=True)
                if hasattr(dcm, "PixelData") and hasattr(dcm, "ImagePositionPatient"):
                    raw_slices.append(dcm)
            except Exception:
                continue

    if not raw_slices:
        raise ValueError("No se encontraron imágenes DICOM válidas con PixelData e ImagePositionPatient.")

    # v3.21.5: no se mezclan series. Se evalúa cada serie/geometría y se elige
    # la que tenga evidencia de ambos miembros por presencia de dos fémures.
    grouped = {}
    for dcm in raw_slices:
        grouped.setdefault(_dicom_group_key(dcm), []).append(dcm)

    candidates = []
    series_qc_rows = []
    print(f"Series/geometrías DICOM candidatas: {len(grouped)}")
    for key, group_slices in grouped.items():
        uid, rows, cols, spacing, orientation, desc = key
        row = {
            "selected": False,
            "series_uid": uid,
            "series_description": desc,
            "rows": rows,
            "columns": cols,
            "n_raw_slices": len(group_slices),
            "n_unique_slices": 0,
            "good_two_femur_slices_sampled": 0,
            "sampled_slices": 0,
            "good_fraction": 0,
            "best_two_femur_z": "",
            "pixel_spacing": spacing,
            "reason": "",
        }
        try:
            if len(group_slices) < 3:
                row["reason"] = "descartada_menos_de_3_cortes"
                series_qc_rows.append(row)
                continue
            slices, real_z_spacing, direction_sign = _sort_unique_dicom_slices(group_slices)
            row["n_unique_slices"] = len(slices)
            if len(slices) < 3:
                row["reason"] = "descartada_menos_de_3_cortes_unicos"
                series_qc_rows.append(row)
                continue
            image_hu, pixel_spacing, affine = _build_volume_from_sorted_slices(slices, real_z_spacing, direction_sign)
            score = _two_femur_score_for_volume(image_hu)
            row["good_two_femur_slices_sampled"] = score["good_slices"]
            row["sampled_slices"] = score["sampled_slices"]
            row["good_fraction"] = score["good_fraction"]
            row["best_two_femur_z"] = "" if score["best_z"] is None else score["best_z"]
            row["pixel_spacing"] = str(pixel_spacing)
            row["reason"] = "evaluada"
            candidates.append({
                "row": row,
                "slices": slices,
                "image_hu": image_hu,
                "pixel_spacing": pixel_spacing,
                "slice_spacing": real_z_spacing,
                "affine": affine,
                "score": score,
            })
            series_qc_rows.append(row)
        except Exception as exc:
            row["reason"] = f"error_evaluando: {exc}"
            series_qc_rows.append(row)

    if not candidates:
        _write_series_qc_csv(series_qc_rows)
        raise ValueError("No se pudo construir ninguna serie DICOM candidata válida.")

    # Selección: máxima evidencia de dos fémures; desempate por número de cortes.
    candidates.sort(
        key=lambda c: (
            int(c["score"]["good_slices"]),
            float(c["score"]["good_fraction"]),
            int(len(c["slices"])),
        ),
        reverse=True,
    )
    selected = candidates[0]
    selected["row"]["selected"] = True
    for row in series_qc_rows:
        if row is selected["row"]:
            row["selected"] = True
    _write_series_qc_csv(series_qc_rows)

    image_hu = selected["image_hu"]
    pixel_spacing = selected["pixel_spacing"]
    real_z_spacing = selected["slice_spacing"]
    affine = selected["affine"]
    slices = selected["slices"]

    print("Serie DICOM seleccionada por QC de dos fémures:")
    print(f"  -> SeriesDescription: {selected['row'].get('series_description')}")
    print(f"  -> SeriesInstanceUID: {selected['row'].get('series_uid')}")
    print(f"  -> Cortes únicos: {len(slices)}")
    print(f"  -> Cortes muestreados con dos fémures: {selected['score']['good_slices']}/{selected['score']['sampled_slices']}")

    # Validación estricta del volumen completo ya seleccionado.
    validate_loaded_volume_has_two_femurs(image_hu)

    print(f"Cortes espaciales únicos: {len(slices)}")
    print(f"  -> Resolución fila/columna: {pixel_spacing} mm")
    print(f"  -> Espaciado físico entre cortes: {real_z_spacing:.4f} mm")
    print(f"  -> Dimensiones del volumen: {image_hu.shape}")

    return image_hu, pixel_spacing, real_z_spacing, affine


# ==============================================================================
# FASE 1.5: AUTO-DETECCIÓN DE LÍMITES Z
# ==============================================================================
def auto_detect_z_limits(image_hu: np.ndarray) -> Tuple[int, int]:
    print("Analizando topología ósea bilateral para auto-detectar límites anatómicos...")
    total_slices = image_hu.shape[2]

    qc_records = scan_volume_two_femur_qc(image_hu)
    valid_z = np.asarray([int(r["z"]) for r in qc_records if r["has_two_femurs"]], dtype=int)

    if valid_z.size == 0:
        raise ValueError(
            "No se pudo detectar un rango Z bilateral con dos fémures. "
            "No se procesará ese stack porque no existe ningún corte bilateral verificable."
        )

    # Usa percentiles para evitar extremos de cadera/rodilla y quedarse con el tramo útil bilateral.
    z_low = int(np.percentile(valid_z, 5))
    z_high = int(np.percentile(valid_z, 95)) + 1
    z_start_final = max(0, z_low)
    z_end_final = min(total_slices, z_high)

    if z_end_final - z_start_final < 10:
        # Si el rango bilateral estricto es corto, usa todo el intervalo donde hubo dos fémures.
        z_start_final = max(0, int(valid_z.min()))
        z_end_final = min(total_slices, int(valid_z.max()) + 1)

    if z_end_final - z_start_final < 5:
        raise ValueError(
            f"El rango Z bilateral es demasiado corto ({z_end_final - z_start_final} cortes). "
            "Revise que la serie DICOM seleccionada contenga ambos muslos completos."
        )

    print(f"  -> Rango bilateral de estudio: Z={z_start_final} a Z={z_end_final - 1}")
    return z_start_final, z_end_final


# ==============================================================================
# FASE 2: EXTRACCIÓN Y DETECCIÓN ADAPTATIVA 1D
# ==============================================================================
def find_fascia_1d(signal_1d: np.ndarray, mm_per_pixel: float) -> int:
    signal_length = len(signal_1d)
    if signal_length < 3:
        return max(0, signal_length - 1)

    blind_idx = max(0, int(BLIND_ZONE_MM / mm_per_pixel))
    max_idx = min(signal_length - 1, int(MAX_MUSCLE_THICKNESS_MM / mm_per_pixel))

    if max_idx <= blind_idx:
        return max_idx

    cwt_res = cwt_1d(
        signal=signal_1d,
        wavelet="mexican_hat",
        n_scales=CWT_N_SCALES,
        min_scale=CWT_MIN_SCALE,
        max_scale=CWT_MAX_SCALE,
    )
    power_1d = np.max(cwt_res.power, axis=0)
    power_1d = gaussian_filter1d(power_1d, sigma=1.0)

    prominence_thr = max(float(np.mean(power_1d) * 1.5), 1e-8)
    peaks, _ = find_peaks(power_1d, prominence=prominence_thr)

    valid_peaks = [int(p) for p in peaks if blind_idx < p < max_idx]
    return valid_peaks[0] if valid_peaks else max_idx


# ==============================================================================
# FASE 3: RAY-CASTING POLAR MÚLTIPLE
# ==============================================================================
def get_bone_components(
    image_slice_hu: np.ndarray,
    roi_side: str,
    hu_threshold: float = BONE_HU_THRESHOLD,
    min_area_px: int = MIN_BONE_COMPONENT_AREA_PX,
) -> List[Dict]:
    roi = get_side_roi_mask(image_slice_hu.shape, roi_side)
    bone_mask = (image_slice_hu > hu_threshold) & roi
    bone_mask = binary_opening(bone_mask, iterations=1)
    labels, num = scipy_label(bone_mask)

    components = []
    for component_id in range(1, num + 1):
        component_mask = labels == component_id
        area = int(np.sum(component_mask))
        if area < min_area_px:
            continue

        ys, xs = np.where(component_mask)
        cy, cx = center_of_mass(component_mask)
        components.append(
            {
                "mask": component_mask,
                "area": area,
                "centroid": (float(cy), float(cx)),
                "bbox": (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())),
            }
        )

    components.sort(key=lambda component: component["area"], reverse=True)
    return components


def get_femur_centroid(image_slice_hu: np.ndarray, roi_side: str):
    components = get_bone_components(image_slice_hu, roi_side)
    if not components:
        return None
    return components[0]["centroid"]


def process_multi_muscle_slice(image_slice_hu: np.ndarray, pixel_spacing: List[float]):
    h, w = image_slice_hu.shape
    slice_masks = {name: np.zeros((h, w), dtype=bool) for name in MUSCLES_CONFIG}
    slice_compartment_masks = {name: np.zeros((h, w), dtype=bool) for name in MUSCLES_CONFIG}
    mean_spacing = float((pixel_spacing[0] + pixel_spacing[1]) / 2.0)
    search_radius_px = max(5, int((MAX_MUSCLE_THICKNESS_MM + 20.0) / mean_spacing))
    radii = np.arange(0, search_radius_px, dtype=float)

    bilateral_centroids = get_bilateral_femur_centroids_for_slice(image_slice_hu)
    if bilateral_centroids is not None:
        centroids = {
            "left": bilateral_centroids["left"],
            "right": bilateral_centroids["right"],
        }
    else:
        centroids = {
            "left": get_femur_centroid(image_slice_hu, "left"),
            "right": get_femur_centroid(image_slice_hu, "right"),
        }

    for muscle_name, config in MUSCLES_CONFIG.items():
        centroid = centroids[config["roi"]]
        if centroid is None:
            continue

        cy, cx = centroid
        angles_rad = np.deg2rad(
            np.linspace(config["angles"][0], config["angles"][1], NUM_RAYS)
        )
        boundary_rows = []
        boundary_cols = []

        for angle in angles_rad:
            y_coords = cy - radii * np.sin(angle)
            x_coords = cx + radii * np.cos(angle)
            ray_signal = map_coordinates(
                image_slice_hu,
                np.vstack((y_coords, x_coords)),
                order=1,
                mode="constant",
                cval=-1000.0,
            )
            peak_idx = find_fascia_1d(ray_signal, mean_spacing)
            peak_idx = int(np.clip(peak_idx, 0, len(radii) - 1))
            boundary_rows.append(y_coords[peak_idx])
            boundary_cols.append(x_coords[peak_idx])

        boundary_rows.append(cy)
        boundary_cols.append(cx)
        rr, cc = polygon(boundary_rows, boundary_cols, shape=(h, w))

        mask = np.zeros((h, w), dtype=bool)
        mask[rr, cc] = True
        slice_compartment_masks[muscle_name] = mask
        # Ventana HU aproximada de tejido muscular "magro" para los trazados previos.
        slice_masks[muscle_name] = mask & (image_slice_hu > -50) & (image_slice_hu < 150)

    return slice_masks, slice_compartment_masks


def segment_all_muscles(
    image_hu: np.ndarray,
    pixel_spacing: List[float],
    z_start: int,
    z_end: int,
):
    print("Ejecutando Ray-Casting simultáneo para 4 músculos...")
    volume_masks = {name: np.zeros_like(image_hu, dtype=bool) for name in MUSCLES_CONFIG}
    compartment_masks = {name: np.zeros_like(image_hu, dtype=bool) for name in MUSCLES_CONFIG}

    for z in range(z_start, z_end):
        slice_result, compartment_result = process_multi_muscle_slice(image_hu[:, :, z], pixel_spacing)
        for name in MUSCLES_CONFIG:
            volume_masks[name][:, :, z] = slice_result[name]
            compartment_masks[name][:, :, z] = compartment_result[name]

        if (z - z_start) % 20 == 0 or z == z_end - 1:
            print(f"  -> Procesado corte {z}/{z_end - 1}")

    return volume_masks, compartment_masks


# ==============================================================================
# FASE 3.5: ANÁLISIS LONGITUDINAL Y LANDMARKS ÓSEOS
# ==============================================================================
def compute_side_bone_profile(
    image_hu: np.ndarray,
    roi_side: str,
    z_start: int,
    z_end: int,
) -> Dict[str, np.ndarray]:
    areas = []
    component_counts = []
    lateral_protrusions = []

    for z in range(z_start, z_end):
        components = get_bone_components(image_hu[:, :, z], roi_side)
        component_counts.append(len(components))

        if not components:
            areas.append(0.0)
            lateral_protrusions.append(0.0)
            continue

        main_component = components[0]
        areas.append(float(main_component["area"]))
        boundary = find_boundaries(main_component["mask"], mode="inner")
        ys, xs = np.where(boundary)
        _, cx = main_component["centroid"]

        if len(xs) == 0:
            lateral_protrusions.append(0.0)
        elif roi_side == "left":
            lateral_protrusions.append(float(cx - xs.min()))
        else:
            lateral_protrusions.append(float(xs.max() - cx))

    areas = np.asarray(areas, dtype=float)
    component_counts = np.asarray(component_counts, dtype=int)
    lateral_protrusions = np.asarray(lateral_protrusions, dtype=float)

    if len(areas) >= 5:
        areas_smooth = gaussian_filter1d(areas, sigma=2.0)
        protrusions_smooth = gaussian_filter1d(lateral_protrusions, sigma=2.0)
    else:
        areas_smooth = areas.copy()
        protrusions_smooth = lateral_protrusions.copy()

    return {
        "areas": areas,
        "areas_smooth": areas_smooth,
        "component_counts": component_counts,
        "lateral_protrusions": lateral_protrusions,
        "lateral_protrusions_smooth": protrusions_smooth,
    }


def infer_proximal_end(profile: Dict[str, np.ndarray]) -> str:
    """Determina si la cadera está hacia índices Z altos o bajos."""
    areas = profile["areas_smooth"]
    if len(areas) == 0:
        return "high_z"

    edge_size = max(3, min(12, len(areas) // 5 or 3))
    low_z_score = float(np.mean(areas[:edge_size]))
    high_z_score = float(np.mean(areas[-edge_size:]))
    return "high_z" if high_z_score >= low_z_score else "low_z"


def choose_end_band(z_start: int, z_end: int, end_name: str, fraction: float) -> List[int]:
    total = max(1, z_end - z_start)
    band_size = max(8, int(round(total * fraction)))

    if end_name == "high_z":
        lower = max(z_start, z_end - band_size)
        return list(range(z_end - 1, lower - 1, -1))

    upper = min(z_end, z_start + band_size)
    return list(range(z_start, upper))


def find_reference_shaft_slice(
    image_hu: np.ndarray,
    roi_side: str,
    z_start: int,
    z_end: int,
):
    """Busca un corte central con un componente óseo dominante compatible con diáfisis."""
    center = (z_start + z_end - 1) // 2
    offsets = [0]
    for delta in range(1, max(center - z_start + 1, z_end - center)):
        offsets.extend([-delta, delta])

    for offset in offsets:
        z = center + offset
        if z < z_start or z >= z_end:
            continue
        components = get_bone_components(image_hu[:, :, z], roi_side)
        if components:
            return int(z), components[0]
    return None, None


def track_bone_component_to_end(
    image_hu: np.ndarray,
    roi_side: str,
    z_start: int,
    z_end: int,
    end_name: str,
):
    """
    Sigue el componente óseo desde la diáfisis hacia un extremo mediante
    continuidad del centroide. Reduce saltos desde el fémur hacia la pelvis.
    """
    reference_z, reference_component = find_reference_shaft_slice(
        image_hu, roi_side, z_start, z_end
    )
    if reference_component is None:
        return {}

    step = 1 if end_name == "high_z" else -1
    stop = z_end if step == 1 else z_start - 1
    current_centroid = np.asarray(reference_component["centroid"], dtype=float)
    tracked = {reference_z: reference_component}

    for z in range(reference_z + step, stop, step):
        components = get_bone_components(image_hu[:, :, z], roi_side)
        if not components:
            continue

        distances = [
            float(np.linalg.norm(np.asarray(c["centroid"], dtype=float) - current_centroid))
            for c in components
        ]
        best_index = int(np.argmin(distances))
        best_distance = distances[best_index]

        # Evita saltar a una estructura lejana; si el hueso reaparece muy lejos,
        # se considera que terminó el trayecto continuo.
        if best_distance > TRACK_MAX_CENTROID_SHIFT_PX:
            continue

        selected = components[best_index]
        tracked[int(z)] = selected
        current_centroid = np.asarray(selected["centroid"], dtype=float)

    return tracked


def select_trochanteric_point(
    image_hu: np.ndarray,
    roi_side: str,
    z_start: int,
    z_end: int,
    profile: Dict[str, np.ndarray],
):
    proximal_end = infer_proximal_end(profile)
    candidate_slices = choose_end_band(
        z_start, z_end, proximal_end, PROXIMAL_SEARCH_FRACTION
    )
    tracked_components = track_bone_component_to_end(
        image_hu, roi_side, z_start, z_end, proximal_end
    )

    best_score = -np.inf
    best_point = None
    best_details = None

    for z in candidate_slices:
        femur_candidate = tracked_components.get(z)
        if femur_candidate is None:
            continue

        boundary = find_boundaries(femur_candidate["mask"], mode="inner")
        ys, xs = np.where(boundary)
        if len(xs) == 0:
            continue

        _, cx = femur_candidate["centroid"]
        if roi_side == "left":
            boundary_idx = int(np.argmin(xs))
            protrusion = float(cx - xs[boundary_idx])
        else:
            boundary_idx = int(np.argmax(xs))
            protrusion = float(xs[boundary_idx] - cx)

        score = protrusion + 0.015 * float(femur_candidate["area"])
        if score > best_score:
            best_score = score
            best_point = (int(ys[boundary_idx]), int(xs[boundary_idx]), int(z))
            best_details = {
                "score": float(score),
                "protrusion_px": protrusion,
                "bone_area_px": int(femur_candidate["area"]),
                "method": "seguimiento_desde_diafisis",
            }

    return best_point, proximal_end, best_details


def _anteromedial_boundary_point(component: Dict, roi_side: str, z: int):
    boundary = find_boundaries(component["mask"], mode="inner")
    ys, xs = np.where(boundary)
    if len(xs) == 0:
        return None

    # En una adquisición axial estándar, menor Y se aproxima a anterior.
    # La dirección medial cambia según la mitad de la matriz.
    anterior_target = float(ys.min())
    medial_target = float(xs.max() if roi_side == "left" else xs.min())

    y_range = max(float(np.ptp(ys)), 1.0)
    x_range = max(float(np.ptp(xs)), 1.0)
    score = ((ys - anterior_target) / y_range) ** 2 + (
        (xs - medial_target) / x_range
    ) ** 2
    idx = int(np.argmin(score))
    return int(ys[idx]), int(xs[idx]), int(z)


def select_tibial_anteromedial_point(
    image_hu: np.ndarray,
    roi_side: str,
    z_start: int,
    z_end: int,
    proximal_end: str,
):
    distal_end = "low_z" if proximal_end == "high_z" else "high_z"
    candidate_slices = choose_end_band(
        z_start, z_end, distal_end, DISTAL_SEARCH_FRACTION
    )
    tracked_components = track_bone_component_to_end(
        image_hu, roi_side, z_start, z_end, distal_end
    )

    fallback = None
    detected = None

    # candidate_slices comienza por el extremo distal y avanza hacia el centro.
    for rank, z in enumerate(candidate_slices):
        components = get_bone_components(image_hu[:, :, z], roi_side)
        tracked_component = tracked_components.get(z)
        if not components or tracked_component is None:
            continue

        point = _anteromedial_boundary_point(tracked_component, roi_side, z)
        if point is None:
            continue

        candidate = {
            "point": point,
            "slice": int(z),
            "component_count": len(components),
            "bone_area_px": int(tracked_component["area"]),
            "search_rank": int(rank),
            "method": "seguimiento_distal_fallback",
        }

        if fallback is None:
            fallback = candidate

        # La coexistencia de dos componentes relevantes apoya la región tibia-fíbula.
        if len(components) >= 2:
            candidate["method"] = "seguimiento_tibia_fibula"
            detected = candidate
            break

    selected = detected or fallback
    if selected is None:
        return None, None
    return selected["point"], selected


def calculate_midpoint(
    p1: Optional[Tuple[int, int, int]],
    p2: Optional[Tuple[int, int, int]],
):
    if p1 is None or p2 is None:
        return None, None
    midpoint_float = tuple((np.asarray(p1, dtype=float) + np.asarray(p2, dtype=float)) / 2.0)
    return midpoint_float, round_point(midpoint_float)


def detect_bony_landmarks(
    image_hu: np.ndarray,
    affine: np.ndarray,
    z_start: int,
    z_end: int,
):
    print("\nDetectando puntos trocantéricos y tibiales anteromediales...")
    results = {}

    for roi_side, side_config in SIDE_CONFIG.items():
        profile = compute_side_bone_profile(image_hu, roi_side, z_start, z_end)
        trochanter, proximal_end, trochanter_details = select_trochanteric_point(
            image_hu, roi_side, z_start, z_end, profile
        )
        tibial, tibial_details = select_tibial_anteromedial_point(
            image_hu, roi_side, z_start, z_end, proximal_end
        )
        midpoint_float, midpoint_voxel = calculate_midpoint(trochanter, tibial)

        trochanter_world = voxel_to_world(affine, trochanter)
        tibial_world = voxel_to_world(affine, tibial)
        midpoint_world = voxel_to_world(affine, midpoint_float)

        distance_mm = None
        if trochanter_world is not None and tibial_world is not None:
            distance_mm = float(
                np.linalg.norm(np.asarray(trochanter_world) - np.asarray(tibial_world))
            )

        warnings = []
        if trochanter is None:
            warnings.append("No se detectó el punto trocantérico.")
        if tibial is None:
            warnings.append("No se detectó el punto tibial anteromedial.")
        if distance_mm is not None and distance_mm < 100.0:
            warnings.append(
                "La distancia entre landmarks es menor de 100 mm; revisar visualmente la detección."
            )
        if midpoint_voxel is not None and not point_inside_volume(midpoint_voxel, image_hu.shape):
            warnings.append("El punto medio quedó por fuera del volumen.")
            midpoint_voxel = None

        results[roi_side] = {
            "patient_side": side_config["patient_side"],
            "muscles": side_config["muscles"],
            "label_values": side_config["label_values"],
            "proximal_end": proximal_end,
            "trochanter_voxel": trochanter,
            "trochanter_world_mm": trochanter_world,
            "trochanter_details": trochanter_details,
            "tibial_voxel": tibial,
            "tibial_world_mm": tibial_world,
            "tibial_details": tibial_details,
            "midpoint_voxel_float": midpoint_float,
            "midpoint_voxel": midpoint_voxel,
            "midpoint_world_mm": midpoint_world,
            "midpoint_slice": midpoint_voxel[2] if midpoint_voxel is not None else None,
            "landmark_distance_mm": distance_mm,
            "warnings": warnings,
        }

        print(f"[{side_config['patient_side']}]")
        print(f"  Punto trocantérico (y,x,z):       {trochanter}")
        print(f"  Punto tibial anteromedial (y,x,z): {tibial}")
        print(
            "  Corte medio:                     "
            + (f"Z={midpoint_voxel[2]}" if midpoint_voxel else "No disponible")
        )
        if midpoint_world is not None:
            print(f"  Posición física del punto medio:  {midpoint_world} mm")
        if distance_mm is not None:
            print(f"  Distancia entre landmarks:        {distance_mm:.2f} mm")
        for warning in warnings:
            print(f"  ADVERTENCIA: {warning}")
        print()

    return results


# ==============================================================================
# FASE 4: ÁREAS TRANSVERSALES EN EL CORTE MEDIO
# ==============================================================================
def compute_area_cm2(mask_slice: np.ndarray, pixel_spacing: List[float]) -> float:
    pixel_area_cm2 = float(pixel_spacing[0] * pixel_spacing[1] / 100.0)
    return float(np.sum(mask_slice) * pixel_area_cm2)


def compute_external_perimeter_mm(
    mask_slice: np.ndarray,
    pixel_spacing: List[float],
) -> float:
    """
    Calcula el perímetro físico del contorno externo principal de una máscara 2D.

    ``find_contours`` entrega coordenadas subpíxel como (fila, columna). La
    distancia de cada segmento se escala con PixelSpacing[0] y PixelSpacing[1]
    para respetar píxeles anisotrópicos. Cuando existen pequeños contornos
    internos o ruido, se conserva únicamente el contorno de mayor longitud, que
    corresponde al límite externo principal del miembro.
    """
    mask = np.asarray(mask_slice, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        return 0.0

    contours = find_contours(mask.astype(np.uint8), level=0.5)
    if not contours:
        return 0.0

    row_spacing_mm = float(pixel_spacing[0])
    col_spacing_mm = float(pixel_spacing[1])
    contour_lengths = []

    for contour in contours:
        if contour.shape[0] < 3:
            continue
        closed = np.vstack([contour, contour[0]])
        delta = np.diff(closed, axis=0)
        segment_lengths_mm = np.sqrt(
            (delta[:, 0] * row_spacing_mm) ** 2
            + (delta[:, 1] * col_spacing_mm) ** 2
        )
        contour_lengths.append(float(np.sum(segment_lengths_mm)))

    return max(contour_lengths) if contour_lengths else 0.0


def compute_midpoint_areas(
    masks_dict: Dict[str, np.ndarray],
    pixel_spacing: List[float],
    landmarks: Dict,
):
    area_report = {}

    for roi_side, info in landmarks.items():
        z_mid = info["midpoint_slice"]
        side_name = info["patient_side"]
        if z_mid is None:
            area_report[roi_side] = None
            continue

        muscle_areas = {}
        total_area = 0.0
        for muscle_name in info["muscles"]:
            area_cm2 = compute_area_cm2(masks_dict[muscle_name][:, :, z_mid], pixel_spacing)
            muscle_areas[muscle_name] = area_cm2
            total_area += area_cm2

        area_report[roi_side] = {
            "patient_side": side_name,
            "slice_z": int(z_mid),
            "muscle_areas_cm2": muscle_areas,
            "total_vasti_area_cm2": float(total_area),
        }

    return area_report


# ==============================================================================
# FASE 4.5: COMPOSICIÓN CORPORAL EN EL CORTE MEDIO MEDIANTE WAVELETS
# ==============================================================================
def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Elimina componentes 2D pequeños sin depender de funciones obsoletas."""
    labels, num = scipy_label(mask)
    cleaned = np.zeros_like(mask, dtype=bool)
    for component_id in range(1, num + 1):
        component = labels == component_id
        if int(np.sum(component)) >= min_area_px:
            cleaned |= component
    return cleaned


def clean_binary_volume_slicewise(mask_volume: np.ndarray, min_area_px: int) -> np.ndarray:
    cleaned = np.zeros_like(mask_volume, dtype=bool)
    for z in range(mask_volume.shape[2]):
        cleaned[:, :, z] = remove_small_components(mask_volume[:, :, z], min_area_px)
    return cleaned


def build_union_mask(mask_dict: Dict[str, np.ndarray], muscle_names: List[str]) -> np.ndarray:
    union = np.zeros_like(next(iter(mask_dict.values())), dtype=bool)
    for name in muscle_names:
        union |= mask_dict[name]
    return union


def _get_full_body_mask_for_limb(image_slice_hu: np.ndarray) -> np.ndarray:
    """Máscara corporal 2D sin dividir por mitad izquierda/derecha de la imagen.

    Corrección v3.21.4:
    antes se aplicaba `get_side_roi_mask`, lo que podía partir un muslo cuando el
    miembro cruzaba la línea media de la matriz o cuando el otro miembro estaba
    parcialmente fuera del campo de visión. Ahora se detecta el cuerpo en todo el
    corte y luego se selecciona el componente anatómico asociado al fémur/landmark.
    """
    body = image_slice_hu > TISSUE_BODY_HU_THRESHOLD
    body = binary_closing(body, iterations=2)
    body = binary_fill_holes(body)
    body = binary_opening(body, iterations=1)
    return body.astype(bool)


def _bone_components_full_field(
    image_slice_hu: np.ndarray,
    hu_threshold: float = BONE_HU_THRESHOLD,
    min_area_px: int = MIN_BONE_COMPONENT_AREA_PX,
) -> List[Dict]:
    """Componentes óseos del corte completo, sin hemicampo."""
    bone_mask = image_slice_hu > hu_threshold
    bone_mask = binary_opening(bone_mask, iterations=1)
    labels, num = scipy_label(bone_mask)
    components = []
    for component_id in range(1, num + 1):
        component_mask = labels == component_id
        area = int(np.sum(component_mask))
        if area < min_area_px:
            continue
        ys, xs = np.where(component_mask)
        cy, cx = center_of_mass(component_mask)
        components.append(
            {
                "mask": component_mask,
                "area": area,
                "centroid": (float(cy), float(cx)),
                "bbox": (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())),
            }
        )
    components.sort(key=lambda component: component["area"], reverse=True)
    return components


def get_femur_centroid_near_point(
    image_slice_hu: np.ndarray,
    target_yx: Tuple[float, float],
    roi_side: Optional[str] = None,
):
    """Busca el fémur más cercano al punto anatómico esperado.

    No usa una división rígida de la imagen por mitades. El parámetro roi_side se
    conserva por compatibilidad, pero la decisión se toma por cercanía al landmark
    del miembro. Esto evita que el fémur quede artificialmente partido por la línea
    media de la matriz.
    """
    bilateral_centroids = get_bilateral_femur_centroids_for_slice(image_slice_hu)
    if bilateral_centroids is not None and roi_side in {"left", "right"}:
        return bilateral_centroids[roi_side]

    components = _bone_components_full_field(image_slice_hu)
    if not components:
        return None

    ty, tx = float(target_yx[0]), float(target_yx[1])
    scored = []
    for component in components:
        cy, cx = component["centroid"]
        dist = float(np.hypot(cy - ty, cx - tx))
        # Se prioriza cercanía al landmark; el área solo desempata.
        score = dist - 0.003 * float(component["area"])
        scored.append((score, dist, -float(component["area"]), component))
    scored.sort(key=lambda item: item[:3])
    return scored[0][3]["centroid"]


def _select_component_containing_or_nearest(
    labels: np.ndarray,
    num: int,
    center_yx: Tuple[float, float],
) -> np.ndarray:
    """Selecciona el componente corporal que contiene o está más cerca del centro."""
    if num <= 0:
        return np.zeros_like(labels, dtype=bool)

    cy = int(np.clip(round(center_yx[0]), 0, labels.shape[0] - 1))
    cx = int(np.clip(round(center_yx[1]), 0, labels.shape[1] - 1))
    center_label = int(labels[cy, cx])
    if center_label > 0:
        return labels == center_label

    best = None
    for component_id in range(1, num + 1):
        component = labels == component_id
        area = int(np.sum(component))
        if area < MIN_TISSUE_COMPONENT_AREA_PX:
            continue
        yy, xx = np.where(component)
        cmy, cmx = center_of_mass(component)
        dist_centroid = float(np.hypot(float(cmy) - center_yx[0], float(cmx) - center_yx[1]))
        # Si el centro está dentro de la caja del componente, se reduce el castigo.
        inside_bbox = yy.min() <= cy <= yy.max() and xx.min() <= cx <= xx.max()
        score = dist_centroid - (50.0 if inside_bbox else 0.0) - 0.001 * area
        item = (score, -area, component_id)
        if best is None or item < best:
            best = item

    if best is None:
        sizes = [int(np.sum(labels == idx)) for idx in range(1, num + 1)]
        return labels == (int(np.argmax(sizes)) + 1)
    return labels == int(best[2])


def _keep_component_around_center(mask: np.ndarray, center_yx: Tuple[float, float]) -> np.ndarray:
    labels, num = scipy_label(mask)
    if num == 0:
        return np.zeros_like(mask, dtype=bool)
    return _select_component_containing_or_nearest(labels, num, center_yx)


def _split_touching_limbs_by_femur_voronoi(
    component_mask: np.ndarray,
    image_slice_hu: np.ndarray,
    target_center_yx: Tuple[float, float],
) -> np.ndarray:
    """Separa muslos que se tocan usando una partición por cercanía al fémur.

    Si el componente corporal incluye los dos miembros por contacto de piel, se
    detectan los centros óseos dentro del componente y se conserva la región más
    cercana al fémur objetivo. Esto es distinto a partir por la mitad de la imagen:
    la frontera se calcula entre centros anatómicos.
    """
    bone_components = _bone_components_full_field(image_slice_hu)
    centers = []
    for comp in bone_components:
        cy, cx = comp["centroid"]
        iy = int(np.clip(round(cy), 0, component_mask.shape[0] - 1))
        ix = int(np.clip(round(cx), 0, component_mask.shape[1] - 1))
        if component_mask[iy, ix]:
            centers.append((float(cy), float(cx), int(comp["area"])))

    if len(centers) <= 1:
        return component_mask

    ty, tx = float(target_center_yx[0]), float(target_center_yx[1])
    distances_to_target = [np.hypot(cy - ty, cx - tx) for cy, cx, _ in centers]
    target_idx = int(np.argmin(distances_to_target))
    target_center = centers[target_idx]
    other_centers = [c for idx, c in enumerate(centers) if idx != target_idx]
    if not other_centers:
        return component_mask

    yy, xx = np.indices(component_mask.shape, dtype=float)
    target_dist2 = (yy - target_center[0]) ** 2 + (xx - target_center[1]) ** 2
    other_dist2 = np.full(component_mask.shape, np.inf, dtype=float)
    for cy, cx, _area in other_centers:
        other_dist2 = np.minimum(other_dist2, (yy - cy) ** 2 + (xx - cx) ** 2)

    split_mask = component_mask & (target_dist2 <= other_dist2)
    split_mask = binary_closing(split_mask, iterations=1)
    split_mask = binary_fill_holes(split_mask)
    split_mask = binary_opening(split_mask, iterations=1)
    split_mask = _keep_component_around_center(split_mask, (target_center[0], target_center[1]))

    # Control de seguridad: si la partición deja un fragmento demasiado pequeño,
    # se mantiene el componente original para no perder el miembro.
    if np.count_nonzero(split_mask) < 0.20 * max(1, np.count_nonzero(component_mask)):
        return component_mask
    return split_mask.astype(bool)


def get_limb_component_mask(
    image_slice_hu: np.ndarray,
    roi_side: str,
    center_yx: Tuple[float, float],
) -> np.ndarray:
    """Obtiene el componente corporal completo del miembro sin cortarlo por la mitad.

    Versión corregida:
    - No aplica hemicampo rígido izquierda/derecha.
    - Selecciona el componente conectado asociado al fémur o landmark del miembro.
    - Si ambos muslos están unidos por contacto, separa con frontera anatómica por
      cercanía a centros femorales, no por la mitad de la imagen.
    """
    body = _get_full_body_mask_for_limb(image_slice_hu)
    labels, num = scipy_label(body)
    if num == 0:
        return np.zeros_like(body, dtype=bool)

    target_center = get_femur_centroid_near_point(image_slice_hu, center_yx, roi_side)
    if target_center is None:
        target_center = center_yx

    component = _select_component_containing_or_nearest(labels, num, target_center)
    component = _split_touching_limbs_by_femur_voronoi(component, image_slice_hu, target_center)
    component = binary_closing(component, iterations=2)
    component = binary_fill_holes(component)
    component = binary_opening(component, iterations=1)
    component = _keep_component_around_center(component, target_center)
    return component.astype(bool)


def ray_radius_to_roi_border(
    center_yx: Tuple[float, float],
    angle: float,
    shape: Tuple[int, int],
    roi_side: str,
) -> float:
    """Distancia máxima de un rayo hasta el borde de la imagen completa.

    Antes el rayo se detenía en el borde del hemicampo, lo que podía generar una
    línea vertical y cortar el miembro. Ahora se permite que el rayo recorra el
    campo completo; la pertenencia al miembro la controla `limb_mask`.
    """
    cy, cx = center_yx
    h, w = shape
    dy = -float(np.sin(angle))
    dx = float(np.cos(angle))

    x_min = 0.0
    x_max = float(w - 1)
    y_min = 0.0
    y_max = float(h - 1)

    candidates = []
    if dy > 1e-9:
        candidates.append((y_max - cy) / dy)
    elif dy < -1e-9:
        candidates.append((y_min - cy) / dy)

    if dx > 1e-9:
        candidates.append((x_max - cx) / dx)
    elif dx < -1e-9:
        candidates.append((x_min - cx) / dx)

    positive = [value for value in candidates if value > 0]
    return max(1.0, min(positive)) if positive else 1.0


def compute_tissue_wavelet_power(signal_1d: np.ndarray) -> np.ndarray:
    """Potencia máxima CWT para localizar interfaces tisulares en un rayo."""
    signal_1d = np.asarray(signal_1d, dtype=float)
    if signal_1d.size < 6:
        return np.zeros_like(signal_1d, dtype=float)

    try:
        cwt_result = cwt_1d(
            signal=signal_1d,
            wavelet="mexican_hat",
            n_scales=TISSUE_CWT_N_SCALES,
            min_scale=TISSUE_CWT_MIN_SCALE,
            max_scale=TISSUE_CWT_MAX_SCALE,
        )
        power = np.max(np.asarray(cwt_result.power, dtype=float), axis=0)
        return gaussian_filter1d(power, sigma=1.0)
    except Exception:
        # Fallback basado en gradiente para que un rayo defectuoso no detenga el estudio.
        smooth = gaussian_filter1d(signal_1d, sigma=1.0)
        return gaussian_filter1d(np.abs(np.gradient(smooth)), sigma=1.0)


def first_sustained_false(mask_1d: np.ndarray, start: int, run_length: int = 3):
    for idx in range(max(0, start), max(0, len(mask_1d) - run_length + 1)):
        if not np.any(mask_1d[idx : idx + run_length]):
            return idx
    return None


def detect_tissue_interfaces_on_ray(
    signal_hu: np.ndarray,
    body_signal: np.ndarray,
    mm_per_pixel: float,
):
    """
    Estima tres radios sobre un rayo:
    1. superficie epidérmica externa,
    2. límite interno dérmico,
    3. fascia profunda que separa grasa subcutánea del compartimento muscular.
    """
    signal_hu = np.asarray(signal_hu, dtype=float)
    body_signal = np.asarray(body_signal, dtype=bool)
    n = signal_hu.size
    if n < 12:
        return None

    smooth = gaussian_filter1d(signal_hu, sigma=1.0)
    power = compute_tissue_wavelet_power(smooth)
    blind_idx = int(round(TISSUE_RAY_BLIND_ZONE_MM / mm_per_pixel))
    blind_idx = int(np.clip(blind_idx, 2, max(2, n - 8)))

    # Primer tramo de aire sostenido al salir del miembro.
    first_air = first_sustained_false(body_signal, blind_idx, run_length=3)
    if first_air is None:
        valid_body = np.where(body_signal)[0]
        if valid_body.size == 0:
            return None
        outer_guess = int(valid_body[-1])
    else:
        outer_guess = max(blind_idx + 2, int(first_air - 1))

    refine_px = max(1, int(round(OUTER_BOUNDARY_REFINEMENT_MM / mm_per_pixel)))
    outer_lo = max(blind_idx + 1, outer_guess - refine_px)
    outer_hi = min(n - 2, outer_guess + refine_px)
    if outer_hi <= outer_lo:
        outer_idx = outer_guess
    else:
        outer_idx = int(outer_lo + np.argmax(power[outer_lo : outer_hi + 1]))

    # Límite interno dérmico: segunda interfaz dentro de un espesor plausible.
    skin_min_px = max(1, int(round(EPIDERMIS_DERMIS_MIN_DEPTH_MM / mm_per_pixel)))
    skin_max_px = max(
        skin_min_px + 1,
        int(round(EPIDERMIS_DERMIS_MAX_DEPTH_MM / mm_per_pixel)),
    )
    dermis_lo = max(blind_idx + 2, outer_idx - skin_max_px)
    dermis_hi = max(dermis_lo, outer_idx - skin_min_px)
    if dermis_hi > dermis_lo:
        dermis_idx = int(dermis_lo + np.argmax(power[dermis_lo : dermis_hi + 1]))
    else:
        fallback_depth = max(
            1, int(round(EPIDERMIS_DERMIS_FALLBACK_DEPTH_MM / mm_per_pixel))
        )
        dermis_idx = int(max(blind_idx + 2, outer_idx - fallback_depth))

    min_subcut_px = max(2, int(round(MIN_SUBCUTANEOUS_LAYER_MM / mm_per_pixel)))
    search_end = int(dermis_idx - min_subcut_px)
    if search_end <= blind_idx + 3:
        return {
            "epidermis": float(outer_idx),
            "dermis": float(dermis_idx),
            "fascia": None,
            "fascia_score": 0.0,
        }

    pre_window = max(2, int(round(FASCIA_PRE_WINDOW_MM / mm_per_pixel)))
    post_window = max(2, int(round(FASCIA_POST_WINDOW_MM / mm_per_pixel)))
    power_max = max(float(np.max(power)), 1e-8)

    best_idx = None
    best_score = -np.inf
    for idx in range(blind_idx + pre_window, search_end):
        inner = smooth[max(blind_idx, idx - pre_window) : idx]
        outer_local = smooth[idx : min(dermis_idx, idx + post_window)]
        outer_full = smooth[idx:dermis_idx]
        if inner.size < 2 or outer_local.size < 2 or outer_full.size < 2:
            continue

        muscle_fraction = float(
            np.mean((inner >= TOTAL_MUSCLE_HU_MIN) & (inner <= TOTAL_MUSCLE_HU_MAX))
        )
        local_fat_fraction = float(
            np.mean(
                (outer_local >= SUBCUTANEOUS_FAT_HU_MIN)
                & (outer_local <= SUBCUTANEOUS_FAT_HU_MAX)
            )
        )
        outer_fat_fraction = float(
            np.mean(
                (outer_full >= SUBCUTANEOUS_FAT_HU_MIN)
                & (outer_full <= SUBCUTANEOUS_FAT_HU_MAX)
            )
        )
        contrast = max(0.0, float(np.mean(inner) - np.mean(outer_local))) / 150.0
        wavelet_score = float(power[idx] / power_max)

        if muscle_fraction < 0.30 or outer_fat_fraction < 0.25:
            continue

        score = (
            2.2 * muscle_fraction
            + 1.7 * local_fat_fraction
            + 2.5 * outer_fat_fraction
            + 0.8 * min(contrast, 2.0)
            + 1.2 * wavelet_score
        )
        if score > best_score:
            best_score = score
            best_idx = int(idx)

    # Fallback: último píxel de rango muscular antes de una capa grasa exterior.
    if best_idx is None:
        for idx in range(search_end - 1, blind_idx + 2, -1):
            inner = smooth[max(blind_idx, idx - pre_window) : idx]
            outer_full = smooth[idx:dermis_idx]
            if inner.size < 2 or outer_full.size < 2:
                continue
            muscle_fraction = float(
                np.mean((inner >= TOTAL_MUSCLE_HU_MIN) & (inner <= TOTAL_MUSCLE_HU_MAX))
            )
            fat_fraction = float(
                np.mean(
                    (outer_full >= SUBCUTANEOUS_FAT_HU_MIN)
                    & (outer_full <= SUBCUTANEOUS_FAT_HU_MAX)
                )
            )
            if muscle_fraction >= 0.35 and fat_fraction >= 0.20:
                best_idx = int(idx)
                best_score = 0.5
                break

    return {
        "epidermis": float(outer_idx),
        "dermis": float(dermis_idx),
        "fascia": float(best_idx) if best_idx is not None else None,
        "fascia_score": float(max(best_score, 0.0)),
    }


def circular_interpolate_and_smooth(
    values: np.ndarray,
    fallback: Optional[np.ndarray] = None,
    sigma: float = TISSUE_BOUNDARY_SMOOTH_SIGMA,
):
    values = np.asarray(values, dtype=float)
    n = len(values)
    valid = np.isfinite(values)
    if np.sum(valid) < 3:
        if fallback is None:
            return None
        filled = np.asarray(fallback, dtype=float).copy()
    else:
        x = np.arange(n, dtype=float)
        xv = x[valid]
        yv = values[valid]
        xv_ext = np.concatenate([xv - n, xv, xv + n])
        yv_ext = np.concatenate([yv, yv, yv])
        filled = np.interp(x, xv_ext, yv_ext)

    return gaussian_filter1d(filled, sigma=sigma, mode="wrap")


def radii_to_polygon_mask(
    shape: Tuple[int, int],
    center_yx: Tuple[float, float],
    angles: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    cy, cx = center_yx
    rows = cy - radii * np.sin(angles)
    cols = cx + radii * np.cos(angles)
    rr, cc = polygon(rows, cols, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def analyze_limb_midpoint_slice(
    image_slice_hu: np.ndarray,
    pixel_spacing: List[float],
    roi_side: str,
    midpoint_yx: Tuple[float, float],
):
    """Analiza composición corporal de un miembro en un único corte axial."""
    mean_spacing = float((pixel_spacing[0] + pixel_spacing[1]) / 2.0)
    # Corrección v3.21.4: el centro se toma del fémur más cercano al landmark,
    # no del fémur recortado por hemicampo. Esto evita cortar el miembro cuando
    # cruza la mitad de la imagen.
    femur_centroid = get_femur_centroid_near_point(image_slice_hu, midpoint_yx, roi_side)
    center = femur_centroid if femur_centroid is not None else midpoint_yx
    limb_mask = get_limb_component_mask(image_slice_hu, roi_side, center)
    if not np.any(limb_mask):
        return None

    angles = np.linspace(0.0, 2.0 * np.pi, TISSUE_NUM_RAYS, endpoint=False)
    outer_radii = np.full(TISSUE_NUM_RAYS, np.nan, dtype=float)
    dermis_radii = np.full(TISSUE_NUM_RAYS, np.nan, dtype=float)
    fascia_radii = np.full(TISSUE_NUM_RAYS, np.nan, dtype=float)
    max_radii = np.zeros(TISSUE_NUM_RAYS, dtype=float)

    cy, cx = center
    for ray_index, angle in enumerate(angles):
        max_radius = ray_radius_to_roi_border(center, angle, image_slice_hu.shape, roi_side)
        max_radius = max(4.0, max_radius - 1.0)
        max_radii[ray_index] = max_radius
        radii = np.arange(0.0, max_radius + 0.5, 1.0, dtype=float)
        rows = cy - radii * np.sin(angle)
        cols = cx + radii * np.cos(angle)

        signal = map_coordinates(
            image_slice_hu,
            np.vstack((rows, cols)),
            order=1,
            mode="constant",
            cval=-1000.0,
        )
        body_signal = map_coordinates(
            limb_mask.astype(np.uint8),
            np.vstack((rows, cols)),
            order=0,
            mode="constant",
            cval=0,
        ).astype(bool)

        interfaces = detect_tissue_interfaces_on_ray(signal, body_signal, mean_spacing)
        if interfaces is None:
            continue
        outer_radii[ray_index] = interfaces["epidermis"]
        dermis_radii[ray_index] = interfaces["dermis"]
        if interfaces["fascia"] is not None:
            fascia_radii[ray_index] = interfaces["fascia"]

    valid_outer = np.isfinite(outer_radii)
    valid_dermis = np.isfinite(dermis_radii)
    valid_fascia = np.isfinite(fascia_radii)
    if np.sum(valid_outer) < max(10, int(0.20 * TISSUE_NUM_RAYS)):
        return None

    outer = circular_interpolate_and_smooth(outer_radii, fallback=max_radii * 0.85)
    dermis = circular_interpolate_and_smooth(
        dermis_radii,
        fallback=outer - EPIDERMIS_DERMIS_FALLBACK_DEPTH_MM / mean_spacing,
    )

    # Si pocos rayos detectan fascia, se usa como apoyo el último tejido muscular
    # sobre cada radio y luego se interpola circularmente.
    if np.sum(valid_fascia) < 3:
        fascia_fallback = dermis * 0.72
    else:
        fascia_fallback = dermis * 0.72
    fascia = circular_interpolate_and_smooth(fascia_radii, fallback=fascia_fallback)

    skin_min_px = max(1.0, EPIDERMIS_DERMIS_MIN_DEPTH_MM / mean_spacing)
    min_subcut_px = max(2.0, MIN_SUBCUTANEOUS_LAYER_MM / mean_spacing)
    outer = np.minimum(outer, max_radii)
    dermis = np.minimum(dermis, outer - skin_min_px)
    dermis = np.maximum(dermis, 3.0)
    fascia = np.minimum(fascia, dermis - min_subcut_px)
    fascia = np.maximum(fascia, TISSUE_RAY_BLIND_ZONE_MM / mean_spacing)

    outer_mask = radii_to_polygon_mask(image_slice_hu.shape, center, angles, outer)
    dermis_inner_mask = radii_to_polygon_mask(image_slice_hu.shape, center, angles, dermis)
    fascia_mask = radii_to_polygon_mask(image_slice_hu.shape, center, angles, fascia)

    # Limita las máscaras al componente anatómico identificado y asegura anidamiento.
    outer_mask &= limb_mask
    dermis_inner_mask &= outer_mask
    fascia_mask &= dermis_inner_mask

    epidermis_contour = find_boundaries(outer_mask, mode="inner")
    dermis_band = outer_mask & ~dermis_inner_mask
    subcutaneous_compartment = dermis_inner_mask & ~fascia_mask

    fat_hu_mask = (
        (image_slice_hu >= SUBCUTANEOUS_FAT_HU_MIN)
        & (image_slice_hu <= SUBCUTANEOUS_FAT_HU_MAX)
    )
    muscle_hu_mask = (
        (image_slice_hu >= TOTAL_MUSCLE_HU_MIN)
        & (image_slice_hu <= TOTAL_MUSCLE_HU_MAX)
    )

    subcutaneous_fat = subcutaneous_compartment & fat_hu_mask
    subcutaneous_fat = binary_opening(subcutaneous_fat, iterations=1)
    subcutaneous_fat = remove_small_components(
        subcutaneous_fat, MIN_TISSUE_COMPONENT_AREA_PX
    )

    total_muscle = fascia_mask & muscle_hu_mask
    total_muscle = binary_opening(total_muscle, iterations=1)
    total_muscle = binary_closing(total_muscle, iterations=1)
    total_muscle = remove_small_components(total_muscle, MIN_TISSUE_COMPONENT_AREA_PX)

    limb_perimeter_mm = compute_external_perimeter_mm(outer_mask, pixel_spacing)
    limb_perimeter_cm = limb_perimeter_mm / 10.0

    fat_area_cm2 = compute_area_cm2(subcutaneous_fat, pixel_spacing)
    muscle_area_cm2 = compute_area_cm2(total_muscle, pixel_spacing)
    dermis_area_cm2 = compute_area_cm2(dermis_band, pixel_spacing)
    ratio_fat_to_muscle = (
        float(fat_area_cm2 / muscle_area_cm2) if muscle_area_cm2 > 0 else None
    )
    ratio_muscle_to_fat = (
        float(muscle_area_cm2 / fat_area_cm2) if fat_area_cm2 > 0 else None
    )
    combined_area = fat_area_cm2 + muscle_area_cm2
    fat_fraction_percent = (
        float(100.0 * fat_area_cm2 / combined_area) if combined_area > 0 else None
    )

    valid_fascia_fraction = float(np.mean(valid_fascia))
    warnings = []
    if valid_fascia_fraction < MIN_VALID_FASCIA_RAY_FRACTION:
        warnings.append(
            "Baja proporción de rayos con detección directa de fascia; parte del contorno fue interpolada."
        )
    if muscle_area_cm2 <= 0:
        warnings.append("No se obtuvo área muscular total válida.")
    if fat_area_cm2 <= 0:
        warnings.append("No se obtuvo área de grasa subcutánea válida.")
    if (
        np.any(outer_mask[0, :])
        or np.any(outer_mask[-1, :])
        or np.any(outer_mask[:, 0])
        or np.any(outer_mask[:, -1])
    ):
        warnings.append(
            "El contorno del miembro toca el borde del campo de visión; revise si el TAC truncó parte del miembro."
        )
    warnings.append(
        "Epidermis y dermis corresponden a límites radiológicos estimados; el TAC no resuelve siempre ambas capas histológicas."
    )

    skin_thickness_mm = (outer - dermis) * mean_spacing
    subcutaneous_thickness_mm = (dermis - fascia) * mean_spacing

    return {
        "center_yx": (float(center[0]), float(center[1])),
        "outer_mask": outer_mask,
        "epidermis_contour": epidermis_contour,
        "dermis_band": dermis_band,
        "dermis_inner_mask": dermis_inner_mask,
        "fascia_mask": fascia_mask,
        "subcutaneous_fat_mask": subcutaneous_fat,
        "total_muscle_mask": total_muscle,
        "fat_area_cm2": float(fat_area_cm2),
        "total_muscle_area_cm2": float(muscle_area_cm2),
        "dermis_band_area_cm2": float(dermis_area_cm2),
        "limb_perimeter_mm": float(limb_perimeter_mm),
        "limb_perimeter_cm": float(limb_perimeter_cm),
        "fat_to_muscle_ratio": ratio_fat_to_muscle,
        "muscle_to_fat_ratio": ratio_muscle_to_fat,
        "fat_fraction_percent": fat_fraction_percent,
        "mean_estimated_skin_thickness_mm": float(np.mean(skin_thickness_mm)),
        "mean_subcutaneous_compartment_thickness_mm": float(
            np.mean(subcutaneous_thickness_mm)
        ),
        "valid_outer_ray_fraction": float(np.mean(valid_outer)),
        "valid_dermis_ray_fraction": float(np.mean(valid_dermis)),
        "valid_fascia_ray_fraction": valid_fascia_fraction,
        "warnings": warnings,
    }


def analyze_midpoint_tissue_composition(
    image_hu: np.ndarray,
    pixel_spacing: List[float],
    landmarks: Dict,
):
    print("\nAnalizando piel, grasa subcutánea y músculo total en los cortes medios...")
    results = {}

    for roi_side, info in landmarks.items():
        z_mid = info.get("midpoint_slice")
        midpoint = info.get("midpoint_voxel_float")
        side_name = info["patient_side"]
        if z_mid is None or midpoint is None:
            results[roi_side] = None
            print(f"[{side_name}] No hay corte medio disponible para composición corporal.")
            continue

        # Control final v3.21.6: si el corte medio trae un solo fémur, buscar
        # automáticamente el corte bilateral más cercano antes de analizar.
        z_mid_int = int(z_mid)
        if not slice_has_two_femurs(image_hu[:, :, z_mid_int]):
            local_qc = scan_volume_two_femur_qc(image_hu)
            replacement_z = _nearest_two_femur_slice(local_qc, z_mid_int)
            if replacement_z is not None:
                print(
                    f"QC RESCATE FINAL: [{side_name}] el corte Z={z_mid_int} no tenía dos fémures; "
                    f"se reemplaza por Z={replacement_z}."
                )
                z_mid_int = int(replacement_z)
            else:
                results[roi_side] = None
                print(f"[{side_name}] Omitido: no se encontró ningún corte con dos fémures para composición corporal.")
                continue

        analysis = analyze_limb_midpoint_slice(
            image_slice_hu=image_hu[:, :, z_mid_int],
            pixel_spacing=pixel_spacing,
            roi_side=roi_side,
            midpoint_yx=(float(midpoint[0]), float(midpoint[1])),
        )
        if analysis is None:
            results[roi_side] = None
            print(f"[{side_name}] No fue posible delimitar el miembro en Z={z_mid}.")
            continue

        analysis.update(
            {
                "patient_side": side_name,
                "slice_z": int(z_mid_int),
                "label_values": TISSUE_LABELS[roi_side],
            }
        )
        results[roi_side] = analysis

        ratio_text = (
            f"{analysis['fat_to_muscle_ratio']:.4f}"
            if analysis["fat_to_muscle_ratio"] is not None
            else "No disponible"
        )
        fat_percent_text = (
            f"{analysis['fat_fraction_percent']:.2f}%"
            if analysis["fat_fraction_percent"] is not None
            else "No disponible"
        )
        print(f"[{side_name}] Z={z_mid_int}")
        print(f"  Perímetro del miembro: {analysis['limb_perimeter_cm']:.2f} cm")
        print(f"  Grasa subcutánea:      {analysis['fat_area_cm2']:.2f} cm²")
        print(f"  Músculo total:         {analysis['total_muscle_area_cm2']:.2f} cm²")
        print(f"  Relación grasa/músculo:{ratio_text}")
        print(f"  Fracción grasa:        {fat_percent_text}")
        print(
            f"  Rayos con fascia válida: {100.0 * analysis['valid_fascia_ray_fraction']:.1f}%"
        )

    return results


def build_tissue_labelmap(
    image_shape: Tuple[int, int, int],
    tissue_composition: Dict,
) -> np.ndarray:
    labelmap = np.zeros(image_shape, dtype=np.uint8)

    for roi_side, analysis in tissue_composition.items():
        if analysis is None:
            continue
        z = analysis["slice_z"]
        labels = analysis["label_values"]
        target = labelmap[:, :, z]

        target[analysis["total_muscle_mask"]] = labels["total_muscle"]
        target[analysis["subcutaneous_fat_mask"]] = labels["subcutaneous_fat"]
        target[analysis["dermis_band"]] = labels["dermis"]
        target[analysis["epidermis_contour"]] = labels["epidermis"]

    return labelmap


def build_tissue_overlay(image_hu: np.ndarray, tissue_composition: Dict) -> np.ndarray:
    overlay = image_hu.astype(np.float32).copy()

    for _, analysis in tissue_composition.items():
        if analysis is None:
            continue
        z = analysis["slice_z"]
        target = overlay[:, :, z]
        target[analysis["total_muscle_mask"]] = 900.0
        target[analysis["subcutaneous_fat_mask"]] = 1400.0
        target[analysis["dermis_band"]] = 2100.0
        target[analysis["epidermis_contour"]] = 3000.0

    return overlay


def window_soft_tissue_image(image_slice_hu: np.ndarray, center: float = 40.0, width: float = 400.0) -> np.ndarray:
    """Convierte un corte HU a imagen 0-1 con ventana de tejidos blandos."""
    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(image_slice_hu.astype(float), low, high)
    return (clipped - low) / max(high - low, 1e-6)


def build_tissue_png_legend_handles():
    return [
        Line2D([0], [0], color="#00FFFF", lw=2.5, label="Piel / contorno epidérmico"),
        Patch(facecolor="#1F77B4", edgecolor="#1F77B4", alpha=0.45, label="Dermis estimada"),
        Patch(facecolor="#FFD700", edgecolor="#FFD700", alpha=0.45, label="Grasa subcutánea"),
        Patch(facecolor="#DC143C", edgecolor="#DC143C", alpha=0.40, label="Músculo total"),
    ]


def _apply_mask_overlay(rgba: np.ndarray, mask: np.ndarray, color_rgb, alpha: float):
    if mask is None or not np.any(mask):
        return
    color = np.asarray(color_rgb, dtype=float)
    rgba[mask, :3] = (1.0 - alpha) * rgba[mask, :3] + alpha * color
    rgba[mask, 3] = 1.0


def make_tissue_overlay_rgba(
    image_slice_hu: np.ndarray,
    analysis: Dict,
) -> np.ndarray:
    """Genera una imagen RGBA con las capas anatómicas coloreadas."""
    gray = window_soft_tissue_image(image_slice_hu)
    rgba = np.dstack([gray, gray, gray, np.ones_like(gray)])

    _apply_mask_overlay(rgba, analysis.get("total_muscle_mask"), (220/255.0, 20/255.0, 60/255.0), 0.40)
    _apply_mask_overlay(rgba, analysis.get("subcutaneous_fat_mask"), (1.0, 215/255.0, 0.0), 0.45)
    _apply_mask_overlay(rgba, analysis.get("dermis_band"), (31/255.0, 119/255.0, 180/255.0), 0.45)

    epidermis = analysis.get("epidermis_contour")
    if epidermis is not None and np.any(epidermis):
        rgba[epidermis, 0] = 0.0
        rgba[epidermis, 1] = 1.0
        rgba[epidermis, 2] = 1.0
        rgba[epidermis, 3] = 1.0

    return rgba


def export_tissue_pngs(
    image_hu: np.ndarray,
    tissue_composition: Dict,
    landmarks: Dict,
    out_dir: str,
):
    """
    Guarda PNG individuales y un resumen combinado.

    La función crea archivos incluso cuando el análisis de un miembro no está
    disponible. En ese caso guarda el corte medio en escala de grises con una
    advertencia diagnóstica, evitando que el fallo pase silenciosamente.
    """
    png_dir = os.path.join(out_dir, PNG_OUTPUT_SUBDIR)
    os.makedirs(png_dir, exist_ok=True)
    print(f"\nGenerando imágenes PNG en: {png_dir}")

    output_paths = {}
    legend_handles = build_tissue_png_legend_handles()
    items = []

    for roi_side in ("left", "right"):
        analysis = tissue_composition.get(roi_side)
        landmark_info = landmarks.get(roi_side, {})
        side_name = (
            analysis.get("patient_side")
            if analysis is not None
            else landmark_info.get("patient_side", roi_side)
        )

        if analysis is not None:
            z = int(analysis["slice_z"])
        else:
            z_value = landmark_info.get("midpoint_slice")
            z = int(z_value) if z_value is not None else int(image_hu.shape[2] // 2)

        z = int(np.clip(z, 0, image_hu.shape[2] - 1))
        items.append((roi_side, side_name, z, analysis))

        fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
        if analysis is not None:
            rgba = make_tissue_overlay_rgba(image_hu[:, :, z], analysis)
            ax.imshow(rgba, origin="upper")

            ratio_fm = analysis.get("fat_to_muscle_ratio")
            ratio_text = f"{ratio_fm:.4f}" if ratio_fm is not None else "No disponible"
            fat_percent = analysis.get("fat_fraction_percent")
            fat_percent_text = (
                f"{fat_percent:.2f}%" if fat_percent is not None else "No disponible"
            )
            text_lines = [
                f"Perímetro: {analysis['limb_perimeter_cm']:.2f} cm",
                f"Grasa subcutánea: {analysis['fat_area_cm2']:.2f} cm²",
                f"Músculo total: {analysis['total_muscle_area_cm2']:.2f} cm²",
                f"Relación grasa/músculo: {ratio_text}",
                f"Porcentaje adiposo: {fat_percent_text}",
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                framealpha=0.90,
                fontsize=8,
            )
        else:
            gray = window_soft_tissue_image(image_hu[:, :, z])
            ax.imshow(gray, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
            text_lines = [
                "SEGMENTACIÓN NO DISPONIBLE",
                "Se guardó este corte para diagnóstico.",
                "Revise los mensajes de detección de fascia y miembro.",
            ]

        ax.set_title(
            f"Miembro {side_name} - Corte medio Z={z}",
            fontsize=12,
            fontweight="bold",
        )
        ax.axis("off")
        ax.text(
            0.02,
            0.02,
            "\n".join(text_lines),
            transform=ax.transAxes,
            fontsize=9,
            color="white",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="black",
                alpha=0.72,
            ),
        )

        png_name = f"{TISSUE_PNG_PREFIX}_{side_name}.png"
        png_path = os.path.join(png_dir, png_name)
        fig.tight_layout()
        fig.savefig(
            png_path,
            format="png",
            dpi=160,
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )
        plt.close(fig)

        if not os.path.isfile(png_path) or os.path.getsize(png_path) == 0:
            raise IOError(f"Matplotlib no creó correctamente el archivo: {png_path}")

        output_paths[roi_side] = png_path
        print(f"  -> PNG miembro {side_name}: {png_path}")

    # Resumen combinado. Se crea aunque uno o ambos análisis hayan fallado.
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=160)
    any_valid_analysis = False

    for ax, (roi_side, side_name, z, analysis) in zip(axes, items):
        if analysis is not None:
            any_valid_analysis = True
            rgba = make_tissue_overlay_rgba(image_hu[:, :, z], analysis)
            ax.imshow(rgba, origin="upper")
            ratio_fm = analysis.get("fat_to_muscle_ratio")
            ratio_text = f"{ratio_fm:.4f}" if ratio_fm is not None else "No disponible"
            info_text = "\n".join(
                [
                    f"Perímetro: {analysis['limb_perimeter_cm']:.2f} cm",
                    f"Grasa: {analysis['fat_area_cm2']:.2f} cm²",
                    f"Músculo: {analysis['total_muscle_area_cm2']:.2f} cm²",
                    f"G/M: {ratio_text}",
                ]
            )
        else:
            gray = window_soft_tissue_image(image_hu[:, :, z])
            ax.imshow(gray, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
            info_text = "Segmentación no disponible\nImagen diagnóstica"

        ax.set_title(f"Miembro {side_name} - Z={z}", fontsize=12, fontweight="bold")
        ax.axis("off")
        ax.text(
            0.02,
            0.02,
            info_text,
            transform=ax.transAxes,
            fontsize=8.5,
            color="white",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="black",
                alpha=0.72,
            ),
        )

    if any_valid_analysis:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=4,
            framealpha=0.92,
            fontsize=9,
        )

    fig.suptitle(
        "Delimitación de piel, dermis, grasa subcutánea y músculo total",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    combined_path = os.path.join(png_dir, TISSUE_PNG_COMBINED_FILENAME)
    fig.savefig(
        combined_path,
        format="png",
        dpi=160,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    plt.close(fig)

    if not os.path.isfile(combined_path) or os.path.getsize(combined_path) == 0:
        raise IOError(f"No se creó correctamente el resumen PNG: {combined_path}")

    output_paths["combined"] = combined_path
    print(f"  -> PNG resumen: {combined_path}")
    print(f"  -> Total PNG creados: {len(output_paths)}")
    return output_paths



def analyze_subcutaneous_fat_volume_all_slices(
    image_hu: np.ndarray,
    pixel_spacing: List[float],
    slice_spacing: float,
    z_start: int,
    z_end: int,
    landmarks: Dict,
):
    """Estima el volumen de grasa subcutánea en cada miembro a lo largo del rango Z."""
    print("\nEstimando volumen de grasa subcutánea en todos los cortes...")
    voxel_volume_cm3 = float(
        pixel_spacing[0] * pixel_spacing[1] * slice_spacing / 1000.0
    )
    results = {}
    # Labelmap 3D de grasa subcutánea en todo el rango analizado.
    # 1 = miembro derecho del paciente, 2 = miembro izquierdo del paciente.
    subcutaneous_labelmap = np.zeros(image_hu.shape, dtype=np.uint8)

    for roi_side, side_info in SIDE_CONFIG.items():
        patient_side = side_info["patient_side"]
        label_value = 1 if roi_side == "left" else 2
        processed_slices = 0
        valid_slices = 0
        fat_voxel_count = 0

        midpoint = landmarks.get(roi_side, {}).get("midpoint_voxel_float")
        if midpoint is not None:
            previous_center = (float(midpoint[0]), float(midpoint[1]))
        else:
            previous_center = (
                float(image_hu.shape[0] / 2.0),
                float(image_hu.shape[1] / 2.0),
            )

        for z in range(z_start, z_end):
            processed_slices += 1
            analysis = analyze_limb_midpoint_slice(
                image_slice_hu=image_hu[:, :, z],
                pixel_spacing=pixel_spacing,
                roi_side=roi_side,
                midpoint_yx=previous_center,
            )
            if analysis is None:
                continue

            valid_slices += 1
            previous_center = analysis["center_yx"]
            fat_mask = analysis["subcutaneous_fat_mask"]
            fat_voxel_count += int(np.count_nonzero(fat_mask))
            subcutaneous_labelmap[:, :, z][fat_mask] = label_value

            if (z - z_start) % 40 == 0 or z == z_end - 1:
                print(
                    f"  -> {patient_side}: grasa subcutánea procesada en Z={z}/{z_end - 1}"
                )

        volume_cm3 = float(fat_voxel_count * voxel_volume_cm3)
        valid_fraction = (
            float(valid_slices / processed_slices) if processed_slices > 0 else 0.0
        )
        results[roi_side] = {
            "patient_side": patient_side,
            "subcutaneous_fat_volume_cm3": volume_cm3,
            "processed_slices": int(processed_slices),
            "valid_slices": int(valid_slices),
            "valid_slice_fraction": valid_fraction,
        }

        print(f"[{patient_side}] Volumen grasa subcutánea: {volume_cm3:.2f} cm³")
        print(
            f"  Cortes válidos: {valid_slices}/{processed_slices} "
            f"({100.0 * valid_fraction:.1f}%)"
        )

    results["_subcutaneous_labelmap"] = {
        "labelmap": subcutaneous_labelmap,
        "labels": {
            0: "Fondo",
            1: "Grasa subcutánea del miembro derecho",
            2: "Grasa subcutánea del miembro izquierdo",
        },
    }
    return results


def compute_intramuscular_fat_metrics(
    image_hu: np.ndarray,
    compartment_masks: Dict[str, np.ndarray],
    pixel_spacing: List[float],
    slice_spacing: float,
):
    """
    Detecta grasa intramuscular en cada corte y músculo.

    Grasa intramuscular: -190 a -30 HU dentro del compartimento segmentado.
    Volumen muscular: -29 a 150 HU dentro del mismo compartimento.
    Los totales por lado y global se calculan con uniones para evitar duplicar
    vóxeles cuando dos compartimentos segmentados se superponen.
    """
    print("\nAnalizando grasa intramuscular en cada corte y músculo...")

    voxel_volume_cm3 = float(
        pixel_spacing[0] * pixel_spacing[1] * slice_spacing / 1000.0
    )
    pixel_area_cm2 = float(pixel_spacing[0] * pixel_spacing[1] / 100.0)
    depth = image_hu.shape[2]

    labelmap = np.zeros(image_hu.shape, dtype=np.uint8)
    muscle_names = list(MUSCLES_CONFIG.keys())

    counters = {
        name: {
            "lean_voxels": 0,
            "fat_voxels": 0,
            "fat_areas_cm2": np.zeros(depth, dtype=np.float32),
        }
        for name in muscle_names
    }

    side_counts = {
        side: {"lean_voxels": 0, "fat_voxels": 0}
        for side in SIDE_CONFIG
    }
    global_lean_voxels = 0
    global_fat_voxels = 0
    area_rows = []

    for z in range(depth):
        image_slice = image_hu[:, :, z]
        fat_hu = (
            (image_slice >= SUBCUTANEOUS_FAT_HU_MIN)
            & (image_slice <= SUBCUTANEOUS_FAT_HU_MAX)
        )
        lean_hu = (
            (image_slice >= TOTAL_MUSCLE_HU_MIN)
            & (image_slice <= TOTAL_MUSCLE_HU_MAX)
        )

        global_lean_union = np.zeros(image_slice.shape, dtype=bool)
        global_fat_union = np.zeros(image_slice.shape, dtype=bool)
        side_lean_union = {
            side: np.zeros(image_slice.shape, dtype=bool) for side in SIDE_CONFIG
        }
        side_fat_union = {
            side: np.zeros(image_slice.shape, dtype=bool) for side in SIDE_CONFIG
        }

        row = {"Z": int(z)}

        for muscle_name in muscle_names:
            compartment = compartment_masks[muscle_name][:, :, z]
            if not np.any(compartment):
                row[f"{muscle_name}_cm2"] = 0.0
                continue

            lean_mask = compartment & lean_hu
            fat_mask = compartment & fat_hu
            fat_mask = remove_small_components(
                fat_mask,
                MIN_INTRAMUSCULAR_FAT_COMPONENT_AREA_PX,
            )

            lean_count = int(np.count_nonzero(lean_mask))
            fat_count = int(np.count_nonzero(fat_mask))
            fat_area_cm2 = float(fat_count * pixel_area_cm2)

            counters[muscle_name]["lean_voxels"] += lean_count
            counters[muscle_name]["fat_voxels"] += fat_count
            counters[muscle_name]["fat_areas_cm2"][z] = fat_area_cm2
            row[f"{muscle_name}_cm2"] = fat_area_cm2

            label_value = INTRAMUSCULAR_FAT_LABELS[muscle_name]
            labelmap[:, :, z][fat_mask] = label_value

            side_key = MUSCLES_CONFIG[muscle_name]["roi"]
            side_lean_union[side_key] |= lean_mask
            side_fat_union[side_key] |= fat_mask
            global_lean_union |= lean_mask
            global_fat_union |= fat_mask

        for side_key in SIDE_CONFIG:
            side_counts[side_key]["lean_voxels"] += int(
                np.count_nonzero(side_lean_union[side_key])
            )
            side_counts[side_key]["fat_voxels"] += int(
                np.count_nonzero(side_fat_union[side_key])
            )

        global_lean_voxels += int(np.count_nonzero(global_lean_union))
        global_fat_voxels += int(np.count_nonzero(global_fat_union))
        row["Total_Grasa_Intramuscular_cm2"] = float(
            np.count_nonzero(global_fat_union) * pixel_area_cm2
        )
        area_rows.append(row)

        if z % 40 == 0 or z == depth - 1:
            print(f"  -> Grasa intramuscular procesada en corte {z}/{depth - 1}")

    per_muscle = {}
    for muscle_name in muscle_names:
        lean_volume = float(counters[muscle_name]["lean_voxels"] * voxel_volume_cm3)
        fat_volume = float(counters[muscle_name]["fat_voxels"] * voxel_volume_cm3)
        ratio = float(fat_volume / lean_volume) if lean_volume > 0 else None
        areas = counters[muscle_name]["fat_areas_cm2"]
        max_z = int(np.argmax(areas)) if areas.size else None
        max_area = float(areas[max_z]) if max_z is not None else 0.0
        nonzero = areas[areas > 0]
        mean_nonzero = float(np.mean(nonzero)) if nonzero.size else 0.0

        per_muscle[muscle_name] = {
            "muscle_volume_cm3": lean_volume,
            "intramuscular_fat_volume_cm3": fat_volume,
            "intramuscular_fat_ratio": ratio,
            "max_intramuscular_fat_area_cm2": max_area,
            "max_intramuscular_fat_area_z": max_z,
            "mean_nonzero_intramuscular_fat_area_cm2": mean_nonzero,
            "label_value": INTRAMUSCULAR_FAT_LABELS[muscle_name],
        }

        ratio_text = f"{ratio:.4f}" if ratio is not None else "No disponible"
        print(
            f"[{muscle_name}] músculo={lean_volume:.2f} cm³ | "
            f"grasa IM={fat_volume:.2f} cm³ | GIM/M={ratio_text}"
        )

    side_summary = {}
    for side_key, counts in side_counts.items():
        lean_volume = float(counts["lean_voxels"] * voxel_volume_cm3)
        fat_volume = float(counts["fat_voxels"] * voxel_volume_cm3)
        side_summary[side_key] = {
            "patient_side": SIDE_CONFIG[side_key]["patient_side"],
            "muscle_volume_cm3": lean_volume,
            "intramuscular_fat_volume_cm3": fat_volume,
            "intramuscular_fat_ratio": (
                float(fat_volume / lean_volume) if lean_volume > 0 else None
            ),
        }

    total_muscle_volume = float(global_lean_voxels * voxel_volume_cm3)
    total_fat_volume = float(global_fat_voxels * voxel_volume_cm3)
    total_ratio = (
        float(total_fat_volume / total_muscle_volume)
        if total_muscle_volume > 0
        else None
    )

    return {
        "per_muscle": per_muscle,
        "side_summary": side_summary,
        "intramuscular_labelmap": labelmap,
        "area_rows": area_rows,
        "total_muscle_volume_cm3": total_muscle_volume,
        "total_intramuscular_fat_volume_cm3": total_fat_volume,
        "total_intramuscular_fat_ratio": total_ratio,
    }


def combine_global_fat_metrics(
    intramuscular_metrics: Dict,
    subcutaneous_volume_results: Dict,
):
    total_muscle = float(intramuscular_metrics["total_muscle_volume_cm3"])
    total_intramuscular = float(
        intramuscular_metrics["total_intramuscular_fat_volume_cm3"]
    )
    total_subcutaneous = float(
        sum(
            result.get("subcutaneous_fat_volume_cm3", 0.0)
            for result in subcutaneous_volume_results.values()
            if result is not None
        )
    )
    total_fat = total_intramuscular + total_subcutaneous

    side_results = {}
    for side_key, side_intramuscular in intramuscular_metrics["side_summary"].items():
        muscle_volume = float(side_intramuscular["muscle_volume_cm3"])
        intramuscular_volume = float(
            side_intramuscular["intramuscular_fat_volume_cm3"]
        )
        subcutaneous_volume = float(
            subcutaneous_volume_results.get(side_key, {}).get(
                "subcutaneous_fat_volume_cm3", 0.0
            )
        )
        total_side_fat = intramuscular_volume + subcutaneous_volume

        side_results[side_key] = {
            "patient_side": side_intramuscular["patient_side"],
            "muscle_volume_cm3": muscle_volume,
            "intramuscular_fat_volume_cm3": intramuscular_volume,
            "subcutaneous_fat_volume_cm3": subcutaneous_volume,
            "total_fat_volume_cm3": total_side_fat,
            "intramuscular_fat_to_muscle_ratio": (
                float(intramuscular_volume / muscle_volume)
                if muscle_volume > 0
                else None
            ),
            "subcutaneous_fat_to_muscle_ratio": (
                float(subcutaneous_volume / muscle_volume)
                if muscle_volume > 0
                else None
            ),
            "total_fat_to_muscle_ratio": (
                float(total_side_fat / muscle_volume)
                if muscle_volume > 0
                else None
            ),
        }

    return {
        "total_muscle_volume_cm3": total_muscle,
        "total_intramuscular_fat_volume_cm3": total_intramuscular,
        "total_subcutaneous_fat_volume_cm3": total_subcutaneous,
        "total_fat_volume_cm3": total_fat,
        "ratio_subcutaneous_to_muscle": (
            float(total_subcutaneous / total_muscle) if total_muscle > 0 else None
        ),
        "ratio_totalfat_to_muscle": (
            float(total_fat / total_muscle) if total_muscle > 0 else None
        ),
        "side_results": side_results,
    }



def build_total_adipose_labelmap(subcutaneous_labelmap: np.ndarray, intramuscular_labelmap: np.ndarray) -> np.ndarray:
    """
    Construye un labelmap 3D unificado de tejido adiposo:
    1 = grasa subcutánea miembro derecho
    2 = grasa subcutánea miembro izquierdo
    3 = grasa intramuscular miembro derecho
    4 = grasa intramuscular miembro izquierdo
    """
    total = np.zeros_like(subcutaneous_labelmap, dtype=np.uint8)
    total[subcutaneous_labelmap == 1] = 1
    total[subcutaneous_labelmap == 2] = 2
    # En INTRAMUSCULAR_FAT_LABELS: 1-2 son vastos derechos, 3-4 vastos izquierdos.
    total[np.isin(intramuscular_labelmap, [1, 2])] = 3
    total[np.isin(intramuscular_labelmap, [3, 4])] = 4
    return total


def export_adipose_labels_description(path: str):
    lines = [
        "ETIQUETAS DEL ARCHIVO Tejido_Adiposo_Total_Volumen.nii.gz",
        "0 = Fondo",
        "1 = Grasa subcutánea del miembro derecho del paciente",
        "2 = Grasa subcutánea del miembro izquierdo del paciente",
        "3 = Grasa intramuscular del miembro derecho del paciente",
        "4 = Grasa intramuscular del miembro izquierdo del paciente",
        "",
        "Archivos asociados:",
        f"- {SUBCUTANEOUS_FAT_LABEL_FILENAME}: solo grasa subcutánea 3D",
        f"- {INTRAMUSCULAR_FAT_LABEL_FILENAME}: grasa intramuscular por músculo",
        f"- {TOTAL_ADIPOSE_LABEL_FILENAME}: grasa subcutánea + intramuscular unificada",
        "",
        "La grasa se clasifica en el rango HU -190 a -30 dentro de compartimentos anatómicos estimados.",
        "Las segmentaciones deben validarse visualmente antes de usarlas como resultado final.",
    ]
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def _append_metric_row(rows: List[Dict], section: str, entity: str, side: str, metric_name: str, value, unit: str, description: str):
    if value is None:
        return
    try:
        value_float = float(value)
    except Exception:
        return
    rows.append({
        "section": section,
        "entity": entity,
        "side": side,
        "metric_name": metric_name,
        "metric_value": value_float,
        "unit_or_scale": unit,
        "description": description,
    })


def export_adipose_metrics_csv(
    path: str,
    tissue_composition: Dict,
    intramuscular_metrics: Dict,
    subcutaneous_volume_results: Dict,
    global_fat_metrics: Dict,
):
    """Exporta todas las métricas de grasa/músculo en formato largo."""
    rows: List[Dict] = []

    # Corte medio por miembro.
    for roi_side, analysis in tissue_composition.items():
        if analysis is None:
            continue
        side = analysis.get("patient_side", roi_side)
        entity = f"miembro_{side.lower()}_corte_medio"
        _append_metric_row(rows, "corte_medio", entity, side, "subcutaneous_fat_area_cm2", analysis.get("fat_area_cm2"), "cm2", "Área de grasa subcutánea en el corte medio.")
        _append_metric_row(rows, "corte_medio", entity, side, "total_muscle_area_cm2", analysis.get("total_muscle_area_cm2"), "cm2", "Área muscular total en el corte medio.")
        _append_metric_row(rows, "corte_medio", entity, side, "fat_to_muscle_ratio", analysis.get("fat_to_muscle_ratio"), "ratio", "Relación grasa subcutánea / músculo en el corte medio.")
        _append_metric_row(rows, "corte_medio", entity, side, "muscle_to_fat_ratio", analysis.get("muscle_to_fat_ratio"), "ratio", "Relación músculo / grasa subcutánea en el corte medio.")
        _append_metric_row(rows, "corte_medio", entity, side, "fat_fraction_percent", analysis.get("fat_fraction_percent"), "%", "Porcentaje adiposo respecto a grasa + músculo en el corte medio.")
        _append_metric_row(rows, "corte_medio", entity, side, "limb_perimeter_cm", analysis.get("limb_perimeter_cm"), "cm", "Perímetro externo estimado del miembro en el corte medio.")
        _append_metric_row(rows, "corte_medio", entity, side, "mean_subcutaneous_thickness_mm", analysis.get("mean_subcutaneous_compartment_thickness_mm"), "mm", "Espesor medio del compartimento subcutáneo estimado.")
        _append_metric_row(rows, "corte_medio", entity, side, "valid_ray_fraction", analysis.get("valid_fascia_fraction"), "0-1", "Fracción de rayos válidos para la detección de fascia.")

    # Grasa subcutánea 3D por miembro.
    for roi_side, result in subcutaneous_volume_results.items():
        if not isinstance(result, dict) or roi_side.startswith("_"):
            continue
        side = result.get("patient_side", roi_side)
        entity = f"miembro_{side.lower()}_volumen"
        _append_metric_row(rows, "volumetria_subcutanea", entity, side, "subcutaneous_fat_volume_cm3", result.get("subcutaneous_fat_volume_cm3"), "cm3", "Volumen 3D de grasa subcutánea dentro del rango Z analizado.")
        _append_metric_row(rows, "volumetria_subcutanea", entity, side, "processed_slices", result.get("processed_slices"), "slices", "Número de cortes procesados para grasa subcutánea.")
        _append_metric_row(rows, "volumetria_subcutanea", entity, side, "valid_slices", result.get("valid_slices"), "slices", "Número de cortes válidos para grasa subcutánea.")
        _append_metric_row(rows, "volumetria_subcutanea", entity, side, "valid_slice_fraction", result.get("valid_slice_fraction"), "0-1", "Fracción de cortes válidos para la estimación de grasa subcutánea.")

    # Grasa intramuscular por músculo.
    for muscle_name, metrics in intramuscular_metrics.get("per_muscle", {}).items():
        side = "Derecho" if muscle_name.endswith("_Der") else "Izquierdo"
        _append_metric_row(rows, "volumetria_intramuscular", muscle_name, side, "muscle_volume_cm3", metrics.get("muscle_volume_cm3"), "cm3", "Volumen muscular magro estimado dentro del compartimento del músculo.")
        _append_metric_row(rows, "volumetria_intramuscular", muscle_name, side, "intramuscular_fat_volume_cm3", metrics.get("intramuscular_fat_volume_cm3"), "cm3", "Volumen de grasa intramuscular en el músculo.")
        _append_metric_row(rows, "volumetria_intramuscular", muscle_name, side, "intramuscular_fat_to_muscle_ratio", metrics.get("intramuscular_fat_ratio"), "ratio", "Relación grasa intramuscular / músculo.")
        _append_metric_row(rows, "volumetria_intramuscular", muscle_name, side, "max_intramuscular_fat_area_cm2", metrics.get("max_intramuscular_fat_area_cm2"), "cm2", "Área máxima de grasa intramuscular por corte.")
        _append_metric_row(rows, "volumetria_intramuscular", muscle_name, side, "mean_nonzero_intramuscular_fat_area_cm2", metrics.get("mean_nonzero_intramuscular_fat_area_cm2"), "cm2", "Área media de grasa intramuscular en cortes con grasa detectada.")

    # Resumen por miembro y global.
    for side_key, side_metrics in global_fat_metrics.get("side_results", {}).items():
        side = side_metrics.get("patient_side", side_key)
        entity = f"miembro_{side.lower()}_resumen"
        _append_metric_row(rows, "resumen_por_miembro", entity, side, "side_muscle_volume_cm3", side_metrics.get("muscle_volume_cm3"), "cm3", "Volumen muscular total del miembro.")
        _append_metric_row(rows, "resumen_por_miembro", entity, side, "side_intramuscular_fat_volume_cm3", side_metrics.get("intramuscular_fat_volume_cm3"), "cm3", "Volumen total de grasa intramuscular del miembro.")
        _append_metric_row(rows, "resumen_por_miembro", entity, side, "side_subcutaneous_fat_volume_cm3", side_metrics.get("subcutaneous_fat_volume_cm3"), "cm3", "Volumen total de grasa subcutánea del miembro.")
        _append_metric_row(rows, "resumen_por_miembro", entity, side, "side_total_fat_volume_cm3", side_metrics.get("total_fat_volume_cm3"), "cm3", "Volumen total de grasa del miembro.")
        _append_metric_row(rows, "resumen_por_miembro", entity, side, "side_intramuscular_fat_to_muscle_ratio", side_metrics.get("intramuscular_fat_to_muscle_ratio"), "ratio", "Relación grasa intramuscular / músculo del miembro.")
        _append_metric_row(rows, "resumen_por_miembro", entity, side, "side_subcutaneous_fat_to_muscle_ratio", side_metrics.get("subcutaneous_fat_to_muscle_ratio"), "ratio", "Relación grasa subcutánea / músculo del miembro.")
        _append_metric_row(rows, "resumen_por_miembro", entity, side, "side_total_fat_to_muscle_ratio", side_metrics.get("total_fat_to_muscle_ratio"), "ratio", "Relación grasa total / músculo del miembro.")

    _append_metric_row(rows, "resumen_global", "global", "bilateral", "global_muscle_volume_cm3", global_fat_metrics.get("total_muscle_volume_cm3"), "cm3", "Volumen muscular total bilateral.")
    _append_metric_row(rows, "resumen_global", "global", "bilateral", "global_intramuscular_fat_volume_cm3", global_fat_metrics.get("total_intramuscular_fat_volume_cm3"), "cm3", "Volumen total bilateral de grasa intramuscular.")
    _append_metric_row(rows, "resumen_global", "global", "bilateral", "global_subcutaneous_fat_volume_cm3", global_fat_metrics.get("total_subcutaneous_fat_volume_cm3"), "cm3", "Volumen total bilateral de grasa subcutánea.")
    _append_metric_row(rows, "resumen_global", "global", "bilateral", "global_total_fat_volume_cm3", global_fat_metrics.get("total_fat_volume_cm3"), "cm3", "Volumen total bilateral de grasa subcutánea + intramuscular.")
    _append_metric_row(rows, "resumen_global", "global", "bilateral", "global_subcutaneous_fat_to_muscle_ratio", global_fat_metrics.get("ratio_subcutaneous_to_muscle"), "ratio", "Relación global grasa subcutánea / músculo.")
    _append_metric_row(rows, "resumen_global", "global", "bilateral", "global_total_fat_to_muscle_ratio", global_fat_metrics.get("ratio_totalfat_to_muscle"), "ratio", "Relación global grasa total / músculo.")

    fieldnames = ["section", "entity", "side", "metric_name", "metric_value", "unit_or_scale", "description"]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def export_intramuscular_labels_description(path: str):
    lines = [
        "ETIQUETAS DEL ARCHIVO Grasa_Intramuscular_Muscular.nii",
        "0 = Fondo",
        "1 = Grasa intramuscular del Vasto Lateral Derecho",
        "2 = Grasa intramuscular del Vasto Medial Derecho",
        "3 = Grasa intramuscular del Vasto Lateral Izquierdo",
        "4 = Grasa intramuscular del Vasto Medial Izquierdo",
        "",
        "Umbral de grasa: -190 a -30 HU dentro del compartimento muscular.",
        "Volumen muscular: -29 a 150 HU dentro del mismo compartimento.",
        "Debe validarse visualmente en 3D Slicer.",
    ]
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def export_intramuscular_area_csv(path: str, area_rows: List[Dict]):
    fieldnames = ["Z"] + [
        f"{name}_cm2" for name in MUSCLES_CONFIG
    ] + ["Total_Grasa_Intramuscular_cm2"]
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in area_rows:
            writer.writerow({key: row.get(key, 0.0) for key in fieldnames})


def build_intramuscular_overlay(
    image_hu: np.ndarray,
    labelmap: np.ndarray,
) -> np.ndarray:
    overlay = image_hu.astype(np.float32).copy()
    display_values = {1: 1600.0, 2: 2000.0, 3: 2400.0, 4: 2800.0}
    for label_value, display_value in display_values.items():
        overlay[labelmap == label_value] = display_value
    return overlay


# ==============================================================================
# FASE 5: DIBUJO 3D DE PUNTOS, LÍNEA Y PUNTO MEDIO
# ==============================================================================
def paint_sphere(
    volume: np.ndarray,
    center: Optional[Tuple[int, int, int]],
    value: int,
    radius: int,
):
    if center is None:
        return

    y0, x0, z0 = center
    h, w, d = volume.shape

    y_min, y_max = max(0, y0 - radius), min(h, y0 + radius + 1)
    x_min, x_max = max(0, x0 - radius), min(w, x0 + radius + 1)
    z_min, z_max = max(0, z0 - radius), min(d, z0 + radius + 1)

    yy, xx, zz = np.ogrid[y_min:y_max, x_min:x_max, z_min:z_max]
    sphere = (yy - y0) ** 2 + (xx - x0) ** 2 + (zz - z0) ** 2 <= radius**2
    subvolume = volume[y_min:y_max, x_min:x_max, z_min:z_max]
    subvolume[sphere] = value


def paint_line(
    volume: np.ndarray,
    p1: Optional[Tuple[int, int, int]],
    p2: Optional[Tuple[int, int, int]],
    value: int,
    radius: int = 1,
):
    if p1 is None or p2 is None:
        return

    start = np.asarray(p1, dtype=float)
    end = np.asarray(p2, dtype=float)
    voxel_distance = float(np.linalg.norm(end - start))
    number_of_samples = max(2, int(np.ceil(voxel_distance * 3.0)))

    for t in np.linspace(0.0, 1.0, number_of_samples):
        point = tuple(int(round(v)) for v in (start * (1.0 - t) + end * t))
        paint_sphere(volume, point, value=value, radius=radius)


def build_landmark_labelmap(image_shape: Tuple[int, int, int], landmarks: Dict) -> np.ndarray:
    labelmap = np.zeros(image_shape, dtype=np.uint8)

    for _, info in landmarks.items():
        labels = info["label_values"]
        trochanter = info["trochanter_voxel"]
        tibial = info["tibial_voxel"]
        midpoint = info["midpoint_voxel"]

        paint_line(
            labelmap,
            trochanter,
            tibial,
            value=labels["line"],
            radius=LINE_RADIUS_PX,
        )
        paint_sphere(
            labelmap,
            trochanter,
            value=labels["trochanter"],
            radius=LANDMARK_RADIUS_PX,
        )
        paint_sphere(
            labelmap,
            tibial,
            value=labels["tibia"],
            radius=LANDMARK_RADIUS_PX,
        )
        paint_sphere(
            labelmap,
            midpoint,
            value=labels["midpoint"],
            radius=MIDPOINT_RADIUS_PX,
        )

    return labelmap


def build_landmark_overlay(image_hu: np.ndarray, landmarks: Dict) -> np.ndarray:
    """Genera un TAC con marcas intensas para revisión rápida como volumen escalar."""
    overlay = image_hu.astype(np.float32).copy()

    for _, info in landmarks.items():
        trochanter = info["trochanter_voxel"]
        tibial = info["tibial_voxel"]
        midpoint = info["midpoint_voxel"]

        paint_line(overlay, trochanter, tibial, value=2200, radius=LINE_RADIUS_PX)
        paint_sphere(overlay, trochanter, value=3000, radius=LANDMARK_RADIUS_PX)
        paint_sphere(overlay, tibial, value=2800, radius=LANDMARK_RADIUS_PX)
        paint_sphere(overlay, midpoint, value=3500, radius=MIDPOINT_RADIUS_PX)

    return overlay


# ==============================================================================
# FASE 6: REPORTE Y EXPORTACIÓN
# ==============================================================================
def save_nifti(data: np.ndarray, affine: np.ndarray, path: str):
    nifti = nib.Nifti1Image(data, affine)
    nib.save(nifti, path)


def export_labels_description(path: str):
    lines = [
        "ETIQUETAS DEL ARCHIVO Landmarks_Linea_CorteMedio.nii",
        "0 = Fondo",
        "1 = Línea miembro derecho",
        "2 = Punto trocantérico derecho",
        "3 = Punto tibial anteromedial derecho",
        "4 = Punto medio derecho",
        "5 = Línea miembro izquierdo",
        "6 = Punto trocantérico izquierdo",
        "7 = Punto tibial anteromedial izquierdo",
        "8 = Punto medio izquierdo",
        "",
        "Los puntos son estimaciones automáticas y deben validarse visualmente.",
    ]
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def export_tissue_labels_description(path: str):
    lines = [
        "ETIQUETAS DEL ARCHIVO Composicion_Corporal_CorteMedio.nii",
        "0 = Fondo",
        "1 = Contorno epidérmico estimado del miembro derecho",
        "2 = Banda dérmica estimada del miembro derecho",
        "3 = Tejido adiposo subcutáneo del miembro derecho",
        "4 = Área muscular total del miembro derecho",
        "5 = Contorno epidérmico estimado del miembro izquierdo",
        "6 = Banda dérmica estimada del miembro izquierdo",
        "7 = Tejido adiposo subcutáneo del miembro izquierdo",
        "8 = Área muscular total del miembro izquierdo",
        "",
        "La separación epidermis/dermis es radiológica y aproximada.",
        "La grasa se clasifica entre -190 y -30 HU dentro del compartimento subcutáneo.",
        "El músculo se clasifica entre -29 y 150 HU dentro de la fascia profunda.",
        "Todas las segmentaciones deben revisarse visualmente antes del análisis final.",
    ]
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def export_and_report_all(
    image_hu: np.ndarray,
    masks_dict: Dict[str, np.ndarray],
    compartment_masks: Dict[str, np.ndarray],
    pixel_spacing: List[float],
    slice_spacing: float,
    affine: np.ndarray,
    out_dir: str,
    z_start: int,
    z_end: int,
    landmarks: Dict,
    tissue_composition: Dict,
    intramuscular_metrics: Dict,
    subcutaneous_volume_results: Dict,
    global_fat_metrics: Dict,
):
    os.makedirs(out_dir, exist_ok=True)

    voxel_volume_cm3 = float(pixel_spacing[0] * pixel_spacing[1] * slice_spacing / 1000.0)
    pixel_area_cm2 = float(pixel_spacing[0] * pixel_spacing[1] / 100.0)
    midpoint_areas = compute_midpoint_areas(masks_dict, pixel_spacing, landmarks)

    report_lines = [
        "=" * 74,
        "REPORTE MORFOMÉTRICO MULTI-MUSCULAR Y LANDMARKS ÓSEOS",
        "=" * 74,
        f"Rango longitudinal analizado: Z={z_start} a Z={z_end - 1}",
        f"Espaciado de píxel: fila={pixel_spacing[0]:.4f} mm, columna={pixel_spacing[1]:.4f} mm",
        f"Espaciado entre cortes: {slice_spacing:.4f} mm",
        "-" * 74,
    ]

    print("\n" + "=" * 58)
    print("REPORTE MORFOMÉTRICO GENERADO")
    print("=" * 58)

    muscle_tracer = image_hu.astype(np.float32).copy()

    for muscle_name, muscle_mask in masks_dict.items():
        volume_cm3 = float(np.sum(muscle_mask) * voxel_volume_cm3)
        areas = np.asarray(
            [np.sum(muscle_mask[:, :, z]) * pixel_area_cm2 for z in range(image_hu.shape[2])],
            dtype=float,
        )
        max_z = int(np.argmax(areas)) if len(areas) else 0
        max_area_cm2 = float(areas[max_z]) if len(areas) else 0.0

        print(f"[{muscle_name}]")
        print(f"  Volumen:      {volume_cm3:.2f} cm³")
        print(f"  Área máxima:  {max_area_cm2:.2f} cm² (Z={max_z})\n")

        report_lines.extend(
            [
                f"MÚSCULO: {muscle_name}",
                f" - Volumen total:              {volume_cm3:.2f} cm³",
                f" - Sección transversal máxima: {max_area_cm2:.2f} cm² (Z={max_z})",
                "-" * 74,
            ]
        )

        for z in range(z_start, z_end):
            if np.any(muscle_mask[:, :, z]):
                boundaries = find_boundaries(muscle_mask[:, :, z], mode="inner")
                muscle_tracer[:, :, z][boundaries] = 1500.0

    # Fondo de aire oscuro para facilitar la inspección.
    muscle_tracer[image_hu < -200] = -1000.0

    report_lines.extend(
        [
            "",
            "LANDMARKS ÓSEOS, LÍNEA 3D Y CORTE MEDIO",
            "=" * 74,
            "Coordenadas voxel expresadas como (fila Y, columna X, corte Z).",
            "Coordenadas físicas expresadas en milímetros según el affine DICOM.",
            "",
        ]
    )

    for roi_side, info in landmarks.items():
        area_info = midpoint_areas.get(roi_side)

        report_lines.extend(
            [
                f"MIEMBRO {info['patient_side'].upper()}",
                f" - Punto trocantérico voxel:          {format_point(info['trochanter_voxel'])}",
                f" - Punto trocantérico físico (mm):    {format_point(info['trochanter_world_mm'])}",
                f" - Punto tibial anteromedial voxel:   {format_point(info['tibial_voxel'])}",
                f" - Punto tibial físico (mm):          {format_point(info['tibial_world_mm'])}",
                f" - Punto medio voxel:                 {format_point(info['midpoint_voxel_float'])}",
                f" - Punto medio físico (mm):           {format_point(info['midpoint_world_mm'])}",
                f" - CORTE TRANSVERSAL MEDIO:           Z={info['midpoint_slice']}",
                " - Distancia entre landmarks:         "
                + (
                    f"{info['landmark_distance_mm']:.2f} mm"
                    if info["landmark_distance_mm"] is not None
                    else "No disponible"
                ),
            ]
        )

        if info["tibial_details"] is not None:
            report_lines.append(
                f" - Método de detección tibial:         {info['tibial_details']['method']}"
            )

        if area_info is None:
            report_lines.append(" - Área transversal en corte medio:   No disponible")
        else:
            report_lines.append(" - Áreas en el corte transversal medio:")
            for muscle_name, area_cm2 in area_info["muscle_areas_cm2"].items():
                report_lines.append(f"    * {muscle_name}: {area_cm2:.2f} cm²")
            report_lines.append(
                f"    * Área combinada de los dos vastos: {area_info['total_vasti_area_cm2']:.2f} cm²"
            )

        if info["warnings"]:
            report_lines.append(" - Advertencias de control de calidad:")
            for warning in info["warnings"]:
                report_lines.append(f"    * {warning}")

        report_lines.extend(["-" * 74, ""])

    report_lines.extend(
        [
            "",
            "COMPOSICIÓN CORPORAL EN EL CORTE MEDIO",
            "=" * 74,
            f"Grasa subcutánea: {SUBCUTANEOUS_FAT_HU_MIN:.0f} a {SUBCUTANEOUS_FAT_HU_MAX:.0f} HU.",
            f"Músculo total: {TOTAL_MUSCLE_HU_MIN:.0f} a {TOTAL_MUSCLE_HU_MAX:.0f} HU.",
            "La fascia profunda, la superficie epidérmica y el límite interno dérmico",
            "se estimaron radialmente mediante CWT de sombrero mexicano.",
            "",
        ]
    )

    for roi_side, analysis in tissue_composition.items():
        side_name = landmarks[roi_side]["patient_side"]
        report_lines.append(f"MIEMBRO {side_name.upper()}")
        if analysis is None:
            report_lines.extend(
                [
                    " - Análisis de composición corporal: No disponible",
                    "-" * 74,
                    "",
                ]
            )
            continue

        ratio_fat_muscle = (
            f"{analysis['fat_to_muscle_ratio']:.4f}"
            if analysis["fat_to_muscle_ratio"] is not None
            else "No disponible"
        )
        ratio_muscle_fat = (
            f"{analysis['muscle_to_fat_ratio']:.4f}"
            if analysis["muscle_to_fat_ratio"] is not None
            else "No disponible"
        )
        fat_fraction = (
            f"{analysis['fat_fraction_percent']:.2f}%"
            if analysis["fat_fraction_percent"] is not None
            else "No disponible"
        )

        report_lines.extend(
            [
                f" - Corte analizado:                         Z={analysis['slice_z']}",
                f" - Perímetro externo del miembro:         {analysis['limb_perimeter_cm']:.2f} cm ({analysis['limb_perimeter_mm']:.2f} mm)",
                f" - Área de grasa subcutánea:              {analysis['fat_area_cm2']:.2f} cm²",
                f" - Área muscular total:                   {analysis['total_muscle_area_cm2']:.2f} cm²",
                f" - Relación grasa/músculo:                {ratio_fat_muscle}",
                f" - Relación músculo/grasa:                {ratio_muscle_fat}",
                f" - Grasa respecto a grasa + músculo:      {fat_fraction}",
                f" - Área de banda dérmica estimada:        {analysis['dermis_band_area_cm2']:.2f} cm²",
                f" - Espesor cutáneo medio estimado:        {analysis['mean_estimated_skin_thickness_mm']:.2f} mm",
                f" - Espesor subcutáneo medio estimado:     {analysis['mean_subcutaneous_compartment_thickness_mm']:.2f} mm",
                f" - Rayos válidos para superficie externa: {100.0 * analysis['valid_outer_ray_fraction']:.1f}%",
                f" - Rayos válidos para límite dérmico:     {100.0 * analysis['valid_dermis_ray_fraction']:.1f}%",
                f" - Rayos válidos para fascia profunda:    {100.0 * analysis['valid_fascia_ray_fraction']:.1f}%",
            ]
        )
        if analysis["warnings"]:
            report_lines.append(" - Advertencias:")
            for warning in analysis["warnings"]:
                report_lines.append(f"    * {warning}")
        report_lines.extend(["-" * 74, ""])

    report_lines.extend(
        [
            "",
            "GRASA INTRAMUSCULAR Y VOLUMETRÍA GLOBAL",
            "=" * 74,
            "La grasa intramuscular se clasifica entre -190 y -30 HU dentro del compartimento",
            "muscular segmentado de cada uno de los cuatro músculos.",
            "",
        ]
    )

    for muscle_name, metrics in intramuscular_metrics["per_muscle"].items():
        ratio_text = (
            f"{metrics['intramuscular_fat_ratio']:.4f}"
            if metrics["intramuscular_fat_ratio"] is not None
            else "No disponible"
        )
        report_lines.extend(
            [
                f"MÚSCULO {muscle_name}",
                f" - Volumen muscular segmentado:          {metrics['muscle_volume_cm3']:.2f} cm³",
                f" - Volumen de grasa intramuscular:       {metrics['intramuscular_fat_volume_cm3']:.2f} cm³",
                f" - Relación grasa IM / volumen músculo:  {ratio_text}",
                f" - Área máxima de grasa IM:              {metrics['max_intramuscular_fat_area_cm2']:.2f} cm² (Z={metrics['max_intramuscular_fat_area_z']})",
                f" - Área media no nula de grasa IM:      {metrics['mean_nonzero_intramuscular_fat_area_cm2']:.2f} cm²",
                "-" * 74,
                "",
            ]
        )

    report_lines.extend(
        [
            "RESUMEN POR MIEMBRO",
            "-" * 74,
        ]
    )
    for side_key, side_metrics in global_fat_metrics["side_results"].items():
        im_ratio = (
            f"{side_metrics['intramuscular_fat_to_muscle_ratio']:.4f}"
            if side_metrics["intramuscular_fat_to_muscle_ratio"] is not None
            else "No disponible"
        )
        sc_ratio = (
            f"{side_metrics['subcutaneous_fat_to_muscle_ratio']:.4f}"
            if side_metrics["subcutaneous_fat_to_muscle_ratio"] is not None
            else "No disponible"
        )
        total_ratio = (
            f"{side_metrics['total_fat_to_muscle_ratio']:.4f}"
            if side_metrics["total_fat_to_muscle_ratio"] is not None
            else "No disponible"
        )
        subcut_info = subcutaneous_volume_results.get(side_key, None)
        valid_slices_text = "No disponible"
        if subcut_info is not None:
            valid_slices_text = f"{subcut_info['valid_slices']}/{subcut_info['processed_slices']} ({100.0 * subcut_info['valid_slice_fraction']:.1f}%)"
        report_lines.extend(
            [
                f"MIEMBRO {side_metrics['patient_side'].upper()}",
                f" - Volumen muscular total del lado:      {side_metrics['muscle_volume_cm3']:.2f} cm³",
                f" - Volumen grasa intramuscular:          {side_metrics['intramuscular_fat_volume_cm3']:.2f} cm³",
                f" - Volumen grasa subcutánea:             {side_metrics['subcutaneous_fat_volume_cm3']:.2f} cm³",
                f" - Volumen total de grasa:               {side_metrics['total_fat_volume_cm3']:.2f} cm³",
                f" - Relación grasa IM / músculo:          {im_ratio}",
                f" - Relación grasa subcutánea / músculo:  {sc_ratio}",
                f" - Relación grasa total / músculo:       {total_ratio}",
                f" - Cortes válidos para grasa subcutánea: {valid_slices_text}",
                "-" * 74,
                "",
            ]
        )

    total_im_ratio = (
        f"{intramuscular_metrics['total_intramuscular_fat_ratio']:.4f}"
        if intramuscular_metrics['total_intramuscular_fat_ratio'] is not None
        else "No disponible"
    )
    total_sc_ratio = (
        f"{global_fat_metrics['ratio_subcutaneous_to_muscle']:.4f}"
        if global_fat_metrics['ratio_subcutaneous_to_muscle'] is not None
        else "No disponible"
    )
    total_fat_ratio = (
        f"{global_fat_metrics['ratio_totalfat_to_muscle']:.4f}"
        if global_fat_metrics['ratio_totalfat_to_muscle'] is not None
        else "No disponible"
    )

    report_lines.extend(
        [
            "RESUMEN GLOBAL DE LOS CUATRO MÚSCULOS",
            "-" * 74,
            f" - Volumen muscular total:               {intramuscular_metrics['total_muscle_volume_cm3']:.2f} cm³",
            f" - Volumen total de grasa intramuscular: {intramuscular_metrics['total_intramuscular_fat_volume_cm3']:.2f} cm³",
            f" - Relación grasa IM / músculo:          {total_im_ratio}",
            f" - Volumen grasa subcutánea / músculos:  {global_fat_metrics['total_subcutaneous_fat_volume_cm3']:.2f} cm³",
            f" - Relación grasa subcutánea / músculo:  {total_sc_ratio}",
            f" - Volumen total de grasa:               {global_fat_metrics['total_fat_volume_cm3']:.2f} cm³",
            f" - Relación grasa total / músculo:       {total_fat_ratio}",
            "-" * 74,
            "",
        ]
    )

    report_lines.extend(
        [
            "CONTROL DE CALIDAD",
            "=" * 74,
            "Los landmarks son estimaciones automáticas basadas en perfiles óseos longitudinales",
            "y geometría del contorno. Las interfaces cutáneas y la fascia profunda son",
            "estimaciones radiológicas asistidas por wavelets y umbrales HU.",
            "Deben revisarse en 3D Slicer antes de usar los valores como medición clínica",
            "o resultado definitivo de investigación.",
        ]
    )

    report_path = os.path.join(out_dir, REPORT_FILENAME)
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines))

    muscle_tracer_path = os.path.join(out_dir, MUSCLE_TRACER_FILENAME)
    save_nifti(muscle_tracer.astype(np.float32), affine, muscle_tracer_path)

    landmark_labelmap = build_landmark_labelmap(image_hu.shape, landmarks)
    landmark_label_path = os.path.join(out_dir, LANDMARK_LABEL_FILENAME)
    save_nifti(landmark_labelmap.astype(np.uint8), affine, landmark_label_path)

    landmark_overlay = build_landmark_overlay(image_hu, landmarks)
    landmark_overlay_path = os.path.join(out_dir, LANDMARK_OVERLAY_FILENAME)
    save_nifti(landmark_overlay.astype(np.float32), affine, landmark_overlay_path)

    labels_path = os.path.join(out_dir, LABELS_FILENAME)
    export_labels_description(labels_path)

    tissue_labelmap = build_tissue_labelmap(image_hu.shape, tissue_composition)
    tissue_label_path = os.path.join(out_dir, TISSUE_LABEL_FILENAME)
    save_nifti(tissue_labelmap.astype(np.uint8), affine, tissue_label_path)

    tissue_overlay = build_tissue_overlay(image_hu, tissue_composition)
    tissue_overlay_path = os.path.join(out_dir, TISSUE_OVERLAY_FILENAME)
    save_nifti(tissue_overlay.astype(np.float32), affine, tissue_overlay_path)

    tissue_labels_path = os.path.join(out_dir, TISSUE_LABELS_FILENAME)
    export_tissue_labels_description(tissue_labels_path)

    intramuscular_labelmap_path = os.path.join(
        out_dir, INTRAMUSCULAR_FAT_LABEL_FILENAME
    )
    save_nifti(
        intramuscular_metrics["intramuscular_labelmap"].astype(np.uint8),
        affine,
        intramuscular_labelmap_path,
    )

    intramuscular_overlay_path = os.path.join(
        out_dir, INTRAMUSCULAR_FAT_OVERLAY_FILENAME
    )
    intramuscular_overlay = build_intramuscular_overlay(
        image_hu,
        intramuscular_metrics["intramuscular_labelmap"],
    )
    save_nifti(
        intramuscular_overlay.astype(np.float32),
        affine,
        intramuscular_overlay_path,
    )
    del intramuscular_overlay

    intramuscular_labels_path = os.path.join(
        out_dir, INTRAMUSCULAR_FAT_LABELS_FILENAME
    )
    export_intramuscular_labels_description(intramuscular_labels_path)

    intramuscular_area_csv_path = os.path.join(
        out_dir, INTRAMUSCULAR_FAT_AREA_CSV_FILENAME
    )
    export_intramuscular_area_csv(
        intramuscular_area_csv_path,
        intramuscular_metrics["area_rows"],
    )

    subcutaneous_labelmap = subcutaneous_volume_results.get("_subcutaneous_labelmap", {}).get("labelmap")
    if subcutaneous_labelmap is None:
        subcutaneous_labelmap = np.zeros_like(image_hu, dtype=np.uint8)
    subcutaneous_labelmap_path = os.path.join(out_dir, SUBCUTANEOUS_FAT_LABEL_FILENAME)
    save_nifti(subcutaneous_labelmap.astype(np.uint8), affine, subcutaneous_labelmap_path)

    total_adipose_labelmap = build_total_adipose_labelmap(
        subcutaneous_labelmap.astype(np.uint8),
        intramuscular_metrics["intramuscular_labelmap"].astype(np.uint8),
    )
    total_adipose_labelmap_path = os.path.join(out_dir, TOTAL_ADIPOSE_LABEL_FILENAME)
    save_nifti(total_adipose_labelmap.astype(np.uint8), affine, total_adipose_labelmap_path)

    adipose_labels_path = os.path.join(out_dir, ADIPOSE_LABELS_FILENAME)
    export_adipose_labels_description(adipose_labels_path)

    adipose_metrics_csv_path = os.path.join(out_dir, ADIPOSE_METRICS_CSV_FILENAME)
    export_adipose_metrics_csv(
        adipose_metrics_csv_path,
        tissue_composition=tissue_composition,
        intramuscular_metrics=intramuscular_metrics,
        subcutaneous_volume_results=subcutaneous_volume_results,
        global_fat_metrics=global_fat_metrics,
    )

    tissue_png_paths = export_tissue_pngs(
        image_hu=image_hu,
        tissue_composition=tissue_composition,
        landmarks=landmarks,
        out_dir=out_dir,
    )

    print(f"-> Reporte:                    {report_path}")
    print(f"-> Trazado muscular:           {muscle_tracer_path}")
    print(f"-> Labelmap de landmarks:      {landmark_label_path}")
    print(f"-> TAC con landmarks:          {landmark_overlay_path}")
    print(f"-> Etiquetas de landmarks:     {labels_path}")
    print(f"-> Labelmap de composición:    {tissue_label_path}")
    print(f"-> TAC con composición:        {tissue_overlay_path}")
    print(f"-> Etiquetas de composición:   {tissue_labels_path}")
    print(f"-> Labelmap grasa intramusc.:  {intramuscular_labelmap_path}")
    print(f"-> TAC grasa intramuscular:    {intramuscular_overlay_path}")
    print(f"-> Etiquetas grasa IM:         {intramuscular_labels_path}")
    print(f"-> Áreas grasa IM por corte:   {intramuscular_area_csv_path}")
    print(f"-> Labelmap grasa subcutánea:  {subcutaneous_labelmap_path}")
    print(f"-> Labelmap tejido adiposo:    {total_adipose_labelmap_path}")
    print(f"-> Etiquetas tejido adiposo:   {adipose_labels_path}")
    print(f"-> Métricas tejido adiposo:    {adipose_metrics_csv_path}")
    for key, png_path in tissue_png_paths.items():
        if key == "combined":
            print(f"-> PNG resumen composición:    {png_path}")
        else:
            side_label = tissue_composition[key]["patient_side"] if tissue_composition.get(key) is not None else key
            print(f"-> PNG composición {side_label}:    {png_path}")

    return {
        "report": report_path,
        "muscle_tracer": muscle_tracer_path,
        "landmark_labelmap": landmark_label_path,
        "landmark_overlay": landmark_overlay_path,
        "landmark_labels": labels_path,
        "tissue_labelmap": tissue_label_path,
        "tissue_overlay": tissue_overlay_path,
        "tissue_labels": tissue_labels_path,
        "intramuscular_labelmap": intramuscular_labelmap_path,
        "intramuscular_overlay": intramuscular_overlay_path,
        "intramuscular_labels": intramuscular_labels_path,
        "intramuscular_area_csv": intramuscular_area_csv_path,
        "subcutaneous_labelmap": subcutaneous_labelmap_path,
        "total_adipose_labelmap": total_adipose_labelmap_path,
        "adipose_labels": adipose_labels_path,
        "adipose_metrics_csv": adipose_metrics_csv_path,
        "tissue_pngs": tissue_png_paths,
    }


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # 1. Leer imágenes y conservar la geometría DICOM.
    image_hu, pixel_spacing, slice_spacing, affine = load_dicom_series(INPUT_DIR)

    # 2. Auto-detectar el rango longitudinal útil para segmentación muscular.
    z_start, z_end = auto_detect_z_limits(image_hu)

    # 3. Buscar landmarks en TODO el TAC para no excluir cadera o tibia.
    landmarks = detect_bony_landmarks(
        image_hu,
        affine,
        z_start=0,
        z_end=image_hu.shape[2],
    )

    # 3.5 QC v3.21.5: antes de calcular composición, exige que los cortes medios
    # tengan dos fémures. Si no, los ajusta al corte cercano más confiable.
    qc_records = scan_volume_two_femur_qc(image_hu)
    landmarks = enforce_midpoint_slices_have_two_femurs(image_hu, landmarks, qc_records)

    # 4. Garantizar que los cortes medios calculados queden segmentados.
    midpoint_slices = [
        info["midpoint_slice"]
        for info in landmarks.values()
        if info["midpoint_slice"] is not None
    ]
    if midpoint_slices:
        z_start = max(
            0,
            min(z_start, min(midpoint_slices) - MUSCLE_MIDPOINT_MARGIN_SLICES),
        )
        z_end = min(
            image_hu.shape[2],
            max(z_end, max(midpoint_slices) + MUSCLE_MIDPOINT_MARGIN_SLICES + 1),
        )
        print(f"Rango muscular final ajustado: Z={z_start} a Z={z_end - 1}")

    # 5. Segmentar vastos laterales y mediales de ambos lados.
    volume_masks, compartment_masks = segment_all_muscles(image_hu, pixel_spacing, z_start, z_end)

    # 6. Analizar epidermis/dermis, grasa subcutánea y músculo total en cada corte medio.
    tissue_composition = analyze_midpoint_tissue_composition(
        image_hu=image_hu,
        pixel_spacing=pixel_spacing,
        landmarks=landmarks,
    )

    # 7. Calcular grasa intramuscular por músculo y volúmenes de grasa subcutánea.
    intramuscular_metrics = compute_intramuscular_fat_metrics(
        image_hu=image_hu,
        compartment_masks=compartment_masks,
        pixel_spacing=pixel_spacing,
        slice_spacing=slice_spacing,
    )
    subcutaneous_volume_results = analyze_subcutaneous_fat_volume_all_slices(
        image_hu=image_hu,
        pixel_spacing=pixel_spacing,
        slice_spacing=slice_spacing,
        z_start=z_start,
        z_end=z_end,
        landmarks=landmarks,
    )
    global_fat_metrics = combine_global_fat_metrics(
        intramuscular_metrics=intramuscular_metrics,
        subcutaneous_volume_results=subcutaneous_volume_results,
    )

    # 8. Reportar áreas, relaciones y exportar todos los NIfTI.
    export_and_report_all(
        image_hu=image_hu,
        masks_dict=volume_masks,
        compartment_masks=compartment_masks,
        pixel_spacing=pixel_spacing,
        slice_spacing=slice_spacing,
        affine=affine,
        out_dir=OUTPUT_DIR,
        z_start=z_start,
        z_end=z_end,
        landmarks=landmarks,
        tissue_composition=tissue_composition,
        intramuscular_metrics=intramuscular_metrics,
        subcutaneous_volume_results=subcutaneous_volume_results,
        global_fat_metrics=global_fat_metrics,
    )

    print("\nPipeline integrado finalizado con éxito.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error en el pipeline: {exc}")
        traceback.print_exc()
