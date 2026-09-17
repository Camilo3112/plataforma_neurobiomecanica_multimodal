"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CORRECCIÓN AUTOMÁTICA DE ROI CORTICAL                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: tools/corregir_roi_cortical_automatico.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.


-----------------------------------------
Corrige máscaras corticales proyectándolas sobre una envolvente cortical
estimada.
El núcleo matemático usa conjuntos voxelizados Ω, operaciones morfológicas
Ω⊕B y Ω⊖B, y búsqueda de traslación que maximiza solapamiento tipo Dice:
Dice(A,B)=2|A∩B|/(|A|+|B|). El remuestreo conserva la grilla anatómica para
que
la ROI corregida pueda compararse con el T1 del paciente.

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

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import (
    affine_transform,
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    label,
)


# =============================================================================
# RUTAS PREDETERMINADAS DEL PACIENTE 3
# =============================================================================
DEFAULT_BRAIN_PATH = Path("/home/humath/Escritorio/datos/paciente 3/Antes/ResultadosFuncional/REFORMATEO/Brain00mm.nii")

DEFAULT_MASK_PATH = Path("/home/humath/Escritorio/resultados/paciente 3/Antes/morfometria/interna/internal_corteza_motora_primaria_izquierda_roi_shell.nii.gz")


# =============================================================================
# PARÁMETROS DEL ALGORITMO
# =============================================================================
MASK_THRESHOLD = 0.5
CORTICAL_SHELL_THICKNESS_MM = 5.0
MAX_TRANSLATION_MM = 65.0
COARSE_STEP_MM = 4.0
FINE_RADIUS_VOXELS = 5
MAX_SEARCH_POINTS = 5000
RANDOM_SEED = 2026

# Escalado automático desactivado por defecto porque una máscara aislada puede
# mejorar artificialmente su puntuación al reducirse. Active --auto-scale solo
# cuando haya evidencia real de un problema de tamaño.
DEFAULT_SCALE_CANDIDATES = (1.0,)
OPTIONAL_SCALE_CANDIDATES = (0.90, 0.95, 1.00, 1.05, 1.10)


@dataclass
class CandidateResult:
    score: float
    shift_vox: tuple[int, int, int]
    scale: float
    closeness: float
    inside_fraction: float
    shell_overlap: float
    outside_penalty: float
    valid_fraction: float
    shift_mm_norm: float
    hemisphere_penalty: float


# =============================================================================
# UTILIDADES
# =============================================================================
def ensure_3d(data: np.ndarray, name: str) -> np.ndarray:
    """Acepta 3D o 4D con un único volumen y devuelve un arreglo 3D."""
    data = np.asarray(data)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{name} debe ser 3D. Forma encontrada: {data.shape}")
    return data


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    labels, number = label(mask)
    if number == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def otsu_threshold(values: np.ndarray, bins: int = 512) -> float:
    """Implementación NumPy de Otsu para evitar una dependencia adicional."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0

    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float(lo if np.isfinite(lo) else 0.0)

    clipped = np.clip(values, lo, hi)
    hist, edges = np.histogram(clipped, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0

    probability = hist.astype(np.float64)
    total = probability.sum()
    if total <= 0:
        return float(lo)
    probability /= total

    omega = np.cumsum(probability)
    mu = np.cumsum(probability * centers)
    mu_total = mu[-1]

    denominator = omega * (1.0 - omega)
    between = np.zeros_like(denominator)
    valid = denominator > 1e-12
    between[valid] = ((mu_total * omega[valid] - mu[valid]) ** 2) / denominator[valid]

    return float(centers[int(np.argmax(between))])


def border_values(volume: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            volume[0, :, :].ravel(),
            volume[-1, :, :].ravel(),
            volume[:, 0, :].ravel(),
            volume[:, -1, :].ravel(),
            volume[:, :, 0].ravel(),
            volume[:, :, -1].ravel(),
        ]
    )


def create_brain_mask(brain: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Segmenta el soporte cerebral.

    Si los bordes son prácticamente constantes, utiliza el fondo de los bordes.
    En caso contrario utiliza Otsu sobre la distancia respecto al fondo.
    """
    brain = np.asarray(brain, dtype=np.float32)
    finite = np.isfinite(brain)
    if not np.any(finite):
        raise ValueError("El archivo del cerebro no contiene valores finitos.")

    border = border_values(brain)
    border = border[np.isfinite(border)]
    background = float(np.median(border)) if border.size else 0.0

    absolute_difference = np.abs(brain - background)
    finite_difference = absolute_difference[finite]
    p99 = float(np.percentile(finite_difference, 99.0))
    epsilon = max(1e-6, p99 * 1e-4)

    border_near_background = float(
        np.mean(np.abs(border - background) <= epsilon)
    ) if border.size else 0.0

    if border_near_background >= 0.80:
        threshold = epsilon
        method = "fondo_constante_en_bordes"
    else:
        threshold = otsu_threshold(finite_difference)
        method = "otsu"

    mask = finite & (absolute_difference > threshold)
    mask = largest_connected_component(mask)
    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)
    mask = largest_connected_component(mask)

    if int(mask.sum()) == 0:
        raise ValueError("No fue posible detectar el volumen cerebral.")

    report = {
        "metodo_segmentacion": method,
        "fondo_estimado": background,
        "umbral": float(threshold),
        "fraccion_borde_fondo": border_near_background,
        "voxeles_cerebro": int(mask.sum()),
    }
    return mask.astype(bool), report


def bounding_box(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("La máscara está vacía.")
    return coordinates.min(axis=0), coordinates.max(axis=0)


def voxel_sizes_from_affine(affine: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.asarray(affine[:3, :3], dtype=float) ** 2, axis=0))


def apply_voxel_transform_to_points(
    points: np.ndarray,
    center: np.ndarray,
    shift_vox: Iterable[float],
    scale: float,
) -> np.ndarray:
    shift_vox = np.asarray(tuple(shift_vox), dtype=float)
    return (points - center) * float(scale) + center + shift_vox


def hemisphere_side(center_vox: np.ndarray, affine: np.ndarray, brain_center_vox: np.ndarray) -> int:
    center_world = nib.affines.apply_affine(affine, center_vox)
    brain_center_world = nib.affines.apply_affine(affine, brain_center_vox)
    delta = float(center_world[0] - brain_center_world[0])
    if abs(delta) < 5.0:
        return 0
    return -1 if delta < 0 else 1


# =============================================================================
# BÚSQUEDA AUTOMÁTICA
# =============================================================================
def evaluate_candidate(
    sample_points: np.ndarray,
    roi_center: np.ndarray,
    shift_vox: tuple[int, int, int],
    scale: float,
    volume_shape: np.ndarray,
    brain_mask: np.ndarray,
    cortical_shell: np.ndarray,
    distance_to_shell: np.ndarray,
    distance_to_brain: np.ndarray,
    brain_affine: np.ndarray,
    maximum_translation_mm: float,
    original_hemisphere: int,
    brain_center_vox: np.ndarray,
) -> CandidateResult:
    transformed = apply_voxel_transform_to_points(
        sample_points,
        center=roi_center,
        shift_vox=shift_vox,
        scale=scale,
    )
    indices = np.rint(transformed).astype(np.int32)

    valid = np.all((indices >= 0) & (indices < volume_shape), axis=1)
    valid_fraction = float(np.mean(valid))
    if valid_fraction < 0.80:
        return CandidateResult(
            score=-1e9,
            shift_vox=shift_vox,
            scale=scale,
            closeness=0.0,
            inside_fraction=0.0,
            shell_overlap=0.0,
            outside_penalty=1.0,
            valid_fraction=valid_fraction,
            shift_mm_norm=float("inf"),
            hemisphere_penalty=1.0,
        )

    idx = indices[valid]
    index_tuple = tuple(idx.T)

    shell_distance = distance_to_shell[index_tuple]
    brain_distance = distance_to_brain[index_tuple]
    inside = brain_mask[index_tuple]
    on_shell = cortical_shell[index_tuple]

    # Cercanía suave a la corteza: 1 en la capa y disminuye con la distancia.
    closeness = float(np.mean(np.exp(-0.5 * (shell_distance / 2.5) ** 2)))
    inside_fraction = float(np.mean(inside))
    shell_overlap = float(np.mean(on_shell))
    outside_penalty = float(np.mean(np.clip(brain_distance / 8.0, 0.0, 1.0)))

    shift_vox_array = np.asarray(shift_vox, dtype=float)
    shift_mm_vector = brain_affine[:3, :3] @ shift_vox_array
    shift_mm_norm = float(np.linalg.norm(shift_mm_vector))

    candidate_center = roi_center + shift_vox_array
    candidate_side = hemisphere_side(candidate_center, brain_affine, brain_center_vox)
    hemisphere_penalty = float(
        original_hemisphere != 0
        and candidate_side != 0
        and candidate_side != original_hemisphere
    )

    scale_penalty = abs(math.log(float(scale)))
    translation_penalty = shift_mm_norm / max(float(maximum_translation_mm), 1.0)

    # La prioridad principal es acercarse a la capa cortical y permanecer en ella.
    score = (
        0.55 * closeness
        + 0.25 * inside_fraction
        + 0.20 * shell_overlap
        - 0.25 * outside_penalty
        - 0.035 * translation_penalty
        - 0.08 * scale_penalty
        - 0.50 * hemisphere_penalty
    )

    return CandidateResult(
        score=float(score),
        shift_vox=tuple(int(v) for v in shift_vox),
        scale=float(scale),
        closeness=closeness,
        inside_fraction=inside_fraction,
        shell_overlap=shell_overlap,
        outside_penalty=outside_penalty,
        valid_fraction=valid_fraction,
        shift_mm_norm=shift_mm_norm,
        hemisphere_penalty=hemisphere_penalty,
    )


def inclusive_range(start: int, stop: int, step: int) -> list[int]:
    if stop < start:
        return []
    values = list(range(int(start), int(stop) + 1, max(1, int(step))))
    if values and values[-1] != stop:
        values.append(int(stop))
    if start <= 0 <= stop and 0 not in values:
        values.append(0)
    return sorted(set(values))


def automatic_search(
    roi_mask: np.ndarray,
    brain_mask: np.ndarray,
    brain_affine: np.ndarray,
    cortical_shell_thickness_mm: float,
    maximum_translation_mm: float,
    coarse_step_mm: float,
    fine_radius_voxels: int,
    scale_candidates: tuple[float, ...],
) -> tuple[CandidateResult, dict]:
    volume_shape = np.asarray(brain_mask.shape, dtype=int)
    zooms = voxel_sizes_from_affine(brain_affine)

    inside_distance = distance_transform_edt(brain_mask, sampling=zooms)
    cortical_shell = brain_mask & (inside_distance <= cortical_shell_thickness_mm)
    distance_to_shell = distance_transform_edt(~cortical_shell, sampling=zooms)
    distance_to_brain = distance_transform_edt(~brain_mask, sampling=zooms)

    all_points = np.argwhere(roi_mask).astype(np.float64)
    if all_points.size == 0:
        raise ValueError("La máscara de la región está vacía.")

    roi_center = np.mean(all_points, axis=0)
    rng = np.random.default_rng(RANDOM_SEED)
    if len(all_points) > MAX_SEARCH_POINTS:
        selected = rng.choice(len(all_points), MAX_SEARCH_POINTS, replace=False)
        sample_points = all_points[selected]
    else:
        sample_points = all_points

    brain_min, brain_max = bounding_box(brain_mask)
    roi_min, roi_max = bounding_box(roi_mask)
    brain_center_vox = (brain_min + brain_max) / 2.0
    original_hemisphere = hemisphere_side(roi_center, brain_affine, brain_center_vox)

    max_shift_vox = np.ceil(maximum_translation_mm / zooms).astype(int)
    margin_vox = np.ceil(10.0 / zooms).astype(int)

    # Limita la búsqueda a traslaciones capaces de acercar la ROI al cerebro.
    lower = np.maximum(
        -max_shift_vox,
        np.floor(brain_min - margin_vox - roi_max).astype(int),
    )
    upper = np.minimum(
        max_shift_vox,
        np.ceil(brain_max + margin_vox - roi_min).astype(int),
    )

    if np.any(lower > upper):
        raise ValueError(
            "El rango máximo de traslación es insuficiente para acercar la máscara "
            "al cerebro. Aumente --max-mm."
        )

    coarse_step_vox = np.maximum(1, np.rint(coarse_step_mm / zooms).astype(int))
    ranges = [
        inclusive_range(lower[i], upper[i], coarse_step_vox[i])
        for i in range(3)
    ]

    best: CandidateResult | None = None
    best_by_scale: dict[str, dict] = {}

    for scale in scale_candidates:
        best_for_scale: CandidateResult | None = None
        for dx in ranges[0]:
            for dy in ranges[1]:
                for dz in ranges[2]:
                    result = evaluate_candidate(
                        sample_points=sample_points,
                        roi_center=roi_center,
                        shift_vox=(dx, dy, dz),
                        scale=scale,
                        volume_shape=volume_shape,
                        brain_mask=brain_mask,
                        cortical_shell=cortical_shell,
                        distance_to_shell=distance_to_shell,
                        distance_to_brain=distance_to_brain,
                        brain_affine=brain_affine,
                        maximum_translation_mm=maximum_translation_mm,
                        original_hemisphere=original_hemisphere,
                        brain_center_vox=brain_center_vox,
                    )
                    if best_for_scale is None or result.score > best_for_scale.score:
                        best_for_scale = result
                    if best is None or result.score > best.score:
                        best = result

        if best_for_scale is not None:
            best_by_scale[str(scale)] = asdict(best_for_scale)

    if best is None:
        raise RuntimeError("La búsqueda gruesa no produjo ninguna posición válida.")

    # Búsqueda fina a resolución de un vóxel alrededor del mejor candidato.
    cx, cy, cz = best.shift_vox
    radius = max(1, int(fine_radius_voxels))
    fine_scales = sorted(
        set(
            [
                best.scale,
                1.0,
                max(0.80, best.scale - 0.025),
                min(1.20, best.scale + 0.025),
            ]
        )
    ) if len(scale_candidates) > 1 else [1.0]

    for scale in fine_scales:
        for dx in range(cx - radius, cx + radius + 1):
            for dy in range(cy - radius, cy + radius + 1):
                for dz in range(cz - radius, cz + radius + 1):
                    if np.any(np.array([dx, dy, dz]) < lower) or np.any(
                        np.array([dx, dy, dz]) > upper
                    ):
                        continue
                    result = evaluate_candidate(
                        sample_points=sample_points,
                        roi_center=roi_center,
                        shift_vox=(dx, dy, dz),
                        scale=scale,
                        volume_shape=volume_shape,
                        brain_mask=brain_mask,
                        cortical_shell=cortical_shell,
                        distance_to_shell=distance_to_shell,
                        distance_to_brain=distance_to_brain,
                        brain_affine=brain_affine,
                        maximum_translation_mm=maximum_translation_mm,
                        original_hemisphere=original_hemisphere,
                        brain_center_vox=brain_center_vox,
                    )
                    if result.score > best.score:
                        best = result

    diagnostics = {
        "zooms_mm": zooms.tolist(),
        "roi_center_vox_initial": roi_center.tolist(),
        "brain_bbox_vox": [brain_min.tolist(), brain_max.tolist()],
        "roi_bbox_vox_initial": [roi_min.tolist(), roi_max.tolist()],
        "search_lower_vox": lower.tolist(),
        "search_upper_vox": upper.tolist(),
        "coarse_step_vox": coarse_step_vox.tolist(),
        "sampled_roi_points": int(len(sample_points)),
        "original_hemisphere": int(original_hemisphere),
        "best_by_scale": best_by_scale,
    }
    return best, diagnostics


# =============================================================================
# APLICACIÓN DE LA TRANSFORMACIÓN
# =============================================================================
def transform_binary_mask(
    mask: np.ndarray,
    output_shape: tuple[int, int, int],
    center_vox: np.ndarray,
    shift_vox: tuple[int, int, int],
    scale: float,
) -> np.ndarray:
    """
    Aplica la transformación directa:
        salida = escala * (entrada - centro) + centro + desplazamiento

    scipy.affine_transform requiere la transformación inversa salida -> entrada.
    """
    inverse_matrix = np.eye(3, dtype=float) / float(scale)
    shift = np.asarray(shift_vox, dtype=float)
    inverse_offset = center_vox - (center_vox + shift) / float(scale)

    corrected = affine_transform(
        input=mask.astype(np.uint8),
        matrix=inverse_matrix,
        offset=inverse_offset,
        output_shape=output_shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    return corrected > 0


def voxel_transform_matrix(center_vox: np.ndarray, shift_vox: np.ndarray, scale: float) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.eye(3) * float(scale)
    matrix[:3, 3] = center_vox + shift_vox - float(scale) * center_vox
    return matrix


def quality_label(result: CandidateResult) -> str:
    if result.shell_overlap >= 0.55 and result.inside_fraction >= 0.75:
        return "alta"
    if result.shell_overlap >= 0.30 and result.inside_fraction >= 0.50:
        return "media"
    return "baja"


# =============================================================================
# VISUALIZACIÓN
# =============================================================================
def intensity_limits(brain: np.ndarray) -> tuple[float, float]:
    valid = brain[np.isfinite(brain)]
    valid = valid[valid != 0]
    if valid.size == 0:
        return float(np.nanmin(brain)), float(np.nanmax(brain))
    return tuple(float(v) for v in np.percentile(valid, [2, 98]))


def mask_center(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(mask)
    if points.size == 0:
        return np.asarray(mask.shape, dtype=float) / 2.0
    return np.mean(points, axis=0)


def save_preview(
    brain: np.ndarray,
    original: np.ndarray,
    corrected: np.ndarray,
    output_path: Path,
) -> None:
    vmin, vmax = intensity_limits(brain)
    figure, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row, (mask, name) in enumerate(
        [(original, "Original remuestreada"), (corrected, "Corregida automática")]
    ):
        center = np.rint(mask_center(mask)).astype(int)
        center = np.clip(center, 0, np.asarray(brain.shape) - 1)
        x, y, z = center

        views = [
            (brain[x, :, :], mask[x, :, :], f"{name} — sagital x={x}"),
            (brain[:, y, :], mask[:, y, :], f"{name} — coronal y={y}"),
            (brain[:, :, z], mask[:, :, z], f"{name} — axial z={z}"),
        ]

        for column, (brain_slice, mask_slice, title) in enumerate(views):
            axis = axes[row, column]
            axis.imshow(np.rot90(brain_slice), cmap="gray", vmin=vmin, vmax=vmax)
            overlay = np.ma.masked_where(mask_slice <= 0, mask_slice)
            axis.imshow(
                np.rot90(overlay),
                cmap="autumn",
                alpha=0.65,
                interpolation="nearest",
            )
            axis.set_title(title)
            axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alinea automáticamente una máscara cortical con Brain00mm."
    )
    parser.add_argument("--brain", type=Path, default=DEFAULT_BRAIN_PATH)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK_PATH)
    parser.add_argument(
        "--max-mm",
        type=float,
        default=MAX_TRANSLATION_MM,
        help="Desplazamiento máximo buscado en milímetros.",
    )
    parser.add_argument(
        "--shell-mm",
        type=float,
        default=CORTICAL_SHELL_THICKNESS_MM,
        help="Espesor de la capa cortical objetivo en milímetros.",
    )
    parser.add_argument(
        "--auto-scale",
        action="store_true",
        help="Prueba escalas isotrópicas entre 0.90 y 1.10. Úselo solo si hay evidencia de tamaño incorrecto.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta opcional del NIfTI corregido.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    brain_path = args.brain.expanduser()
    mask_path = args.mask.expanduser()

    if not brain_path.exists():
        raise FileNotFoundError(f"No existe el cerebro: {brain_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"No existe la máscara: {mask_path}")

    output_path = args.output or mask_path.with_name(
        f"{mask_path.name.removesuffix('.gz').removesuffix('.nii')}_corregida_auto.nii.gz"
    )
    preview_path = output_path.with_name(f"{output_path.stem.removesuffix('.nii')}_preview.png")
    report_path = output_path.with_name(f"{output_path.stem.removesuffix('.nii')}_reporte.json")

    print("\nCargando archivos...")
    brain_img = nib.load(str(brain_path))
    mask_img = nib.load(str(mask_path))

    brain = ensure_3d(np.asarray(brain_img.dataobj, dtype=np.float32), "El cerebro")
    original_mask_data = ensure_3d(
        np.asarray(mask_img.dataobj, dtype=np.float32), "La máscara"
    )

    # Primero respeta las coordenadas físicas declaradas en ambos encabezados.
    mask_binary_img = nib.Nifti1Image(
        (original_mask_data > MASK_THRESHOLD).astype(np.uint8),
        mask_img.affine,
        header=mask_img.header.copy(),
    )
    resampled_mask_img = resample_from_to(
        mask_binary_img,
        (brain_img.shape[:3], brain_img.affine),
        order=0,
        mode="constant",
        cval=0.0,
    )
    resampled_mask = ensure_3d(
        np.asarray(resampled_mask_img.dataobj) > 0.5,
        "La máscara remuestreada",
    )

    if int(resampled_mask.sum()) == 0:
        raise ValueError(
            "La máscara quedó vacía al remuestrearla al espacio del cerebro. "
            "Revise las matrices affine o aumente el campo de visión."
        )

    print("Detectando el volumen cerebral...")
    brain_mask, segmentation_report = create_brain_mask(brain)

    scales = OPTIONAL_SCALE_CANDIDATES if args.auto_scale else DEFAULT_SCALE_CANDIDATES
    print("Buscando la mejor posición cortical automáticamente...")
    best, search_report = automatic_search(
        roi_mask=resampled_mask,
        brain_mask=brain_mask,
        brain_affine=brain_img.affine,
        cortical_shell_thickness_mm=float(args.shell_mm),
        maximum_translation_mm=float(args.max_mm),
        coarse_step_mm=COARSE_STEP_MM,
        fine_radius_voxels=FINE_RADIUS_VOXELS,
        scale_candidates=scales,
    )

    initial_center = np.mean(np.argwhere(resampled_mask), axis=0)
    corrected = transform_binary_mask(
        mask=resampled_mask,
        output_shape=tuple(int(v) for v in brain.shape),
        center_vox=initial_center,
        shift_vox=best.shift_vox,
        scale=best.scale,
    )

    if int(corrected.sum()) == 0:
        raise RuntimeError("La transformación calculada produjo una máscara vacía.")

    # Matrices y desplazamiento en coordenadas físicas RAS+.
    shift_vox_array = np.asarray(best.shift_vox, dtype=float)
    shift_mm_ras = brain_img.affine[:3, :3] @ shift_vox_array
    transform_vox = voxel_transform_matrix(initial_center, shift_vox_array, best.scale)
    transform_world = brain_img.affine @ transform_vox @ np.linalg.inv(brain_img.affine)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_header = brain_img.header.copy()
    output_header.set_data_dtype(np.uint8)
    output_img = nib.Nifti1Image(
        corrected.astype(np.uint8),
        brain_img.affine,
        header=output_header,
    )

    qform, qcode = brain_img.get_qform(coded=True)
    sform, scode = brain_img.get_sform(coded=True)
    if qform is not None:
        output_img.set_qform(qform, int(qcode or 1))
    if sform is not None:
        output_img.set_sform(sform, int(scode or 1))

    nib.save(output_img, str(output_path))
    save_preview(brain, resampled_mask, corrected, preview_path)

    report = {
        "brain_path": str(brain_path),
        "mask_path": str(mask_path),
        "output_path": str(output_path),
        "preview_path": str(preview_path),
        "brain_shape": list(brain.shape),
        "mask_shape_original": list(original_mask_data.shape),
        "mask_voxels_resampled": int(resampled_mask.sum()),
        "mask_voxels_corrected": int(corrected.sum()),
        "segmentation": segmentation_report,
        "automatic_result": asdict(best),
        "quality": quality_label(best),
        "translation_voxels_brain_grid": list(best.shift_vox),
        "translation_mm_RAS": shift_mm_ras.tolist(),
        "isotropic_scale": float(best.scale),
        "voxel_transform_matrix": transform_vox.tolist(),
        "world_transform_matrix_RAS": transform_world.tolist(),
        "search": search_report,
        "warning": (
            "Validar visualmente. El ajuste geométrico a la superficie cortical no "
            "sustituye un registro anatómico atlas-paciente."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ RESULTADO ================")
    print(f"Traslación en vóxeles: {best.shift_vox}")
    print(f"Traslación en mm RAS+: {np.round(shift_mm_ras, 3).tolist()}")
    print(f"Escala isotrópica: {best.scale:.4f}")
    print(f"Puntuación: {best.score:.4f}")
    print(f"Solapamiento con capa cortical: {best.shell_overlap:.3f}")
    print(f"Fracción dentro del cerebro: {best.inside_fraction:.3f}")
    print(f"Calidad geométrica estimada: {quality_label(best)}")
    print(f"\nMáscara corregida: {output_path}")
    print(f"Vista previa: {preview_path}")
    print(f"Reporte: {report_path}")
    print("\nRevise siempre la vista previa antes de usar la máscara en el análisis final.")


if __name__ == "__main__":
    main()
