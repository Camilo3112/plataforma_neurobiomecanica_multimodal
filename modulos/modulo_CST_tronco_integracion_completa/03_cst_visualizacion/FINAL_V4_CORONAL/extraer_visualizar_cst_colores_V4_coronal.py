"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  EXTRAER VISUALIZAR CST COLORES V4 CORONAL                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: modulos/modulo_CST_tronco_integracion_completa/03_cst_visualizacion/FINAL_V4_CORONAL/extraer_visualizar_cst_colores_V4_coronal.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Extrae, fusiona y visualiza la vía corticoespinal. Las streamlines son curvas
3D discretas filtradas por intersección con máscaras anatómicas. El color RGB
se
calcula por orientación local: c = |dr/ds| normalizado, donde componentes
R,G,B
representan ejes izquierda-derecha, anterior-posterior y superior-inferior. La
densidad voxelizada cuenta el número de fibras que atraviesan cada voxel.

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

import json
import math
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    import nibabel as nib
    import numpy as np
    from nibabel.affines import apply_affine
    from nibabel.processing import resample_from_to
    from nibabel.streamlines import Tractogram, TrkFile
    from nibabel.streamlines.trk import Field
    from scipy.ndimage import distance_transform_edt
except ImportError as exc:
    print("\nFalta una dependencia de Python:", exc)
    print("\nInstálalas con:")
    print("py -m pip install numpy scipy nibabel matplotlib")
    raise SystemExit(1)


# ======================================================================
# CONFIGURACIÓN
# ======================================================================

PROJECT_ROOT = Path(os.environ.get("VCE_PROJECT_ROOT", r"D:\EAFIT\01-2026\proyecto"))
PATIENT_ID = os.environ.get("VCE_PATIENT_ID", "paciente 6")
STAGE_NAME = os.environ.get("VCE_STAGE", "Antes")

PATIENT_DATA = PROJECT_ROOT / "datos" / PATIENT_ID / STAGE_NAME
PATIENT_RESULTS = PROJECT_ROOT / "resultados" / PATIENT_ID / STAGE_NAME

# Permite sobreescribir rutas para integrarlo a la suite mayor.
_t1_env = os.environ.get("VCE_T1_PATH", "").strip()
if _t1_env:
    T1_PATH = Path(_t1_env)
else:
    T1_PATH = PATIENT_DATA / "Resultados funcional" / "REFORMATEO" / "rT1.nii"
    if not T1_PATH.exists():
        alt = PATIENT_DATA / "ResultadosFuncional" / "REFORMATEO" / "rT1.nii"
        if alt.exists():
            T1_PATH = alt

TRACTOGRAPHY_DIR = Path(os.environ.get("VCE_TRACTOGRAPHY_DIR", str(PATIENT_RESULTS / "tractografia_propia")))
WHOLEBRAIN_TRK = Path(os.environ.get("VCE_WHOLEBRAIN_TRK", str(TRACTOGRAPHY_DIR / "wholebrain_T1.trk")))
BRAINSTEM_DIR = Path(os.environ.get("VCE_BRAINSTEM_DIR", str(TRACTOGRAPHY_DIR / "brainstem_substructures_validated")))

MIDBRAIN_PATH = BRAINSTEM_DIR / "midbrain_T1.nii.gz"
PONS_PATH = BRAINSTEM_DIR / "pons_T1.nii.gz"
MEDULLA_PATH = BRAINSTEM_DIR / "medulla_T1.nii.gz"

MOTOR_MASK_DIR = Path(os.environ.get(
    "VCE_MOTOR_MASK_DIR",
    str(PATIENT_RESULTS / "morfometria" / "interna" / "corregidas_corticales"),
))

OUTPUT_DIR = Path(os.environ.get(
    "VCE_CST_OUTPUT_DIR",
    str(TRACTOGRAPHY_DIR / "cst_visualizacion_ventral_medial"),
))

# True: M1 + M2 como origen cortical.
# False: solo M1.
INCLUIR_M2 = True

# Expansión anatómica para compensar que las streamlines suelen detenerse
# cerca del límite sustancia gris / sustancia blanca.
DILATACION_CORTEZA_MM = float(os.environ.get("VCE_CST_DILATACION_CORTEZA_MM", "8.0"))
DILATACION_TRONCO_MM = float(os.environ.get("VCE_CST_DILATACION_TRONCO_MM", "4.0"))
CST_RESCATE_MESENCEFALO_ENABLE = os.environ.get("VCE_CST_RESCATE_MESENCEFALO_ENABLE", "1") != "0"
CST_RESCATE_MESENCEFALO_TOP_N = int(os.environ.get("VCE_CST_RESCATE_MESENCEFALO_TOP_N", "2500"))
CST_RESCATE_MESENCEFALO_LONGITUD_MINIMA_MM = float(os.environ.get("VCE_CST_RESCATE_MESENCEFALO_LONGITUD_MINIMA_MM", "30.0"))
CST_RESCATE_MESENCEFALO_Z_MIN_MM = float(os.environ.get("VCE_CST_RESCATE_MESENCEFALO_Z_MIN_MM", "22.0"))

# Banda compartida alrededor de la línea media para no cortar fibras
# cercanas al centro del puente o del bulbo.
TOLERANCIA_LINEA_MEDIA_MM = float(os.environ.get("VCE_CST_TOLERANCIA_LINEA_MEDIA_MM", "6.0"))

# Reglas anatómicas del segmento aceptado.
LONGITUD_MINIMA_MM = float(os.environ.get("VCE_CST_LONGITUD_MINIMA_MM", "45.0"))
RANGO_SUPERIOR_INFERIOR_MIN_MM = float(os.environ.get("VCE_CST_RANGO_Z_MIN_MM", "35.0"))
FRACCION_IPSILATERAL_MINIMA = float(os.environ.get("VCE_CST_FRACCION_IPSILATERAL_MINIMA", "0.60"))
FRACCION_MAXIMA_ANTES_CORTEZA = 0.35

# Selección anterior/ventral del tronco para evitar que se acepten
# fibras posteriores en el corte sagital. En coordenadas RAS, Y mayor
# corresponde a una posición más anterior.
FRACCION_ANTERIOR_MESENCEFALO = 0.65
FRACCION_ANTERIOR_PUENTE = 0.60
FRACCION_ANTERIOR_BULBO = 0.65

# Refinamiento mediolateral para el plano coronal.
# La fracción indica cuánto de cada hemiestructura, empezando desde la
# línea media, se conserva como waypoint.
FRACCION_MEDIAL_MESENCEFALO = 0.75
FRACCION_MEDIAL_PUENTE = 0.70
FRACCION_MEDIAL_BULBO = 0.55

# En el tronco se usa una tolerancia menor para reducir cruces artificiales
# de la línea media en la visualización coronal.
TOLERANCIA_CORONAL_LINEA_MEDIA_MM = 2.5

# Separación máxima entre puntos usada para probar la intersección con ROI.
PASO_MUESTREO_MM = 0.75

# Vista previa.
MAX_FIBRAS_POR_LADO_EN_PREVIA = 600
MOSTRAR_VENTANA_3D = True

# Colores solicitados.
COLOR_IZQUIERDA = (255, 55, 55)       # rojo
COLOR_DERECHA = (45, 125, 255)        # azul
COLOR_SUPERPOSICION = (220, 55, 220)  # magenta


# ======================================================================
# ESTRUCTURAS
# ======================================================================

@dataclass
class Candidate:
    streamline: np.ndarray
    side_fraction: float
    length_mm: float
    z_span_mm: float
    strict: bool


@dataclass
class SideMasks:
    cortex: np.ndarray
    midbrain: np.ndarray
    pons: np.ndarray
    medulla: np.ndarray


# ======================================================================
# UTILIDADES
# ======================================================================

def log(text: str) -> None:
    print(text, flush=True)


def require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"\nNo se encontró {description}:\n{path}"
        )
    return path


def find_motor_mask(
    side_spanish: str,
    area: str,
) -> Path | None:
    """Busca la máscara anatómica refinada de M1 o M2."""
    patterns = [
        f"*{area}_{side_spanish}*atlas_refinada.nii.gz",
        f"*{area}_{side_spanish}*corregida*.nii.gz",
        f"*{area}_{side_spanish}*.nii.gz",
    ]

    candidates: list[Path] = []

    for pattern in patterns:
        candidates.extend(MOTOR_MASK_DIR.glob(pattern))

    # Eliminar duplicados conservando rutas reales.
    unique = sorted(
        {path.resolve() for path in candidates if path.is_file()},
        key=lambda path: (
            "atlas_refinada" not in path.name.lower(),
            "overlap" not in path.name.lower(),
            len(path.name),
        ),
    )

    return unique[0] if unique else None


def same_geometry(
    img_a: nib.spatialimages.SpatialImage,
    img_b: nib.spatialimages.SpatialImage,
) -> bool:
    return (
        tuple(img_a.shape[:3]) == tuple(img_b.shape[:3])
        and np.allclose(img_a.affine, img_b.affine, atol=1e-3)
    )


def load_mask_in_t1(
    path: Path,
    t1_img: nib.Nifti1Image,
) -> np.ndarray:
    img = nib.load(str(path))

    if not same_geometry(img, t1_img):
        log(f"  Remuestreando a rT1: {path.name}")
        img = resample_from_to(
            img,
            (t1_img.shape[:3], t1_img.affine),
            order=0,
        )

    mask = np.asarray(img.dataobj) > 0

    if not np.any(mask):
        raise RuntimeError(f"La máscara está vacía: {path}")

    return mask


def dilate_mask_mm(
    mask: np.ndarray,
    affine: np.ndarray,
    distance_mm: float,
) -> np.ndarray:
    if distance_mm <= 0:
        return mask.copy()

    voxel_sizes = nib.affines.voxel_sizes(affine)
    distances = distance_transform_edt(
        ~mask,
        sampling=voxel_sizes,
    )
    return distances <= distance_mm


def mask_centroid_world(
    mask: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    indices = np.argwhere(mask)

    if len(indices) == 0:
        raise RuntimeError("No se puede calcular el centroide de una máscara vacía.")

    return apply_affine(affine, indices).mean(axis=0)


def split_mask_by_world_x(
    mask: np.ndarray,
    affine: np.ndarray,
    midline_x: float,
    side: str,
    tolerance_mm: float,
) -> np.ndarray:
    indices = np.argwhere(mask)
    points = apply_affine(affine, indices)
    x = points[:, 0]

    if side == "left":
        keep = x <= midline_x + tolerance_mm
    elif side == "right":
        keep = x >= midline_x - tolerance_mm
    else:
        raise ValueError(side)

    result = np.zeros(mask.shape, dtype=bool)
    selected = indices[keep]

    if len(selected):
        result[
            selected[:, 0],
            selected[:, 1],
            selected[:, 2],
        ] = True

    return result



def keep_anterior_fraction_per_level(
    mask: np.ndarray,
    affine: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """
    Conserva la porción anterior de una estructura en cada nivel axial.

    La selección se realiza en coordenadas mundiales RAS:
        X = izquierda/derecha
        Y = posterior/anterior
        Z = inferior/superior

    Un valor de fraction=0.60 conserva aproximadamente el 60 % más
    anterior de cada nivel de la máscara.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            "fraction debe estar entre 0 y 1."
        )

    indices = np.argwhere(mask)

    if len(indices) == 0:
        raise RuntimeError(
            "No se puede subdividir una máscara vacía."
        )

    # Identificar el eje de vóxel que más contribuye al eje superior-inferior.
    z_axis = int(
        np.argmax(
            np.abs(
                np.asarray(affine, dtype=float)[2, :3]
            )
        )
    )

    world = apply_affine(
        affine,
        indices,
    )
    world_y = world[:, 1]
    levels = indices[:, z_axis]

    result = np.zeros(
        mask.shape,
        dtype=bool,
    )

    for level in np.unique(levels):
        group = levels == level
        group_indices = indices[group]
        group_y = world_y[group]

        if len(group_indices) <= 2:
            keep = np.ones(
                len(group_indices),
                dtype=bool,
            )
        else:
            threshold = float(
                np.quantile(
                    group_y,
                    1.0 - fraction,
                )
            )
            keep = group_y >= threshold

        selected = group_indices[keep]

        if len(selected):
            result[
                selected[:, 0],
                selected[:, 1],
                selected[:, 2],
            ] = True

    if not np.any(result):
        raise RuntimeError(
            "La subdivisión anterior produjo una máscara vacía."
        )

    return result



def estimate_brainstem_midline_x(
    mask: np.ndarray,
    affine: np.ndarray,
) -> float:
    """
    Estima la línea media en X usando el centro del ancho del tronco
    en cada nivel axial y luego toma la mediana de esos centros.

    Es más estable para el plano coronal que promediar los centroides de
    M1, porque M1 puede ser asimétrica entre hemisferios.
    """
    indices = np.argwhere(mask)

    if len(indices) == 0:
        raise RuntimeError(
            "No se puede estimar la línea media con una máscara vacía."
        )

    z_axis = int(
        np.argmax(
            np.abs(
                np.asarray(affine, dtype=float)[2, :3]
            )
        )
    )

    world = apply_affine(
        affine,
        indices,
    )
    world_x = world[:, 0]
    levels = indices[:, z_axis]

    centers: list[float] = []

    for level in np.unique(levels):
        group_x = world_x[levels == level]

        if len(group_x) < 2:
            continue

        centers.append(
            float(
                (np.min(group_x) + np.max(group_x)) / 2.0
            )
        )

    if not centers:
        return float(np.median(world_x))

    return float(np.median(centers))


def keep_medial_fraction_per_level(
    mask: np.ndarray,
    affine: np.ndarray,
    midline_x: float,
    side: str,
    fraction: float,
) -> np.ndarray:
    """
    Conserva, por cada nivel axial, la fracción de una hemimáscara más
    cercana a la línea media.

    En RAS:
        X menor = izquierda
        X mayor = derecha

    No traslada la tractografía. Solo hace más específico el waypoint
    anatómico en dirección mediolateral.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            "fraction debe estar entre 0 y 1."
        )

    if side not in {"left", "right"}:
        raise ValueError(
            "side debe ser 'left' o 'right'."
        )

    indices = np.argwhere(mask)

    if len(indices) == 0:
        raise RuntimeError(
            "No se puede subdividir una hemimáscara vacía."
        )

    z_axis = int(
        np.argmax(
            np.abs(
                np.asarray(affine, dtype=float)[2, :3]
            )
        )
    )

    world = apply_affine(
        affine,
        indices,
    )
    distances = np.abs(
        world[:, 0] - midline_x
    )
    levels = indices[:, z_axis]

    result = np.zeros(
        mask.shape,
        dtype=bool,
    )

    for level in np.unique(levels):
        group = levels == level
        group_indices = indices[group]
        group_distances = distances[group]

        if len(group_indices) <= 2:
            keep = np.ones(
                len(group_indices),
                dtype=bool,
            )
        else:
            threshold = float(
                np.quantile(
                    group_distances,
                    fraction,
                )
            )
            keep = group_distances <= threshold

        selected = group_indices[keep]

        if len(selected):
            result[
                selected[:, 0],
                selected[:, 1],
                selected[:, 2],
            ] = True

    if not np.any(result):
        raise RuntimeError(
            "La subdivisión medial produjo una máscara vacía."
        )

    return result


def streamline_length_mm(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(
        np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    )


def densify_streamline(
    points: np.ndarray,
    max_step_mm: float,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)

    if len(points) < 2:
        return points

    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)

    if float(np.max(lengths, initial=0.0)) <= max_step_mm:
        return points

    output = [points[0]]

    for start, end, length in zip(
        points[:-1],
        points[1:],
        lengths,
    ):
        divisions = max(1, int(math.ceil(float(length) / max_step_mm)))

        for index in range(1, divisions + 1):
            alpha = index / divisions
            output.append(
                start + alpha * (end - start)
            )

    return np.asarray(output, dtype=np.float32)


def points_to_voxels(
    points: np.ndarray,
    inverse_affine: np.ndarray,
    shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    vox_float = apply_affine(inverse_affine, points)
    vox = np.floor(vox_float + 0.5).astype(np.int32)

    valid = (
        (vox[:, 0] >= 0)
        & (vox[:, 1] >= 0)
        & (vox[:, 2] >= 0)
        & (vox[:, 0] < shape[0])
        & (vox[:, 1] < shape[1])
        & (vox[:, 2] < shape[2])
    )

    return vox, valid


def hit_array(
    mask: np.ndarray,
    voxels: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    hits = np.zeros(len(voxels), dtype=bool)

    valid_voxels = voxels[valid]

    if len(valid_voxels):
        hits[valid] = mask[
            valid_voxels[:, 0],
            valid_voxels[:, 1],
            valid_voxels[:, 2],
        ]

    return hits


def ordered_first_hits(
    arrays: Sequence[np.ndarray],
) -> list[int] | None:
    current = -1
    selected: list[int] = []

    for array in arrays:
        positions = np.flatnonzero(array)

        positions = positions[positions > current]

        if len(positions) == 0:
            return None

        current = int(positions[0])
        selected.append(current)

    return selected


def evaluate_orientation(
    points: np.ndarray,
    masks: SideMasks,
    side: str,
    midline_x: float,
    require_medulla: bool,
    inverse_affine: np.ndarray,
    shape: Sequence[int],
) -> Candidate | None:
    voxels, valid = points_to_voxels(
        points,
        inverse_affine,
        shape,
    )

    valid_fraction = float(np.mean(valid))

    if valid_fraction < 0.70:
        return None

    cortex_hits = hit_array(
        masks.cortex,
        voxels,
        valid,
    )
    midbrain_hits = hit_array(
        masks.midbrain,
        voxels,
        valid,
    )
    pons_hits = hit_array(
        masks.pons,
        voxels,
        valid,
    )
    medulla_hits = hit_array(
        masks.medulla,
        voxels,
        valid,
    )

    arrays = [
        cortex_hits,
        midbrain_hits,
        pons_hits,
    ]

    if require_medulla:
        arrays.append(medulla_hits)

    ordered = ordered_first_hits(arrays)

    if ordered is None:
        return None

    cortex_index = ordered[0]

    if require_medulla:
        end_index = ordered[-1]
    else:
        # El tractograma actual sí alcanza el puente, pero no intersecta el
        # bulbo. Para visualizar todo el tramo realmente reconstruido,
        # conservamos la trayectoria desde la corteza hasta su punto más
        # inferior después de entrar al puente.
        pons_index_for_tail = ordered[2]
        tail = points[pons_index_for_tail:]

        if len(tail) > 1:
            end_index = (
                pons_index_for_tail
                + int(np.argmin(tail[:, 2]))
            )
        else:
            end_index = pons_index_for_tail

        if end_index <= pons_index_for_tail:
            end_index = len(points) - 1

    # La corteza debe encontrarse cerca de uno de los extremos.
    if cortex_index > FRACCION_MAXIMA_ANTES_CORTEZA * (len(points) - 1):
        return None

    if end_index <= cortex_index + 5:
        return None

    segment = points[cortex_index:end_index + 1].copy()

    length_mm = streamline_length_mm(segment)
    z_span_mm = float(np.ptp(segment[:, 2]))

    minimum_length = (
        LONGITUD_MINIMA_MM
        if require_medulla
        else max(35.0, LONGITUD_MINIMA_MM - 15.0)
    )

    minimum_z_span = (
        RANGO_SUPERIOR_INFERIOR_MIN_MM
        if require_medulla
        else max(30.0, RANGO_SUPERIOR_INFERIOR_MIN_MM - 10.0)
    )

    if length_mm < minimum_length:
        return None

    if z_span_mm < minimum_z_span:
        return None

    # Evalúa el predominio ipsilateral entre corteza y puente.
    pons_index = ordered[2]
    upper_segment = points[cortex_index:pons_index + 1]
    x = upper_segment[:, 0]

    if side == "left":
        side_fraction = float(
            np.mean(x <= midline_x + TOLERANCIA_LINEA_MEDIA_MM)
        )
    else:
        side_fraction = float(
            np.mean(x >= midline_x - TOLERANCIA_LINEA_MEDIA_MM)
        )

    if side_fraction < FRACCION_IPSILATERAL_MINIMA:
        return None

    return Candidate(
        streamline=segment,
        side_fraction=side_fraction,
        length_mm=length_mm,
        z_span_mm=z_span_mm,
        strict=require_medulla,
    )


def evaluate_streamline(
    original: np.ndarray,
    masks: SideMasks,
    side: str,
    midline_x: float,
    require_medulla: bool,
    inverse_affine: np.ndarray,
    shape: Sequence[int],
) -> Candidate | None:
    points = densify_streamline(
        original,
        PASO_MUESTREO_MM,
    )

    if len(points) < 10:
        return None

    candidates = []

    forward = evaluate_orientation(
        points,
        masks,
        side,
        midline_x,
        require_medulla,
        inverse_affine,
        shape,
    )

    if forward is not None:
        candidates.append(forward)

    reverse = evaluate_orientation(
        points[::-1].copy(),
        masks,
        side,
        midline_x,
        require_medulla,
        inverse_affine,
        shape,
    )

    if reverse is not None:
        candidates.append(reverse)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda candidate: (
            candidate.side_fraction,
            candidate.length_mm,
        ),
    )




def evaluate_midbrain_first_orientation(
    points: np.ndarray,
    masks: SideMasks,
    side: str,
    midline_x: float,
    inverse_affine: np.ndarray,
    shape: Sequence[int],
) -> Candidate | None:
    """Rescate anatómico centrado en mesencéfalo.

    Primero exige corteza motora -> mesencéfalo. Si además atraviesa puente
    o bulbo, extiende el segmento hasta la porción más inferior disponible.
    No inventa fibras: solo recorta streamlines de wholebrain_T1.trk.
    """
    voxels, valid = points_to_voxels(points, inverse_affine, shape)
    valid_fraction = float(np.mean(valid)) if len(valid) else 0.0
    if valid_fraction < 0.60:
        return None

    cortex_hits = hit_array(masks.cortex, voxels, valid)
    midbrain_hits = hit_array(masks.midbrain, voxels, valid)
    pons_hits = hit_array(masks.pons, voxels, valid)
    medulla_hits = hit_array(masks.medulla, voxels, valid)

    ordered = ordered_first_hits([cortex_hits, midbrain_hits])
    if ordered is None:
        return None

    cortex_index, midbrain_index = ordered
    if cortex_index > 0.45 * max(1, len(points) - 1):
        return None

    # Si después del mesencéfalo hay puente o bulbo, usarlo como final.
    later_pons = np.flatnonzero(pons_hits)
    later_pons = later_pons[later_pons > midbrain_index]
    later_medulla = np.flatnonzero(medulla_hits)
    later_medulla = later_medulla[later_medulla > midbrain_index]

    if len(later_medulla):
        end_index = int(later_medulla[0])
        strict = True
    elif len(later_pons):
        end_index = int(later_pons[0])
        strict = False
    else:
        # Extender desde mesencéfalo hacia el punto más inferior de la cola.
        tail = points[midbrain_index:]
        end_index = midbrain_index + int(np.argmin(tail[:, 2])) if len(tail) else midbrain_index
        strict = False

    if end_index <= cortex_index + 5:
        return None

    segment = points[cortex_index:end_index + 1].copy()
    length_mm = streamline_length_mm(segment)
    z_span_mm = float(np.ptp(segment[:, 2])) if len(segment) else 0.0

    if length_mm < CST_RESCATE_MESENCEFALO_LONGITUD_MINIMA_MM:
        return None
    if z_span_mm < CST_RESCATE_MESENCEFALO_Z_MIN_MM:
        return None

    upper_segment = points[cortex_index:midbrain_index + 1]
    x = upper_segment[:, 0]
    if side == "left":
        side_fraction = float(np.mean(x <= midline_x + TOLERANCIA_LINEA_MEDIA_MM))
    else:
        side_fraction = float(np.mean(x >= midline_x - TOLERANCIA_LINEA_MEDIA_MM))

    if side_fraction < max(0.50, FRACCION_IPSILATERAL_MINIMA - 0.10):
        return None

    return Candidate(segment, side_fraction, length_mm, z_span_mm, strict)


def evaluate_midbrain_first_streamline(
    original: np.ndarray,
    masks: SideMasks,
    side: str,
    midline_x: float,
    inverse_affine: np.ndarray,
    shape: Sequence[int],
) -> Candidate | None:
    points = densify_streamline(original, PASO_MUESTREO_MM)
    if len(points) < 10:
        return None
    candidates = []
    forward = evaluate_midbrain_first_orientation(points, masks, side, midline_x, inverse_affine, shape)
    if forward is not None:
        candidates.append(forward)
    reverse = evaluate_midbrain_first_orientation(points[::-1].copy(), masks, side, midline_x, inverse_affine, shape)
    if reverse is not None:
        candidates.append(reverse)
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c.strict, c.side_fraction, c.length_mm))


def rescue_cst_from_midbrain_first(
    wholebrain_path: Path,
    left_masks: SideMasks,
    right_masks: SideMasks,
    source_header: dict,
    midline_x: float,
    inverse_affine: np.ndarray,
    shape: Sequence[int],
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, int]]:
    """Extrae una CST de respaldo priorizando mesencéfalo completo.

    Se usa cuando el filtro estricto corteza->mesencéfalo->puente/bulbo no
    encuentra fibras. La salida se marca explícitamente como rescate.
    """
    trk_file = nib.streamlines.load(str(wholebrain_path), lazy_load=True)
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    reviewed = 0
    left_hits = 0
    right_hits = 0

    for reviewed, streamline in enumerate(trk_file.tractogram.streamlines, start=1):
        streamline = np.asarray(streamline, dtype=np.float32)
        if len(streamline) < 10:
            continue
        lc = evaluate_midbrain_first_streamline(streamline, left_masks, "left", midline_x, inverse_affine, shape)
        rc = evaluate_midbrain_first_streamline(streamline, right_masks, "right", midline_x, inverse_affine, shape)
        if lc is not None:
            left_hits += 1
        if rc is not None:
            right_hits += 1
        if lc is not None and rc is not None:
            if lc.side_fraction >= rc.side_fraction:
                rc = None
            else:
                lc = None
        if lc is not None and len(left) < CST_RESCATE_MESENCEFALO_TOP_N:
            left.append(lc.streamline)
        if rc is not None and len(right) < CST_RESCATE_MESENCEFALO_TOP_N:
            right.append(rc.streamline)
        if len(left) >= CST_RESCATE_MESENCEFALO_TOP_N and len(right) >= CST_RESCATE_MESENCEFALO_TOP_N:
            break

    return left, right, {"reviewed": int(reviewed), "left_hits": int(left_hits), "right_hits": int(right_hits)}

def save_mask(
    reference: nib.Nifti1Image,
    data: np.ndarray,
    path: Path,
    dtype: np.dtype,
) -> None:
    image = nib.Nifti1Image(
        np.asarray(data, dtype=dtype),
        reference.affine,
    )
    image.set_qform(reference.affine, code=1)
    image.set_sform(reference.affine, code=1)
    nib.save(image, str(path))


def build_trk_header(
    source_header: dict,
    t1_img: nib.Nifti1Image,
    number_streamlines: int,
) -> dict:
    header = source_header.copy()

    header[Field.DIMENSIONS] = np.asarray(
        t1_img.shape[:3],
        dtype=np.int16,
    )
    header[Field.VOXEL_SIZES] = np.asarray(
        t1_img.header.get_zooms()[:3],
        dtype=np.float32,
    )
    header[Field.VOXEL_TO_RASMM] = np.asarray(
        t1_img.affine,
        dtype=np.float32,
    )
    header[Field.VOXEL_ORDER] = "".join(
        nib.aff2axcodes(t1_img.affine)
    ).encode("ascii")
    header[Field.NB_STREAMLINES] = int(number_streamlines)

    return header


def save_trk(
    streamlines: Sequence[np.ndarray],
    source_header: dict,
    t1_img: nib.Nifti1Image,
    path: Path,
) -> None:
    tractogram = Tractogram(
        streamlines,
        affine_to_rasmm=np.eye(4),
    )

    header = build_trk_header(
        source_header,
        t1_img,
        len(streamlines),
    )

    trk_file = TrkFile(
        tractogram,
        header=header,
    )
    trk_file.save(str(path))


def _clean_streamlines_for_export(streamlines: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Normaliza streamlines para exportación a Slicer/VTK.

    Conserva las coordenadas RAS-mm ya transformadas al espacio T1, elimina
    streamlines vacías o con NaN/Inf y fuerza float32. No reubica ni inventa
    fibras.
    """
    cleaned: list[np.ndarray] = []
    for sl in streamlines:
        arr = np.asarray(sl, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
            continue
        if not np.isfinite(arr).all():
            continue
        cleaned.append(arr)
    return cleaned


def save_slicer_compatible_trk(
    streamlines: Sequence[np.ndarray],
    t1_img: nib.Nifti1Image,
    path: Path,
) -> int:
    """Guarda un .trk con header limpio basado únicamente en el rT1/rAnatomico.

    Algunos visores, especialmente 3D Slicer, rechazan .trk con encabezados
    heredados de TrackVis/dcm2niix aunque las fibras sean correctas. Esta salida
    usa un header nuevo, dimensiones/voxel size/affine del T1 y streamlines en
    RAS-mm.
    """
    cleaned = _clean_streamlines_for_export(streamlines)
    header = TrkFile.create_empty_header()
    header[Field.DIMENSIONS] = np.asarray(t1_img.shape[:3], dtype=np.int16)
    header[Field.VOXEL_SIZES] = np.asarray(t1_img.header.get_zooms()[:3], dtype=np.float32)
    header[Field.VOXEL_TO_RASMM] = np.asarray(t1_img.affine, dtype=np.float32)
    try:
        header[Field.VOXEL_ORDER] = "".join(nib.aff2axcodes(t1_img.affine)).encode("ascii")
    except Exception:
        header[Field.VOXEL_ORDER] = b"RAS"
    header[Field.NB_STREAMLINES] = int(len(cleaned))
    tractogram = Tractogram(cleaned, affine_to_rasmm=np.eye(4))
    TrkFile(tractogram, header=header).save(str(path))
    return len(cleaned)


def _orientation_rgb_for_streamline(sl: np.ndarray) -> np.ndarray:
    """Color RGB por orientación local de la fibra, estilo tractografía.

    R = izquierda/derecha, G = anterior/posterior, B = superior/inferior.
    Se usa valor absoluto para que ambos sentidos de una misma orientación
    tengan el mismo color.
    """
    arr = np.asarray(sl, dtype=np.float32)
    n = arr.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if n == 1:
        return np.array([[0.2, 0.2, 1.0]], dtype=np.float32)
    diffs = np.diff(arr, axis=0)
    dirs = np.vstack([diffs[0], diffs])
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    rgb = np.abs(dirs / norms)
    rgb = np.clip(rgb * 1.25, 0.0, 1.0)
    return rgb.astype(np.float32)


def save_legacy_vtk_rgb_streamlines(
    streamlines: Sequence[np.ndarray],
    path: Path,
) -> int:
    """Guarda un VTK PolyData ASCII compatible con 3D Slicer.

    Es un respaldo para cuando Slicer no abre el .trk. El archivo se carga como
    Model y trae COLOR_SCALARS RGB por punto.
    """
    cleaned = _clean_streamlines_for_export(streamlines)
    points: list[list[float]] = []
    lines: list[list[int]] = []
    colors: list[list[float]] = []

    for sl in cleaned:
        start = len(points)
        n = len(sl)
        points.extend(sl.astype(float).tolist())
        lines.append([n] + list(range(start, start + n)))
        colors.extend(_orientation_rgb_for_streamline(sl).astype(float).tolist())

    with path.open("w", encoding="ascii") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("via_cortico_espinal_completa_slicer\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        total_line_size = sum(len(line) for line in lines)
        f.write(f"LINES {len(lines)} {total_line_size}\n")
        for line in lines:
            f.write(" ".join(str(v) for v in line) + "\n")
        f.write(f"POINT_DATA {len(points)}\n")
        f.write("COLOR_SCALARS RGB 3\n")
        for r, g, b in colors:
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")
    return len(cleaned)


def voxelize_bundle(
    streamlines: Sequence[np.ndarray],
    shape: Sequence[int],
    affine: np.ndarray,
) -> np.ndarray:
    density = np.zeros(shape, dtype=np.uint32)
    inverse_affine = np.linalg.inv(affine)

    for streamline in streamlines:
        points = densify_streamline(
            streamline,
            PASO_MUESTREO_MM,
        )
        voxels, valid = points_to_voxels(
            points,
            inverse_affine,
            shape,
        )
        voxels = voxels[valid]

        if len(voxels) == 0:
            continue

        # Cada streamline cuenta una sola vez por vóxel.
        voxels = np.unique(voxels, axis=0)

        density[
            voxels[:, 0],
            voxels[:, 1],
            voxels[:, 2],
        ] += 1

    return density


def direction_rgb_map(
    streamlines: Sequence[np.ndarray],
    density: np.ndarray,
    shape: Sequence[int],
    affine: np.ndarray,
) -> np.ndarray:
    """
    Convención RGB direccional:
        rojo  = izquierda-derecha
        verde = anterior-posterior
        azul  = superior-inferior
    """
    accumulator = np.zeros(
        tuple(shape) + (3,),
        dtype=np.float32,
    )
    weights = np.zeros(shape, dtype=np.float32)
    inverse_affine = np.linalg.inv(affine)

    for streamline in streamlines:
        points = densify_streamline(
            streamline,
            PASO_MUESTREO_MM,
        )

        if len(points) < 2:
            continue

        vectors = np.diff(points, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        valid_segments = lengths > 1e-6

        vectors = vectors[valid_segments]
        lengths = lengths[valid_segments]

        if len(vectors) == 0:
            continue

        directions = np.abs(
            vectors / lengths[:, None]
        )
        midpoints = (
            points[:-1][valid_segments]
            + points[1:][valid_segments]
        ) / 2.0

        voxels, valid_voxels = points_to_voxels(
            midpoints,
            inverse_affine,
            shape,
        )

        voxels = voxels[valid_voxels]
        directions = directions[valid_voxels]
        lengths = lengths[valid_voxels]

        for channel in range(3):
            np.add.at(
                accumulator[..., channel],
                (
                    voxels[:, 0],
                    voxels[:, 1],
                    voxels[:, 2],
                ),
                directions[:, channel] * lengths,
            )

        np.add.at(
            weights,
            (
                voxels[:, 0],
                voxels[:, 1],
                voxels[:, 2],
            ),
            lengths,
        )

    nonzero = weights > 0
    rgb = np.zeros_like(accumulator)

    rgb[nonzero] = (
        accumulator[nonzero]
        / weights[nonzero, None]
    )

    positive_density = density[density > 0]

    if len(positive_density):
        scale = float(
            np.percentile(positive_density, 99)
        )
        scale = max(scale, 1.0)
        brightness = np.clip(
            np.log1p(density.astype(np.float32))
            / np.log1p(scale),
            0.0,
            1.0,
        )
        rgb *= brightness[..., None]

    return np.round(
        np.clip(rgb, 0.0, 1.0) * 255.0
    ).astype(np.uint8)




def save_final_cst_outputs_to_tractography_root(
    *,
    bilateral: Sequence[np.ndarray],
    source_header: dict,
    t1_img: nib.Nifti1Image,
    density: np.ndarray,
    labelmap: np.ndarray,
    rgb: np.ndarray,
    preview_path: Path | None = None,
) -> dict:
    """Guarda la vía corticoespinal bilateral final en la raíz de tractografia_propia.

    La carpeta cst_visualizacion_ventral_medial conserva los archivos intermedios
    por lado. Esta función crea los nombres finales pedidos por el usuario.
    El .nii es una representación voxelizada RGB; la geometría real queda en .trk.
    """
    root = TRACTOGRAPHY_DIR
    root.mkdir(parents=True, exist_ok=True)

    final_trk = root / "via_cortico_espinal_completa.trk"
    final_rgb_nii = root / "via_cortico_espinal_completa.nii"
    final_rgb_gz = root / "via_cortico_espinal_completa_rgb_direccion_T1.nii.gz"
    final_mask = root / "via_cortico_espinal_completa_mask.nii.gz"
    final_density = root / "via_cortico_espinal_completa_density.nii.gz"
    final_labelmap = root / "via_cortico_espinal_completa_colores_itksnap_T1.nii.gz"
    final_labels = root / "via_cortico_espinal_completa_labels_itksnap.txt"
    final_preview = root / "via_cortico_espinal_completa_preview.png"
    final_slicer_trk = root / "via_cortico_espinal_completa_slicer.trk"
    final_slicer_vtk = root / "via_cortico_espinal_completa_slicer.vtk"

    save_trk(bilateral, source_header, t1_img, final_trk)
    slicer_streamline_count = save_slicer_compatible_trk(bilateral, t1_img, final_slicer_trk)
    vtk_streamline_count = save_legacy_vtk_rgb_streamlines(bilateral, final_slicer_vtk)

    # RGB direccional tipo tractografía. Se guarda en .nii sin comprimir porque
    # el usuario pidió explícitamente via_cortico_espinal_completa.nii.
    save_mask(t1_img, rgb, final_rgb_nii, np.uint8)
    save_mask(t1_img, rgb, final_rgb_gz, np.uint8)
    save_mask(t1_img, (density > 0).astype(np.uint8), final_mask, np.uint8)
    save_mask(t1_img, density.astype(np.float32), final_density, np.float32)
    save_mask(t1_img, labelmap.astype(np.uint8), final_labelmap, np.uint8)
    write_itksnap_labels(final_labels)

    if preview_path is not None and preview_path.exists():
        try:
            shutil.copy2(preview_path, final_preview)
        except Exception:
            pass

    readme = root / "LEEME_via_cortico_espinal_completa.txt"
    readme.write_text(
        "SALIDAS FINALES DE LA VIA CORTICOESPINAL COMPLETA\n\n"
        "via_cortico_espinal_completa.trk\n"
        "  Streamlines reales fusionadas: izquierda + derecha. Este es el archivo principal para fibras.\n\n"
        "via_cortico_espinal_completa.nii\n"
        "  Representación voxelizada RGB de la vía fusionada, con colores tipo tractografía.\n"
        "  No reemplaza al .trk; sirve para visualización como volumen/overlay RGB.\n\n"
        "via_cortico_espinal_completa_mask.nii.gz\n"
        "  Máscara binaria de la vía completa.\n\n"
        "via_cortico_espinal_completa_density.nii.gz\n"
        "  Mapa de densidad de streamlines de la vía completa.\n\n"
        "via_cortico_espinal_completa_colores_itksnap_T1.nii.gz\n"
        "  Labelmap para ITK-SNAP: izquierda/derecha/superposición. Usar con el archivo labels.\n\n"
        "via_cortico_espinal_completa_preview.png\n"
        "  Vista rápida para revisar visualmente la forma de la vía.\n\n"
        "via_cortico_espinal_completa_slicer.trk\n"
        "  TRK con header limpio basado en rT1/rAnatomico. Es el primero que debes probar en 3D Slicer.\n\n"
        "via_cortico_espinal_completa_slicer.vtk\n"
        "  Respaldo para 3D Slicer si el .trk falla. Se carga como Model con colores RGB por orientación.\n\n"
        "IMPORTANTE\n"
        "  El .nii es una imagen voxelizada coloreada; la trayectoria real está en el .trk.\n"
        "  Validar visualmente sobre rT1, FA_en_T1 y ColorFA_en_T1.\n",
        encoding="utf-8",
    )

    return {
        "final_trk": str(final_trk),
        "final_rgb_nii": str(final_rgb_nii),
        "final_rgb_gz": str(final_rgb_gz),
        "final_mask": str(final_mask),
        "final_density": str(final_density),
        "final_labelmap": str(final_labelmap),
        "final_labels": str(final_labels),
        "final_preview": str(final_preview),
        "slicer_trk": str(final_slicer_trk),
        "slicer_vtk": str(final_slicer_vtk),
        "slicer_streamline_count": int(slicer_streamline_count),
        "vtk_streamline_count": int(vtk_streamline_count),
        "readme": str(readme),
    }

def write_itksnap_labels(path: Path) -> None:
    left = COLOR_IZQUIERDA
    right = COLOR_DERECHA
    overlap = COLOR_SUPERPOSICION

    text = f"""################################################
# ITK-SnAP Label Description File
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
0       0    0    0    0     0   0    "Clear Label"
1       {left[0]}  {left[1]}  {left[2]}  1     1   1    "CST izquierda"
2       {right[0]}  {right[1]}  {right[2]}  1     1   1    "CST derecha"
3       {overlap[0]}  {overlap[1]}  {overlap[2]}  1     1   1    "Superposición CST"
"""
    path.write_text(
        text,
        encoding="utf-8",
    )


def save_preview_3d(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
    path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log(
            "No se generó la vista 3D porque falta matplotlib. "
            "Los NIfTI y TRK sí fueron guardados."
        )
        return

    rng = np.random.default_rng(42)

    def sample_bundle(
        bundle: Sequence[np.ndarray],
        limit: int,
    ) -> list[np.ndarray]:
        if len(bundle) <= limit:
            return list(bundle)

        selected = rng.choice(
            len(bundle),
            size=limit,
            replace=False,
        )
        return [bundle[int(index)] for index in selected]

    left_sample = sample_bundle(
        left,
        MAX_FIBRAS_POR_LADO_EN_PREVIA,
    )
    right_sample = sample_bundle(
        right,
        MAX_FIBRAS_POR_LADO_EN_PREVIA,
    )

    figure = plt.figure(figsize=(10, 10))
    axis = figure.add_subplot(111, projection="3d")

    for streamline in left_sample:
        axis.plot(
            streamline[:, 0],
            streamline[:, 1],
            streamline[:, 2],
            color=np.asarray(COLOR_IZQUIERDA) / 255.0,
            linewidth=0.55,
            alpha=0.60,
        )

    for streamline in right_sample:
        axis.plot(
            streamline[:, 0],
            streamline[:, 1],
            streamline[:, 2],
            color=np.asarray(COLOR_DERECHA) / 255.0,
            linewidth=0.55,
            alpha=0.60,
        )

    all_streamlines = left_sample + right_sample

    if all_streamlines:
        all_points = np.concatenate(all_streamlines, axis=0)
        minimum = all_points.min(axis=0)
        maximum = all_points.max(axis=0)
        ranges = np.maximum(maximum - minimum, 1.0)
        centers = (minimum + maximum) / 2.0
        radius = float(np.max(ranges) / 2.0)

        axis.set_xlim(
            centers[0] - radius,
            centers[0] + radius,
        )
        axis.set_ylim(
            centers[1] - radius,
            centers[1] + radius,
        )
        axis.set_zlim(
            centers[2] - radius,
            centers[2] + radius,
        )

    axis.set_xlabel("X: izquierda-derecha (mm)")
    axis.set_ylabel("Y: posterior-anterior (mm)")
    axis.set_zlabel("Z: inferior-superior (mm)")
    axis.set_title(
        "CST refinada hacia el corredor anterior y paramediano\n"
        "Rojo: izquierda | Azul: derecha"
    )
    axis.view_init(elev=8, azim=-90)
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    log(f"Vista previa: {path}")

    if MOSTRAR_VENTANA_3D:
        plt.show()
    else:
        plt.close(figure)


def create_readme(
    path: Path,
    t1_path: Path,
    labelmap_path: Path,
    labels_path: Path,
) -> None:
    text = f"""
RESULTADOS DE LA CST REFINADA EN PLANOS SAGITAL Y CORONAL

COLORES
- Rojo: CST izquierda.
- Azul: CST derecha.
- Magenta: vóxeles recorridos por ambos lados.

ABRIR EN ITK-SNAP

1. File -> Open Main Image:
   {t1_path}

2. Segmentation -> Open Segmentation:
   {labelmap_path}

3. Segmentation -> Import/Load Label Descriptions:
   {labels_path}

ARCHIVOS TRK
- cst_ventral_medial_izquierda_T1.trk
- cst_ventral_medial_derecha_T1.trk
- cst_ventral_medial_bilateral_T1.trk

Los archivos TRK contienen las trayectorias reales. ITK-SNAP muestra
mejor la versión voxelizada NIfTI; para visualizar líneas 3D se puede
usar TrackVis, MI-Brain, FURY o MRview.

MAPA RGB DIRECCIONAL
- cst_rgb_direccion_T1.nii.gz

Convención:
- rojo: izquierda-derecha;
- verde: anterior-posterior;
- azul: superior-inferior.

IMPORTANTE
La extracción no usa la tractografía ni el CST generados por el
resonador. Parte exclusivamente de wholebrain_T1.trk y de las ROI
anatómicas. Si no existen fibras hasta el bulbo, el archivo muestra
solamente el segmento comprobado hasta el puente y lo declara en el
reporte JSON.
"""
    path.write_text(
        textwrap.dedent(text).strip() + "\n",
        encoding="utf-8",
    )


# ======================================================================
# PROCESO PRINCIPAL
# ======================================================================

def main() -> int:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("=" * 76)
    log("CST COLOREADA CON REFINAMIENTO VENTRAL Y MEDIAL")
    log("=" * 76)

    require_file(T1_PATH, "el rT1")
    require_file(WHOLEBRAIN_TRK, "la tractografía propia wholebrain_T1.trk")
    require_file(MIDBRAIN_PATH, "la máscara de mesencéfalo")
    require_file(PONS_PATH, "la máscara de puente")
    require_file(MEDULLA_PATH, "la máscara de bulbo")

    if "tractografia_propia" not in str(
        WHOLEBRAIN_TRK
    ).lower():
        raise RuntimeError(
            "Por seguridad, el tractograma fuente debe estar dentro "
            "de tractografia_propia. No se usarán resultados del resonador."
        )

    m1_left_path = find_motor_mask(
        "izquierda",
        "m1",
    )
    m1_right_path = find_motor_mask(
        "derecha",
        "m1",
    )

    if m1_left_path is None or m1_right_path is None:
        raise FileNotFoundError(
            "\nNo se localizaron las dos máscaras M1 refinadas en:\n"
            f"{MOTOR_MASK_DIR}"
        )

    m2_left_path = find_motor_mask(
        "izquierda",
        "m2",
    )
    m2_right_path = find_motor_mask(
        "derecha",
        "m2",
    )

    log("\nEntradas:")
    log(f"  rT1: {T1_PATH}")
    log(f"  Tractografía propia: {WHOLEBRAIN_TRK}")
    log(f"  M1 izquierda: {m1_left_path.name}")
    log(f"  M1 derecha: {m1_right_path.name}")

    if INCLUIR_M2:
        if m2_left_path and m2_right_path:
            log(f"  M2 izquierda: {m2_left_path.name}")
            log(f"  M2 derecha: {m2_right_path.name}")
        else:
            log(
                "  ADVERTENCIA: faltó una máscara M2. "
                "Se continuará únicamente con M1."
            )

    t1_img = nib.load(str(T1_PATH))
    shape = tuple(int(value) for value in t1_img.shape[:3])
    inverse_affine = np.linalg.inv(t1_img.affine)

    log("\nCargando y alineando ROI...")
    m1_left = load_mask_in_t1(
        m1_left_path,
        t1_img,
    )
    m1_right = load_mask_in_t1(
        m1_right_path,
        t1_img,
    )

    use_m2_actual = bool(
        INCLUIR_M2
        and m2_left_path is not None
        and m2_right_path is not None
    )

    if use_m2_actual:
        m2_left = load_mask_in_t1(
            m2_left_path,
            t1_img,
        )
        m2_right = load_mask_in_t1(
            m2_right_path,
            t1_img,
        )
    else:
        m2_left = np.zeros(shape, dtype=bool)
        m2_right = np.zeros(shape, dtype=bool)

    midbrain = load_mask_in_t1(
        MIDBRAIN_PATH,
        t1_img,
    )
    pons = load_mask_in_t1(
        PONS_PATH,
        t1_img,
    )
    medulla = load_mask_in_t1(
        MEDULLA_PATH,
        t1_img,
    )

    left_centroid = mask_centroid_world(
        m1_left,
        t1_img.affine,
    )
    right_centroid = mask_centroid_world(
        m1_right,
        t1_img.affine,
    )

    motor_midline_x = float(
        (left_centroid[0] + right_centroid[0]) / 2.0
    )

    brainstem_union = (
        midbrain
        | pons
        | medulla
    )

    brainstem_midline_x = estimate_brainstem_midline_x(
        brainstem_union,
        t1_img.affine,
    )

    # Para el corredor inferior del CST se toma la línea media anatómica
    # del tronco. El punto medio de M1 queda como control diagnóstico.
    midline_x = brainstem_midline_x

    log(
        f"Línea media del tronco en X = {brainstem_midline_x:.2f} mm."
    )
    log(
        f"Punto medio entre M1 izquierda/derecha = "
        f"{motor_midline_x:.2f} mm."
    )

    midline_difference = abs(
        brainstem_midline_x - motor_midline_x
    )

    if midline_difference > 2.0:
        log(
            "AVISO: M1 y el tronco difieren "
            f"{midline_difference:.2f} mm en la línea media. "
            "Se usará el tronco para evitar desplazamiento coronal."
        )

    if left_centroid[0] >= right_centroid[0]:
        log(
            "ADVERTENCIA: el centroide etiquetado como izquierdo no "
            "está a la izquierda del derecho en RAS. Revisa las etiquetas."
        )

    # El CST atraviesa la porción anterior/ventral del tronco.
    # Las máscaras completas de mesencéfalo, puente y bulbo también incluyen
    # regiones posteriores; por eso se crean waypoints ventrales por nivel.
    midbrain_ventral = keep_anterior_fraction_per_level(
        midbrain,
        t1_img.affine,
        FRACCION_ANTERIOR_MESENCEFALO,
    )
    pons_ventral = keep_anterior_fraction_per_level(
        pons,
        t1_img.affine,
        FRACCION_ANTERIOR_PUENTE,
    )
    medulla_ventral = keep_anterior_fraction_per_level(
        medulla,
        t1_img.affine,
        FRACCION_ANTERIOR_BULBO,
    )

    save_mask(
        t1_img,
        midbrain_ventral,
        OUTPUT_DIR / "waypoint_mesencefalo_ventral_T1.nii.gz",
        np.uint8,
    )
    save_mask(
        t1_img,
        pons_ventral,
        OUTPUT_DIR / "waypoint_puente_ventral_T1.nii.gz",
        np.uint8,
    )
    save_mask(
        t1_img,
        medulla_ventral,
        OUTPUT_DIR / "waypoint_bulbo_ventral_T1.nii.gz",
        np.uint8,
    )

    log(
        "Waypoints sagitales refinados: "
        "mesencéfalo anterior, puente ventral y bulbo anterior."
    )

    cortex_left = m1_left | m2_left
    cortex_right = m1_right | m2_right

    cortex_left = dilate_mask_mm(
        cortex_left,
        t1_img.affine,
        DILATACION_CORTEZA_MM,
    )
    cortex_right = dilate_mask_mm(
        cortex_right,
        t1_img.affine,
        DILATACION_CORTEZA_MM,
    )

    # Primero se separa cada waypoint por hemisferio usando la línea
    # media del tronco y luego se conserva su fracción paramediana.
    midbrain_left_half = split_mask_by_world_x(
        midbrain_ventral,
        t1_img.affine,
        midline_x,
        "left",
        TOLERANCIA_CORONAL_LINEA_MEDIA_MM,
    )
    midbrain_right_half = split_mask_by_world_x(
        midbrain_ventral,
        t1_img.affine,
        midline_x,
        "right",
        TOLERANCIA_CORONAL_LINEA_MEDIA_MM,
    )

    pons_left_half = split_mask_by_world_x(
        pons_ventral,
        t1_img.affine,
        midline_x,
        "left",
        TOLERANCIA_CORONAL_LINEA_MEDIA_MM,
    )
    pons_right_half = split_mask_by_world_x(
        pons_ventral,
        t1_img.affine,
        midline_x,
        "right",
        TOLERANCIA_CORONAL_LINEA_MEDIA_MM,
    )

    medulla_left_half = split_mask_by_world_x(
        medulla_ventral,
        t1_img.affine,
        midline_x,
        "left",
        TOLERANCIA_CORONAL_LINEA_MEDIA_MM,
    )
    medulla_right_half = split_mask_by_world_x(
        medulla_ventral,
        t1_img.affine,
        midline_x,
        "right",
        TOLERANCIA_CORONAL_LINEA_MEDIA_MM,
    )

    midbrain_left_corridor = keep_medial_fraction_per_level(
        midbrain_left_half,
        t1_img.affine,
        midline_x,
        "left",
        FRACCION_MEDIAL_MESENCEFALO,
    )
    midbrain_right_corridor = keep_medial_fraction_per_level(
        midbrain_right_half,
        t1_img.affine,
        midline_x,
        "right",
        FRACCION_MEDIAL_MESENCEFALO,
    )

    pons_left_corridor = keep_medial_fraction_per_level(
        pons_left_half,
        t1_img.affine,
        midline_x,
        "left",
        FRACCION_MEDIAL_PUENTE,
    )
    pons_right_corridor = keep_medial_fraction_per_level(
        pons_right_half,
        t1_img.affine,
        midline_x,
        "right",
        FRACCION_MEDIAL_PUENTE,
    )

    medulla_left_corridor = keep_medial_fraction_per_level(
        medulla_left_half,
        t1_img.affine,
        midline_x,
        "left",
        FRACCION_MEDIAL_BULBO,
    )
    medulla_right_corridor = keep_medial_fraction_per_level(
        medulla_right_half,
        t1_img.affine,
        midline_x,
        "right",
        FRACCION_MEDIAL_BULBO,
    )

    # Guardar los corredores exactos para inspección en ITK-SNAP.
    corridor_masks = {
        "waypoint_mesencefalo_ventral_medial_izquierdo_T1.nii.gz":
            midbrain_left_corridor,
        "waypoint_mesencefalo_ventral_medial_derecho_T1.nii.gz":
            midbrain_right_corridor,
        "waypoint_puente_ventral_medial_izquierdo_T1.nii.gz":
            pons_left_corridor,
        "waypoint_puente_ventral_medial_derecho_T1.nii.gz":
            pons_right_corridor,
        "waypoint_bulbo_ventral_medial_izquierdo_T1.nii.gz":
            medulla_left_corridor,
        "waypoint_bulbo_ventral_medial_derecho_T1.nii.gz":
            medulla_right_corridor,
    }

    for filename, corridor_mask in corridor_masks.items():
        save_mask(
            t1_img,
            corridor_mask,
            OUTPUT_DIR / filename,
            np.uint8,
        )

    log(
        "Waypoints coronales refinados: "
        "ventrales y paramedianos por hemisferio."
    )

    left_masks = SideMasks(
        cortex=cortex_left,
        midbrain=dilate_mask_mm(
            midbrain_left_corridor,
            t1_img.affine,
            DILATACION_TRONCO_MM,
        ),
        pons=dilate_mask_mm(
            pons_left_corridor,
            t1_img.affine,
            DILATACION_TRONCO_MM,
        ),
        medulla=dilate_mask_mm(
            medulla_left_corridor,
            t1_img.affine,
            DILATACION_TRONCO_MM,
        ),
    )

    right_masks = SideMasks(
        cortex=cortex_right,
        midbrain=dilate_mask_mm(
            midbrain_right_corridor,
            t1_img.affine,
            DILATACION_TRONCO_MM,
        ),
        pons=dilate_mask_mm(
            pons_right_corridor,
            t1_img.affine,
            DILATACION_TRONCO_MM,
        ),
        medulla=dilate_mask_mm(
            medulla_right_corridor,
            t1_img.affine,
            DILATACION_TRONCO_MM,
        ),
    )

    log("\nCargando streamlines en RAS+ milímetros...")
    trk_file = nib.streamlines.load(
        str(WHOLEBRAIN_TRK),
        lazy_load=True,
    )

    source_header = trk_file.header
    source_streamlines = trk_file.tractogram.streamlines

    left_strict: list[np.ndarray] = []
    right_strict: list[np.ndarray] = []
    left_partial: list[np.ndarray] = []
    right_partial: list[np.ndarray] = []

    total = 0
    points_in_bounds = 0
    points_checked = 0

    log(
        "\nFiltrando con orden obligatorio: "
        "corteza -> mesencéfalo -> puente -> bulbo..."
    )

    for total, streamline in enumerate(
        source_streamlines,
        start=1,
    ):
        streamline = np.asarray(
            streamline,
            dtype=np.float32,
        )

        if len(streamline) < 10:
            continue

        # Control rápido de compatibilidad espacial.
        if total <= 500:
            sampled = streamline[::max(1, len(streamline) // 20)]
            voxels, valid = points_to_voxels(
                sampled,
                inverse_affine,
                shape,
            )
            points_in_bounds += int(valid.sum())
            points_checked += int(len(valid))

        left_candidate = evaluate_streamline(
            streamline,
            left_masks,
            "left",
            midline_x,
            True,
            inverse_affine,
            shape,
        )
        right_candidate = evaluate_streamline(
            streamline,
            right_masks,
            "right",
            midline_x,
            True,
            inverse_affine,
            shape,
        )

        # Evita duplicar una fibra ambigua en ambos hemisferios.
        if left_candidate and right_candidate:
            if (
                left_candidate.side_fraction
                >= right_candidate.side_fraction
            ):
                right_candidate = None
            else:
                left_candidate = None

        if left_candidate:
            left_strict.append(
                left_candidate.streamline
            )
        elif len(left_partial) < 5000:
            partial = evaluate_streamline(
                streamline,
                left_masks,
                "left",
                midline_x,
                False,
                inverse_affine,
                shape,
            )
            if partial:
                left_partial.append(
                    partial.streamline
                )

        if right_candidate:
            right_strict.append(
                right_candidate.streamline
            )
        elif len(right_partial) < 5000:
            partial = evaluate_streamline(
                streamline,
                right_masks,
                "right",
                midline_x,
                False,
                inverse_affine,
                shape,
            )
            if partial:
                right_partial.append(
                    partial.streamline
                )

        if total % 5000 == 0:
            log(
                f"  Revisadas: {total:,} | "
                f"CST izquierda: {len(left_strict):,} | "
                f"CST derecha: {len(right_strict):,}"
            )

    in_bounds_fraction = (
        points_in_bounds / points_checked
        if points_checked
        else 0.0
    )

    if in_bounds_fraction < 0.70:
        raise RuntimeError(
            "\nEl tractograma no parece estar correctamente alineado con rT1. "
            f"Solo {in_bounds_fraction:.1%} de los puntos de control están "
            "dentro de la matriz del T1. No se guardará un CST engañoso."
        )

    log("\nResultado anatómico:")
    log(f"  Streamlines revisadas: {total:,}")
    log(f"  Completas hasta bulbo - izquierda: {len(left_strict):,}")
    log(f"  Completas hasta bulbo - derecha: {len(right_strict):,}")
    log(f"  Parciales hasta puente - izquierda: {len(left_partial):,}")
    log(f"  Parciales hasta puente - derecha: {len(right_partial):,}")
    log(
        f"  Compatibilidad espacial con rT1: "
        f"{in_bounds_fraction:.1%}"
    )

    # No se inventan fibras que no existen. Si el tractograma no alcanza
    # el bulbo, se visualiza el segmento anatómicamente respaldado hasta
    # el puente y se deja explícita esa limitación en los nombres y reporte.
    if left_strict:
        left_selected = left_strict
        left_status = "completa_hasta_bulbo"
    elif left_partial:
        left_selected = left_partial
        left_status = "segmento_hasta_puente"
        log(
            "  AVISO: la izquierda se visualizará hasta el puente; "
            "wholebrain_T1.trk no contiene fibras que lleguen al bulbo."
        )
    else:
        left_selected = []
        left_status = "ausente"

    if right_strict:
        right_selected = right_strict
        right_status = "completa_hasta_bulbo"
    elif right_partial:
        right_selected = right_partial
        right_status = "segmento_hasta_puente"
        log(
            "  AVISO: la derecha se visualizará hasta el puente; "
            "wholebrain_T1.trk no contiene fibras que lleguen al bulbo."
        )
    else:
        right_selected = []
        right_status = "ausente"

    rescue_mesencefalo_stats = None
    rescue_mesencefalo_usado = False

    if not left_selected and not right_selected and CST_RESCATE_MESENCEFALO_ENABLE:
        log(
            "\nNo hubo segmento corteza -> mesencéfalo -> puente. "
            "Activando rescate anatómico centrado en mesencéfalo completo..."
        )
        left_rescue, right_rescue, rescue_mesencefalo_stats = rescue_cst_from_midbrain_first(
            WHOLEBRAIN_TRK,
            left_masks,
            right_masks,
            source_header,
            midline_x,
            inverse_affine,
            shape,
        )
        if left_rescue:
            left_selected = left_rescue
            left_status = "rescate_corteza_mesencefalo"
            rescue_mesencefalo_usado = True
        if right_rescue:
            right_selected = right_rescue
            right_status = "rescate_corteza_mesencefalo"
            rescue_mesencefalo_usado = True
        log(f"  Rescate mesencéfalo - izquierda: {len(left_selected):,}")
        log(f"  Rescate mesencéfalo - derecha: {len(right_selected):,}")

    if not left_selected and not right_selected:
        raise RuntimeError(
            "\nNo se encontró ni siquiera un segmento ordenado "
            "corteza -> mesencéfalo. Revisa primero la máscara midbrain_T1.nii.gz "
            "sobre el rT1 y la alineación de wholebrain_T1.trk."
        )

    bilateral = left_selected + right_selected

    log("\nGuardando trayectorias respaldadas por los datos...")
    left_trk_path = OUTPUT_DIR / "cst_ventral_medial_izquierda_T1.trk"
    right_trk_path = OUTPUT_DIR / "cst_ventral_medial_derecha_T1.trk"
    bilateral_trk_path = OUTPUT_DIR / "cst_ventral_medial_bilateral_T1.trk"

    save_trk(
        left_selected,
        source_header,
        t1_img,
        left_trk_path,
    )
    save_trk(
        right_selected,
        source_header,
        t1_img,
        right_trk_path,
    )
    save_trk(
        bilateral,
        source_header,
        t1_img,
        bilateral_trk_path,
    )

    log("Voxelizando para ITK-SNAP...")
    left_density = voxelize_bundle(
        left_selected,
        shape,
        t1_img.affine,
    )
    right_density = voxelize_bundle(
        right_selected,
        shape,
        t1_img.affine,
    )
    combined_density = left_density + right_density

    labelmap = np.zeros(
        shape,
        dtype=np.uint8,
    )
    labelmap[left_density > 0] = 1
    labelmap[right_density > 0] = 2
    labelmap[
        (left_density > 0)
        & (right_density > 0)
    ] = 3

    labelmap_path = (
        OUTPUT_DIR
        / "cst_ventral_medial_colores_itksnap_T1.nii.gz"
    )
    labels_path = (
        OUTPUT_DIR
        / "cst_labels_itksnap.txt"
    )

    save_mask(
        t1_img,
        labelmap,
        labelmap_path,
        np.uint8,
    )
    save_mask(
        t1_img,
        left_density,
        OUTPUT_DIR / "cst_ventral_medial_densidad_izquierda_T1.nii.gz",
        np.float32,
    )
    save_mask(
        t1_img,
        right_density,
        OUTPUT_DIR / "cst_ventral_medial_densidad_derecha_T1.nii.gz",
        np.float32,
    )
    save_mask(
        t1_img,
        combined_density,
        OUTPUT_DIR / "cst_ventral_medial_densidad_bilateral_T1.nii.gz",
        np.float32,
    )

    write_itksnap_labels(labels_path)

    log("Creando mapa RGB direccional...")
    rgb = direction_rgb_map(
        bilateral,
        combined_density,
        shape,
        t1_img.affine,
    )
    save_mask(
        t1_img,
        rgb,
        OUTPUT_DIR / "cst_ventral_medial_rgb_direccion_T1.nii.gz",
        np.uint8,
    )

    report = {
        "patient": PATIENT_ID,
        "timepoint": STAGE_NAME,
        "scanner_tractography_used": False,
        "source_tractogram": str(WHOLEBRAIN_TRK),
        "reference_t1": str(T1_PATH),
        "motor_origin": (
            "M1+M2"
            if use_m2_actual
            else "M1"
        ),
        "waypoint_order": [
            "motor_cortex",
            "midbrain",
            "pons",
            "medulla",
        ],
        "primary_waypoint": "midbrain_T1.nii.gz / mesencéfalo completo delimitado primero",
        "colors": {
            "left_cst_rgb": list(COLOR_IZQUIERDA),
            "right_cst_rgb": list(COLOR_DERECHA),
            "overlap_rgb": list(COLOR_SUPERPOSICION),
            "direction_rgb_convention": {
                "R": "left-right",
                "G": "anterior-posterior",
                "B": "superior-inferior",
            },
        },
        "parameters": {
            "cortex_dilation_mm": DILATACION_CORTEZA_MM,
            "brainstem_dilation_mm": DILATACION_TRONCO_MM,
            "midbrain_first_rescue_enabled": CST_RESCATE_MESENCEFALO_ENABLE,
            "midbrain_first_rescue_top_n": CST_RESCATE_MESENCEFALO_TOP_N,
            "midbrain_first_rescue_min_length_mm": CST_RESCATE_MESENCEFALO_LONGITUD_MINIMA_MM,
            "midbrain_first_rescue_min_z_span_mm": CST_RESCATE_MESENCEFALO_Z_MIN_MM,
            "midline_tolerance_mm": TOLERANCIA_LINEA_MEDIA_MM,
            "minimum_length_mm": LONGITUD_MINIMA_MM,
            "minimum_superior_inferior_span_mm":
                RANGO_SUPERIOR_INFERIOR_MIN_MM,
            "minimum_ipsilateral_fraction":
                FRACCION_IPSILATERAL_MINIMA,
            "sampling_step_mm": PASO_MUESTREO_MM,
            "anterior_fraction_midbrain":
                FRACCION_ANTERIOR_MESENCEFALO,
            "anterior_fraction_pons":
                FRACCION_ANTERIOR_PUENTE,
            "anterior_fraction_medulla":
                FRACCION_ANTERIOR_BULBO,
            "medial_fraction_midbrain":
                FRACCION_MEDIAL_MESENCEFALO,
            "medial_fraction_pons":
                FRACCION_MEDIAL_PUENTE,
            "medial_fraction_medulla":
                FRACCION_MEDIAL_BULBO,
            "coronal_midline_tolerance_mm":
                TOLERANCIA_CORONAL_LINEA_MEDIA_MM,
            "motor_midline_x_mm":
                motor_midline_x,
            "brainstem_midline_x_mm":
                brainstem_midline_x,
            "midline_difference_mm":
                midline_difference,
            "sagittal_screen_note": (
                "En ITK-SNAP, la derecha de la vista sagital suele "
                "corresponder a P (posterior), no al hemisferio derecho."
            ),
        },
        "counts": {
            "wholebrain_reviewed": total,
            "left_strict": len(left_strict),
            "right_strict": len(right_strict),
            "bilateral_selected": len(bilateral),
            "left_selected": len(left_selected),
            "right_selected": len(right_selected),
            "left_partial_to_pons": len(left_partial),
            "right_partial_to_pons": len(right_partial),
        },
        "reconstruction_status": {
            "left": left_status,
            "right": right_status,
            "full_cst_to_medulla_available": bool(
                left_strict or right_strict
            ),
            "midbrain_first_rescue_used": bool(rescue_mesencefalo_usado),
            "midbrain_first_rescue_stats": rescue_mesencefalo_stats,
        },
        "geometry": {
            "t1_shape": list(shape),
            "orientation": list(
                nib.aff2axcodes(t1_img.affine)
            ),
            "streamline_points_inside_t1_fraction":
                in_bounds_fraction,
            "estimated_midline_x_mm": midline_x,
            "midline_basis": "brainstem_per_level_bbox_center_median",
            "motor_midline_x_mm": motor_midline_x,
            "brainstem_midline_x_mm": brainstem_midline_x,
            "left_m1_centroid_ras_mm":
                left_centroid.tolist(),
            "right_m1_centroid_ras_mm":
                right_centroid.tolist(),
        },
        "inputs": {
            "m1_left": str(m1_left_path),
            "m1_right": str(m1_right_path),
            "m2_left": (
                str(m2_left_path)
                if m2_left_path
                else None
            ),
            "m2_right": (
                str(m2_right_path)
                if m2_right_path
                else None
            ),
            "midbrain": str(MIDBRAIN_PATH),
            "pons": str(PONS_PATH),
            "medulla": str(MEDULLA_PATH),
            "derived_waypoints": {
                "midbrain_ventral": str(
                    OUTPUT_DIR
                    / "waypoint_mesencefalo_ventral_T1.nii.gz"
                ),
                "pons_ventral": str(
                    OUTPUT_DIR
                    / "waypoint_puente_ventral_T1.nii.gz"
                ),
                "medulla_ventral": str(
                    OUTPUT_DIR
                    / "waypoint_bulbo_ventral_T1.nii.gz"
                ),
            },
        },
        "outputs": {
            "itksnap_labelmap": str(labelmap_path),
            "itksnap_labels": str(labels_path),
            "left_trk": str(left_trk_path),
            "right_trk": str(right_trk_path),
            "bilateral_trk": str(bilateral_trk_path),
            "direction_rgb": str(
                OUTPUT_DIR / "cst_ventral_medial_rgb_direccion_T1.nii.gz"
            ),
        },
        "interpretation_warning": (
            "El labelmap representa únicamente las trayectorias presentes "
            "en wholebrain_T1.trk. La V3 no traslada artificialmente las "
            "streamlines: selecciona las que atraviesan la porción "
            "anterior/ventral del tronco. Cuando el estado es "
            "segmento_hasta_puente, no debe interpretarse como una "
            "reconstrucción completa hasta el bulbo."
        ),
    }

    report_path = OUTPUT_DIR / "cst_ventral_medial_reporte.json"
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    create_readme(
        OUTPUT_DIR / "LEEME_CST_COLORES.txt",
        T1_PATH,
        labelmap_path,
        labels_path,
    )

    preview_path = OUTPUT_DIR / "cst_ventral_medial_vista_3D_colores.png"
    save_preview_3d(
        left_selected,
        right_selected,
        preview_path,
    )

    log("Guardando salida final fusionada en tractografia_propia/ ...")
    final_complete_outputs = save_final_cst_outputs_to_tractography_root(
        bilateral=bilateral,
        source_header=source_header,
        t1_img=t1_img,
        density=combined_density,
        labelmap=labelmap,
        rgb=rgb,
        preview_path=preview_path,
    )

    # Actualizar reporte con la salida final pedida por el usuario.
    try:
        report["outputs"]["via_cortico_espinal_completa"] = final_complete_outputs
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass

    log("\n" + "=" * 76)
    log("PROCESO TERMINADO")
    log("=" * 76)
    log(f"Resultados: {OUTPUT_DIR}")
    log(f"ITK-SNAP: {labelmap_path.name}")
    log(f"Colores: {labels_path.name}")
    log(f"TRK bilateral: {bilateral_trk_path.name}")
    try:
        log(f"Salida final TRK: {Path(final_complete_outputs['final_trk']).name}")
        log(f"Salida final NII RGB: {Path(final_complete_outputs['final_rgb_nii']).name}")
        log(f"Salida Slicer TRK: {Path(final_complete_outputs['slicer_trk']).name}")
        log(f"Salida Slicer VTK: {Path(final_complete_outputs['slicer_vtk']).name}")
    except Exception:
        pass
    log(f"Reporte: {report_path.name}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n" + "=" * 76)
        print("ERROR")
        print("=" * 76)
        print(exc)
        print(
            "\nNo se usó ni se modificó ninguna tractografía "
            "generada por el resonador."
        )
        raise SystemExit(1)
