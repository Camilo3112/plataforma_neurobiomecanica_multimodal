"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      VALIDAR BRAINSTEM SUBSTRUCTURES T1                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: modulos/modulo_CST_tronco_integracion_completa/04_validacion/validar_brainstem_substructures_T1.py
Versión: v3.21.21

Descripción
-----------
Valida máscaras anatómicas del tronco encefálico y sus etiquetas derivadas.

Fundamento físico-matemático implementado
-----------------------------------------
Valida subestructuras del tronco encefálico en el T1 del paciente. Las
etiquetas
FreeSurfer se interpretan como un campo categórico L(i,j,k). Se extraen
máscaras
M_l = 1[L=l] para mesencéfalo, puente, bulbo y SCP; luego se cuantifican
volumen,
centroide y orden anatómico superior-inferior para detectar errores de
registro.

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
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import label as cc_label


# ============================================================
# RUTAS
# ============================================================

RT1_PATH = Path(
    r"D:\EAFIT\01-2026\proyecto\datos\paciente 6\Antes"
    r"\Resultados funcional\REFORMATEO\rT1.nii"
)

# Pon aquí el archivo generado en Linux.
# También puede ser brainstemSsLabels.v12.FSvoxelSpace.mgz.
BRAINSTEM_SEGMENTATION_PATH = Path(
    r"D:\EAFIT\01-2026\proyecto\resultados\paciente 6\Antes"
    r"\tractografia_propia\brainstem_substructures"
    r"\brainstem_substructures_T1.nii.gz"
)

OUTPUT_DIR = Path(
    r"D:\EAFIT\01-2026\proyecto\resultados\paciente 6\Antes"
    r"\tractografia_propia\brainstem_substructures_validated"
)

LABELS = {
    "midbrain": 173,
    "pons": 174,
    "medulla": 175,
    "scp": 178,
}

# Etiquetas de salida para ITK-SNAP.
OUTPUT_LABELS = {
    "midbrain": 5,
    "pons": 6,
    "medulla": 7,
    "scp": 8,
}


# ============================================================
# UTILIDADES
# ============================================================

def largest_component(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    labeled, count = cc_label(mask)

    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def save_like(
    reference: nib.Nifti1Image,
    data: np.ndarray,
    path: Path,
    dtype=np.uint8,
) -> None:
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    nib.save(
        nib.Nifti1Image(
            np.asarray(data, dtype=dtype),
            reference.affine,
            header,
        ),
        str(path),
    )


def mask_metrics(
    mask: np.ndarray,
    affine: np.ndarray,
) -> dict[str, Any]:
    points = np.argwhere(mask)

    if points.size == 0:
        return {
            "voxel_count": 0,
            "volume_mm3": 0.0,
            "bbox_mm": [0.0, 0.0, 0.0],
            "center_world_mm": [None, None, None],
            "component_count": 0,
        }

    zooms = nib.affines.voxel_sizes(affine)
    voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))

    lo = points.min(axis=0)
    hi = points.max(axis=0)
    bbox_vox = hi - lo + 1
    bbox_mm = bbox_vox * zooms

    center_voxel = points.mean(axis=0)
    center_world = nib.affines.apply_affine(
        affine,
        center_voxel,
    )

    _, component_count = cc_label(mask)

    return {
        "voxel_count": int(points.shape[0]),
        "volume_mm3": float(points.shape[0] * voxel_volume),
        "bbox_voxels": [int(v) for v in bbox_vox],
        "bbox_mm": [float(v) for v in bbox_mm],
        "center_world_mm": [float(v) for v in center_world],
        "component_count": int(component_count),
    }


def validate_order(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    centers = {
        name: values["center_world_mm"][2]
        for name, values in metrics.items()
        if values["center_world_mm"][2] is not None
    }

    missing = [
        name
        for name in ("midbrain", "pons", "medulla")
        if name not in centers
    ]

    if missing:
        return {
            "valid": False,
            "reason": f"faltan centros para: {missing}",
            "centers_z_mm": centers,
        }

    # En RAS, z aumenta hacia superior:
    # medulla < pons < midbrain.
    valid = (
        centers["medulla"]
        < centers["pons"]
        < centers["midbrain"]
    )

    return {
        "valid": bool(valid),
        "expected": "medulla_z < pons_z < midbrain_z",
        "centers_z_mm": centers,
    }


# ============================================================
# PROGRAMA
# ============================================================

def main() -> int:
    if not RT1_PATH.exists():
        raise FileNotFoundError(f"No existe rT1: {RT1_PATH}")

    if not BRAINSTEM_SEGMENTATION_PATH.exists():
        raise FileNotFoundError(
            "No existe la segmentación de FreeSurfer:\n"
            f"{BRAINSTEM_SEGMENTATION_PATH}\n\n"
            "Primero ejecuta generar_brainstem_substructures_freesurfer.sh "
            "en Linux y copia su carpeta de resultados."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t1_img = nib.load(str(RT1_PATH))
    seg_img = nib.load(str(BRAINSTEM_SEGMENTATION_PATH))

    # Vecino más cercano para preservar etiquetas enteras.
    seg_t1_img = resample_from_to(
        seg_img,
        t1_img,
        order=0,
    )

    seg_t1 = np.rint(
        np.asarray(seg_t1_img.dataobj)
    ).astype(np.int16)

    save_like(
        t1_img,
        seg_t1,
        OUTPUT_DIR / "brainstem_substructures_T1.nii.gz",
        dtype=np.int16,
    )

    masks: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for name, label_value in LABELS.items():
        mask = seg_t1 == label_value
        mask = largest_component(mask)

        masks[name] = mask
        metrics[name] = mask_metrics(mask, t1_img.affine)

        save_like(
            t1_img,
            mask.astype(np.uint8),
            OUTPUT_DIR / f"{name}_T1.nii.gz",
            dtype=np.uint8,
        )

        if metrics[name]["voxel_count"] == 0:
            warnings.append(
                f"{name} está vacío; no puede usarse para tractografía."
            )

        if metrics[name]["component_count"] > 1:
            warnings.append(
                f"{name} tiene múltiples componentes."
            )

    order_validation = validate_order(metrics)
    if not order_validation["valid"]:
        warnings.append(
            "El orden superior-inferior no es anatómicamente correcto: "
            "debe cumplirse bulbo < puente < mesencéfalo."
        )

    # Labelmap visual, sin solapamientos.
    labelmap = np.zeros(t1_img.shape[:3], dtype=np.uint8)

    for name in ("midbrain", "pons", "medulla", "scp"):
        labelmap[masks[name]] = OUTPUT_LABELS[name]

    save_like(
        t1_img,
        labelmap,
        OUTPUT_DIR / "brainstem_substructures_labelmap_T1.nii.gz",
        dtype=np.uint8,
    )

    # Unión del tronco sin SCP, útil como referencia pero no como waypoint único.
    core = (
        masks["midbrain"]
        | masks["pons"]
        | masks["medulla"]
    )

    save_like(
        t1_img,
        core.astype(np.uint8),
        OUTPUT_DIR / "brainstem_core_T1.nii.gz",
        dtype=np.uint8,
    )

    valid = (
        all(metrics[name]["voxel_count"] > 0
            for name in ("midbrain", "pons", "medulla"))
        and order_validation["valid"]
        and len(warnings) == 0
    )

    report = {
        "input_segmentation": str(BRAINSTEM_SEGMENTATION_PATH),
        "t1_reference": str(RT1_PATH),
        "orientation": list(nib.aff2axcodes(t1_img.affine)),
        "labels_freesurfer": LABELS,
        "output_labels_itksnap": OUTPUT_LABELS,
        "metrics": metrics,
        "superior_inferior_order": order_validation,
        "valid_for_cst_waypoints": bool(valid),
        "warnings": warnings,
        "important": (
            "Para CST no se recomienda usar brainstem_core_T1 como único "
            "waypoint. Deben aplicarse por separado mesencéfalo, puente y "
            "bulbo, preferiblemente divididos por hemisferio."
        ),
    }

    (OUTPUT_DIR / "brainstem_substructures_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    label_description = """################################################
# ITK-SnAP Label Description File
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
0       0    0    0    0     0   0    "Clear Label"
5       242  104  76   1     1   1    "Mesencéfalo"
6       206  195  58   1     1   1    "Puente"
7       119  159  176  1     1   1    "Bulbo raquídeo"
8       142  182  0    1     1   1    "Pedúnculo cerebeloso superior"
"""

    (OUTPUT_DIR / "brainstem_labels_itksnap.txt").write_text(
        label_description,
        encoding="utf-8",
    )

    print("=" * 72)
    print("SUBESTRUCTURAS DEL TRONCO GENERADAS")
    print("=" * 72)
    print(f"Válidas para waypoints CST: {valid}")
    print(f"Salida: {OUTPUT_DIR}")

    for name in ("midbrain", "pons", "medulla", "scp"):
        item = metrics[name]
        print(
            f"- {name}: {item['voxel_count']} vóxeles | "
            f"{item['volume_mm3']:.1f} mm³ | "
            f"centro={item['center_world_mm']}"
        )

    if warnings:
        print("\nADVERTENCIAS:")
        for warning in warnings:
            print(f"- {warning}")

    print(
        "\nAbre en ITK-SNAP:\n"
        f"{OUTPUT_DIR / 'brainstem_substructures_labelmap_T1.nii.gz'}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
