"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  VALIDAR BRAINSTEM SUBSTRUCTURES T1 WINDOWS                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: modulos/modulo_CST_tronco_integracion_completa/02_segmentacion_tronco/FINAL_V9/validar_brainstem_substructures_T1_windows.py
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

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import label as cc_label


LABELS = {
    "midbrain": 173,
    "pons": 174,
    "medulla": 175,
    "scp": 178,
}

OUTPUT_LABELS = {
    "midbrain": 5,
    "pons": 6,
    "medulla": 7,
    "scp": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1", required=True, type=Path)
    parser.add_argument("--seg", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = cc_label(mask.astype(bool))

    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def save_like(
    reference: nib.Nifti1Image,
    data: np.ndarray,
    path: Path,
    dtype: np.dtype,
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
    bbox_voxels = hi - lo + 1
    bbox_mm = bbox_voxels * zooms

    center_voxel = points.mean(axis=0)
    center_world = nib.affines.apply_affine(
        affine,
        center_voxel,
    )

    _, component_count = cc_label(mask)

    return {
        "voxel_count": int(points.shape[0]),
        "volume_mm3": float(points.shape[0] * voxel_volume),
        "bbox_voxels": [int(value) for value in bbox_voxels],
        "bbox_mm": [float(value) for value in bbox_mm],
        "center_world_mm": [float(value) for value in center_world],
        "component_count": int(component_count),
    }


def validate_superior_inferior_order(
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    centers_z = {
        name: values["center_world_mm"][2]
        for name, values in metrics.items()
        if values["center_world_mm"][2] is not None
    }

    missing = [
        name
        for name in ("midbrain", "pons", "medulla")
        if name not in centers_z
    ]

    if missing:
        return {
            "valid": False,
            "reason": f"Faltan estructuras: {missing}",
            "centers_z_mm": centers_z,
        }

    valid = (
        centers_z["medulla"]
        < centers_z["pons"]
        < centers_z["midbrain"]
    )

    return {
        "valid": bool(valid),
        "expected": "medulla_z < pons_z < midbrain_z",
        "centers_z_mm": centers_z,
    }


def main() -> int:
    args = parse_args()

    if not args.t1.exists():
        raise FileNotFoundError(f"No existe T1: {args.t1}")

    if not args.seg.exists():
        raise FileNotFoundError(
            f"No existe segmentación: {args.seg}"
        )

    args.out.mkdir(parents=True, exist_ok=True)

    t1_img = nib.load(str(args.t1))
    seg_img = nib.load(str(args.seg))

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
        args.out / "brainstem_substructures_T1.nii.gz",
        np.int16,
    )

    masks: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for name, label_value in LABELS.items():
        mask = largest_component(seg_t1 == label_value)
        masks[name] = mask
        metrics[name] = mask_metrics(mask, t1_img.affine)

        save_like(
            t1_img,
            mask,
            args.out / f"{name}_T1.nii.gz",
            np.uint8,
        )

        if metrics[name]["voxel_count"] == 0:
            warnings.append(f"{name} está vacío.")

    order = validate_superior_inferior_order(metrics)

    if not order["valid"]:
        warnings.append(
            "No se cumple el orden bulbo < puente < mesencéfalo."
        )

    labelmap = np.zeros(t1_img.shape[:3], dtype=np.uint8)

    for name in ("midbrain", "pons", "medulla", "scp"):
        labelmap[masks[name]] = OUTPUT_LABELS[name]

    save_like(
        t1_img,
        labelmap,
        args.out / "brainstem_substructures_labelmap_T1.nii.gz",
        np.uint8,
    )

    brainstem_core = (
        masks["midbrain"]
        | masks["pons"]
        | masks["medulla"]
    )

    save_like(
        t1_img,
        brainstem_core,
        args.out / "brainstem_core_T1.nii.gz",
        np.uint8,
    )

    valid = (
        all(
            metrics[name]["voxel_count"] > 0
            for name in ("midbrain", "pons", "medulla")
        )
        and order["valid"]
        and not warnings
    )

    report = {
        "input_segmentation": str(args.seg),
        "t1_reference": str(args.t1),
        "orientation": list(nib.aff2axcodes(t1_img.affine)),
        "freesurfer_labels": LABELS,
        "output_labels_itksnap": OUTPUT_LABELS,
        "metrics": metrics,
        "superior_inferior_order": order,
        "valid_for_cst_waypoints": bool(valid),
        "warnings": warnings,
        "important": (
            "Para CST se deben usar por separado mesencéfalo, "
            "puente y bulbo, no solo la unión completa."
        ),
    }

    report_path = (
        args.out / "brainstem_substructures_report.json"
    )

    report_path.write_text(
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

    (
        args.out / "brainstem_labels_itksnap.txt"
    ).write_text(
        label_description,
        encoding="utf-8",
    )

    print("=" * 72)
    print("VALIDACIÓN DEL TRONCO COMPLETADA")
    print("=" * 72)
    print(f"Válido para waypoints CST: {valid}")
    print(f"Reporte: {report_path}")
    print(
        "Labelmap: "
        f"{args.out / 'brainstem_substructures_labelmap_T1.nii.gz'}"
    )

    for name, values in metrics.items():
        print(
            f"{name}: {values['voxel_count']} vóxeles | "
            f"{values['volume_mm3']:.1f} mm³"
        )

    if warnings:
        print("\nADVERTENCIAS:")
        for warning in warnings:
            print(f"- {warning}")

    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
