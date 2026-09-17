#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        CONVERSIÓN CST PARA 3D SLICER                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: tools/convertir_cst_para_slicer.py
Versión: v3.21.21

Descripción
-----------
Genera versiones TRK/VTK compatibles con 3D Slicer.


-----------------------------------------
Convierte tractogramas a formatos compatibles con 3D Slicer. El tracto se
modela
como polilíneas r_i(s) en RAS-mm. El encabezado TRK se reconstruye usando la
matriz
afín del T1, mientras que el VTK guarda puntos, líneas y colores RGB por
orientación
local d = |Δr|/||Δr|| para visualización anatómica.

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

import nibabel as nib
import numpy as np
from nibabel.streamlines import Tractogram, load
from nibabel.streamlines.trk import TrkFile, Field


def clean_streamlines(streamlines):
    cleaned = []
    for sl in streamlines:
        arr = np.asarray(sl, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
            continue
        if not np.isfinite(arr).all():
            continue
        cleaned.append(arr)
    return cleaned


def orientation_rgb(sl: np.ndarray) -> np.ndarray:
    n = sl.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if n == 1:
        return np.array([[0.2, 0.2, 1.0]], dtype=np.float32)
    diffs = np.diff(sl, axis=0)
    dirs = np.vstack([diffs[0], diffs])
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.clip(np.abs(dirs / norms) * 1.25, 0.0, 1.0).astype(np.float32)


def save_slicer_trk(streamlines, t1_img, out_path: Path):
    header = TrkFile.create_empty_header()
    header[Field.DIMENSIONS] = np.asarray(t1_img.shape[:3], dtype=np.int16)
    header[Field.VOXEL_SIZES] = np.asarray(t1_img.header.get_zooms()[:3], dtype=np.float32)
    header[Field.VOXEL_TO_RASMM] = np.asarray(t1_img.affine, dtype=np.float32)
    try:
        header[Field.VOXEL_ORDER] = "".join(nib.aff2axcodes(t1_img.affine)).encode("ascii")
    except Exception:
        header[Field.VOXEL_ORDER] = b"RAS"
    header[Field.NB_STREAMLINES] = int(len(streamlines))
    tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
    TrkFile(tractogram, header=header).save(str(out_path))


def save_vtk(streamlines, out_path: Path):
    points = []
    lines = []
    colors = []
    for sl in streamlines:
        start = len(points)
        n = len(sl)
        points.extend(sl.astype(float).tolist())
        lines.append([n] + list(range(start, start + n)))
        colors.extend(orientation_rgb(sl).astype(float).tolist())

    with out_path.open("w", encoding="ascii") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("via_cortico_espinal_completa_slicer\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        total = sum(len(line) for line in lines)
        f.write(f"LINES {len(lines)} {total}\n")
        for line in lines:
            f.write(" ".join(str(v) for v in line) + "\n")
        f.write(f"POINT_DATA {len(points)}\n")
        f.write("COLOR_SCALARS RGB 3\n")
        for r, g, b in colors:
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")


def find_t1(tractografia_dir: Path, explicit: str | None):
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"No existe T1 explícito: {p}")
    resumen = tractografia_dir / "resumen_tractografia.json"
    if resumen.exists():
        data = json.loads(resumen.read_text(encoding="utf-8"))
        for key in ("t1_reference", "t1_path"):
            val = data.get(key)
            if val and Path(val).exists():
                return Path(val)
    # último respaldo: buscar NIfTI anatómico cerca del árbol de datos si existe en el JSON de CST
    raise FileNotFoundError("No pude deducir el rT1/rAnatomico. Usa --t1-path.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tractografia-dir", required=True)
    ap.add_argument("--t1-path", default=None)
    args = ap.parse_args()

    out = Path(args.tractografia_dir)
    in_trk = out / "via_cortico_espinal_completa.trk"
    if not in_trk.exists():
        raise FileNotFoundError(f"No existe {in_trk}")
    t1_path = find_t1(out, args.t1_path)
    t1 = nib.load(str(t1_path))
    trk = load(str(in_trk))
    streamlines = clean_streamlines(trk.streamlines)
    if not streamlines:
        raise RuntimeError("No hay streamlines válidas para exportar.")

    out_trk = out / "via_cortico_espinal_completa_slicer.trk"
    out_vtk = out / "via_cortico_espinal_completa_slicer.vtk"
    save_slicer_trk(streamlines, t1, out_trk)
    save_vtk(streamlines, out_vtk)

    print("T1 referencia:", t1_path)
    print("Streamlines exportadas:", len(streamlines))
    print("TRK Slicer:", out_trk)
    print("VTK Slicer:", out_vtk)


if __name__ == "__main__":
    main()
