"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                REFINAMIENTO DE ATLAS MOTOR EN ESPACIO NATIVO                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/motor_atlas_refinement.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Refina regiones motoras integrando priors anatómicos y restricciones
geométricas
del paciente. Las ROI se tratan como conjuntos discretos Ω en la grilla T1. El
ajuste usa operaciones morfológicas, intersecciones Ω_refinada = Ω_atlas ∩
Ω_corteza
y remuestreo por afines para mantener correspondencia entre atlas y anatomía
nativa.

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

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt, label

from .neuroimage import motor_cortex_atlas_prior
from .io_utils import safe_mkdir, save_json


def _ensure_3d(data: np.ndarray, name: str) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f'{name} debe ser 3D. Forma: {data.shape}')
    return data


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    labels, number = label(mask)
    if number == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def _border_values(volume: np.ndarray) -> np.ndarray:
    return np.concatenate([
        volume[0, :, :].ravel(), volume[-1, :, :].ravel(),
        volume[:, 0, :].ravel(), volume[:, -1, :].ravel(),
        volume[:, :, 0].ravel(), volume[:, :, -1].ravel(),
    ])


def _otsu_threshold(values: np.ndarray, bins: int = 512) -> float:
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
    prob = hist.astype(np.float64)
    total = prob.sum()
    if total <= 0:
        return float(lo)
    prob /= total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    between = np.zeros_like(denom)
    valid = denom > 1e-12
    between[valid] = ((mu_total * omega[valid] - mu[valid]) ** 2) / denom[valid]
    return float(centers[int(np.argmax(between))])


def create_brain_mask_simple(brain: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    brain = np.asarray(brain, dtype=np.float32)
    finite = np.isfinite(brain)
    if not np.any(finite):
        raise ValueError('El cerebro no contiene valores finitos.')
    border = _border_values(brain)
    border = border[np.isfinite(border)]
    background = float(np.median(border)) if border.size else 0.0
    diff = np.abs(brain - background)
    p99 = float(np.percentile(diff[finite], 99.0))
    epsilon = max(1e-6, p99 * 1e-4)
    frac_border_bg = float(np.mean(np.abs(border - background) <= epsilon)) if border.size else 0.0
    if frac_border_bg >= 0.80:
        threshold = epsilon
        method = 'fondo_constante_en_bordes'
    else:
        threshold = _otsu_threshold(diff[finite])
        method = 'otsu_sobre_diferencia_fondo'
    mask = finite & (diff > threshold)
    mask = _largest_connected_component(mask)
    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)
    mask = _largest_connected_component(mask)
    return mask.astype(bool), {
        'metodo_segmentacion_cerebro': method,
        'fondo_estimado': background,
        'umbral': float(threshold),
        'fraccion_borde_fondo': frac_border_bg,
        'voxeles_cerebro': int(mask.sum()),
    }


def voxel_sizes_from_affine(affine: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.asarray(affine[:3, :3], dtype=float) ** 2, axis=0))


def infer_side_region_from_name(name: str) -> tuple[str, str]:
    n = name.lower()
    if any(t in n for t in ['izquierda', 'izq', 'left']):
        side = 'izquierda'
    elif any(t in n for t in ['derecha', 'der', 'right']):
        side = 'derecha'
    else:
        side = 'izquierda'
    if any(t in n for t in ['secundaria', 'premotora', 'm2', 'sma']):
        region = 'm2'
    else:
        region = 'm1'
    return side, region


def _save_nifti(data: np.ndarray, reference_img: nib.Nifti1Image, path: Path, dtype=np.uint8) -> None:
    safe_mkdir(path.parent)
    hdr = reference_img.header.copy()
    hdr.set_data_dtype(dtype)
    img = nib.Nifti1Image(np.asarray(data).astype(dtype), reference_img.affine, header=hdr)
    qform, qcode = reference_img.get_qform(coded=True)
    sform, scode = reference_img.get_sform(coded=True)
    if qform is not None:
        img.set_qform(qform, int(qcode or 1))
    if sform is not None:
        img.set_sform(sform, int(scode or 1))
    nib.save(img, str(path))


def _mask_center(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.asarray(mask.shape, dtype=float) / 2.0
    return np.mean(pts, axis=0)


def _intensity_limits(brain: np.ndarray) -> tuple[float, float]:
    valid = brain[np.isfinite(brain)]
    valid = valid[valid != 0]
    if valid.size == 0:
        return float(np.nanmin(brain)), float(np.nanmax(brain))
    return tuple(float(v) for v in np.percentile(valid, [2, 98]))


def save_atlas_preview(brain: np.ndarray, corrected: np.ndarray, atlas_target: np.ndarray, refined: np.ndarray, output_path: Path) -> None:
    vmin, vmax = _intensity_limits(brain)
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    items = [(corrected, 'corregida por shell'), (atlas_target, 'atlas-prior target'), (refined, 'refinada atlas+shell')]
    for row, (mask, label_txt) in enumerate(items):
        center = np.rint(_mask_center(mask)).astype(int)
        center = np.clip(center, 0, np.asarray(brain.shape) - 1)
        x, y, z = center
        views = [
            (brain[x, :, :], mask[x, :, :], f'{label_txt} · sagital x={x}'),
            (brain[:, y, :], mask[:, y, :], f'{label_txt} · coronal y={y}'),
            (brain[:, :, z], mask[:, :, z], f'{label_txt} · axial z={z}'),
        ]
        for col, (bs, ms, title) in enumerate(views):
            ax = axes[row, col]
            ax.imshow(np.rot90(bs), cmap='gray', vmin=vmin, vmax=vmax)
            overlay = np.ma.masked_where(ms <= 0, ms)
            ax.imshow(np.rot90(overlay), cmap='autumn', alpha=0.65, interpolation='nearest')
            ax.set_title(title)
            ax.axis('off')
    fig.tight_layout()
    safe_mkdir(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def refine_cortical_mask_with_atlas_prior(
    brain_path: Path,
    corrected_mask_path: Path,
    output_dir: Path | None = None,
    shell_mm: float = 5.0,
    prior_threshold: float = 0.10,
    sigma_scale: float = 1.0,
) -> dict[str, Any]:
    """Refina una máscara de corteza motora con una prior atlas heurística M1/M2.

    No es un registro atlas MNI real. Es una guía anatómica probabilística en el espacio
    del paciente para que la máscara corregida no quede solo 'sobre corteza', sino también
    cerca de la zona motora esperada por lado y región.
    """
    brain_path = Path(brain_path)
    corrected_mask_path = Path(corrected_mask_path)
    output_dir = Path(output_dir) if output_dir else corrected_mask_path.parent
    safe_mkdir(output_dir)

    brain_img = nib.load(str(brain_path))
    mask_img = nib.load(str(corrected_mask_path))
    brain = _ensure_3d(np.asarray(brain_img.dataobj, dtype=np.float32), 'cerebro')
    mask_data = _ensure_3d(np.asarray(mask_img.dataobj, dtype=np.float32), 'mascara') > 0.5

    if mask_data.shape != brain.shape or not np.allclose(mask_img.affine, brain_img.affine):
        mask_resampled = resample_from_to(
            nib.Nifti1Image(mask_data.astype(np.uint8), mask_img.affine, header=mask_img.header.copy()),
            (brain_img.shape[:3], brain_img.affine),
            order=0,
            mode='constant',
            cval=0,
        )
        mask_data = _ensure_3d(np.asarray(mask_resampled.dataobj), 'mascara remuestreada') > 0.5

    if int(mask_data.sum()) == 0:
        raise ValueError(f'Máscara vacía para refinar: {corrected_mask_path}')

    brain_mask, seg_report = create_brain_mask_simple(brain)
    zooms = voxel_sizes_from_affine(brain_img.affine)
    inside_distance = distance_transform_edt(brain_mask, sampling=zooms)
    cortical_shell = brain_mask & (inside_distance <= float(shell_mm))

    side, region = infer_side_region_from_name(corrected_mask_path.name)
    prior = motor_cortex_atlas_prior(brain.shape, side=side, region=region, sigma_scale=float(sigma_scale))
    prior_target_pool = cortical_shell & (prior >= float(prior_threshold))

    target_voxels = int(mask_data.sum())
    score = np.where(cortical_shell, prior, 0.0).astype(np.float32)
    candidates = np.argwhere(score > 0)
    if candidates.size == 0:
        atlas_target = np.zeros_like(mask_data, dtype=bool)
    else:
        values = score[tuple(candidates.T)]
        k = min(max(target_voxels, 1), len(values))
        # Top-k por score sin ordenar todo el volumen.
        idx = np.argpartition(values, -k)[-k:]
        atlas_target = np.zeros_like(mask_data, dtype=bool)
        atlas_target[tuple(candidates[idx].T)] = True

    intersection = mask_data & prior_target_pool
    overlap_fraction_original = float(intersection.sum() / max(mask_data.sum(), 1))
    shell_fraction_original = float((mask_data & cortical_shell).sum() / max(mask_data.sum(), 1))
    prior_mean_original = float(np.mean(prior[mask_data])) if np.any(mask_data) else 0.0

    # Decisión: si la máscara corregida ya coincide bien con el atlas-prior, conserva la intersección.
    # Si no, usa el target atlas-prior con el mismo volumen aproximado para estabilizar la región.
    if int(intersection.sum()) >= max(20, int(0.35 * target_voxels)):
        refined = intersection
        decision = 'interseccion_mascara_corregida_shell_atlas_prior'
    elif int(atlas_target.sum()) > 0:
        refined = atlas_target
        decision = 'reubicacion_por_atlas_prior_mismo_volumen_aproximado'
    else:
        refined = mask_data & cortical_shell
        if int(refined.sum()) == 0:
            refined = mask_data
        decision = 'fallback_shell_o_mascara_corregida'

    refined = _largest_connected_component(refined)
    if int(refined.sum()) == 0:
        refined = mask_data
        decision = 'fallback_mascara_corregida_por_refinada_vacia'

    stem = corrected_mask_path.name.replace('.nii.gz', '').replace('.nii', '')
    prior_path = output_dir / f'{stem}_atlas_prior_{region}_{side}.nii.gz'
    shell_path = output_dir / f'{stem}_cortical_shell_objetivo.nii.gz'
    atlas_target_path = output_dir / f'{stem}_atlas_target_same_volume.nii.gz'
    refined_path = output_dir / f'{stem}_atlas_refinada.nii.gz'
    preview_path = output_dir / f'{stem}_atlas_refinada_preview.png'
    report_path = output_dir / f'{stem}_atlas_refinada_reporte.json'

    _save_nifti(prior, brain_img, prior_path, dtype=np.float32)
    _save_nifti(cortical_shell.astype(np.uint8), brain_img, shell_path, dtype=np.uint8)
    _save_nifti(atlas_target.astype(np.uint8), brain_img, atlas_target_path, dtype=np.uint8)
    _save_nifti(refined.astype(np.uint8), brain_img, refined_path, dtype=np.uint8)
    save_atlas_preview(brain, mask_data, atlas_target, refined, preview_path)

    refined_prior_mean = float(np.mean(prior[refined])) if np.any(refined) else 0.0
    refined_shell_fraction = float((refined & cortical_shell).sum() / max(refined.sum(), 1))
    report = {
        'brain_path': str(brain_path),
        'corrected_mask_path': str(corrected_mask_path),
        'refined_mask_path': str(refined_path),
        'atlas_prior_path': str(prior_path),
        'atlas_target_path': str(atlas_target_path),
        'cortical_shell_path': str(shell_path),
        'preview_path': str(preview_path),
        'side': side,
        'region': region,
        'method': 'cortical_shell_plus_motor_atlas_prior_heuristic',
        'decision': decision,
        'warning': 'Atlas-prior heurístico en espacio del paciente; no reemplaza registro anatómico atlas-paciente ni FreeSurfer/aparc.',
        'shell_mm': float(shell_mm),
        'prior_threshold': float(prior_threshold),
        'sigma_scale': float(sigma_scale),
        'mask_voxels_corrected': int(mask_data.sum()),
        'mask_voxels_refined': int(refined.sum()),
        'atlas_target_voxels': int(atlas_target.sum()),
        'overlap_fraction_original_with_prior_pool': overlap_fraction_original,
        'shell_fraction_original': shell_fraction_original,
        'prior_mean_original_mask': prior_mean_original,
        'prior_mean_refined_mask': refined_prior_mean,
        'shell_fraction_refined': refined_shell_fraction,
        'segmentation': seg_report,
        'quality_hint': 'alta' if refined_prior_mean >= 0.45 and refined_shell_fraction >= 0.80 else ('media' if refined_prior_mean >= 0.25 else 'baja'),
    }
    save_json(report, report_path)
    return report
