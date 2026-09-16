"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ANÁLISIS AVANZADO DE SEÑALES                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/advanced_signal.py
Versión: v3.21.21

Descripción
-----------
Calcula métricas espectrales, correlacionales y multiescala.

Fundamento físico-matemático implementado
-----------------------------------------
Amplía el análisis de señales con representaciones temporales, espectrales y
multiescala. La FFT permite estimar potencia por banda, la autocorrelación
mide
periodicidad interna y la correlación cruzada estima retardo electromecánico.
Las
señales heterogéneas se alinean sobre una grilla temporal común mediante
interpolación lineal antes de comparar amplitud, fase o energía.

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

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.ndimage import gaussian_laplace

from .io_utils import safe_mkdir, save_dataframe


def _zscore(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y - np.nanmean(y))
    return y / (np.nanstd(y) + 1e-12)


def cwt_like_scalogram(y: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Escalograma tipo CWT con sombrero mexicano/LoG.

    Está inspirado en los módulos originales `cwt_engine.py` y `wavelet_3d.py`,
    pero usa una implementación ligera y estable para no romper compatibilidad de SciPy.
    """
    y = _zscore(np.asarray(y, dtype=float))
    out = []
    for s in scales:
        resp = gaussian_laplace(y, sigma=max(float(s), 0.5))
        out.append(np.abs(resp) ** 2)
    return np.vstack(out)


def shannon_entropy(y: np.ndarray, bins: int = 32) -> float:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 4:
        return math.nan
    hist, _ = np.histogram(y, bins=bins, density=False)
    p = hist.astype(float) / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def multiscale_entropy_curve(y: np.ndarray, max_scale: int = 12) -> pd.DataFrame:
    y = _zscore(y)
    rows = []
    for scale in range(1, max_scale + 1):
        n = (len(y) // scale) * scale
        if n < scale * 4:
            continue
        coarse = y[:n].reshape(-1, scale).mean(axis=1)
        rows.append({"scale": scale, "entropy_shannon": shannon_entropy(coarse)})
    return pd.DataFrame(rows)


def hurst_rs(y: np.ndarray, min_window: int = 8) -> float:
    """Estimación Hurst R/S robusta para señales EMG/DIN."""
    y = _zscore(y)
    n = len(y)
    if n < min_window * 4:
        return math.nan
    max_window = max(min(n // 2, 256), min_window + 1)
    windows = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), 12).astype(int))
    xs, ys = [], []
    for w in windows:
        if w < 4 or n // w < 2:
            continue
        rs_vals = []
        for start in range(0, n - w + 1, w):
            seg = y[start:start+w]
            seg = seg - np.mean(seg)
            z = np.cumsum(seg)
            R = np.max(z) - np.min(z)
            S = np.std(seg)
            if S > 1e-12 and R > 0:
                rs_vals.append(R / S)
        if rs_vals:
            xs.append(np.log(w))
            ys.append(np.log(np.mean(rs_vals)))
    if len(xs) < 2:
        return math.nan
    return float(np.polyfit(xs, ys, 1)[0])


def spectral_summary(t: np.ndarray, y: np.ndarray, max_freq_hz: float = 20.0) -> dict:
    y = _zscore(y)
    if len(t) < 8:
        return {}
    dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 1.0
    fs = 1.0 / max(dt, 1e-12)
    freqs, psd = scipy_signal.welch(y, fs=fs, nperseg=min(512, len(y)))
    mask = freqs <= max_freq_hz
    if not np.any(mask):
        mask = np.ones_like(freqs, dtype=bool)
    freqs2, psd2 = freqs[mask], psd[mask]
    total = float(np.trapz(psd2, freqs2)) if len(freqs2) > 1 else 0.0
    peak_f = float(freqs2[np.argmax(psd2)]) if len(freqs2) else math.nan
    # Distribución de potencia por bandas útiles en señales neuromusculares lentas del proyecto.
    bands = [(0, 0.5), (0.5, 2), (2, 5), (5, 10), (10, max_freq_hz)]
    out = {"spectral_total_band_power": total, "spectral_peak_freq_band_hz": peak_f}
    for lo, hi in bands:
        b = (freqs >= lo) & (freqs < hi)
        power = float(np.trapz(psd[b], freqs[b])) if np.sum(b) > 1 else 0.0
        key = f"bandpower_{str(lo).replace('.', 'p')}_{str(hi).replace('.', 'p')}_hz"
        out[key] = power
        out[key + "_rel"] = power / (total + 1e-12)
    return out


def coherence_summary(t: np.ndarray, f: np.ndarray, g: np.ndarray, max_freq_hz: float = 20.0) -> dict:
    if len(f) < 16 or len(g) < 16 or len(t) < 16:
        return {}
    n = min(len(f), len(g), len(t))
    f, g, t = _zscore(f[:n]), _zscore(g[:n]), t[:n]
    dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 1.0
    fs = 1.0 / max(dt, 1e-12)
    freqs, coh = scipy_signal.coherence(f, g, fs=fs, nperseg=min(256, n))
    mask = freqs <= max_freq_hz
    if not np.any(mask):
        mask = np.ones_like(freqs, dtype=bool)
    freqs2, coh2 = freqs[mask], coh[mask]
    return {
        "coherence_mean": float(np.nanmean(coh2)) if len(coh2) else math.nan,
        "coherence_max": float(np.nanmax(coh2)) if len(coh2) else math.nan,
        "coherence_peak_freq_hz": float(freqs2[int(np.nanargmax(coh2))]) if len(coh2) else math.nan,
    }


def phase_locking_value(f: np.ndarray, g: np.ndarray) -> float:
    n = min(len(f), len(g))
    if n < 16:
        return math.nan
    af = scipy_signal.hilbert(_zscore(f[:n]))
    ag = scipy_signal.hilbert(_zscore(g[:n]))
    phase_diff = np.angle(af) - np.angle(ag)
    return float(np.abs(np.mean(np.exp(1j * phase_diff))))


def plot_advanced_pair(t: np.ndarray, f: np.ndarray, g: np.ndarray, label_f: str, label_g: str, out_dir: Path, base: str, max_freq_hz: float = 20.0) -> dict:
    safe_mkdir(out_dir)
    n = min(len(t), len(f), len(g))
    t, f, g = t[:n], _zscore(f[:n]), _zscore(g[:n])
    scales = np.geomspace(1, max(2, min(64, n / 8)), 32)
    sf = cwt_like_scalogram(f, scales)
    sg = cwt_like_scalogram(g, scales)

    # Guardar matrices para análisis externo.
    np.save(out_dir / f"{base}_cwt_{label_f.replace(' ', '_')}.npy", sf)
    np.save(out_dir / f"{base}_cwt_{label_g.replace(' ', '_')}.npy", sg)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    axes[0].plot(t, f, lw=0.8, label=label_f)
    axes[0].plot(t, g, lw=0.8, label=label_g, alpha=0.8)
    axes[0].set_title("Señales normalizadas para análisis avanzado")
    axes[0].set_xlabel("Tiempo (s)")
    axes[0].legend(fontsize=8)

    im1 = axes[1].imshow(sf, aspect="auto", origin="lower", extent=[t[0], t[-1], scales[0], scales[-1]], cmap="plasma")
    axes[1].set_title(f"Escalograma CWT-like · {label_f}")
    axes[1].set_ylabel("Escala")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(sg, aspect="auto", origin="lower", extent=[t[0], t[-1], scales[0], scales[-1]], cmap="plasma")
    axes[2].set_title(f"Escalograma CWT-like · {label_g}")
    axes[2].set_xlabel("Tiempo (s)")
    axes[2].set_ylabel("Escala")
    fig.colorbar(im2, ax=axes[2], shrink=0.8)
    path_scalo = out_dir / f"{base}_cwt_escalogramas.png"
    fig.savefig(path_scalo, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Coherencia
    dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 1.0
    fs = 1.0 / max(dt, 1e-12)
    freqs, coh = scipy_signal.coherence(f, g, fs=fs, nperseg=min(256, n))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, coh, lw=1.0)
    ax.set_xlim(0, max_freq_hz)
    ax.set_ylim(0, 1.05)
    ax.set_title("Coherencia espectral")
    ax.set_xlabel("Hz")
    ax.set_ylabel("Coherencia")
    ax.grid(alpha=0.25)
    path_coh = out_dir / f"{base}_coherencia.png"
    fig.savefig(path_coh, dpi=150, bbox_inches="tight")
    plt.close(fig)

    mse_f = multiscale_entropy_curve(f)
    mse_g = multiscale_entropy_curve(g)
    if not mse_f.empty:
        mse_f["senal"] = label_f
    if not mse_g.empty:
        mse_g["senal"] = label_g
    mse = pd.concat([mse_f, mse_g], ignore_index=True) if not mse_f.empty or not mse_g.empty else pd.DataFrame()
    if not mse.empty:
        save_dataframe(mse, out_dir / f"{base}_entropia_multiescala.csv")
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, grp in mse.groupby("senal"):
            ax.plot(grp["scale"], grp["entropy_shannon"], marker="o", label=name)
        ax.set_title("Entropía multiescala")
        ax.set_xlabel("Escala de coarse-graining")
        ax.set_ylabel("Entropía Shannon")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(out_dir / f"{base}_entropia_multiescala.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {
        "advanced_cwt_png": str(path_scalo),
        "advanced_coherence_png": str(path_coh),
        "f_hurst_rs": hurst_rs(f),
        "g_hurst_rs": hurst_rs(g),
        "f_entropy_shannon": shannon_entropy(f),
        "g_entropy_shannon": shannon_entropy(g),
        "phase_locking_value": phase_locking_value(f, g),
        **{f"f_{k}": v for k, v in spectral_summary(t, f, max_freq_hz).items()},
        **{f"g_{k}": v for k, v in spectral_summary(t, g, max_freq_hz).items()},
        **coherence_summary(t, f, g, max_freq_hz),
    }
