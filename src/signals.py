"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                PROCESAMIENTO DE SEÑALES EMG Y DINAMOMÉTRICAS                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/signals.py
Versión: v3.21.21

Descripción
-----------
Procesa señales temporales de EMG y dinamometría.

Fundamento físico-matemático implementado
-----------------------------------------
Procesa señales temporales EMG y dinamométricas como series discretas x[n].
Incluye centrado, normalización, RMS = sqrt(mean(x^2)), correlación cruzada
R_xy[τ]=Σ x[n]y[n+τ], espectro por FFT X[k]=Σ x[n]e^{-j2πkn/N} y
sincronización
por interpolación cuando las frecuencias de muestreo difieren.

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

import math
import re
import unicodedata
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as scipy_signal

from .io_utils import safe_mkdir
from .advanced_signal import plot_advanced_pair

Side = Literal["derecha", "izquierda"]
Muscle = Literal["lateral", "medial"]
SignalKind = Literal["EMG", "DIN"]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace("µ", "u").replace("μ", "u")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path or not Path(path).exists():
        raise FileNotFoundError(path)
    path = Path(path)
    # EMG viene normalmente con ; y decimales con coma. Dinamometría puede venir con coma.
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, sep=None, engine="python")


def _to_float_series(series: pd.Series) -> np.ndarray:
    return (
        series.astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .astype(float)
        .to_numpy()
    )


def _find_col(df: pd.DataFrame, must_contain: list[str], alternatives: list[list[str]] | None = None) -> str:
    norm_cols = {_norm(c): c for c in df.columns}
    for n, original in norm_cols.items():
        if all(token in n for token in must_contain):
            return original
    if alternatives:
        for tokens in alternatives:
            for n, original in norm_cols.items():
                if all(token in n for token in tokens):
                    return original
    raise KeyError(f"No encontré columna con tokens {must_contain}. Columnas disponibles: {list(df.columns)}")


def side_from_filename(path: Path, fallback: Side) -> Side:
    name = _norm(Path(path).stem)
    if "derecha" in name or "right" in name or re.search(r"\bder\b", name):
        return "derecha"
    if "izquierda" in name or "left" in name or re.search(r"\bizq\b", name):
        return "izquierda"
    return fallback


def load_emg_signal(path: Path, side: Side, muscle: Muscle) -> tuple[np.ndarray, np.ndarray, dict]:
    """Carga una señal EMG respetando la lateralidad del archivo.

    Regla clave del proyecto: si el archivo se llama derecha_*.csv, usa solo columnas
    de vastos derechos; si se llama izquierda_*.csv, usa solo columnas de vastos izquierdos.
    """
    df = _read_csv_flexible(Path(path)).copy()
    effective_side = side_from_filename(Path(path), side)
    side_token = "derecho" if effective_side == "derecha" else "izquierdo"
    side_token_alt = "der" if effective_side == "derecha" else "izq"
    muscle_token = "lateral" if muscle == "lateral" else "medial"

    if "time" in [_norm(c) for c in df.columns]:
        time_col = next(c for c in df.columns if _norm(c) == "time")
        t = _to_float_series(df[time_col])
    else:
        frames_col = _find_col(df, ["frames"], alternatives=[["frame"], ["tiempo"]])
        frames = _to_float_series(df[frames_col])
        t = frames * 0.05

    col = _find_col(
        df,
        ["vasto", muscle_token, side_token],
        alternatives=[
            ["rms", "vasto", muscle_token, side_token],
            ["vasto", muscle_token, side_token_alt],
            [muscle_token, side_token],
        ],
    )
    y = _to_float_series(df[col])
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(t) == 0:
        raise ValueError(f"La señal EMG quedó vacía: {path}")
    if np.nanmax(t) == np.nanmin(t):
        t = np.arange(len(y)) * 0.05
    return t, y.astype(float), {
        "path": str(path),
        "kind": "EMG",
        "requested_side": side,
        "effective_side": effective_side,
        "muscle": muscle,
        "column": col,
        "n": int(len(y)),
    }


def load_din_signal(path: Path, side: Side) -> tuple[np.ndarray, np.ndarray, dict]:
    df = _read_csv_flexible(Path(path)).copy()
    side_en = "right" if side == "derecha" else "left"
    side_es = "derech" if side == "derecha" else "izquierd"

    # Tiempo
    try:
        tcol = _find_col(df, ["seconds"], alternatives=[["time"], ["tiempo"], ["segundos"]])
        t = _to_float_series(df[tcol])
    except Exception:
        t = np.arange(len(df), dtype=float)
        tcol = "inferido_indice"

    # Fuerza: idealmente suma absoluta de inner + outer del lado solicitado.
    candidates = []
    norm_map = {_norm(c): c for c in df.columns}
    for n, original in norm_map.items():
        if (side_en in n or side_es in n or (side == "derecha" and "der" in n) or (side == "izquierda" and "izq" in n)) and (
            "force" in n or "fuerza" in n
        ):
            candidates.append(original)

    if not candidates:
        # fallback: columnas numéricas que no sean tiempo
        for c in df.columns:
            if c != tcol:
                try:
                    _ = _to_float_series(df[c])
                    candidates.append(c)
                except Exception:
                    pass

    if not candidates:
        raise KeyError(f"No encontré columnas de fuerza en {path}. Columnas: {list(df.columns)}")

    forces = []
    for c in candidates:
        try:
            forces.append(np.abs(_to_float_series(df[c])))
        except Exception:
            pass
    if not forces:
        raise ValueError(f"No fue posible convertir fuerzas a número: {path}")
    y = np.nansum(np.vstack(forces), axis=0)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    return t.astype(float), y.astype(float), {
        "path": str(path),
        "kind": "DIN",
        "side": side,
        "columns": candidates,
        "time_column": tcol,
        "n": int(len(y)),
    }


def load_signal(path: Path, kind: SignalKind, side: Side, muscle: Muscle | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    if kind.upper() == "EMG":
        if muscle is None:
            raise ValueError("Para EMG debes indicar muscle='lateral' o 'medial'.")
        return load_emg_signal(path, side, muscle)
    return load_din_signal(path, side)


def detrend_and_normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)] if y.ndim else y
    y = scipy_signal.detrend(y, type="constant")
    scale = np.nanstd(y)
    return y / (scale + 1e-12)


def resample_to_common_time(t1: np.ndarray, y1: np.ndarray, t2: np.ndarray, y2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t1, y1, t2, y2 = map(np.asarray, (t1, y1, t2, y2))
    if len(t1) < 2 or len(t2) < 2:
        raise ValueError("Señales demasiado cortas para alinear.")
    dt1 = np.nanmedian(np.diff(t1))
    dt2 = np.nanmedian(np.diff(t2))
    dt = float(np.nanmin([abs(dt1), abs(dt2)]))
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.05
    t0 = max(float(np.nanmin(t1)), float(np.nanmin(t2)))
    t_end = min(float(np.nanmax(t1)), float(np.nanmax(t2)))
    if t_end <= t0:
        n = min(len(y1), len(y2))
        return np.arange(n) * dt, y1[:n], y2[:n]
    t = np.arange(t0, t_end, dt)
    y1i = np.interp(t, t1, y1)
    y2i = np.interp(t, t2, y2)
    return t - t[0], y1i, y2i


def align_by_xcorr(t: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    z1 = detrend_and_normalize(y1)
    z2 = detrend_and_normalize(y2)
    corr = scipy_signal.correlate(z1, z2, mode="full", method="fft")
    lags = scipy_signal.correlation_lags(len(z1), len(z2), mode="full")
    lag = int(lags[np.nanargmax(np.abs(corr))])
    dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 1.0
    if lag > 0:
        f = y1[lag:]
        g = y2[: len(f)]
        tt = t[: len(f)]
    elif lag < 0:
        g = y2[-lag:]
        f = y1[: len(g)]
        tt = t[: len(f)]
    else:
        n = min(len(y1), len(y2))
        tt, f, g = t[:n], y1[:n], y2[:n]
    n = min(len(f), len(g), len(tt))
    return tt[:n], f[:n], g[:n], lag * dt


def signal_features(t: np.ndarray, y: np.ndarray, kind: SignalKind) -> dict:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return {}
    dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 1.0
    fs = 1.0 / max(dt, 1e-12)
    y0 = np.nan_to_num(y - np.nanmean(y))
    abs_y = np.abs(y0)
    duration = float(t[-1] - t[0]) if len(t) else float(len(y) * dt)
    out = {
        "n": int(len(y)),
        "duration_s": duration,
        "mean": float(np.nanmean(y)),
        "std": float(np.nanstd(y)),
        "rms": float(np.sqrt(np.nanmean(y0**2))),
        "mean_abs": float(np.nanmean(abs_y)),
        "peak_abs": float(np.nanmax(abs_y)),
        "auc_abs": float(np.trapz(abs_y, dx=dt)),
    }
    if len(y) >= 8:
        freqs, psd = scipy_signal.welch(y0, fs=fs, nperseg=min(256, len(y)))
        if np.any(psd > 0):
            out["dominant_freq_hz"] = float(freqs[int(np.nanargmax(psd))])
            cdf = np.cumsum(psd)
            out["median_freq_hz"] = float(freqs[np.searchsorted(cdf, cdf[-1] / 2)])
            out["spectral_power"] = float(np.trapz(psd, freqs))
    if kind.upper() == "DIN":
        out["max_force"] = float(np.nanmax(y))
        out["mean_force"] = float(np.nanmean(y))
        idx = int(np.nanargmax(y))
        out["time_to_peak_s"] = float(t[idx] - t[0]) if len(t) > idx else math.nan
        if idx > 0 and len(t) > idx:
            out["rfd_to_peak"] = float((y[idx] - y[0]) / max(t[idx] - t[0], 1e-12))
        out["impulse"] = float(np.trapz(y, t)) if len(t) == len(y) else float(np.trapz(y, dx=dt))
    else:
        out["iemg"] = float(np.trapz(abs_y, dx=dt))
        out["zero_crossings"] = int(np.sum(np.diff(np.signbit(y0)) != 0))
    return out


def pair_metrics(t: np.ndarray, f: np.ndarray, g: np.ndarray) -> dict:
    if len(f) < 2 or len(g) < 2:
        return {}
    zf = detrend_and_normalize(f)
    zg = detrend_and_normalize(g)
    n = min(len(zf), len(zg))
    zf, zg = zf[:n], zg[:n]
    pearson = float(np.corrcoef(zf, zg)[0, 1]) if n > 2 else np.nan
    rmse = float(np.sqrt(np.mean((zf - zg) ** 2)))
    corr = scipy_signal.correlate(zf, zg, mode="full", method="fft")
    lags = scipy_signal.correlation_lags(n, n, mode="full")
    idx = int(np.nanargmax(np.abs(corr)))
    dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 1.0
    max_corr = float(corr[idx] / (np.linalg.norm(zf) * np.linalg.norm(zg) + 1e-12))
    return {
        "pearson_aligned": pearson,
        "rmse_zscore": rmse,
        "xcorr_max_norm": max_corr,
        "xcorr_lag_s": float(lags[idx] * dt),
    }


def plot_pair(t: np.ndarray, f: np.ndarray, g: np.ndarray, label_f: str, label_g: str, title: str, out_png: Path, max_freq_hz: float = 20.0) -> None:
    safe_mkdir(out_png.parent)
    dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 1.0
    fs = 1.0 / max(dt, 1e-12)
    zf = detrend_and_normalize(f)
    zg = detrend_and_normalize(g)
    n = min(len(zf), len(zg))
    zf, zg, tt = zf[:n], zg[:n], t[:n]

    bg = "#050d1a"
    fig = plt.figure(figsize=(18, 11), facecolor=bg)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.35)
    fig.suptitle(title, color="white", fontsize=15, fontweight="bold")

    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot(tt, zf, label=label_f, linewidth=1)
    ax0.plot(tt, zg, label=label_g, linewidth=1, alpha=0.8)
    ax0.set_title("Señales alineadas (z-score)")
    ax0.set_xlabel("Tiempo (s)")
    ax0.legend(fontsize=8)

    ax1 = fig.add_subplot(gs[0, 2:])
    freqs = np.fft.rfftfreq(n, 1 / fs)
    ax1.plot(freqs, np.abs(np.fft.rfft(zf)), linewidth=1, label=label_f)
    ax1.plot(freqs, np.abs(np.fft.rfft(zg)), linewidth=1, alpha=0.7, label=label_g)
    ax1.set_xlim(0, max_freq_hz)
    ax1.set_title("FFT · magnitud")
    ax1.set_xlabel("Hz")

    ax2 = fig.add_subplot(gs[1, :2])
    corr = scipy_signal.correlate(zf, zg, mode="full", method="fft")
    lags = scipy_signal.correlation_lags(n, n, mode="full") * dt
    ax2.plot(lags, corr / (np.max(np.abs(corr)) + 1e-12), linewidth=1)
    ax2.set_title("Correlación cruzada FFT")
    ax2.set_xlabel("Lag (s)")

    ax3 = fig.add_subplot(gs[1, 2:])
    acf = scipy_signal.correlate(zf, zf, mode="full", method="fft")
    acf_lags = scipy_signal.correlation_lags(n, n, mode="full") * dt
    ax3.plot(acf_lags, acf / (np.max(np.abs(acf)) + 1e-12), linewidth=1)
    ax3.set_xlim(-2, 2)
    ax3.set_title(f"Autocorrelación · {label_f}")
    ax3.set_xlabel("Lag (s)")

    ax4 = fig.add_subplot(gs[2, :2])
    f_psd, pxx_f = scipy_signal.welch(zf, fs=fs, nperseg=min(256, n))
    g_psd, pxx_g = scipy_signal.welch(zg, fs=fs, nperseg=min(256, n))
    ax4.semilogy(f_psd, pxx_f + 1e-12, linewidth=1, label=label_f)
    ax4.semilogy(g_psd, pxx_g + 1e-12, linewidth=1, alpha=0.8, label=label_g)
    ax4.set_xlim(0, max_freq_hz)
    ax4.set_title("Densidad espectral de potencia")
    ax4.set_xlabel("Hz")

    ax5 = fig.add_subplot(gs[2, 2:])
    cross = np.fft.rfft(zf) * np.conj(np.fft.rfft(zg))
    phase = np.angle(cross)
    ax5.scatter(freqs, phase, s=6, c=np.abs(cross), cmap="viridis")
    ax5.set_xlim(0, max_freq_hz)
    ax5.set_title("Fase cruzada")
    ax5.set_xlabel("Hz")

    for ax in fig.get_axes():
        ax.set_facecolor("#0b1929")
        ax.tick_params(colors="#d8e2ef", labelsize=8)
        ax.xaxis.label.set_color("#d8e2ef")
        ax.yaxis.label.set_color("#d8e2ef")
        ax.title.set_color("white")
        ax.grid(alpha=0.15)
        for spine in ax.spines.values():
            spine.set_color("#253858")
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor("#0b1929")
            for txt in leg.get_texts():
                txt.set_color("white")

    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close(fig)


def analyze_pair(
    path_f: Path,
    kind_f: SignalKind,
    side_f: Side,
    muscle_f: Muscle | None,
    label_f: str,
    path_g: Path,
    kind_g: SignalKind,
    side_g: Side,
    muscle_g: Muscle | None,
    label_g: str,
    title: str,
    out_png: Path,
    max_freq_plot_hz: float = 20.0,
) -> dict:
    t1, y1, meta1 = load_signal(path_f, kind_f, side_f, muscle_f)
    t2, y2, meta2 = load_signal(path_g, kind_g, side_g, muscle_g)
    t, y1r, y2r = resample_to_common_time(t1, y1, t2, y2)
    t, f, g, lag_align_s = align_by_xcorr(t, y1r, y2r)
    if len(f) < 8 or len(g) < 8:
        raise ValueError(f"Muy pocos datos después de alinear: {len(f)} muestras")
    plot_pair(t, f, g, label_f, label_g, title, out_png, max_freq_hz=max_freq_plot_hz)
    advanced_dir = out_png.parent / "analisis_avanzado"
    advanced_metrics = {}
    try:
        advanced_metrics = plot_advanced_pair(
            t, f, g, label_f, label_g, advanced_dir, out_png.stem, max_freq_hz=max_freq_plot_hz
        )
    except Exception as exc:
        advanced_metrics = {"advanced_signal_error": str(exc)}

    metrics = {
        "title": title,
        "out_png": str(out_png),
        "alignment_lag_s": float(lag_align_s),
        "signal_f": meta1,
        "signal_g": meta2,
    }
    metrics.update(advanced_metrics)
    metrics.update({f"f_{k}": v for k, v in signal_features(t, f, kind_f).items()})
    metrics.update({f"g_{k}": v for k, v in signal_features(t, g, kind_g).items()})
    metrics.update(pair_metrics(t, f, g))
    return metrics
