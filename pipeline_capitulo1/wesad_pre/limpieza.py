"""
limpieza.py — Limpieza por señal (etapa 3) y manejo de huecos (etapa 5).

Se limpian las señales del modelo (config.MODEL_SIGNALS, solo pecho) y, solo para el EDA,
la EDA y la temperatura de muñeca (config.CLEAN_SIGNALS).
EMG y acelerómetros no se limpian para el modelo: se usan en el EDA y para marcar
movimiento.

Cómo se trata una muestra marcada como "excluir" en el QC:
  1. Antes de filtrar, se rellena por interpolación lineal. Esto NO la "arregla":
     solo evita que un valor absurdo (p. ej. -273 °C) se riegue por el filtro a
     las muestras vecinas.
  2. Después de filtrar, la muestra sigue marcada como inválida, salvo que sea un
     hueco corto en una señal lenta (EDA o temperatura, <= 2 s), que sí se acepta
     como corregido.
"""
from fractions import Fraction

import numpy as np
from scipy import signal as sps

from .config import CLEAN, MAX_GAP_INTERP_S, SIGNALS
from .etiquetas import find_blocks

try:
    import neurokit2 as nk
except ImportError:  # el paquete funciona sin NeuroKit2 salvo Resp
    nk = None


# ---------------------------------------------------------------------------
# Herramientas básicas
# ---------------------------------------------------------------------------

def fill_invalid(x, invalid):
    """Rellena las muestras inválidas por interpolación lineal (solo para poder filtrar)."""
    x = np.asarray(x, dtype=np.float64).copy()
    invalid = np.asarray(invalid, dtype=bool) | ~np.isfinite(x)
    if not invalid.any():
        return x
    good = np.where(~invalid)[0]
    if good.size == 0:
        return np.zeros_like(x)
    x[invalid] = np.interp(np.where(invalid)[0], good, x[good])
    return x


def butter_filter(x, fs, kind, cutoff, order):
    """Butterworth de fase cero (sosfiltfilt). kind: 'lowpass' | 'highpass' | 'bandpass'."""
    sos = sps.butter(order, cutoff, btype=kind, fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def notch(x, fs, f0, q=30.0):
    """Filtro notch de fase cero en f0 Hz (interferencia de la red eléctrica)."""
    b, a = sps.iirnotch(f0, q, fs=fs)
    return sps.filtfilt(b, a, x)


def resample(x, fs_in, fs_out):
    """Cambia la frecuencia de muestreo con filtro anti-aliasing (resample_poly)."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float64)
    fr = Fraction(fs_out, fs_in).limit_denominator(1000)
    return sps.resample_poly(x, fr.numerator, fr.denominator)


def resample_mask(mask, fs_in, fs_out, n_out):
    """Lleva una máscara a otra frecuencia: la muestra de salida queda marcada si
    CUALQUIER muestra de entrada que representa está marcada."""
    mask = np.asarray(mask, dtype=bool)
    if fs_in == fs_out:
        return mask[:n_out].copy()
    c = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
    edges = np.minimum(np.round(np.arange(n_out + 1) * fs_in / fs_out).astype(np.int64), mask.size)
    lo, hi = edges[:-1], np.maximum(edges[1:], edges[:-1] + 1)
    hi = np.minimum(hi, mask.size)
    return (c[hi] - c[lo]) > 0


def short_gaps(invalid, max_len):
    """Tramos inválidos de longitud <= max_len muestras (los que se pueden interpolar)."""
    out = np.zeros_like(invalid, dtype=bool)
    for s, e in find_blocks(invalid):
        if e - s <= max_len:
            out[s:e] = True
    return out


# ---------------------------------------------------------------------------
# Limpieza por señal
# ---------------------------------------------------------------------------

def clean_signal(x, key, invalid):
    """Limpia una señal según config.CLEAN.

    Parámetros
      x       : señal en unidades físicas (1D), a su frecuencia original
      key     : nombre de la señal, p. ej. 'chest_ECG'
      invalid : máscara del QC (True = excluir)

    Devuelve un diccionario:
      y            : señal limpia a la frecuencia de trabajo fs
      fs           : frecuencia de trabajo
      invalid      : máscara de muestras inválidas a esa frecuencia
      interpolated : muestras que eran huecos cortos y se aceptaron como corregidas
    """
    p = CLEAN[key]
    fs_in = SIGNALS[key]["fs"]
    fs_out = p["fs_out"]
    xf = fill_invalid(x, invalid)

    if key == "chest_ECG":
        y = notch(xf, fs_in, p["notch_hz"])
        y = butter_filter(y, fs_in, "bandpass", p["band"], p["order"])

    elif key in ("chest_EDA", "wrist_EDA"):
        y = butter_filter(xf, fs_in, "lowpass", p["lowpass"], p["order"])
        y = resample(y, fs_in, fs_out)
        y = np.clip(y, 0, None)          # la conductancia no puede ser negativa

    elif key == "chest_Resp":
        y = resample(xf, fs_in, fs_out)
        y = nk.rsp_clean(y, sampling_rate=fs_out, method=p["method"])

    elif key in ("chest_Temp", "wrist_TEMP"):
        y = resample(xf, fs_in, fs_out)
        y = butter_filter(y, fs_out, "lowpass", p["lowpass"], p["order"])

    else:
        raise ValueError(f"{key} no tiene limpieza configurada (rol: {SIGNALS[key]['role']})")

    y = np.asarray(y, dtype=np.float64)
    inv = resample_mask(invalid, fs_in, fs_out, y.size)

    interpolated = np.zeros_like(inv)
    if key in MAX_GAP_INTERP_S:
        interpolated = short_gaps(inv, int(MAX_GAP_INTERP_S[key] * fs_out))
        inv = inv & ~interpolated
    return dict(y=y, fs=fs_out, invalid=inv, interpolated=interpolated)


def motion_index(acc, fs, window_s=1.0):
    """Índice de movimiento: desviación estándar de la magnitud del acelerómetro
    en ventanas móviles de window_s segundos (en g). 0 = quieto."""
    from .calidad import rolling_std
    mag = np.linalg.norm(acc, axis=1) if acc.ndim == 2 else acc
    return rolling_std(mag, int(window_s * fs))
