"""
eventos.py — Eventos fisiológicos y calidad (etapa 4).

De cada señal limpia se extraen los "eventos" que luego alimentan las
características del objetivo 2:
  - ECG  -> picos R e intervalos RR (base de la frecuencia cardíaca y la HRV)
  - EDA  -> componente tónico (SCL), componente fásico y respuestas SCR
  - Resp -> respiraciones (inicio, pico, amplitud, duración)

Todas las funciones reciben la máscara de muestras inválidas para no producir
eventos en tramos excluidos por el QC.
"""
import warnings

import numpy as np
import pandas as pd

from .config import EVENTS

try:
    import neurokit2 as nk
except ImportError:
    nk = None


def _span_invalid(invalid, a, b):
    """True si hay alguna muestra inválida entre los índices a y b (inclusive)."""
    c = np.concatenate([[0], np.cumsum(invalid, dtype=np.int64)])
    a = np.asarray(a)
    b = np.minimum(np.asarray(b) + 1, invalid.size)
    return (c[b] - c[a]) > 0


def _fix_peaks(peaks, fs):
    """Corrige latidos perdidos, extra y ectópicos (Lipponen y Tarvainen 2019,
    implementado en NeuroKit2 como method='Kubios'). Devuelve (picos, n_corregidos)."""
    if len(peaks) < 10:
        return np.asarray(peaks), 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info, fixed = nk.signal_fixpeaks(peaks, sampling_rate=fs, iterative=True, method="Kubios")
    n_fix = sum(len(info.get(k, [])) for k in ("ectopic", "missed", "extra", "longshort"))
    return np.asarray(fixed, dtype=np.int64), int(n_fix)


def _interval_table(peaks, fs, invalid, quality=None, qmin=None):
    """Tabla de intervalos entre latidos consecutivos, con su validez."""
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size < 2:
        return pd.DataFrame(columns=["t_s", "ibi_ms", "valido"])
    a, b = peaks[:-1], peaks[1:]
    ibi = (b - a) / fs * 1000.0
    lo, hi = EVENTS["rr_ms_range"]
    ok = (ibi >= lo) & (ibi <= hi) & ~_span_invalid(invalid, a, b)
    if quality is not None:
        ok &= (quality[a] >= qmin) & (quality[b] >= qmin)
    return pd.DataFrame({"t_s": b / fs, "ibi_ms": ibi, "valido": ok})


# ---------------------------------------------------------------------------
# ECG
# ---------------------------------------------------------------------------

ZHAO_CODES = {"Excellent": 2, "Barely acceptable": 1, "Unacceptable": 0}


def ecg_sqi_zhao(y, fs, segment_s):
    """Calidad del ECG por segmentos de segment_s segundos con el método de Zhao y
    Zhang (2018): 2 = excelente, 1 = aceptable, 0 = inaceptable.
    Devuelve un código por segmento."""
    w = int(segment_s * fs)
    codes = []
    for i in range(0, y.size, w):
        seg = y[i:i + w]
        if seg.size < 0.5 * w:
            codes.append(codes[-1] if codes else 0)
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                codes.append(ZHAO_CODES.get(nk.ecg_quality(seg, sampling_rate=fs, method="zhao2018"), 0))
        except Exception:
            codes.append(0)
    return np.asarray(codes, dtype=np.int8)


def ecg_events(y, fs, invalid):
    """Picos R, intervalos RR y calidad del ECG limpio."""
    _, info = nk.ecg_peaks(y, sampling_rate=fs, method="neurokit")
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    peaks, n_fix = _fix_peaks(peaks, fs)
    peaks = peaks[(peaks >= 0) & (peaks < y.size)]
    seg_s = EVENTS["ecg_sqi_segment_s"]
    sqi = ecg_sqi_zhao(y, fs, seg_s)
    # calidad por muestra: 1 si el segmento es excelente o aceptable
    good = np.repeat(sqi >= 1, int(seg_s * fs))[:y.size]
    if good.size < y.size:
        good = np.r_[good, np.repeat(good[-1:], y.size - good.size)]
    rr = _interval_table(peaks, fs, invalid, good.astype(float), 0.5)
    return dict(peaks=peaks, rr=rr, sqi_segmentos=sqi, sqi_segmento_s=seg_s, n_corregidos=n_fix)


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def eda_events(y, fs, invalid, blocks):
    """Separa SCL (tónico) y componente fásico con cvxEDA (Greco et al. 2016) dentro
    de cada bloque de condición, y detecta las SCR.

    blocks: tabla de etiquetas.condition_blocks (inicio_s, fin_s).
    Fuera de los bloques el tónico/fásico queda en NaN (no se usa para el modelo)."""
    tonic = np.full(y.size, np.nan)
    phasic = np.full(y.size, np.nan)
    scr_rows = []
    method = "cvxeda"
    for _, b in blocks.iterrows():
        s, e = int(b.inicio_s * fs), min(int(b.fin_s * fs), y.size)
        seg = y[s:e]
        if seg.size < 10 * fs:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dec = nk.eda_phasic(seg, sampling_rate=fs, method=method)
        except Exception:
            # Si cvxopt no está instalado o el optimizador falla: filtro pasa-altas 0.05 Hz
            method = "highpass"
            dec = nk.eda_phasic(seg, sampling_rate=fs, method=method)
        tonic[s:e] = dec["EDA_Tonic"].values
        phasic[s:e] = dec["EDA_Phasic"].values
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, info = nk.eda_peaks(dec["EDA_Phasic"].values, sampling_rate=fs,
                                       method="neurokit", amplitude_min=0.1)
        except Exception:
            continue
        for pk, amp in zip(info["SCR_Peaks"], info["SCR_Amplitude"]):
            if np.isfinite(amp) and amp >= EVENTS["scr_min_amplitude_us"] and not invalid[s + int(pk)]:
                scr_rows.append(dict(t_s=(s + int(pk)) / fs, amplitud_us=float(amp)))
    scr = pd.DataFrame(scr_rows, columns=["t_s", "amplitud_us"])
    return dict(tonic=tonic, phasic=phasic, scr=scr, metodo=method)


# ---------------------------------------------------------------------------
# Respiración
# ---------------------------------------------------------------------------

def rsp_events(y, fs, invalid):
    """Respiraciones: cada ciclo va de un valle (inicio de inhalación) al siguiente."""
    _, info = nk.rsp_peaks(y, sampling_rate=fs, method="khodadad2018")
    peaks = np.asarray(info["RSP_Peaks"], dtype=np.int64)
    troughs = np.asarray(info["RSP_Troughs"], dtype=np.int64)
    rows = []
    for i in range(len(troughs) - 1):
        a, b = troughs[i], troughs[i + 1]
        inside = peaks[(peaks > a) & (peaks < b)]
        if inside.size == 0:
            continue
        pk = inside[0]
        dur = (b - a) / fs
        rows.append(dict(t_s=b / fs, duracion_s=dur, frecuencia_rpm=60.0 / dur,
                         amplitud=float(y[pk] - y[a]),
                         valido=bool(~_span_invalid(invalid, np.array([a]), np.array([b]))[0])
                         and 1.5 <= dur <= 20.0))
    breaths = pd.DataFrame(rows, columns=["t_s", "duracion_s", "frecuencia_rpm", "amplitud", "valido"])
    return dict(peaks=peaks, troughs=troughs, breaths=breaths)
