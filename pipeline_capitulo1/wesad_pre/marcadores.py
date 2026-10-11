"""
marcadores.py — Marcadores fisiológicos por sujeto y condición (para el EDA).

No son todavía las características del modelo (eso es el objetivo 2): son un
resumen por condición que permite ver si el estrés se nota en la fisiología y
qué señales lo separan mejor.
"""
import numpy as np
import pandas as pd

from .config import COND_ORDER


def _in_condition(t, blocks, cond):
    """Máscara de tiempos t (s) que caen dentro de algún bloque de la condición."""
    m = np.zeros(len(t), dtype=bool)
    for _, b in blocks[blocks.condicion == cond].iterrows():
        m |= (t >= b.inicio_s) & (t < b.fin_s)
    return m


def _cond_samples(sig, blocks, cond):
    """Muestras válidas de una señal continua dentro de la condición."""
    t = np.arange(sig["y"].size) / sig["fs"]
    m = _in_condition(t, blocks, cond)
    if "invalid" in sig:
        m &= ~sig["invalid"]
    return np.asarray(sig["y"], dtype=float)[m]


def _minutes(blocks, cond):
    return blocks.loc[blocks.condicion == cond, "duracion_s"].sum() / 60.0


def hrv_time(ibi_df):
    """FC, SDNN, RMSSD y pNN50 a partir de una tabla de intervalos (t_s, ibi_ms, valido)."""
    v = ibi_df[ibi_df.valido]
    if len(v) < 20:
        return dict(fc_lpm=np.nan, sdnn_ms=np.nan, rmssd_ms=np.nan, pnn50_pct=np.nan)
    ibi = ibi_df.ibi_ms.values
    ok = ibi_df.valido.values
    # diferencias sucesivas solo entre dos intervalos válidos y contiguos
    pair = ok[1:] & ok[:-1]
    d = np.diff(ibi)[pair]
    return dict(fc_lpm=60000.0 / v.ibi_ms.mean(),
                sdnn_ms=v.ibi_ms.std(ddof=1),
                rmssd_ms=np.sqrt(np.mean(d ** 2)) if d.size else np.nan,
                pnn50_pct=100.0 * np.mean(np.abs(d) > 50) if d.size else np.nan)


def subject_markers(proc):
    """Tabla larga: una fila por (sujeto, condición, marcador)."""
    blocks = proc["bloques"]
    s, ev = proc["señales"], proc["eventos"]
    rows = []

    def add(cond, signal, name, value, unit):
        rows.append(dict(sujeto=proc["sujeto"], condicion=cond, señal=signal,
                         marcador=name, valor=value, unidad=unit))

    for cond in COND_ORDER:
        if not (blocks.condicion == cond).any():
            continue
        mins = _minutes(blocks, cond)

        # ECG y BVP: frecuencia cardíaca y variabilidad
        for key, tab in (("chest_ECG", ev["chest_ECG"]["rr"]), ("wrist_BVP", ev["wrist_BVP"]["ibi"])):
            sub = tab[_in_condition(tab.t_s.values, blocks, cond)]
            h = hrv_time(sub)
            add(cond, key, "FC", h["fc_lpm"], "lpm")
            add(cond, key, "RMSSD", h["rmssd_ms"], "ms")
            add(cond, key, "SDNN", h["sdnn_ms"], "ms")
            add(cond, key, "pNN50", h["pnn50_pct"], "%")
            add(cond, key, "% intervalos válidos", 100 * sub.valido.mean() if len(sub) else np.nan, "%")

        # EDA: nivel tónico y respuestas fásicas
        for key in ("chest_EDA", "wrist_EDA"):
            tonic = dict(y=ev[key]["tonic"], fs=s[key]["fs"], invalid=s[key]["invalid"])
            scl = _cond_samples(tonic, blocks, cond)
            scr = ev[key]["scr"]
            scr_c = scr[_in_condition(scr.t_s.values, blocks, cond)]
            add(cond, key, "SCL", np.nanmean(scl) if scl.size else np.nan, "µS")
            add(cond, key, "SCR por minuto", len(scr_c) / mins if mins else np.nan, "1/min")
            add(cond, key, "Amplitud SCR", scr_c.amplitud_us.mean() if len(scr_c) else np.nan, "µS")

        # Respiración
        br = ev["chest_Resp"]["breaths"]
        br_c = br[_in_condition(br.t_s.values, blocks, cond) & br.valido.values]
        add(cond, "chest_Resp", "Frecuencia respiratoria", br_c.frecuencia_rpm.mean() if len(br_c) else np.nan, "rpm")
        add(cond, "chest_Resp", "Amplitud respiratoria", br_c.amplitud.mean() if len(br_c) else np.nan, "u.a.")
        add(cond, "chest_Resp", "Variabilidad respiratoria (CV)",
            br_c.duracion_s.std() / br_c.duracion_s.mean() if len(br_c) > 2 else np.nan, "")

        # Temperatura
        for key in ("chest_Temp", "wrist_TEMP"):
            x = _cond_samples(s[key], blocks, cond)
            add(cond, key, "Temperatura media", np.mean(x) if x.size else np.nan, "°C")
            slopes = []
            for _, b in blocks[blocks.condicion == cond].iterrows():
                fs = s[key]["fs"]
                seg = np.asarray(s[key]["y"][int(b.inicio_s * fs):int(b.fin_s * fs)], dtype=float)
                inv = s[key]["invalid"][int(b.inicio_s * fs):int(b.fin_s * fs)]
                if seg.size > 10 and (~inv).mean() > 0.8:
                    tt = np.arange(seg.size) / fs / 60.0
                    slopes.append(np.polyfit(tt[~inv], seg[~inv], 1)[0])
            add(cond, key, "Pendiente de temperatura", np.mean(slopes) if slopes else np.nan, "°C/min")

        # Exploratorias: EMG y movimiento
        emg = _cond_samples(proc["emg_rms"], blocks, cond)
        add(cond, "chest_EMG", "RMS de EMG", np.mean(emg) if emg.size else np.nan, "mV")
        for key in ("chest_ACC", "wrist_ACC"):
            mi = _cond_samples(proc["movimiento"][key], blocks, cond)
            add(cond, key, "Índice de movimiento", np.mean(mi) if mi.size else np.nan, "g")
    return pd.DataFrame(rows)
