"""
ventanas.py — Segmentación en ventanas (etapa 6).

Opción estándar de WESAD (Schmidt et al. 2018): ventanas de 60 s que avanzan
0.25 s. Cada ventana queda completamente dentro de UN bloque de condición
(nunca mezcla, por ejemplo, el final de la línea base con el inicio del estrés).

La tabla de ventanas NO contiene la señal: contiene el "índice" (sujeto, inicio,
fin, etiqueta) y la calidad de cada modalidad dentro de la ventana. En el
objetivo 2, la extracción de características recorre esta tabla.
"""
import numpy as np
import pandas as pd

from .config import MODEL_SIGNALS, TASKS, WINDOW


def _window_fraction(mask, fs, starts, ends):
    """Fracción de muestras True de `mask` dentro de cada ventana [start, end)."""
    c = np.concatenate([[0], np.cumsum(np.asarray(mask, dtype=np.int64))])
    a = np.clip((starts * fs).astype(np.int64), 0, mask.size)
    b = np.clip((ends * fs).astype(np.int64), 0, mask.size)
    return (c[b] - c[a]) / np.maximum(b - a, 1)


def _window_mean(x, fs, starts, ends):
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    c = np.concatenate([[0.0], np.cumsum(x)])
    a = np.clip((starts * fs).astype(np.int64), 0, x.size)
    b = np.clip((ends * fs).astype(np.int64), 0, x.size)
    return (c[b] - c[a]) / np.maximum(b - a, 1)


def subject_windows(proc, cfg=WINDOW):
    """Tabla de ventanas de un sujeto con su etiqueta y la calidad por modalidad."""
    L, step = cfg["length_s"], cfg["step_s"]
    parts = []
    for _, b in proc["bloques"].iterrows():
        first = b.inicio_s + cfg["trim_start_s"]
        if b.fin_s - first < L:
            continue
        starts = np.arange(first, b.fin_s - L + 1e-9, step)
        parts.append(pd.DataFrame(dict(sujeto=proc["sujeto"], condicion=b.condicion,
                                       bloque=int(b.bloque), inicio_s=starts, fin_s=starts + L)))
    if not parts:
        return pd.DataFrame()
    w = pd.concat(parts, ignore_index=True)
    st, en = w.inicio_s.values, w.fin_s.values

    for task, mapping in TASKS.items():
        w[task] = w.condicion.map(mapping)          # meditación queda en NaN

    for key in MODEL_SIGNALS:
        sig = proc["señales"][key]
        frac_bad = _window_fraction(sig["invalid"], sig["fs"], st, en)
        w[f"frac_valida__{key}"] = 1.0 - frac_bad
        w[f"frac_marcada__{key}"] = _window_fraction(sig["marcar"], sig["fs"], st, en)
        w[f"valida__{key}"] = w[f"frac_valida__{key}"] >= cfg["min_valid_frac"]

    # Calidad fisiológica adicional
    ev = proc["eventos"]
    ecg = ev["chest_ECG"]
    # fracción de la ventana en segmentos de ECG aceptables o excelentes (Zhao y Zhang 2018)
    w["ecg_frac_sqi_ok"] = _window_fraction(ecg["sqi_segmentos"] >= 1, 1.0 / ecg["sqi_segmento_s"], st, en)
    # Solo el ACC de pecho: el modelo usa solo sensores de pecho (el de muñeca queda para el EDA)
    for key in ("chest_ACC",):
        m = proc["movimiento"][key]
        w[f"movimiento__{key}"] = _window_mean(m["y"], m["fs"], st, en)

    w["n_modalidades_validas"] = w[[f"valida__{k}" for k in MODEL_SIGNALS]].sum(axis=1)
    w["todas_validas"] = w["n_modalidades_validas"] == len(MODEL_SIGNALS)
    return w


# ---------------------------------------------------------------------------
# Normalización (etapa 7) — se aplica en el objetivo 2, DENTRO de cada pliegue
# ---------------------------------------------------------------------------

def zscore_por_sujeto(df, cols, ref_condition="baseline"):
    """Opción A (estandarización dentro del sujeto, Ben-Shakhar 1985; ver también la calibración
    personalizada de Nkurikiyeyezu et al. 2019): cada sujeto se normaliza con la media y la
    desviación de SU PROPIA línea base. Supone que al usar el sistema con una persona
    nueva se graba primero un reposo corto de calibración."""
    out = df.copy()
    for sid, g in df.groupby("sujeto"):
        ref = g[g.condicion == ref_condition][cols]
        mu, sd = ref.mean(), ref.std(ddof=0).replace(0, np.nan)
        out.loc[g.index, cols] = (g[cols] - mu) / sd
    return out


class ZScoreGlobal:
    """Opción B: media y desviación calculadas SOLO con los sujetos de entrenamiento del
    pliegue y aplicadas tal cual al sujeto de prueba (sin calibración)."""

    def fit(self, train_df, cols):
        self.cols = cols
        self.mu = train_df[cols].mean()
        self.sd = train_df[cols].std(ddof=0).replace(0, np.nan)
        return self

    def transform(self, df):
        out = df.copy()
        out[self.cols] = (df[self.cols] - self.mu) / self.sd
        return out
