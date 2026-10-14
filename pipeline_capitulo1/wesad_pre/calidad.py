"""
calidad.py — Control de calidad a nivel de muestra (etapa 2).

Regla de oro: aquí SOLO SE DETECTA. Ninguna función modifica la señal; todas
devuelven máscaras booleanas (True = la muestra tiene ese problema). La decisión
de qué hacer con cada problema está en config.QC ("excluir" o "marcar").

El QC se calcula una sola vez sobre los 15 sujetos, antes de cualquier división
entrenamiento/prueba, para que las dos particiones tengan exactamente el mismo
criterio.
"""
import numpy as np

from .config import QC, SIGNALS
from .etiquetas import find_blocks


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def keep_long_runs(mask, min_len):
    """Deja en True solo los tramos contiguos de al menos min_len muestras."""
    mask = np.asarray(mask, dtype=bool)
    if min_len <= 1:
        return mask.copy()
    out = np.zeros_like(mask)
    for s, e in find_blocks(mask):
        if e - s >= min_len:
            out[s:e] = True
    return out


def rolling_std(x, w):
    """Desviación estándar en una ventana móvil centrada de w muestras (vía sumas acumuladas)."""
    x = np.asarray(x, dtype=np.float64)
    w = max(int(w), 2)
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    n = x.size
    half = w // 2
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) - half + w, 0, n)
    cnt = hi - lo
    mean = (c1[hi] - c1[lo]) / cnt
    var = (c2[hi] - c2[lo]) / cnt - mean ** 2
    return np.sqrt(np.clip(var, 0, None))


def dilate(mask, pad):
    """Extiende cada tramo marcado `pad` muestras hacia ambos lados."""
    mask = np.asarray(mask, dtype=bool)
    pad = int(pad)
    if pad <= 0 or not mask.any():
        return mask
    # suma móvil de ancho 2*pad+1 con sumas acumuladas (rápido incluso a 700 Hz)
    c = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
    n = mask.size
    idx = np.arange(n)
    lo = np.clip(idx - pad, 0, n)
    hi = np.clip(idx + pad + 1, 0, n)
    return (c[hi] - c[lo]) > 0


# ---------------------------------------------------------------------------
# Los criterios
# ---------------------------------------------------------------------------

def detect_nan(x):
    """NaN o infinito (en cualquier eje)."""
    bad = ~np.isfinite(x)
    return bad.any(axis=1) if bad.ndim == 2 else bad


def detect_out_of_range(x, low, high):
    """Valor imposible: fuera de [low, high] (en cualquier eje)."""
    bad = (x < low) | (x > high)
    return bad.any(axis=1) if bad.ndim == 2 else bad


def detect_rail(x, rail, min_len):
    """Saturación: |x| pegado al tope del sensor durante >= min_len muestras."""
    at = np.abs(x) >= rail
    at = at.any(axis=1) if at.ndim == 2 else at
    return keep_long_runs(at, min_len)


def detect_flat_std(x, thr, min_len):
    """Señal plana: variación casi nula (std móvil < thr) durante >= min_len muestras."""
    s = rolling_std(np.nan_to_num(x), min_len)
    return dilate(s < thr, min_len // 2)


def detect_flat_equal(x, min_len):
    """Sensor pegado: el mismo valor exacto repetido durante >= min_len muestras."""
    same = np.r_[False, np.diff(x) == 0]
    runs = keep_long_runs(same, min_len)
    # incluir la muestra que inicia cada tramo
    return runs | np.r_[runs[1:], False]


def detect_slope(x, fs, max_slope, pad_len):
    """Salto abrupto: |x(t+0.25 s) - x(t)| / 0.25 s > max_slope (Kleckner et al. 2018)."""
    lag = max(int(round(0.25 * fs)), 1)
    slope = np.zeros(x.size)
    slope[lag:] = np.abs(x[lag:] - x[:-lag]) / (lag / fs)
    return dilate(slope > max_slope, pad_len)


# ---------------------------------------------------------------------------
# Todo junto
# ---------------------------------------------------------------------------

def run_qc(x, key, cfg=QC):
    """Aplica a la señal `key` todos los criterios configurados.

    Devuelve:
      masks : {criterio: máscara booleana}
      levels: {criterio: "excluir" | "marcar"}
    """
    fs = SIGNALS[key]["fs"]
    c = cfg.get(key, {})
    masks, levels = {}, {}

    masks["nan"] = detect_nan(x)
    levels["nan"] = "excluir"
    xf = np.where(np.isfinite(x), x, 0.0)
    x1 = np.linalg.norm(xf, axis=1) if xf.ndim == 2 else xf

    if "range" in c:
        masks["fuera_de_rango"] = detect_out_of_range(xf, *c["range"])
        levels["fuera_de_rango"] = "excluir"
    if "rail" in c:
        r = c["rail"]
        masks["saturacion"] = detect_rail(xf, r["value"], int(r["min_s"] * fs))
        levels["saturacion"] = r["level"]
    if "rail_breve" in c:
        at = np.abs(xf) >= c["rail_breve"]["value"]
        at = at.any(axis=1) if at.ndim == 2 else at
        masks["saturacion_breve"] = at & ~masks.get("saturacion", np.zeros_like(at))
        levels["saturacion_breve"] = c["rail_breve"]["level"]
    if "flat_std" in c:
        f = c["flat_std"]
        masks["plana"] = detect_flat_std(x1, f["value"], int(f["min_s"] * fs))
        levels["plana"] = f["level"]
    if "flat_equal_s" in c:
        f = c["flat_equal_s"]
        masks["pegada"] = detect_flat_equal(x1, int(f["value"] * fs))
        levels["pegada"] = f["level"]
    if "slope" in c:
        s = c["slope"]
        masks["salto"] = detect_slope(x1, fs, s["value"], int(s["pad_s"] * fs))
        levels["salto"] = s["level"]
    return masks, levels


def combine(masks, levels, level):
    """Une (OR) todas las máscaras de un nivel ("excluir" o "marcar")."""
    out = None
    for name, m in masks.items():
        if levels[name] == level:
            out = m.copy() if out is None else (out | m)
    if out is None:
        n = len(next(iter(masks.values())))
        out = np.zeros(n, dtype=bool)
    return out


def summarize(masks, levels, key, sid):
    """Una fila de resumen: % de muestras afectadas por cada criterio."""
    n = len(next(iter(masks.values())))
    row = dict(sujeto=f"S{sid}", señal=key, n_muestras=n)
    for name, m in masks.items():
        row[f"pct_{name}"] = 100.0 * m.sum() / n
    row["pct_excluido"] = 100.0 * combine(masks, levels, "excluir").sum() / n
    row["pct_marcado"] = 100.0 * combine(masks, levels, "marcar").sum() / n
    return row


def segments_table(masks, key, sid):
    """Lista de tramos afectados (inicio, fin, duración, criterio) para el anexo."""
    fs = SIGNALS[key]["fs"]
    rows = []
    for name, m in masks.items():
        for s, e in find_blocks(m):
            rows.append(dict(sujeto=f"S{sid}", señal=key, criterio=name,
                             inicio_s=s / fs, fin_s=e / fs, duracion_s=(e - s) / fs))
    return rows


# ---------------------------------------------------------------------------
# Calidad fisiológica (etapa 3): ¿la forma de onda sigue siendo interpretable?
# ---------------------------------------------------------------------------
def calidad_fisiologica(proc):
    """Indicadores de calidad fisiológica de las 4 señales del modelo, por bloque de condición.

    A diferencia del QC por muestra, estos indicadores miden si los eventos fisiológicos
    (latidos, respiraciones, nivel de sudoración, temperatura de piel) son interpretables:
      - ECG: % de segmentos de 10 s con calidad excelente o aceptable (Zhao y Zhang, 2018)
        y % de intervalos RR válidos tras la corrección de latidos.
      - Respiración: % de ciclos válidos (duración de 1.5 a 20 s, sin muestras excluidas).
      - EDA de pecho: % de muestras válidas según el QC de Kleckner et al. (2018).
      - Temperatura de pecho: mediana del bloque (°C), para detectar sensores mal adheridos.
    Devuelve una lista de filas (una por bloque); se resume por sujeto y condición en el notebook 03.
    """
    ev, sig = proc["eventos"], proc["señales"]
    seg = ev["chest_ECG"]["sqi_segmentos"]
    seg_s = ev["chest_ECG"]["sqi_segmento_s"]
    t_seg = np.arange(seg.size) * seg_s + seg_s / 2
    rr = ev["chest_ECG"]["rr"]
    br = ev["chest_Resp"]["breaths"]
    rows = []
    for _, b in proc["bloques"].iterrows():
        a, z = b.inicio_s, b.fin_s
        m_seg = (t_seg >= a) & (t_seg < z)
        rr_b = rr[(rr.t_s >= a) & (rr.t_s < z)]
        br_b = br[(br.t_s >= a) & (br.t_s < z)]

        def pct_validas(key):
            s = sig[key]
            i0, i1 = int(a * s["fs"]), int(z * s["fs"])
            return 100.0 * (1.0 - s["invalid"][i0:i1].mean())

        t = sig["chest_Temp"]
        rows.append(dict(
            sujeto=proc["sujeto"], condicion=b.condicion, bloque=int(b.bloque), duracion_s=float(b.duracion_s),
            ecg_pct_seg_aceptables=100.0 * (seg[m_seg] >= 1).mean(),
            ecg_pct_seg_excelentes=100.0 * (seg[m_seg] == 2).mean(),
            ecg_pct_rr_validos=100.0 * rr_b.valido.mean(),
            resp_pct_ciclos_validos=100.0 * br_b.valido.mean(),
            eda_pecho_pct_validas=pct_validas("chest_EDA"),
            temp_pecho_mediana_c=float(np.median(t["y"][int(a * t["fs"]):int(z * t["fs"])])),
        ))
    return rows
