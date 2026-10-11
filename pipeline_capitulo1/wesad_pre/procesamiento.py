"""
procesamiento.py — Orquesta las etapas 2 a 5 para un sujeto y guarda el resultado.

Se usa desde los notebooks 02 y 03. Procesa un sujeto a la vez y libera memoria,
igual que los notebooks originales.
"""
import gc
import gzip
import os
import pickle

import numpy as np

from . import calidad, carga, eventos, etiquetas, limpieza
from .config import ALL_SIGNALS, EVENTS, MODEL_SIGNALS, OUT_DIRS, SIGNALS


def qc_path(sid, root="."):
    return os.path.join(root, OUT_DIRS["qc"], f"S{sid}_qc.npz")


def clean_path(sid, root="."):
    return os.path.join(root, OUT_DIRS["clean"], f"S{sid}_procesado.pkl.gz")


# ---------------------------------------------------------------------------
# Etapa 2: QC de las 10 señales
# ---------------------------------------------------------------------------

def qc_subject(data, sid, root="."):
    """Corre el QC sobre las 10 señales de un sujeto. Guarda, por señal, las máscaras
    'excluir' y 'marcar' en salidas/qc/S{id}_qc.npz y devuelve (filas_resumen, tramos)."""
    os.makedirs(os.path.join(root, OUT_DIRS["qc"]), exist_ok=True)
    resumen, tramos, store = [], [], {}
    for key in ALL_SIGNALS:
        x = carga.get_signal(data, key)
        masks, levels = calidad.run_qc(x, key)
        resumen.append(calidad.summarize(masks, levels, key, sid))
        tramos.extend(calidad.segments_table(masks, key, sid))
        store[f"{key}__excluir"] = np.packbits(calidad.combine(masks, levels, "excluir"))
        store[f"{key}__marcar"] = np.packbits(calidad.combine(masks, levels, "marcar"))
        store[f"{key}__n"] = np.array(len(x))
    np.savez_compressed(qc_path(sid, root), **store)
    return resumen, tramos


def load_qc(sid, root="."):
    """Lee las máscaras guardadas por qc_subject. Devuelve {señal: {'excluir', 'marcar'}}."""
    z = np.load(qc_path(sid, root))
    out = {}
    for key in ALL_SIGNALS:
        n = int(z[f"{key}__n"])
        out[key] = {lvl: np.unpackbits(z[f"{key}__{lvl}"])[:n].astype(bool)
                    for lvl in ("excluir", "marcar")}
    return out


# ---------------------------------------------------------------------------
# Etapas 3-5: limpieza, eventos y huecos de las señales del modelo
# ---------------------------------------------------------------------------

def process_subject(data, sid, root=".", qc=None):
    """Limpia las señales del modelo, extrae eventos y calcula el índice de movimiento.
    Guarda todo en salidas/limpias/S{id}_procesado.pkl.gz y lo devuelve."""
    if qc is None:
        qc = load_qc(sid, root)
    label = np.asarray(data["label"])
    blocks = etiquetas.condition_blocks(label)
    out = dict(sujeto=f"S{sid}", bloques=blocks, n_label=len(label), señales={}, eventos={})

    for key in MODEL_SIGNALS:
        x = carga.get_signal(data, key)
        res = limpieza.clean_signal(x, key, qc[key]["excluir"])
        marcar = limpieza.resample_mask(qc[key]["marcar"], SIGNALS[key]["fs"], res["fs"], res["y"].size)
        out["señales"][key] = dict(y=res["y"].astype(np.float32), fs=res["fs"],
                                   invalid=res["invalid"], marcar=marcar,
                                   interpolated=res["interpolated"])

    s = out["señales"]
    ev = out["eventos"]
    ev["chest_ECG"] = eventos.ecg_events(s["chest_ECG"]["y"].astype(np.float64), s["chest_ECG"]["fs"],
                                         s["chest_ECG"]["invalid"])
    ev["wrist_BVP"] = eventos.ppg_events(s["wrist_BVP"]["y"].astype(np.float64), s["wrist_BVP"]["fs"],
                                         s["wrist_BVP"]["invalid"])
    for key in ("chest_EDA", "wrist_EDA"):
        ev[key] = eventos.eda_events(s[key]["y"].astype(np.float64), s[key]["fs"],
                                     s[key]["invalid"], blocks)
    ev["chest_Resp"] = eventos.rsp_events(s["chest_Resp"]["y"].astype(np.float64), s["chest_Resp"]["fs"],
                                          s["chest_Resp"]["invalid"])

    # Movimiento (no es predictor: sirve para marcar ventanas) a 4 Hz
    out["movimiento"] = {}
    for key in ("chest_ACC", "wrist_ACC"):
        acc = carga.get_signal(data, key)
        fs = SIGNALS[key]["fs"]
        mi = limpieza.motion_index(acc, fs, EVENTS["motion_window_s"])
        out["movimiento"][key] = dict(y=limpieza.resample(mi, fs, 4).astype(np.float32), fs=4)

    # EMG (exploratoria): envolvente RMS de 1 s tras pasa-altas 30 Hz, a 4 Hz, solo para el EDA
    emg = carga.get_signal(data, "chest_EMG")
    emg = limpieza.butter_filter(limpieza.notch(emg, 700, 50), 700, "highpass", 30.0, 4)
    from .calidad import rolling_std
    out["emg_rms"] = dict(y=limpieza.resample(rolling_std(emg, 700), 700, 4).astype(np.float32), fs=4)

    os.makedirs(os.path.join(root, OUT_DIRS["clean"]), exist_ok=True)
    # comprimido con gzip: ~20 MB por sujeto en vez de ~40 MB
    with gzip.open(clean_path(sid, root), "wb", compresslevel=6) as f:
        pickle.dump(out, f, protocol=4)
    gc.collect()
    return out


def load_processed(sid, root="."):
    """Abre salidas/limpias/S{id}_procesado.pkl.gz (lo guarda el notebook 03)."""
    with gzip.open(clean_path(sid, root), "rb") as f:
        return pickle.load(f)
