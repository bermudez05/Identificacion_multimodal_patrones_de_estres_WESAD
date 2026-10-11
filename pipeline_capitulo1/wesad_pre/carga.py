"""
carga.py — Lectura de los archivos de WESAD (etapa 0).

- load_subject : abre SX.pkl (un sujeto a la vez, para no llenar la memoria).
- get_signal   : devuelve una señal en unidades físicas, como arreglo float.
- read_quest   : lee SX_quest.csv (orden del protocolo y cuestionarios).
- read_notes   : lee las notas del investigador en SX_readme.txt.
"""
import os
import pickle

import numpy as np
import pandas as pd

from .config import SIGNALS


def load_subject(data_path, sid):
    """Abre WESAD/S{sid}/S{sid}.pkl. Devuelve el diccionario original del dataset:
    {'signal': {'chest': {...}, 'wrist': {...}}, 'label': array, 'subject': 'S2'}."""
    file_path = os.path.join(data_path, f"S{sid}", f"S{sid}.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def get_signal(data, key):
    """Devuelve la señal `key` (p. ej. 'chest_ECG') en unidades físicas.

    - Señales de un canal -> arreglo 1D.
    - Acelerómetros -> arreglo 2D (n, 3) en g.
    Los .pkl ya traen el pecho en unidades físicas; solo el ACC de muñeca
    necesita conversión (el Empatica E4 lo guarda en unidades de 1/64 g)."""
    spec = SIGNALS[key]
    x = np.asarray(data["signal"][spec["device"]][spec["channel"]], dtype=np.float64)
    if key == "wrist_ACC":
        x = x / 64.0
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    return x


def magnitude(x):
    """Magnitud euclidiana de un acelerómetro de 3 ejes (o la señal tal cual si es 1D)."""
    x = np.asarray(x)
    return np.linalg.norm(x, axis=1) if x.ndim == 2 else x


# ---------------------------------------------------------------------------
# Cuestionarios y notas
# ---------------------------------------------------------------------------

# Condiciones del protocolo tal como aparecen en la fila "# ORDER" de SX_quest.csv
QUEST_TO_COND = {"Base": "baseline", "TSST": "stress", "Fun": "amusement",
                 "Medi 1": "meditation_1", "Medi 2": "meditation_2"}


def read_quest(data_path, sid):
    """Lee SX_quest.csv. Devuelve un diccionario con:
      - 'protocolo': DataFrame (condición, inicio_min, fin_min) en el orden en que se aplicaron.
      - 'PANAS', 'STAI', 'DIM' (SAM: valencia y activación): DataFrame con una fila por
        condición afectiva (Base, TSST, Medi 1, Fun, Medi 2, en el orden aplicado).
      - 'SSSQ': ítems del cuestionario de estrés, aplicado solo después del TSST.
    Las lecturas (sRead, fRead) no tienen cuestionario."""
    file_path = os.path.join(data_path, f"S{sid}", f"S{sid}_quest.csv")
    rows = {}
    with open(file_path, encoding="latin1") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(";")]
            if not parts or not parts[0].startswith("#"):
                continue
            tag = parts[0].lstrip("#").strip()
            values = [p for p in parts[1:] if p != ""]
            rows.setdefault(tag, []).append(values)

    order = rows["ORDER"][0]
    start = [float(v) for v in rows["START"][0]]
    end = [float(v) for v in rows["END"][0]]
    protocolo = pd.DataFrame({"fase": order, "inicio_min": start, "fin_min": end})

    # Los cuestionarios se aplicaron después de cada fase afectiva, en el orden del protocolo
    affective = [o for o in order if o in QUEST_TO_COND]
    out = {"protocolo": protocolo}
    for tag in ("PANAS", "STAI", "DIM"):
        mat = rows.get(tag, [])
        if len(mat) == len(affective):
            width = max(len(r) for r in mat)
            arr = np.full((len(mat), width), np.nan)
            for i, r in enumerate(mat):
                arr[i, :len(r)] = [float(v) for v in r]
            df = pd.DataFrame(arr, columns=[f"item_{j + 1}" for j in range(width)])
            df.insert(0, "condicion", [QUEST_TO_COND[a] for a in affective])
            out[tag] = df
    if "SSSQ" in rows:
        out["SSSQ"] = pd.DataFrame([[float(v) for v in rows["SSSQ"][0]]],
                                   columns=[f"item_{j + 1}" for j in range(len(rows["SSSQ"][0]))])
    return out


def read_notes(data_path, sid):
    """Devuelve el texto de 'Additional notes' de SX_readme.txt (o '' si no hay notas)."""
    file_path = os.path.join(data_path, f"S{sid}", f"S{sid}_readme.txt")
    if not os.path.exists(file_path):
        return ""
    with open(file_path, encoding="latin1") as f:
        text = f.read()
    if "Additional notes" not in text:
        return ""
    notes = text.split("Additional notes")[-1].split("###")[-1].strip()
    return "" if notes in ("", "-") else " ".join(notes.split())


def read_info(data_path, sid):
    """Datos personales y prerrequisitos del readme (edad, sexo, café, etc.)."""
    file_path = os.path.join(data_path, f"S{sid}", f"S{sid}_readme.txt")
    info = {}
    with open(file_path, encoding="latin1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "?" in line:                       # "Did you drink coffee today? NO"
                k, v = line.split("?", 1)
                info[k.strip() + "?"] = v.strip()
            elif ":" in line:                     # "Age: 27"
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    return info
