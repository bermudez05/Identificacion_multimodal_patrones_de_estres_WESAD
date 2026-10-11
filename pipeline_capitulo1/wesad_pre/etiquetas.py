"""
etiquetas.py — Etiquetas del protocolo y bloques de condición (etapa 1).

Las etiquetas de WESAD vienen a 700 Hz (una por muestra del pecho). Como cada
señal tiene su propia frecuencia de muestreo, aquí se "traduce" la etiqueta al
tiempo de cada señal y se arman los bloques contiguos de cada condición.
"""
import numpy as np
import pandas as pd

from .config import CONDITIONS, FS_LABEL


def find_blocks(mask):
    """Lista de (inicio, fin) de los tramos contiguos donde mask es True (fin exclusivo)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [mask.size]
    return list(zip(starts, ends))


def condition_blocks(label, fs_label=FS_LABEL):
    """Tabla con un renglón por bloque contiguo de condición (solo 1-4).
    Columnas: cond_id, condicion, bloque (1, 2... si la condición se repite),
    inicio_s, fin_s, duracion_s."""
    rows = []
    for cid, name in CONDITIONS.items():
        for k, (s, e) in enumerate(find_blocks(label == cid), start=1):
            rows.append(dict(cond_id=cid, condicion=name, bloque=k,
                             inicio_s=s / fs_label, fin_s=e / fs_label,
                             duracion_s=(e - s) / fs_label))
    return pd.DataFrame(rows).sort_values("inicio_s").reset_index(drop=True)


def label_at_fs(label, fs, n, fs_label=FS_LABEL):
    """Etiqueta para cada una de las n muestras de una señal muestreada a fs Hz."""
    idx = np.minimum((np.arange(n) * fs_label / fs).astype(np.int64), len(label) - 1)
    return np.asarray(label)[idx]


def time_axis(n, fs):
    """Tiempo en segundos de cada muestra."""
    return np.arange(n) / fs
