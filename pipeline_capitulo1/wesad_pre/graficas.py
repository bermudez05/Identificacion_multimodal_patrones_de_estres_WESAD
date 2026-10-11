"""
graficas.py — Funciones de gráficos compartidas por los notebooks.

Todas las figuras se guardan además en salidas/figuras/ (PNG, 200 dpi) para
poder pegarlas directamente en el documento del trabajo de grado.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import welch

from .config import COND_ORDER, OUT_DIRS, SIGNALS

# Un color fijo por condición (paleta validada para daltonismo; la condición
# siempre tiene el mismo color en todos los notebooks).
COND_COLORS = {"baseline": "#2a78d6", "stress": "#eb6834",
               "amusement": "#1baf7a", "meditation": "#eda100"}
COND_ES = {"baseline": "Línea base", "stress": "Estrés",
           "amusement": "Diversión", "meditation": "Meditación"}
INK, MUTED = "#0b0b0b", "#52514e"


def setup():
    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False,
                         "axes.spines.right": False, "grid.color": "#e6e5e0",
                         "axes.titleweight": "bold", "axes.titlesize": 10,
                         "lines.linewidth": 1.0})


def savefig(fig, name, root="."):
    folder = os.path.join(root, OUT_DIRS["figures"])
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, f"{name}.png"), dpi=200, bbox_inches="tight")


def shade_conditions(ax, blocks, alpha=0.12, labels=True):
    """Sombrea en el eje de tiempo (s) los bloques de cada condición."""
    seen = set()
    for _, b in blocks.iterrows():
        lab = COND_ES[b.condicion] if labels and b.condicion not in seen else None
        ax.axvspan(b.inicio_s, b.fin_s, color=COND_COLORS[b.condicion], alpha=alpha, lw=0, label=lab)
        seen.add(b.condicion)


def plot_session(signals, blocks, title, decimate_to=4.0):
    """Sesión completa de un sujeto: una fila por señal, condiciones sombreadas.
    signals: lista de (nombre, arreglo_1d, fs, unidad)."""
    fig, axes = plt.subplots(len(signals), 1, figsize=(13, 1.35 * len(signals)), sharex=True)
    for ax, (name, x, fs, unit) in zip(np.atleast_1d(axes), signals):
        step = max(int(fs / decimate_to), 1)
        xs = x[::step]
        t = np.arange(xs.size) * step / fs
        ax.plot(t / 60, xs, color=INK, lw=0.5)
        shade = blocks.assign(inicio_s=blocks.inicio_s / 60, fin_s=blocks.fin_s / 60)
        shade_conditions(ax, shade, labels=(ax is np.atleast_1d(axes)[0]))
        ax.set_ylabel(f"{name}\n({unit})", fontsize=8, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("Tiempo (min)")
    np.atleast_1d(axes)[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.9), ncol=4, fontsize=8, frameon=False)
    fig.suptitle(title, y=1.0 + 0.25 / len(signals), fontweight="bold")
    fig.tight_layout()
    return fig


def psd(x, fs, nperseg_s=8.0):
    """Densidad espectral de potencia (Welch) con ventanas de nperseg_s segundos."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    nper = int(min(len(x), max(nperseg_s * fs, 256)))
    return welch(x - x.mean(), fs=fs, nperseg=nper)


def plot_markers_by_condition(markers, items, conds=("baseline", "stress", "amusement", "meditation"),
                              ncols=4, title=None):
    """Pequeños múltiplos: un panel por marcador. Cada línea gris = un sujeto;
    puntos de color = mediana por condición."""
    n = len(items)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    xs = np.arange(len(conds))
    for ax, (sig, mk) in zip(axes, items):
        g = markers[(markers.señal == sig) & (markers.marcador == mk)]
        wide = g.pivot_table(index="sujeto", columns="condicion", values="valor").reindex(columns=list(conds))
        for _, r in wide.iterrows():
            ax.plot(xs, r.values, color="#b9b8b2", lw=0.8, marker="o", ms=2.5)
        med = wide.median()
        for i, c in enumerate(conds):
            ax.plot(i, med[c], "o", ms=8, color=COND_COLORS[c], mec="white", mew=1.5, zorder=3)
        ax.plot(xs, med.values, color=INK, lw=1.6, zorder=2)
        ax.set_xticks(xs, [COND_ES[c] for c in conds], rotation=25, fontsize=7.5)
        unit = g.unidad.iloc[0] if len(g) else ""
        ax.set_title(f"{mk}\n{SIGNALS[sig]['name']}", fontsize=8.5)
        ax.set_ylabel(unit, fontsize=8)
        ax.tick_params(axis="y", labelsize=7.5)
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def subject_sort_key(s):
    return int(str(s).lstrip("S"))


def order_subjects(df, col="sujeto"):
    cats = sorted(df[col].unique(), key=subject_sort_key)
    return pd.Categorical(df[col], categories=cats, ordered=True)
