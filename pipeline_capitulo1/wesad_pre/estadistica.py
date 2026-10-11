"""
estadistica.py — Pruebas para comparar condiciones (bloque G del EDA).

Se usan pruebas NO paramétricas y PAREADAS porque:
  - hay solo 15 sujetos (no se puede suponer normalidad),
  - cada sujeto pasa por todas las condiciones (medidas repetidas).
Se usa UN valor por sujeto y condición (no miles de ventanas), para no inflar
artificialmente la significancia con ventanas que se solapan.
"""
import numpy as np
import pandas as pd
from scipy import stats


def holm(pvals):
    """Corrección de Holm-Bonferroni para comparaciones múltiples."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def rank_biserial(x, y):
    """Tamaño de efecto para Wilcoxon pareado (correlación rango-biserial, -1 a 1).
    Positivo = y tiende a ser mayor que x. |r| ~0.1 pequeño, ~0.3 mediano, >=0.5 grande."""
    d = np.asarray(y) - np.asarray(x)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def compare_conditions(markers, conds=("baseline", "stress", "amusement"),
                       pairs=(("baseline", "stress"), ("stress", "amusement"), ("baseline", "amusement"))):
    """Para cada (señal, marcador): Friedman entre condiciones + Wilcoxon por pares.

    markers: tabla larga de marcadores.subject_markers (sujeto, condicion, señal, marcador, valor).
    Devuelve una fila por (señal, marcador) con medianas, p-valores y tamaños de efecto."""
    rows = []
    for (sig, mk), g in markers.groupby(["señal", "marcador"]):
        wide = g.pivot_table(index="sujeto", columns="condicion", values="valor").reindex(columns=list(conds))
        wide = wide.dropna()
        row = dict(señal=sig, marcador=mk, unidad=g.unidad.iloc[0], n_sujetos=len(wide))
        for c in conds:
            row[f"mediana_{c}"] = wide[c].median() if len(wide) else np.nan
        if len(wide) >= 5:
            chi2, p = stats.friedmanchisquare(*[wide[c].values for c in conds])
            row["friedman_p"] = p
            row["kendall_W"] = chi2 / (len(wide) * (len(conds) - 1))   # 0 = nada, 1 = máximo acuerdo
            ps = []
            for a, b in pairs:
                try:
                    p_ab = stats.wilcoxon(wide[a], wide[b]).pvalue
                except ValueError:
                    p_ab = 1.0
                ps.append(p_ab)
                row[f"r_{a}_vs_{b}"] = rank_biserial(wide[a], wide[b])
            for (a, b), padj in zip(pairs, holm(ps)):
                row[f"p_holm_{a}_vs_{b}"] = padj
        rows.append(row)
    return pd.DataFrame(rows)


def bland_altman(a, b):
    """Concordancia entre dos mediciones del mismo fenómeno (p. ej. FC por ECG y por BVP).
    Devuelve sesgo (media de a - b) y límites de acuerdo al 95 %."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    bias, sd = d.mean(), d.std(ddof=1)
    return dict(sesgo=bias, lim_inf=bias - 1.96 * sd, lim_sup=bias + 1.96 * sd, n=int(ok.sum()))


def between_subject_cv(markers, cond="baseline"):
    """Coeficiente de variación ENTRE sujetos de cada marcador en la línea base (%).
    Un CV alto indica que cada persona tiene su propio 'nivel' y justifica normalizar por sujeto."""
    g = markers[markers.condicion == cond].groupby(["señal", "marcador"]).valor
    return (100 * g.std() / g.mean().abs()).rename("cv_entre_sujetos_pct").reset_index()
