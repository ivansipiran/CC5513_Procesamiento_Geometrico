"""Registro de métodos: cómo obtener una malla simplificada con cada uno."""
import time

import numpy as np

from . import baselines
from .decimate import PairCollapse

# nombre → (tipo, parámetros, etiqueta corta para tablas)
METHODS = {
    "qem":            ("collapse", dict(cost="qem", placement="optimal"), "QEM (v̄ óptimo)"),
    "qem-svd":        ("collapse", dict(cost="qem", placement="svd"), "QEM (pseudo-inversa)"),
    "qem-subset":     ("collapse", dict(cost="qem", placement="subset"), "QEM (mejor de v₁,v₂,medio)"),
    "qem-midpoint":   ("collapse", dict(cost="qem", placement="midpoint"), "QEM (punto medio)"),
    "qem-noarea":     ("collapse", dict(cost="qem", weighting="none"), "QEM sin ponderar por área"),
    "qem-noborder":   ("collapse", dict(cost="qem", boundary_weight=0.0), "QEM sin restricción de borde"),
    "length":         ("collapse", dict(cost="length"), "arista más corta"),
    "clustering":     ("cluster", dict(representative="mean"), "clustering (promedio)"),
    "clustering-qem": ("cluster", dict(representative="qem"), "clustering + cuádricas"),
    "open3d":         ("open3d", dict(), "Open3D quadric"),
}
DEFAULT_COMPARE = ["qem", "qem-subset", "qem-midpoint", "length", "clustering",
                   "clustering-qem", "open3d"]


def collapse_history(V, F, overrides, base=None, progress=None):
    """Corre PairCollapse hasta el final y devuelve el historial (+ tiempos)."""
    kw = dict(base or {})
    kw.update(overrides)
    t0 = time.perf_counter()
    pc = PairCollapse(V, F, **kw)
    t_init = time.perf_counter() - t0
    h = pc.simplify(0, progress=progress)
    h.stats["time_init"] = t_init
    return h


def run_at_targets(name, V, F, targets, base=None, progress=None):
    """Mallas de `name` para cada objetivo de caras. Devuelve lista de (V, F, segundos)."""
    kind, kw, _ = METHODS[name]
    out = []
    if kind == "collapse":
        h = collapse_history(V, F, kw, base, progress)
        per_step = h.stats["time"] / max(h.steps, 1)
        for t in targets:
            k = h.step_for_faces(t)
            Vs, Fs, _, _ = h.state(k)
            out.append((Vs, Fs, h.stats["time_init"] + per_step * k))
    elif kind == "cluster":
        for t in targets:
            t0 = time.perf_counter()
            Vs, Fs, _ = baselines.clustering_to_target(V, F, t, **kw)
            out.append((Vs, Fs, time.perf_counter() - t0))
    elif kind == "open3d":
        if not baselines.has_open3d():
            return None
        for t in targets:
            t0 = time.perf_counter()
            Vs, Fs = baselines.open3d_quadric(V, F, t)
            out.append((Vs, Fs, time.perf_counter() - t0))
    return out


def available(names):
    return [n for n in names if METHODS[n][0] != "open3d" or baselines.has_open3d()]


def label(name):
    return METHODS[name][2]


def _width(s):
    import unicodedata
    return sum(0 if unicodedata.combining(ch) else 1 for ch in s)


def _pad(s, w, left):
    fill = " " * max(0, w - _width(s))
    return s + fill if left else fill + s


def fmt_table(rows, cols):
    """Tabla de texto simple: rows = lista de dicts, cols = [(clave, título, formato)]."""
    head = [t for _, t, _ in cols]
    body = [[(f.format(r[k]) if r.get(k) is not None and not (isinstance(r[k], float) and np.isnan(r[k]))
              else "—") for k, _, f in cols] for r in rows]
    w = [max(_width(h), *(_width(b[i]) for b in body)) for i, h in enumerate(head)]
    line = lambda xs: "  ".join(_pad(x, w[i], i == 0) for i, x in enumerate(xs))
    return "\n".join([line(head), line(["-" * x for x in w])] + [line(b) for b in body])
