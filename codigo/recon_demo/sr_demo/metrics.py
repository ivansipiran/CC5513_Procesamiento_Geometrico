"""Metricas: topologia de la malla y error contra la superficie real."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def mesh_stats(V: np.ndarray, T: np.ndarray) -> dict:
    """chi = V - E + F, componentes, aristas de borde / no-manifold y genero.

    Para una superficie cerrada y orientable con c componentes:
        chi = 2c - 2g   =>   g = (2c - chi) / 2   (genero total)
    Si hay aristas de borde la malla es abierta y el genero no esta definido
    por esa formula (se informa el numero de aristas de borde).
    """
    if len(T) == 0:
        return dict(V=0, E=0, F=0, chi=0, comps=0, boundary=0, nonmanifold=0,
                    genus=None, closed=False, big_comps=0, area=0.0)
    used = np.unique(T)
    E = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    ue, cnt = np.unique(E, axis=0, return_counts=True)
    nV, nE, nF = len(used), len(ue), len(T)
    chi = nV - nE + nF
    n = len(V)
    A = coo_matrix((np.ones(len(ue)), (ue[:, 0], ue[:, 1])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    lab_used = lab[used]
    comps, sizes = np.unique(lab_used, return_counts=True)
    boundary = int((cnt == 1).sum())
    nonman = int((cnt > 2).sum())
    closed = boundary == 0 and nonman == 0
    genus = (2 * len(comps) - chi) // 2 if closed else None
    tri = V[T]
    area = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                               tri[:, 2] - tri[:, 0]), axis=1).sum())
    big = int((sizes >= 0.01 * nV).sum())
    return dict(V=nV, E=nE, F=nF, chi=int(chi), comps=int(len(comps)), big_comps=big,
                boundary=boundary, nonmanifold=nonman, genus=genus, closed=closed,
                area=area)


def topology_label(s: dict) -> str:
    if s["F"] == 0:
        return "vacia"
    if s["closed"]:
        if s["comps"] == 1:
            return f"cerrada, genero {s['genus']}"
        return f"cerrada, {s['comps']} comp., genero total {s['genus']}"
    parts = []
    if s["boundary"]:
        parts.append(f"{s['boundary']} aristas de borde")
    if s["nonmanifold"]:
        parts.append(f"{s['nonmanifold']} aristas no-manifold")
    parts.append(f"{s['comps']} comp.")
    kind = "abierta" if s["boundary"] else "no-manifold"
    return f"{kind} (" + ", ".join(parts) + ")"


def accuracy(surface, V: np.ndarray, diag: float) -> np.ndarray:
    """Distancia de cada vertice reconstruido a la superficie real (frac. diag)."""
    if len(V) == 0:
        return np.zeros(0)
    return surface.distance(V) / diag


def completeness(gt_samples: np.ndarray, V: np.ndarray, T: np.ndarray, diag: float) -> np.ndarray:
    """Distancia de muestras de la superficie REAL a la malla reconstruida.

    La precision (vertices -> verdad) no penaliza que falte superficie; esta
    si. Se aproxima con la distancia a la nube formada por vertices y
    baricentros de la malla.
    """
    if len(T) == 0:
        return np.full(len(gt_samples), np.inf)
    Q = np.vstack([V[np.unique(T)], V[T].mean(1)])
    d, _ = cKDTree(Q).query(gt_samples, workers=-1)
    return d / diag
