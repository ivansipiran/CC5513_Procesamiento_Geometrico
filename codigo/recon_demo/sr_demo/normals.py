"""Normales: PCA local, orientacion consistente y filtro de outliers (Bloque 1)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree

ORIENT_METHODS = {
    "true":     "verdaderas (ground truth)",
    "mst":      "PCA + MST (Hoppe 92)",
    "centroid": "PCA + hacia afuera del centroide",
    "none":     "PCA sin orientar (signo arbitrario)",
}


@dataclass
class NormalParams:
    method: str = "mst"     # true | mst | centroid | none
    k: int = 20             # vecinos para PCA
    k_graph: int = 10       # vecinos del grafo riemanniano (MST)
    filter: bool = False    # filtro estadistico de outliers
    alpha: float = 2.0      # umbral: media + alpha * desv. estandar
    k_filter: int = 12


@dataclass
class NormalResult:
    P: np.ndarray            # nube despues del filtro
    N: np.ndarray            # normales orientadas
    keep: np.ndarray         # mascara sobre la nube original
    variation: np.ndarray    # variacion de superficie lambda0 / sum(lambda)
    N_true: np.ndarray
    is_outlier: np.ndarray
    ang_err: np.ndarray      # error angular SIN signo (calidad del plano), grados
    flipped: np.ndarray      # normal apuntando hacia adentro
    t_pca: float = 0.0
    t_orient: float = 0.0
    t_filter: float = 0.0
    filter_precision: float = float("nan")
    filter_recall: float = float("nan")

    def summary(self) -> dict:
        ok = ~self.is_outlier
        e = self.ang_err[ok]
        return dict(median_deg=float(np.median(e)) if len(e) else np.nan,
                    p95_deg=float(np.percentile(e, 95)) if len(e) else np.nan,
                    oriented_ok=float(1 - self.flipped[ok].mean()) if ok.any() else np.nan,
                    n=len(self.P), outliers_left=int(self.is_outlier.sum()))


def statistical_filter(P: np.ndarray, k: int = 12, alpha: float = 2.0) -> np.ndarray:
    """Descarta puntos cuya distancia media a sus k vecinos es atipica."""
    d, _ = cKDTree(P).query(P, k=k + 1, workers=-1)
    m = d[:, 1:].mean(1)
    return m <= m.mean() + alpha * m.std()


def pca_normals(P: np.ndarray, k: int):
    """Normal = vector propio del menor valor propio de la covarianza local.

    Devuelve (normales sin orientar, centroides locales, variacion de superficie).
    """
    k = int(max(3, min(k, len(P))))
    _, idx = cKDTree(P).query(P, k=k, workers=-1)
    nb = P[idx]
    o = nb.mean(1)
    D = nb - o[:, None]
    C = np.einsum("nki,nkj->nij", D, D) / k
    lam, vec = np.linalg.eigh(C)                     # ascendente
    var = lam[:, 0] / np.maximum(lam.sum(1), 1e-30)
    return vec[:, :, 0].copy(), o, var


def orient_mst(P: np.ndarray, N: np.ndarray, k: int = 10) -> np.ndarray:
    """Propagacion sobre el MST del grafo riemanniano (peso 1 - |ni.nj|).

    La semilla de cada componente es el punto con mayor z, cuya normal se
    orienta hacia +z (ese punto esta en la "tapa" del objeto).
    """
    N = N.copy()
    n = len(P)
    k = min(k, n - 1)
    _, idx = cKDTree(P).query(P, k=k + 1, workers=-1)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    w = 1.0 - np.abs((N[rows] * N[cols]).sum(1)) + 1e-6   # > 0 (0 = arista ausente)
    G = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    G = G.maximum(G.T)                                     # grafo simetrico
    T = minimum_spanning_tree(G)
    T = T + T.T
    ncomp, lab = connected_components(T, directed=False)
    for c in range(ncomp):
        members = np.flatnonzero(lab == c)
        seed = members[np.argmax(P[members, 2])]
        if N[seed, 2] < 0:
            N[seed] *= -1
        order, pred = breadth_first_order(T, seed, directed=False)
        for i in order[1:]:                                # los padres van primero
            if N[i] @ N[pred[i]] < 0:
                N[i] *= -1
    return N


def orient_centroid(P: np.ndarray, N: np.ndarray) -> np.ndarray:
    N = N.copy()
    flip = ((P - P.mean(0)) * N).sum(1) < 0
    N[flip] *= -1
    return N


def estimate(P: np.ndarray, N_true: np.ndarray, is_outlier: np.ndarray,
             prm: NormalParams, seed: int = 0) -> NormalResult:
    keep = np.ones(len(P), bool)
    t0 = time.perf_counter()
    prec = rec = float("nan")
    if prm.filter:
        keep = statistical_filter(P, prm.k_filter, prm.alpha)
        removed = ~keep
        if is_outlier.any():
            tp = (removed & is_outlier).sum()
            prec = float(tp / max(removed.sum(), 1))
            rec = float(tp / is_outlier.sum())
    t_filter = time.perf_counter() - t0
    Pk, Nt, out = P[keep], N_true[keep], is_outlier[keep]

    t0 = time.perf_counter()
    N, _, var = pca_normals(Pk, prm.k)
    t_pca = time.perf_counter() - t0

    t0 = time.perf_counter()
    if prm.method == "true":
        N = np.where(out[:, None], N, Nt)
        # los outliers no tienen normal verdadera: se orientan hacia afuera
        N[out] = orient_centroid(Pk[out], N[out]) if out.any() else N[out]
    elif prm.method == "mst":
        N = orient_mst(Pk, N, prm.k_graph)
    elif prm.method == "centroid":
        N = orient_centroid(Pk, N)
    elif prm.method == "none":
        s = np.random.default_rng(seed).choice([-1.0, 1.0], size=len(N))
        N = N * s[:, None]
    else:
        raise ValueError(prm.method)
    t_orient = time.perf_counter() - t0

    dots = np.nan_to_num((N * Nt).sum(1), nan=1.0)
    ang = np.degrees(np.arccos(np.clip(np.abs(dots), 0, 1)))
    ang[out] = np.nan
    flipped = (dots < 0) & ~out
    return NormalResult(P=Pk, N=N, keep=keep, variation=var, N_true=Nt,
                        is_outlier=out, ang_err=ang, flipped=flipped,
                        t_pca=t_pca, t_orient=t_orient, t_filter=t_filter,
                        filter_precision=prec, filter_recall=rec)
