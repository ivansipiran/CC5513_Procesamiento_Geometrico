"""
De correspondencias a transformacion (slides 26-29).

Tres estrategias, todas implementadas para poder compararlas:

1. `geometric_consistency_grouping`
   El criterio de Johnson & Hebert que aparece en las slides. Para dos
   correspondencias C1 = (s1, m1) y C2 = (s2, m2):

       d_gc(C1, C2) = || S_{m2}(m1) - S_{s2}(s1) ||
       w_gc(C1, C2) = d_gc / (1 - exp(-(||S_{m2}(m1)|| + ||S_{s2}(s1)||) / (2 gamma)))
       W_gc(C1, C2) = max(w_gc(C1,C2), w_gc(C2,C1))

   donde S_A(B) son las coordenadas spin (alpha, beta) de B en la base local
   del punto orientado A. El denominador castiga pares de correspondencias
   demasiado cercanas entre si (no restringen la pose).

2. `distance_consistency_groups`
   Criterio clasico, mas simple y sin normales:
       | ||s_i - s_j|| - ||m_i - m_j|| | < eps
   Se buscan cliques grandes de correspondencias mutuamente consistentes.

3. `ransac_pose`
   RANSAC sobre ternas de correspondencias con verificacion LCP.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
# transformacion rigida optima (Umeyama / Horn)
# --------------------------------------------------------------------------- #
def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """R,t que minimizan sum ||dst_i - (R src_i + t)||^2."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cd - R @ cs
    return T


# --------------------------------------------------------------------------- #
# verificacion (LCP: largest common pointset)
# --------------------------------------------------------------------------- #
class Verifier:
    """Cuenta que fraccion de `source` cae sobre `target` tras aplicar T.

    Usa scipy.cKDTree: la consulta se hace por lotes en C, lo que importa
    porque 4PCS evalua miles de candidatos.
    """

    def __init__(self, target: o3d.geometry.PointCloud, tau: float,
                 cache_points: np.ndarray | None = None,
                 quick: int = 250):
        self.tree = cKDTree(np.asarray(target.points))
        self.tau = tau
        self.quick = quick
        self._sub_cache: dict[tuple[int, int], np.ndarray] = {}

    def _subset(self, P: np.ndarray, n: int | None, rng) -> np.ndarray:
        if n is None or len(P) <= n:
            return P
        key = (id(P), n)
        if key not in self._sub_cache:
            r = rng or np.random.default_rng(0)
            self._sub_cache[key] = P[r.choice(len(P), n, replace=False)]
        return self._sub_cache[key]

    def score(self, src_pts: np.ndarray, T: np.ndarray,
              subsample: int | None = 1500, rng=None) -> float:
        P = self._subset(src_pts, subsample, rng)
        Q = (T[:3, :3] @ P.T).T + T[:3, 3]
        d, _ = self.tree.query(Q, k=1, distance_upper_bound=self.tau,
                               workers=-1)
        return float(np.isfinite(d).mean())

    def score_two_stage(self, src_pts: np.ndarray, T: np.ndarray,
                        best_so_far: float, subsample: int = 1500,
                        rng=None) -> float:
        """Primero una estimacion barata; solo si promete, la completa."""
        q = self.score(src_pts, T, subsample=self.quick, rng=rng)
        # cota superior optimista con el error de muestreo
        if q + 3.0 / np.sqrt(self.quick) < best_so_far:
            return q
        return self.score(src_pts, T, subsample=subsample, rng=rng)


# --------------------------------------------------------------------------- #
# 1. criterio de Johnson & Hebert (el de las slides)
# --------------------------------------------------------------------------- #
def spin_coords(p: np.ndarray, n: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(alpha, beta) de q en la base local del punto orientado (p, n)."""
    d = q - p
    beta = d @ n
    alpha = np.sqrt(np.maximum((d * d).sum(axis=-1) - beta ** 2, 0.0))
    return np.stack([alpha, beta], axis=-1)


def wgc_matrix(src_pts, src_nrm, tgt_pts, tgt_nrm, pairs, gamma):
    """Matriz W_gc entre todas las correspondencias (menor = mas consistente)."""
    si = pairs[:, 0]
    mi = pairs[:, 1]
    S = src_pts[si]
    Sn = src_nrm[si]
    M = tgt_pts[mi]
    Mn = tgt_nrm[mi]
    n = len(pairs)

    # S_{s_j}(s_i)  y  S_{m_j}(m_i)  para todo (i, j)
    ds = S[:, None, :] - S[None, :, :]
    bs = np.einsum("ijk,jk->ij", ds, Sn)
    as_ = np.sqrt(np.maximum((ds * ds).sum(axis=2) - bs ** 2, 0.0))
    dm = M[:, None, :] - M[None, :, :]
    bm = np.einsum("ijk,jk->ij", dm, Mn)
    am = np.sqrt(np.maximum((dm * dm).sum(axis=2) - bm ** 2, 0.0))

    d_gc = np.sqrt((as_ - am) ** 2 + (bs - bm) ** 2)
    norm_s = np.sqrt(as_ ** 2 + bs ** 2)
    norm_m = np.sqrt(am ** 2 + bm ** 2)
    den = 1.0 - np.exp(-(norm_s + norm_m) / (2.0 * gamma))
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(den > 1e-9, d_gc / den, np.inf)
    W = np.maximum(w, w.T)
    np.fill_diagonal(W, np.inf)
    return W


def geometric_consistency_grouping(src_pts, src_nrm, tgt_pts, tgt_nrm,
                                   pairs, gamma, thresh,
                                   min_group: int = 3,
                                   max_groups: int | None = None):
    """Crecimiento de grupos segun la slide 27.

    Para cada correspondencia se inicializa un grupo y se agrega repetidamente
    la correspondencia que minimiza W_gc(C_j, G_i) mientras este bajo el umbral.
    """
    if len(pairs) < min_group:
        return []
    W = wgc_matrix(src_pts, src_nrm, tgt_pts, tgt_nrm, pairs, gamma)
    n = len(pairs)
    groups = []
    seeds = range(n) if max_groups is None else range(min(n, max_groups))
    for i in seeds:
        members = [i]
        used = np.zeros(n, dtype=bool)
        used[i] = True
        while True:
            # W_gc(C_j, G) = max sobre los miembros del grupo
            cost = W[np.ix_(np.arange(n), members)].max(axis=1)
            cost[used] = np.inf
            j = int(np.argmin(cost))
            if not np.isfinite(cost[j]) or cost[j] > thresh:
                break
            members.append(j)
            used[j] = True
        if len(members) >= min_group:
            groups.append(np.asarray(members))
    # ordenar por tamano y quitar duplicados
    groups.sort(key=len, reverse=True)
    seen = set()
    uniq = []
    for g in groups:
        key = tuple(sorted(g.tolist()))
        if key not in seen:
            seen.add(key)
            uniq.append(g)
    return uniq


# --------------------------------------------------------------------------- #
# 2. consistencia de distancias (clique greedy)
# --------------------------------------------------------------------------- #
def distance_consistency_groups(src_pts, tgt_pts, pairs, eps,
                                min_group: int = 3, max_groups: int = 60):
    if len(pairs) < min_group:
        return []
    S = src_pts[pairs[:, 0]]
    M = tgt_pts[pairs[:, 1]]
    ds = np.linalg.norm(S[:, None] - S[None, :], axis=2)
    dm = np.linalg.norm(M[:, None] - M[None, :], axis=2)
    A = np.abs(ds - dm) < eps
    np.fill_diagonal(A, False)

    n = len(pairs)
    order = np.argsort(-A.sum(axis=1))
    groups = []
    for seed in order[:max_groups]:
        members = [int(seed)]
        cand = np.flatnonzero(A[seed])
        # greedy: agregar el candidato conectado a todo el grupo con mas grado
        while True:
            ok = [c for c in cand if A[c, members].all()]
            if not ok:
                break
            c = max(ok, key=lambda x: A[x].sum())
            members.append(int(c))
            cand = np.array([x for x in cand if x != c])
        if len(members) >= min_group:
            groups.append(np.asarray(members))
    groups.sort(key=len, reverse=True)
    return groups


# --------------------------------------------------------------------------- #
# 3. RANSAC
# --------------------------------------------------------------------------- #
def ransac_pose(src_pts, tgt_pts, pairs, verifier: Verifier,
                src_all: np.ndarray,
                n_iter: int = 20000, eps: float = 0.02,
                edge_ratio: float = 0.9, seed: int = 0):
    """RANSAC clasico sobre ternas de correspondencias, con prefiltro de
    longitud de aristas y verificacion LCP."""
    rng = np.random.default_rng(seed)
    S = src_pts[pairs[:, 0]]
    M = tgt_pts[pairs[:, 1]]
    n = len(pairs)
    best: tuple[float, np.ndarray] = (-1.0, np.eye(4))
    if n < 3:
        return np.eye(4), -1.0
    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        s, m = S[idx], M[idx]
        ds = np.linalg.norm(s[[0, 1, 2]] - s[[1, 2, 0]], axis=1)
        dm = np.linalg.norm(m[[0, 1, 2]] - m[[1, 2, 0]], axis=1)
        if np.any(np.minimum(ds, dm) / np.maximum(np.maximum(ds, dm), 1e-12) < edge_ratio):
            continue
        if ds.min() < eps:
            continue
        T = umeyama(s, m)
        # inliers entre las correspondencias
        pred = (T[:3, :3] @ S.T).T + T[:3, 3]
        inl = np.linalg.norm(pred - M, axis=1) < eps
        if inl.sum() < 3:
            continue
        T = umeyama(S[inl], M[inl])
        sc = verifier.score(src_all, T)
        if sc > best[0]:
            best = (sc, T)
    return best[1], best[0]


# --------------------------------------------------------------------------- #
# seleccion final entre grupos candidatos
# --------------------------------------------------------------------------- #
@dataclass
class PoseCandidate:
    T: np.ndarray
    score: float
    size: int
    residual: float


def poses_from_groups(src_pts, tgt_pts, pairs, groups,
                      verifier: Verifier, src_all: np.ndarray,
                      max_eval: int = 80) -> list[PoseCandidate]:
    S = src_pts[pairs[:, 0]]
    M = tgt_pts[pairs[:, 1]]
    out = []
    for g in groups[:max_eval]:
        if len(g) < 3:
            continue
        T = umeyama(S[g], M[g])
        pred = (T[:3, :3] @ S[g].T).T + T[:3, 3]
        res = float(np.sqrt(((pred - M[g]) ** 2).sum(axis=1).mean()))
        out.append(PoseCandidate(T, verifier.score(src_all, T), len(g), res))
    out.sort(key=lambda c: -c.score)
    return out
