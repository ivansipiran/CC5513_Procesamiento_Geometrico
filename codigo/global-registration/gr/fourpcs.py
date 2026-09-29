"""
4PCS  (Aiger, Mitra & Cohen-Or, SIGGRAPH 2008)
Super4PCS (Mellado, Aiger & Mitra, SGP 2014)

Registro global SIN caracteristicas. La observacion clave (slide 35): las
razones afines de una base de 4 puntos coplanares son invariantes a
transformaciones rigidas, asi que se puede buscar el conjunto congruente en
O(n^2) en vez de O(n^3).

Base B = {p1, p2, p3, p4} coplanar, con las diagonales p1p2 y p3p4 que se
cortan en e:

    r1 = ||e - p1|| / ||p2 - p1||        r2 = ||e - p3|| / ||p4 - p3||

Dado un par (q_a, q_b) de Q a distancia d1 = ||p1 - p2||, los dos puntos
    e = q_a + r1 (q_b - q_a)   y   e = q_b + r1 (q_a - q_b)
son candidatos a ser la interseccion. Lo mismo con d2 y r2. Un conjunto
congruente aparece cuando un candidato de la familia d1 coincide (a menos de
delta) con uno de la familia d2.

Diferencia 4PCS / Super4PCS implementada aqui:

  * extraccion de pares a distancia d
      - 4PCS      : todas las distancias, O(n^2)
      - Super4PCS : grilla espacial + rasterizacion de la cascara esferica,
                    O(n + #pares)
  * extraccion de congruentes
      - 4PCS      : kd-tree sobre los puntos intermedios, O(m log m)
      - Super4PCS : grilla hash, O(m)   + filtro por angulo entre normales
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .grouping import Verifier, umeyama


# --------------------------------------------------------------------------- #
# extraccion de pares a una distancia dada
# --------------------------------------------------------------------------- #
def pairs_naive(P: np.ndarray, d: float, delta: float) -> np.ndarray:
    """Todos los pares (i, j), i<j, con | ||pi-pj|| - d | <= delta.  O(n^2)."""
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    iu = np.triu_indices(len(P), 1)
    m = np.abs(D[iu] - d) <= delta
    return np.stack([iu[0][m], iu[1][m]], axis=1)


class ShellGrid:
    """Grilla espacial uniforme para consultar cascaras esfericas.

    Es el nucleo de la 'extraccion inteligente de pares' de Super4PCS: en vez
    de mirar todos los puntos dentro de una bola de radio d+delta (que para d
    grande es casi toda la nube, o sea O(n^2)), solo se visitan las celdas que
    intersectan la cascara [d-delta, d+delta]. El costo pasa a ser
    O(n + #pares).

    Implementacion: se itera sobre CELDAS ocupadas, no sobre puntos, y para
    cada par de celdas se calculan las distancias en bloque con numpy.
    """

    def __init__(self, P: np.ndarray, cell_frac: float = 1.0 / 3.0):
        self.P = P
        self.cell_frac = cell_frac
        self._cache: dict[float, tuple] = {}
        self._offsets_cache: dict[tuple, np.ndarray] = {}

    def _build(self, cell: float):
        if cell in self._cache:
            return self._cache[cell]
        P = self.P
        origin = P.min(axis=0) - cell
        keys = np.floor((P - origin) / cell).astype(np.int64)
        dims = keys.max(axis=0) + 3
        lin = np.ravel_multi_index(keys.T, dims)
        order = np.argsort(lin, kind="stable")
        lin_s = lin[order]
        starts = np.flatnonzero(np.r_[True, lin_s[1:] != lin_s[:-1]])
        uc_lin = lin_s[starts]
        uc_key = np.stack(np.unravel_index(uc_lin, dims), axis=1)
        ends = np.r_[starts[1:], len(lin_s)]
        self._cache[cell] = (origin, dims, order, uc_lin, uc_key, starts, ends)
        return self._cache[cell]

    def _shell_offsets(self, d: float, delta: float, cell: float) -> np.ndarray:
        """Offsets de celda cuyo cubo puede contener puntos de la cascara."""
        key = (round(d / cell, 4), round(delta / cell, 4))
        if key in self._offsets_cache:
            return self._offsets_cache[key]
        R = int(np.ceil((d + delta) / cell)) + 1
        g = np.arange(-R, R + 1)
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        off = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        lo = np.maximum(np.abs(off) - 1, 0) * cell    # dist minima entre cubos
        hi = (np.abs(off) + 1) * cell                 # dist maxima
        keep = ((np.linalg.norm(hi, axis=1) >= d - delta) &
                (np.linalg.norm(lo, axis=1) <= d + delta))
        self._offsets_cache[key] = off[keep]
        return off[keep]

    def pairs(self, d: float, delta: float, max_pairs: int = 400000) -> np.ndarray:
        """Pares a distancia d +- delta visitando solo la cascara rasterizada."""
        cell = max(delta, d * self.cell_frac)
        origin, dims, order, uc_lin, uc_key, starts, ends = self._build(cell)
        off = self._shell_offsets(d, delta, cell)
        P = self.P

        # join vectorizado celda-a-celda: para cada celda ocupada, que celdas
        # de la cascara estan tambien ocupadas
        U, O = len(uc_key), len(off)
        q = uc_key[:, None, :] + off[None, :, :]
        good = np.all((q >= 0) & (q < dims), axis=2)
        qlin = np.full((U, O), -1, dtype=np.int64)
        qlin[good] = np.ravel_multi_index(q[good].T, dims)
        pos = np.searchsorted(uc_lin, qlin)
        pos_c = np.clip(pos, 0, len(uc_lin) - 1)
        hit = good & (uc_lin[pos_c] == qlin)
        ci, oj = np.nonzero(hit)
        cj = pos_c[ci, oj]
        keep = cj >= ci                       # cada par de celdas una sola vez
        ci, cj = ci[keep], cj[keep]

        out_i, out_j, total = [], [], 0
        for a, b in zip(ci, cj):
            ia = order[starts[a]:ends[a]]
            ib = order[starts[b]:ends[b]]
            D = np.linalg.norm(P[ia][:, None, :] - P[ib][None, :, :], axis=2)
            m = np.abs(D - d) <= delta
            if a == b:                        # dentro de una celda: i < j
                m &= ia[:, None] < ib[None, :]
            if not m.any():
                continue
            aa, bb = np.nonzero(m)
            lo = np.minimum(ia[aa], ib[bb])
            hi = np.maximum(ia[aa], ib[bb])
            out_i.append(lo)
            out_j.append(hi)
            total += len(aa)
            if total > max_pairs:
                break
        if not out_i:
            return np.zeros((0, 2), dtype=np.int64)
        return np.stack([np.concatenate(out_i), np.concatenate(out_j)], axis=1)


def grid_join(A: np.ndarray, B: np.ndarray, r: float,
              max_out: int = 500000) -> tuple[np.ndarray, np.ndarray]:
    """Pares (i de A, j de B) con ||A_i - B_j|| <= r, via grilla hash.

    Join por celdas: O(|A| + |B| + #pares). Es la version vectorizada de la
    'extraccion inteligente' de Super4PCS aplicada a los puntos intermedios.
    """
    if len(A) == 0 or len(B) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    origin = np.minimum(A.min(axis=0), B.min(axis=0)) - 2 * r
    dims = np.floor((np.maximum(A.max(axis=0), B.max(axis=0)) - origin) / r
                    ).astype(np.int64) + 4
    ka = np.floor((A - origin) / r).astype(np.int64)
    kb = np.floor((B - origin) / r).astype(np.int64)
    lin_a = np.ravel_multi_index(ka.T, dims)
    order = np.argsort(lin_a)
    lin_a_s = lin_a[order]

    oi, oj = [], []
    total = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                q = np.ravel_multi_index((kb + [dx, dy, dz]).T, dims)
                lo = np.searchsorted(lin_a_s, q, "left")
                hi = np.searchsorted(lin_a_s, q, "right")
                cnt = hi - lo
                sel = np.flatnonzero(cnt > 0)
                if not len(sel):
                    continue
                reps = np.repeat(sel, cnt[sel])
                pos = np.concatenate([np.arange(lo[s], hi[s]) for s in sel])
                ia = order[pos]
                d = np.linalg.norm(A[ia] - B[reps], axis=1)
                m = d <= r
                oi.append(ia[m])
                oj.append(reps[m])
                total += int(m.sum())
                if total > max_out:
                    break
    if not oi:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(oi), np.concatenate(oj)


# --------------------------------------------------------------------------- #
# seleccion de la base coplanar
# --------------------------------------------------------------------------- #
def line_intersection(p1, p2, p3, p4):
    """Punto de maxima cercania entre las rectas p1p2 y p3p4.

    Devuelve (e, r1, r2, gap) con e el punto medio del segmento de minima
    distancia y r1, r2 los parametros a lo largo de cada recta.
    """
    u = p2 - p1
    v = p4 - p3
    w = p1 - p3
    a, b, c = u @ u, u @ v, v @ v
    d, e_ = u @ w, v @ w
    den = a * c - b * b
    if abs(den) < 1e-14:
        return None, None, None, np.inf
    s = (b * e_ - c * d) / den
    t = (a * e_ - b * d) / den
    A = p1 + s * u
    B = p3 + t * v
    return 0.5 * (A + B), s, t, float(np.linalg.norm(A - B))


def select_coplanar_base(P: np.ndarray, rng, diameter: float,
                         delta: float, width_frac=(0.3, 0.9),
                         max_try: int = 200):
    """Elige una base de 4 puntos aproximadamente coplanares y bien separados."""
    n = len(P)
    for _ in range(max_try):
        i1, i2 = rng.choice(n, 2, replace=False)
        d1 = np.linalg.norm(P[i1] - P[i2])
        if not (width_frac[0] * diameter <= d1 <= width_frac[1] * diameter):
            continue
        # tercer punto lejos de la recta p1p2
        i3 = int(rng.integers(n))
        u = (P[i2] - P[i1]) / d1
        w = P[i3] - P[i1]
        h = np.linalg.norm(w - (w @ u) * u)
        if h < 0.2 * d1:
            continue
        # cuarto punto: el mas cercano al plano, con diagonales que se cruzan
        nrm = np.cross(P[i2] - P[i1], P[i3] - P[i1])
        nn = np.linalg.norm(nrm)
        if nn < 1e-12:
            continue
        nrm = nrm / nn
        dist_plane = np.abs((P - P[i1]) @ nrm)
        cand = np.flatnonzero(dist_plane < delta)
        cand = cand[(cand != i1) & (cand != i2) & (cand != i3)]
        if len(cand) == 0:
            continue
        rng.shuffle(cand)
        for i4 in cand[:60]:
            for (a, b, c, d) in ((i1, i2, i3, i4), (i1, i3, i2, i4),
                                 (i1, i4, i2, i3)):
                e, r1, r2, gap = line_intersection(P[a], P[b], P[c], P[d])
                if e is None or gap > delta:
                    continue
                if not (0.05 < r1 < 0.95 and 0.05 < r2 < 0.95):
                    continue
                dd1 = np.linalg.norm(P[b] - P[a])
                dd2 = np.linalg.norm(P[d] - P[c])
                if min(dd1, dd2) < 0.2 * diameter:
                    continue
                return np.array([a, b, c, d]), float(r1), float(r2), dd1, dd2
    return None


# --------------------------------------------------------------------------- #
# extraccion de conjuntos congruentes
# --------------------------------------------------------------------------- #
def _intermediates(Q, pairs, r):
    """Los dos puntos intermedios de cada par, con la orientacion usada."""
    a, b = Q[pairs[:, 0]], Q[pairs[:, 1]]
    e1 = a + r * (b - a)
    e2 = b + r * (a - b)
    E = np.vstack([e1, e2])
    idx = np.vstack([pairs, pairs[:, ::-1]])       # (origen, destino)
    return E, idx


def find_congruent(Q: np.ndarray,
                   pairs1: np.ndarray, r1: float,
                   pairs2: np.ndarray, r2: float,
                   delta: float,
                   base_angle: float | None = None,
                   angle_tol: float = 0.15,
                   backend: str = "grid",
                   max_sets: int = 20000) -> np.ndarray:
    """Devuelve los conjuntos congruentes como filas [i1, i2, i3, i4]."""
    if len(pairs1) == 0 or len(pairs2) == 0:
        return np.zeros((0, 4), dtype=np.int64)
    E1, I1 = _intermediates(Q, pairs1, r1)
    E2, I2 = _intermediates(Q, pairs2, r2)

    if backend == "grid":                    # Super4PCS: join por grilla, O(m)
        ha, hb = grid_join(E1, E2, delta, max_out=max_sets)
    else:                                    # 4PCS: kd-tree, O(m log m)
        tree = cKDTree(E1)
        lists = tree.query_ball_point(E2, delta, workers=-1)
        hb = np.repeat(np.arange(len(E2)), [len(x) for x in lists])
        ha = np.array([i for x in lists for i in x], dtype=np.int64)
        if len(ha) > max_sets:
            ha, hb = ha[:max_sets], hb[:max_sets]

    if len(ha) == 0:
        return np.zeros((0, 4), dtype=np.int64)
    S = np.stack([I1[ha, 0], I1[ha, 1], I2[hb, 0], I2[hb, 1]], axis=1)
    S = S[(S[:, 0] != S[:, 2]) & (S[:, 0] != S[:, 3]) &
          (S[:, 1] != S[:, 2]) & (S[:, 1] != S[:, 3])]
    if base_angle is not None and len(S):
        u = Q[S[:, 1]] - Q[S[:, 0]]
        v = Q[S[:, 3]] - Q[S[:, 2]]
        cu = np.abs((u * v).sum(axis=1) /
                    (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-12))
        S = S[np.abs(cu - abs(base_angle)) < angle_tol]
    return S


# --------------------------------------------------------------------------- #
# algoritmo completo
# --------------------------------------------------------------------------- #
@dataclass
class FourPCSConfig:
    delta: float = 0.02              # tolerancia de aproximacion
    overlap: float = 0.5             # estimacion del solapamiento
    n_samples: int = 800             # submuestreo de P y Q para la busqueda
    n_bases: int = 100               # numero de bases (iteraciones L)
    variant: str = "super4pcs"       # 'super4pcs' | '4pcs'
    use_normals: bool = True         # filtro por angulo entre normales (Super4PCS)
    normal_tol: float = 0.25
    width_frac: tuple = (0.3, 0.9)
    max_sets: int = 20000
    max_eval_per_base: int = 400
    verify_subsample: int = 1200
    early_stop_lcp: float = 0.95
    seed: int = 0


@dataclass
class FourPCSResult:
    T: np.ndarray
    lcp: float
    n_bases_used: int
    n_candidates: int
    timings: dict = field(default_factory=dict)


def _pair_filter_normals(Q, Qn, pairs, target_cos, tol):
    if len(pairs) == 0:
        return pairs
    c = (Qn[pairs[:, 0]] * Qn[pairs[:, 1]]).sum(axis=1)
    return pairs[np.abs(np.abs(c) - abs(target_cos)) < tol]


def register(source: o3d.geometry.PointCloud,
             target: o3d.geometry.PointCloud,
             cfg: FourPCSConfig) -> FourPCSResult:
    """Registro global 4PCS / Super4PCS.  source -> target."""
    rng = np.random.default_rng(cfg.seed)
    P_all = np.asarray(source.points)
    Q_all = np.asarray(target.points)

    ip = rng.choice(len(P_all), min(cfg.n_samples, len(P_all)), replace=False)
    iq = rng.choice(len(Q_all), min(cfg.n_samples, len(Q_all)), replace=False)
    P, Q = P_all[ip], Q_all[iq]
    Pn = np.asarray(source.normals)[ip] if source.has_normals() else None
    Qn = np.asarray(target.normals)[iq] if target.has_normals() else None

    diameter = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
    verifier = Verifier(target, cfg.delta)

    t_pairs = t_cong = t_verify = 0.0
    shell = ShellGrid(Q) if cfg.variant == "super4pcs" else None

    best_T, best_lcp, n_cand = np.eye(4), -1.0, 0
    used = 0
    for _ in range(cfg.n_bases):
        base = select_coplanar_base(P, rng, diameter, cfg.delta, cfg.width_frac)
        if base is None:
            continue
        used += 1
        idx, r1, r2, d1, d2 = base
        B = P[idx]
        u = B[1] - B[0]
        v = B[3] - B[2]
        base_angle = float((u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))

        t0 = time.time()
        if cfg.variant == "super4pcs":
            pairs1 = shell.pairs(d1, cfg.delta)
            pairs2 = shell.pairs(d2, cfg.delta)
        else:
            pairs1 = pairs_naive(Q, d1, cfg.delta)
            pairs2 = pairs_naive(Q, d2, cfg.delta)
        if cfg.use_normals and Qn is not None and Pn is not None:
            c1 = float(Pn[idx[0]] @ Pn[idx[1]])
            c2 = float(Pn[idx[2]] @ Pn[idx[3]])
            pairs1 = _pair_filter_normals(Q, Qn, pairs1, c1, cfg.normal_tol)
            pairs2 = _pair_filter_normals(Q, Qn, pairs2, c2, cfg.normal_tol)
        t_pairs += time.time() - t0

        t0 = time.time()
        S = find_congruent(Q, pairs1, r1, pairs2, r2, cfg.delta,
                           base_angle=base_angle,
                           backend="grid" if cfg.variant == "super4pcs" else "kdtree",
                           max_sets=cfg.max_sets)
        t_cong += time.time() - t0
        n_cand += len(S)

        t0 = time.time()
        if len(S) > cfg.max_eval_per_base:
            S = S[rng.choice(len(S), cfg.max_eval_per_base, replace=False)]
        for s in S:
            T = umeyama(B, Q[s])
            # descarte barato: la pose debe reproducir la base
            pred = (T[:3, :3] @ B.T).T + T[:3, 3]
            if np.linalg.norm(pred - Q[s], axis=1).max() > 2 * cfg.delta:
                continue
            lcp = verifier.score_two_stage(P_all, T, best_lcp,
                                           subsample=cfg.verify_subsample,
                                           rng=rng)
            if lcp > best_lcp:
                best_lcp, best_T = lcp, T
        t_verify += time.time() - t0

        if best_lcp >= cfg.early_stop_lcp:
            break

    return FourPCSResult(T=best_T, lcp=best_lcp, n_bases_used=used,
                         n_candidates=n_cand,
                         timings=dict(pairs=t_pairs, congruent=t_cong,
                                      verify=t_verify))


def register_with_icp(source, target, cfg: FourPCSConfig, icp_tau: float):
    from . import metrics
    r = register(source, target, cfg)
    T = metrics.icp_refine(source, target, r.T, icp_tau)
    return r, T
