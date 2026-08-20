"""Nucleo del algoritmo ICP (Iterative Closest Point).

Estructura clasica del algoritmo (Besl & McKay 1992):

    repetir hasta converger:
        1. CORRESPONDENCIA : para cada punto de la fuente, buscar el punto mas
                             cercano de la nube objetivo  -> modulo neighbors
        2. RECHAZO         : descartar pares poco confiables (distancia grande,
                             normales incompatibles, trimming por percentil)
        3. MINIMIZACION    : hallar la transformacion rigida que minimiza el
                             error sobre los pares aceptados
        4. ACTUALIZACION   : componer la transformacion y repetir

El paso 3 admite dos metricas:

  * point-to-point (Besl & McKay): minimiza  sum ||R p_i + t - q_i||^2.
    Solucion cerrada por SVD (Arun/Horn). Convergencia lineal y lenta cuando
    las nubes deslizan sobre superficies planas.

  * point-to-plane (Chen & Medioni): minimiza  sum ((R p_i + t - q_i) . n_i)^2,
    donde n_i es la normal del objetivo (plano tangente). No tiene solucion
    cerrada; se linealiza la rotacion para angulos pequenos y se resuelve un
    sistema 6x6. Converge en muchas menos iteraciones porque permite el
    deslizamiento tangencial "gratis".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .geometry import (apply_transform, make_transform, pose_error,
                       project_to_rotation, rodrigues, rotation_angle,
                       transform_normals)
from .neighbors import make_backend

VARIANTS = ("point2point", "point2plane")


# --------------------------------------------------------------------------- #
# Paso 3: los dos solvers
# --------------------------------------------------------------------------- #
def solve_point_to_point(P: np.ndarray, Q: np.ndarray,
                         w: np.ndarray | None = None) -> np.ndarray:
    """Alineamiento de Procrustes rigido (Arun et al. / Horn), solucion cerrada.

    Minimiza sum w_i ||R p_i + t - q_i||^2.
      1. se centran ambas nubes en sus centroides (esto elimina t del problema)
      2. H = sum w_i (p_i - p_bar)(q_i - q_bar)^T
      3. SVD: H = U S V^T  ->  R = V diag(1,1,det(V U^T)) U^T
         (el determinante evita obtener una reflexion)
      4. t = q_bar - R p_bar
    """
    if w is None:
        w = np.ones(len(P))
    w = w / max(w.sum(), 1e-20)
    p_bar = w @ P
    q_bar = w @ Q
    Pc = P - p_bar
    Qc = Q - q_bar
    H = (Pc * w[:, None]).T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = q_bar - R @ p_bar
    return make_transform(R, t)


def solve_point_to_plane(P: np.ndarray, Q: np.ndarray, N: np.ndarray,
                         w: np.ndarray | None = None) -> np.ndarray:
    """Minimizacion punto-a-plano linealizada (Chen & Medioni / Low 2004).

    Con R ~= I + [omega]_x para angulos pequenos, el residual
        ((R p + t - q) . n)
    se vuelve lineal en x = [omega, t] (6 incognitas):
        (p x n) . omega + n . t = (q - p) . n
    Se arma A x = b con A_i = [p_i x n_i, n_i] y se resuelven las ecuaciones
    normales A^T A x = A^T b (sistema 6x6). Luego omega se convierte en una
    rotacion exacta con Rodrigues.
    """
    if w is None:
        w = np.ones(len(P))
    sw = np.sqrt(w)[:, None]
    C = np.cross(P, N)                     # (n,3)
    A = np.hstack([C, N]) * sw             # (n,6)
    b = (np.sum((Q - P) * N, axis=1) * sw[:, 0])

    AtA = A.T @ A
    Atb = A.T @ b
    # regularizacion minima por si el sistema queda mal condicionado
    AtA[np.diag_indices(6)] += 1e-12
    try:
        x = np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(AtA, Atb, rcond=None)[0]

    omega, t = x[:3], x[3:]
    R = rodrigues(omega)
    if not np.all(np.isfinite(R)):
        return np.eye(4)
    R = project_to_rotation(R)
    return make_transform(R, t)


# --------------------------------------------------------------------------- #
# Parametros, historia y resultado
# --------------------------------------------------------------------------- #
@dataclass
class ICPParams:
    variant: str = "point2point"        # point2point | point2plane
    nn_backend: str = "kdtree"          # kdtree | brute | brute-loop
    max_iter: int = 50
    sample_size: int | None = None      # submuestreo de la fuente por iteracion
    max_corr_dist: float | None = None  # umbral absoluto de rechazo
    auto_reject_factor: float | None = 3.0   # umbral = factor * mediana(dist)
    trim_ratio: float = 1.0             # fraccion de pares conservados (trimmed ICP)
    normal_reject_deg: float | None = None   # rechazo por angulo entre normales
    tol_rmse: float = 1e-7
    tol_rot_deg: float = 1e-5
    tol_trans: float = 1e-8
    seed: int = 0
    store_corr: int = 500               # correspondencias guardadas para visualizar

    def label(self) -> str:
        return f"{self.variant}/{self.nn_backend}"


@dataclass
class IterationRecord:
    it: int
    T: np.ndarray                 # pose acumulada mostrada en esta iteracion
    rmse: float                   # RMSE de las correspondencias en esta pose
    inlier_ratio: float
    n_pairs: int
    t_query: float = 0.0
    t_solve: float = 0.0
    t_total: float = 0.0
    delta_rot_deg: float = 0.0
    delta_trans: float = 0.0
    rot_err_deg: float | None = None    # respecto al ground truth
    trans_err: float | None = None
    gt_rmse: float | None = None
    corr_src: np.ndarray | None = None  # indices (en la nube fuente) para viz
    corr_tgt: np.ndarray | None = None  # indices (en la nube objetivo) para viz
    corr_inlier: np.ndarray | None = None


@dataclass
class ICPResult:
    T: np.ndarray
    params: ICPParams
    history: list[IterationRecord] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0
    total_time: float = 0.0
    build_time: float = 0.0
    query_time: float = 0.0
    solve_time: float = 0.0

    @property
    def rmse(self) -> float:
        return self.history[-1].rmse if self.history else float("nan")

    def rmse_curve(self) -> np.ndarray:
        return np.array([h.rmse for h in self.history], dtype=np.float64)

    def gt_curve(self) -> np.ndarray:
        return np.array([np.nan if h.gt_rmse is None else h.gt_rmse for h in self.history])

    def summary(self) -> str:
        h = self.history[-1] if self.history else None
        s = (f"{self.params.label():26s} iter={self.n_iter:3d} "
             f"rmse={self.rmse:.3e} t={self.total_time*1e3:8.1f} ms "
             f"(nn={self.query_time*1e3:7.1f} ms, solve={self.solve_time*1e3:6.1f} ms)")
        if h is not None and h.rot_err_deg is not None:
            s += f"  err_rot={h.rot_err_deg:6.3f} deg  err_trans={h.trans_err:.3e}"
        s += "  conv" if self.converged else "  (max_iter)"
        return s


# --------------------------------------------------------------------------- #
# Algoritmo
# --------------------------------------------------------------------------- #
def icp(source: np.ndarray,
        target: np.ndarray,
        params: ICPParams | None = None,
        target_normals: np.ndarray | None = None,
        source_normals: np.ndarray | None = None,
        T_init: np.ndarray | None = None,
        T_gt: np.ndarray | None = None,
        verbose: bool = False) -> ICPResult:
    """Registra `source` sobre `target` y devuelve la transformacion y su historia.

    source, target      : nubes (n,3) y (m,3)
    target_normals      : obligatorio para la variante point2plane
    T_init              : pose inicial (identidad por defecto)
    T_gt                : transformacion verdadera, solo para medir el error
    """
    params = params or ICPParams()
    if params.variant not in VARIANTS:
        raise ValueError(f"Variante '{params.variant}' desconocida: {VARIANTS}")
    if params.variant == "point2plane" and target_normals is None:
        raise ValueError("point2plane requiere normales de la nube objetivo")

    rng = np.random.default_rng(params.seed)
    source = np.ascontiguousarray(source, dtype=np.float64)
    target = np.ascontiguousarray(target, dtype=np.float64)

    nn = make_backend(params.nn_backend)
    nn.build(target)          # el arbol se construye UNA sola vez (objetivo fijo)

    T = np.eye(4) if T_init is None else np.array(T_init, dtype=np.float64)
    result = ICPResult(T=T.copy(), params=params, build_time=nn.build_time)
    t_start = time.perf_counter()

    n_src = len(source)
    viz_pool = np.sort(rng.permutation(n_src)[:max(0, params.store_corr)])

    def correspondences(T_cur):
        """Paso 1 + 2: vecino mas cercano y rechazo de pares."""
        moved = apply_transform(T_cur, source)
        if params.sample_size and params.sample_size < n_src:
            sel = np.sort(rng.choice(n_src, size=params.sample_size, replace=False))
        else:
            sel = np.arange(n_src)
        dist, idx = nn.query(moved[sel])
        t_q = nn.last_query_time

        keep = np.ones(len(sel), dtype=bool)
        thr = params.max_corr_dist
        if thr is None and params.auto_reject_factor:
            med = float(np.median(dist)) if len(dist) else 0.0
            thr = params.auto_reject_factor * max(med, 1e-12)
        if thr is not None:
            keep &= dist <= thr
        if params.trim_ratio < 1.0 and keep.any():
            k = max(6, int(params.trim_ratio * keep.sum()))
            order = np.argsort(dist)
            cut = np.zeros(len(sel), dtype=bool)
            cut[order[:k]] = True
            keep &= cut
        if (params.normal_reject_deg is not None and target_normals is not None
                and source_normals is not None):
            ns = transform_normals(T_cur, source_normals[sel])
            cosang = np.sum(ns * target_normals[idx], axis=1)
            keep &= cosang >= np.cos(np.deg2rad(params.normal_reject_deg))
        return moved, sel, dist, idx, keep, t_q

    converged = False
    it = 0
    for it in range(params.max_iter + 1):
        t_it = time.perf_counter()
        moved, sel, dist, idx, keep, t_q = correspondences(T)
        result.query_time += t_q

        n_pairs = int(keep.sum())
        rmse = float(np.sqrt(np.mean(dist[keep] ** 2))) if n_pairs else float("inf")

        rec = IterationRecord(
            it=it, T=T.copy(), rmse=rmse,
            inlier_ratio=n_pairs / max(len(sel), 1), n_pairs=n_pairs,
            t_query=t_q,
        )
        if T_gt is not None:
            rec.rot_err_deg, rec.trans_err = pose_error(T, T_gt)
            gt_moved = apply_transform(T_gt, source)
            rec.gt_rmse = float(np.sqrt(np.mean(np.sum((moved - gt_moved) ** 2, axis=1))))

        # subconjunto de correspondencias guardado para la animacion
        if params.store_corr and len(viz_pool):
            pos = np.searchsorted(sel, viz_pool) if len(sel) < n_src else viz_pool
            if len(sel) < n_src:
                ok = (pos < len(sel)) & (sel[np.minimum(pos, len(sel) - 1)] == viz_pool)
                pos = pos[ok]
            rec.corr_src = sel[pos]
            rec.corr_tgt = idx[pos]
            rec.corr_inlier = keep[pos]

        # ultima pasada: solo evaluamos, no actualizamos
        if it == params.max_iter or n_pairs < 6 or converged:
            rec.t_total = time.perf_counter() - t_it
            result.history.append(rec)
            break

        # Paso 3: minimizacion
        t0 = time.perf_counter()
        P = moved[sel][keep]
        Q = target[idx][keep]
        if params.variant == "point2point":
            dT = solve_point_to_point(P, Q)
        else:
            dT = solve_point_to_plane(P, Q, target_normals[idx][keep])
        t_s = time.perf_counter() - t0
        result.solve_time += t_s
        rec.t_solve = t_s

        # Paso 4: actualizacion de la pose acumulada
        T = dT @ T
        rec.delta_rot_deg = np.rad2deg(rotation_angle(dT))
        rec.delta_trans = float(np.linalg.norm(dT[:3, 3]))
        rec.t_total = time.perf_counter() - t_it
        result.history.append(rec)

        if verbose:
            print(f"  it {it:3d}  rmse={rmse:.6e}  pares={n_pairs:6d}  "
                  f"d_rot={rec.delta_rot_deg:.4f} deg  d_t={rec.delta_trans:.2e}")

        # criterio de parada: la actualizacion ya no mueve nada
        small_step = (rec.delta_rot_deg < params.tol_rot_deg
                      and rec.delta_trans < params.tol_trans)
        small_gain = (len(result.history) > 1
                      and abs(result.history[-2].rmse - rmse) < params.tol_rmse)
        if small_step or small_gain:
            converged = True

    result.T = T
    result.converged = converged
    result.n_iter = max(0, len(result.history) - 1)
    result.total_time = time.perf_counter() - t_start
    return result
