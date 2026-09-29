"""Metricas de evaluacion: pose, registro y repetibilidad de keypoints."""
from __future__ import annotations

import copy

import numpy as np
import open3d as o3d


# --------------------------------------------------------------------------- #
# error de pose
# --------------------------------------------------------------------------- #
def rotation_error_deg(T_est: np.ndarray, T_gt: np.ndarray) -> float:
    R = T_est[:3, :3] @ T_gt[:3, :3].T
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def translation_error(T_est: np.ndarray, T_gt: np.ndarray) -> float:
    return float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))


def rmse_on_gt(source: o3d.geometry.PointCloud,
               T_est: np.ndarray, T_gt: np.ndarray) -> float:
    """RMSE entre la nube llevada por T_est y la misma nube llevada por T_gt.

    Es la metrica honesta: no depende del emparejamiento por vecino mas cercano.
    """
    P = np.asarray(source.points)
    A = (T_est[:3, :3] @ P.T).T + T_est[:3, 3]
    B = (T_gt[:3, :3] @ P.T).T + T_gt[:3, 3]
    return float(np.sqrt(((A - B) ** 2).sum(axis=1).mean()))


def inlier_rmse_and_fitness(source, target, T, tau):
    r = o3d.pipelines.registration.evaluate_registration(source, target, tau, T)
    return float(r.inlier_rmse), float(r.fitness)


def is_success(T_est, T_gt, rot_thresh_deg=5.0, trans_thresh=0.05) -> bool:
    return (rotation_error_deg(T_est, T_gt) < rot_thresh_deg
            and translation_error(T_est, T_gt) < trans_thresh)


# --------------------------------------------------------------------------- #
# repetibilidad de keypoints
# --------------------------------------------------------------------------- #
def in_overlap(pts: np.ndarray, other: o3d.geometry.PointCloud,
               T: np.ndarray, tau: float) -> np.ndarray:
    """Mascara: puntos que, tras aplicar T, caen sobre la otra nube."""
    q = (T[:3, :3] @ pts.T).T + T[:3, 3]
    tree = o3d.geometry.KDTreeFlann(other)
    return np.array([tree.search_radius_vector_3d(p, tau)[0] > 0 for p in q])


def repeatability(kp_src: np.ndarray,
                  kp_tgt: np.ndarray,
                  T_gt: np.ndarray,
                  source: o3d.geometry.PointCloud,
                  target: o3d.geometry.PointCloud,
                  eps: float,
                  overlap_tau: float | None = None) -> dict:
    """Repetibilidad absoluta y relativa (Salti et al., 2011).

    Un keypoint de `source` es repetible si, llevado por T_gt, tiene un keypoint
    de `target` a distancia < eps. Solo se cuentan los keypoints que caen en la
    zona de solapamiento.
    """
    overlap_tau = overlap_tau if overlap_tau is not None else eps
    mask = in_overlap(kp_src, target, T_gt, overlap_tau)
    kp_in = kp_src[mask]
    if len(kp_in) == 0 or len(kp_tgt) == 0:
        return dict(absolute=0, relative=0.0, n_overlap=0, n_src=len(kp_src),
                    n_tgt=len(kp_tgt))
    q = (T_gt[:3, :3] @ kp_in.T).T + T_gt[:3, 3]
    tree = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(kp_tgt)))
    hits = sum(tree.search_radius_vector_3d(p, eps)[0] > 0 for p in q)
    return dict(absolute=int(hits), relative=hits / len(kp_in),
                n_overlap=int(len(kp_in)), n_src=len(kp_src), n_tgt=len(kp_tgt))


# --------------------------------------------------------------------------- #
# calidad de las correspondencias
# --------------------------------------------------------------------------- #
def correspondence_inliers(src_pts: np.ndarray,
                           tgt_pts: np.ndarray,
                           pairs: np.ndarray,
                           T_gt: np.ndarray,
                           eps: float) -> np.ndarray:
    """Mascara de correspondencias correctas bajo el ground truth."""
    if len(pairs) == 0:
        return np.zeros(0, dtype=bool)
    a = src_pts[pairs[:, 0]]
    b = tgt_pts[pairs[:, 1]]
    a = (T_gt[:3, :3] @ a.T).T + T_gt[:3, 3]
    return np.linalg.norm(a - b, axis=1) < eps


def icp_refine(source, target, T_init, tau, max_iter=60, point_to_plane=True):
    est = (o3d.pipelines.registration.TransformationEstimationPointToPlane()
           if point_to_plane else
           o3d.pipelines.registration.TransformationEstimationPointToPoint())
    res = o3d.pipelines.registration.registration_icp(
        source, target, tau, T_init, est,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    return np.asarray(res.transformation)


def summarize(rows: list[dict]) -> dict:
    """Promedios y tasa de exito de una lista de resultados."""
    if not rows:
        return {}
    ok = np.array([r["success"] for r in rows], dtype=bool)
    out = dict(n=len(rows), recall=float(ok.mean()))
    for key in ("rot_err", "trans_err", "rmse", "time"):
        vals = np.array([r[key] for r in rows], dtype=float)
        out[f"{key}_median"] = float(np.median(vals))
        if ok.any():
            out[f"{key}_median_ok"] = float(np.median(vals[ok]))
    return out
