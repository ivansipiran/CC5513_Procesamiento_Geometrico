"""
Global registration basado en features -- la receta de las slides completa.

    Point Sampling   ->  Harris 3D  (Sipiran & Bustos)
    Point Description->  Spin Images (Johnson & Hebert)
    Point Matching   ->  correlacion C(P,Q)
    Find Transform   ->  agrupamiento geometrico / RANSAC  + ICP

Todos los pasos son intercambiables para poder aislar donde se rompe el
pipeline (por eso `detector`, `descriptor` y `solver` son parametros).
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from . import grouping, harris3d, metrics
from . import spin_images as si


# --------------------------------------------------------------------------- #
@dataclass
class FeatureConfig:
    voxel: float = 0.008
    n_keypoints: int = 300
    nms_mult: float = 4.0
    # detector: 'harris' | 'harris_noble' | 'iss' | 'uniform'
    detector: str = "harris"
    harris_k_ring: int = 60
    # descriptor: 'spin' | 'fpfh'
    descriptor: str = "spin"
    spin_bin_mult: float = 2.0
    spin_width: int = 16
    spin_support_angle: float = 60.0
    # matching
    match_mode: str = "mutual"          # 'mutual' | 'global' | 'topk'
    top_k: int = 2
    # solver: 'grouping' | 'distance' | 'ransac' | 'o3d_ransac'
    solver: str = "grouping"
    gc_gamma_mult: float = 6.0
    gc_thresh_mult: float = 1.5
    dc_eps_mult: float = 2.0
    ransac_iter: int = 20000
    inlier_mult: float = 2.5            # eps de inlier en unidades de voxel
    verify_mult: float = 2.0
    icp: bool = True
    icp_mult: float = 3.0


@dataclass
class FeatureResult:
    T: np.ndarray
    T_before_icp: np.ndarray
    score: float
    n_keypoints: tuple[int, int]
    n_correspondences: int
    timings: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
def detect(pcd, cfg: FeatureConfig) -> np.ndarray:
    """Devuelve indices de keypoints en la nube."""
    nms = cfg.nms_mult * cfg.voxel
    if cfg.detector in ("harris", "harris_noble"):
        resp = "noble" if cfg.detector == "harris_noble" else "harris"
        kp = harris3d.detect_pointcloud(pcd, cfg.n_keypoints,
                                        k_ring=cfg.harris_k_ring,
                                        nms_radius=nms, response=resp)
        return kp.indices
    if cfg.detector == "iss":
        k = o3d.geometry.keypoint.compute_iss_keypoints(
            pcd, salient_radius=6 * cfg.voxel, non_max_radius=nms,
            gamma_21=0.975, gamma_32=0.975, min_neighbors=8)
        tree = o3d.geometry.KDTreeFlann(pcd)
        idx = [tree.search_knn_vector_3d(p, 1)[1][0] for p in np.asarray(k.points)]
        return np.unique(np.asarray(idx))
    if cfg.detector == "uniform":
        n = len(pcd.points)
        rng = np.random.default_rng(0)
        return rng.choice(n, min(cfg.n_keypoints, n), replace=False)
    raise ValueError(cfg.detector)


def describe(pcd, idx, cfg: FeatureConfig) -> np.ndarray:
    if cfg.descriptor == "spin":
        par = si.SpinImageParams(bin_size=cfg.spin_bin_mult * cfg.voxel,
                                 image_width=cfg.spin_width,
                                 support_angle_deg=cfg.spin_support_angle)
        return si.describe(pcd, idx, par)
    if cfg.descriptor == "fpfh":
        f = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * cfg.voxel,
                                                      max_nn=100))
        return np.asarray(f.data).T[idx]
    raise ValueError(cfg.descriptor)


def similarity(Ds, Dt, cfg: FeatureConfig) -> np.ndarray:
    if cfg.descriptor == "spin":
        return si.spin_similarity_matrix(Ds, Dt)
    d = ((Ds[:, None, :] - Dt[None, :, :]) ** 2).sum(axis=2)
    return -d


# --------------------------------------------------------------------------- #
def register(source: o3d.geometry.PointCloud,
             target: o3d.geometry.PointCloud,
             cfg: FeatureConfig,
             seed: int = 0) -> FeatureResult:
    t = {}
    v = cfg.voxel
    # RANSAC de Open3D tiene su propio RNG: sin esto el resultado cambia entre
    # corridas, que es justo lo que no se quiere en un demo ni en un benchmark.
    try:
        o3d.utility.random.seed(int(seed))
    except AttributeError:                    # Open3D < 0.16
        pass
    pts_s = np.asarray(source.points)
    pts_t = np.asarray(target.points)
    nrm_s = np.asarray(source.normals)
    nrm_t = np.asarray(target.normals)

    t0 = time.time()
    is_ = detect(source, cfg)
    it_ = detect(target, cfg)
    t["detect"] = time.time() - t0

    t0 = time.time()
    Ds = describe(source, is_, cfg)
    Dt = describe(target, it_, cfg)
    t["describe"] = time.time() - t0

    t0 = time.time()
    C = similarity(Ds, Dt, cfg)
    pairs = si.match(C, mode=cfg.match_mode, top_k=cfg.top_k)
    t["match"] = time.time() - t0

    KS, KT = pts_s[is_], pts_t[it_]
    NS, NT = nrm_s[is_], nrm_t[it_]

    verifier = grouping.Verifier(target, cfg.verify_mult * v)
    t0 = time.time()
    if len(pairs) < 3:
        T, score = np.eye(4), 0.0
        cands = []
    elif cfg.solver == "grouping":
        groups = grouping.geometric_consistency_grouping(
            KS, NS, KT, NT, pairs,
            gamma=cfg.gc_gamma_mult * v, thresh=cfg.gc_thresh_mult * v,
            min_group=3)
        cands = grouping.poses_from_groups(KS, KT, pairs, groups, verifier, pts_s)
        T, score = (cands[0].T, cands[0].score) if cands else (np.eye(4), 0.0)
    elif cfg.solver == "distance":
        groups = grouping.distance_consistency_groups(
            KS, KT, pairs, eps=cfg.dc_eps_mult * v, min_group=3)
        cands = grouping.poses_from_groups(KS, KT, pairs, groups, verifier, pts_s)
        T, score = (cands[0].T, cands[0].score) if cands else (np.eye(4), 0.0)
    elif cfg.solver == "ransac":
        T, score = grouping.ransac_pose(KS, KT, pairs, verifier, pts_s,
                                        n_iter=cfg.ransac_iter,
                                        eps=cfg.inlier_mult * v, seed=seed)
        cands = []
    elif cfg.solver == "o3d_ransac":
        corr = o3d.utility.Vector2iVector(pairs.astype(np.int32))
        kps = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(KS))
        kpt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(KT))
        res = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            kps, kpt, corr, cfg.inlier_mult * v,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                 cfg.inlier_mult * v)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(cfg.ransac_iter, 0.999))
        T = np.asarray(res.transformation)
        score = verifier.score(pts_s, T)
        cands = []
    else:
        raise ValueError(cfg.solver)
    t["solve"] = time.time() - t0

    T_pre = T.copy()
    if cfg.icp:
        t0 = time.time()
        T = metrics.icp_refine(source, target, T, cfg.icp_mult * v)
        t["icp"] = time.time() - t0

    return FeatureResult(T=T, T_before_icp=T_pre, score=score,
                         n_keypoints=(len(is_), len(it_)),
                         n_correspondences=len(pairs), timings=t,
                         extra=dict(kp_src=KS, kp_tgt=KT, pairs=pairs,
                                    n_candidates=len(cands)))
