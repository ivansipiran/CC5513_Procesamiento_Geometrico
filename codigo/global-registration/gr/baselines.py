"""Baselines de referencia de Open3D: FPFH+RANSAC y Fast Global Registration."""
from __future__ import annotations

import numpy as np
import open3d as o3d

from . import metrics


def fpfh(pcd, voxel, radius_mult=5.0):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_mult * voxel,
                                                  max_nn=100))


def fpfh_ransac(source, target, voxel, tau_mult=1.5, n_iter=100000,
                icp=True, seed=0):
    fs, ft = fpfh(source, voxel), fpfh(target, voxel)
    tau = tau_mult * voxel
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, fs, ft, True, tau,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(tau)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(n_iter, 0.999))
    T = np.asarray(res.transformation)
    if icp:
        T = metrics.icp_refine(source, target, T, 3 * voxel)
    return T


def fgr(source, target, voxel, icp=True):
    fs, ft = fpfh(source, voxel), fpfh(target, voxel)
    opt = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=1.5 * voxel)
    res = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source, target, fs, ft, opt)
    T = np.asarray(res.transformation)
    if icp:
        T = metrics.icp_refine(source, target, T, 3 * voxel)
    return T
