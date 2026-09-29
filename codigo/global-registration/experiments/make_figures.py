"""Figuras de las etapas del pipeline, para el reporte y para las slides."""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/experiments/", 1)[0])
from gr import data, harris3d, metrics, viz                      # noqa: E402
from gr import pipeline_feature as pf                            # noqa: E402
from gr import spin_images as si                                 # noqa: E402

VOXEL = 0.008
VIEW = (18, -62)


def fig_stages(mesh_name="armadillo", dphi=60, seed=7):
    mesh = data.load_mesh(mesh_name)
    pair = data.make_pair(mesh, mesh_name, view_a=(70, 0), view_b=(70, dphi),
                          voxel=VOXEL, seed=seed, keep_meshes=False)
    cfg = pf.FeatureConfig(voxel=VOXEL, solver="o3d_ransac")
    res = pf.register(pair.source, pair.target, cfg)
    KS, KT, pairs = res.extra["kp_src"], res.extra["kp_tgt"], res.extra["pairs"]
    inl = metrics.correspondence_inliers(KS, KT, pairs, pair.T_gt, 3 * VOXEL)

    fig = plt.figure(figsize=(15, 4.2))
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    viz.registration(ax, pair.source, pair.target, np.eye(4), view=VIEW)
    ax.set_title("1. entrada: dos scans\n(pose desconocida)", fontsize=10)

    ax = fig.add_subplot(1, 4, 2, projection="3d")
    h = harris3d.harris_response_pointcloud(np.asarray(pair.target.points))
    viz.cloud(ax, pair.target, scalar=h, s=2.0, view=VIEW)
    ax.scatter(*KT.T, c="k", s=9, depthshade=False)
    ax.set_title("2. Harris 3D\nrespuesta + keypoints", fontsize=10)

    ax = fig.add_subplot(1, 4, 3, projection="3d")
    viz.correspondences(ax, KS, KT, pairs, inl, view=VIEW,
                        source=pair.source, target=pair.target)
    ax.set_title(f"3. correspondencias por spin images\n"
                 f"{int(inl.sum())}/{len(pairs)} correctas", fontsize=10)

    ax = fig.add_subplot(1, 4, 4, projection="3d")
    viz.registration(ax, pair.source, pair.target, res.T, view=VIEW)
    ax.set_title(f"4. pose + ICP\nerror {metrics.rotation_error_deg(res.T, pair.T_gt):.2f} deg",
                 fontsize=10)
    return viz.save(fig, "figs/fig_stages.png")


def fig_corr_overlap(mesh_name="bunny"):
    mesh = data.load_mesh(mesh_name)
    fig = plt.figure(figsize=(13, 4.4))
    for i, dphi in enumerate((45, 90, 135)):
        pair = data.make_pair(mesh, mesh_name, view_a=(70, 0), view_b=(70, dphi),
                              voxel=VOXEL, seed=7, keep_meshes=False)
        cfg = pf.FeatureConfig(voxel=VOXEL, solver="o3d_ransac")
        res = pf.register(pair.source, pair.target, cfg)
        KS, KT, prs = res.extra["kp_src"], res.extra["kp_tgt"], res.extra["pairs"]
        inl = metrics.correspondence_inliers(KS, KT, prs, pair.T_gt, 3 * VOXEL)
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        viz.correspondences(ax, KS, KT, prs, inl, view=VIEW, max_lines=90,
                            source=pair.source, target=pair.target)
        ax.set_title(f"solapamiento {pair.overlap:.2f}\n"
                     f"inliers {inl.mean():.0%} ({int(inl.sum())}/{len(prs)})",
                     fontsize=10)
    return viz.save(fig, "figs/fig_corr_overlap.png")


def fig_spin(mesh_name="armadillo"):
    mesh = data.load_mesh(mesh_name)
    pair = data.make_pair(mesh, mesh_name, view_a=(70, 0), view_b=(70, 45),
                          voxel=VOXEL, seed=7, keep_meshes=False)
    kp = harris3d.detect_pointcloud(pair.source, 200, nms_radius=4 * VOXEL)
    par = si.SpinImageParams(bin_size=2 * VOXEL, image_width=16)
    D = si.describe(pair.source, kp.indices, par)
    fig, axes = plt.subplots(2, 8, figsize=(13, 3.6))
    for a, i in zip(axes.ravel(), range(16)):
        a.imshow(D[i].reshape(16, 16), cmap="gray_r")
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Spin images de 16 keypoints  (eje horizontal: alpha, vertical: beta)",
                 fontsize=10)
    return viz.save(fig, "figs/fig_spin.png")


def fig_harris_mesh(mesh_name="bunny"):
    mesh = data.load_mesh(mesh_name)
    V = np.asarray(mesh.vertices)
    h = harris3d.harris_response_mesh(mesh, delta_frac=0.01)
    kp = harris3d.select_keypoints(V, h, 120, nms_radius=0.025)
    fig = plt.figure(figsize=(11, 4.6))
    for i, vw in enumerate([(15, -60), (15, 120)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        viz.mesh_scalar(ax, mesh, h, keypoints=kp, view=vw)
    fig.suptitle("Harris 3D sobre malla: vecindades adaptativas por anillos "
                 "(respuesta percentilizada; puntos negros = keypoints)", fontsize=10)
    return viz.save(fig, "figs/fig_harris_mesh.png")


if __name__ == "__main__":
    for f in (fig_harris_mesh, fig_spin, fig_stages, fig_corr_overlap):
        print(f())
