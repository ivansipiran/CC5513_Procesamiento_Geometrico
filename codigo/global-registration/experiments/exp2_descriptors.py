"""
Experimento 2 -- ?Que tan buenas son las correspondencias?

Fijamos los mismos keypoints (Harris 3D) y cambiamos solo el descriptor:
spin images (la receta de las slides) vs FPFH. Medimos la razon de inliers
de las correspondencias, que es lo que determina si RANSAC / el agrupamiento
geometrico pueden encontrar la pose.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, str(__file__).rsplit("/experiments/", 1)[0])
from gr import data, harris3d, metrics, spin_images as si    # noqa: E402

VOXEL = 0.008
NKP = 300
EPS = 3 * VOXEL


def fpfh_desc(pcd, idx, voxel, radius_mult=5.0):
    f = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_mult * voxel,
                                                  max_nn=100))
    return np.asarray(f.data).T[idx]


def match_l2(A, B, mode="mutual"):
    d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    C = -d                                  # mayor = mejor
    return si.match(C, mode=mode)


def run_pair(pair, spin_params_list):
    pts_s = np.asarray(pair.source.points)
    pts_t = np.asarray(pair.target.points)
    nms = 4 * pair.voxel
    ks = harris3d.detect_pointcloud(pair.source, NKP, nms_radius=nms)
    kt = harris3d.detect_pointcloud(pair.target, NKP, nms_radius=nms)
    KS, KT = pts_s[ks.indices], pts_t[kt.indices]

    rows = []

    for tag, par in spin_params_list:
        t0 = time.time()
        Ds = si.describe(pair.source, ks.indices, par)
        Dt = si.describe(pair.target, kt.indices, par)
        C = si.spin_similarity_matrix(Ds, Dt)
        dt = time.time() - t0
        for mode in ("mutual", "global"):
            pairs = si.match(C, mode=mode)
            inl = metrics.correspondence_inliers(KS, KT, pairs, pair.T_gt, EPS)
            rows.append(dict(desc=f"spin[{tag}]", mode=mode, n=len(pairs),
                             inliers=int(inl.sum()),
                             ratio=float(inl.mean()) if len(pairs) else 0.0,
                             top50=float(inl[:50].mean()) if len(pairs) else 0.0,
                             time=dt))

    t0 = time.time()
    Fs = fpfh_desc(pair.source, ks.indices, pair.voxel)
    Ft = fpfh_desc(pair.target, kt.indices, pair.voxel)
    dt = time.time() - t0
    for mode in ("mutual", "global"):
        pairs = match_l2(Fs, Ft, mode)
        inl = metrics.correspondence_inliers(KS, KT, pairs, pair.T_gt, EPS)
        rows.append(dict(desc="FPFH", mode=mode, n=len(pairs),
                         inliers=int(inl.sum()),
                         ratio=float(inl.mean()) if len(pairs) else 0.0,
                         top50=float(inl[:50].mean()) if len(pairs) else 0.0,
                         time=dt))
    return rows


def main():
    spin_params = [
        ("W16 b2v", si.SpinImageParams(bin_size=2 * VOXEL, image_width=16)),
        ("W16 b4v", si.SpinImageParams(bin_size=4 * VOXEL, image_width=16)),
        ("W8  b4v", si.SpinImageParams(bin_size=4 * VOXEL, image_width=8)),
        ("W20 b3v", si.SpinImageParams(bin_size=3 * VOXEL, image_width=20)),
        ("W16 b4v sinNorm",
         si.SpinImageParams(bin_size=4 * VOXEL, image_width=16, normalize=False)),
    ]
    out = []
    for mn in ("bunny", "armadillo"):
        mesh = data.load_mesh(mn)
        for dphi in (45, 90, 135):
            pair = data.make_pair(mesh, f"{mn}_{dphi}", view_a=(70, 0),
                                  view_b=(70, dphi), voxel=VOXEL, seed=7,
                                  keep_meshes=False)
            print(f"\n=== {pair.summary()} ===", flush=True)
            for r in run_pair(pair, spin_params):
                r.update(mesh=mn, view=dphi, overlap=pair.overlap)
                out.append(r)
                print(f"  {r['desc']:22s} {r['mode']:7s} "
                      f"n={r['n']:4d} inliers={r['inliers']:4d} "
                      f"ratio={r['ratio']:.3f} top50={r['top50']:.3f} "
                      f"({r['time']:.1f}s)", flush=True)

    with open("experiments/results_descriptors.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nguardado -> experiments/results_descriptors.json")


if __name__ == "__main__":
    main()
