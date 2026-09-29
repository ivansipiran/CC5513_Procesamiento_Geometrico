"""
Experimento 1 -- Puerta de entrada de la receta:
?los keypoints de Harris 3D son repetibles entre dos scans parciales?

Si dos scans no eligen los mismos puntos fisicos, ningun descriptor puede
salvar el pipeline: no hay correspondencias correctas que encontrar.

Comparamos Harris 3D (nube y malla) contra ISS y contra muestreo aleatorio.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, str(__file__).rsplit("/experiments/", 1)[0])
from gr import data, harris3d, metrics                      # noqa: E402

VOXEL = 0.008
NKP = 200
NMS = 4 * VOXEL
EPS_LIST = [1, 2, 3, 4, 6]          # en unidades de voxel


def kp_harris_cloud(pcd, voxel):
    kp = harris3d.detect_pointcloud(pcd, n_keypoints=NKP, k_ring=60,
                                    nms_radius=NMS)
    return kp.points


def kp_harris_cloud_noble(pcd, voxel):
    kp = harris3d.detect_pointcloud(pcd, n_keypoints=NKP, k_ring=60,
                                    nms_radius=NMS, response="noble")
    return kp.points


def kp_iss(pcd, voxel):
    k = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd, salient_radius=6 * voxel, non_max_radius=NMS,
        gamma_21=0.975, gamma_32=0.975, min_neighbors=8)
    P = np.asarray(k.points)
    if len(P) > NKP:
        P = P[np.random.default_rng(0).choice(len(P), NKP, replace=False)]
    return P


def kp_random(pcd, voxel):
    P = np.asarray(pcd.points)
    rng = np.random.default_rng(0)
    return P[rng.choice(len(P), min(NKP, len(P)), replace=False)]


DETECTORS = {
    "Harris3D (nube)": kp_harris_cloud,
    "Harris3D (Noble)": kp_harris_cloud_noble,
    "ISS": kp_iss,
    "aleatorio": kp_random,
}


def main():
    mesh_names = ["bunny", "armadillo"]
    views = [(70, 0), (70, 45), (70, 90), (70, 135)]
    rows = []
    for mn in mesh_names:
        mesh = data.load_mesh(mn)
        for vb in views[1:]:
            pair = data.make_pair(mesh, f"{mn}_{vb[1]}", view_a=views[0],
                                  view_b=vb, voxel=VOXEL, seed=7,
                                  keep_meshes=False)
            print(f"\n=== {pair.summary()} ===", flush=True)
            for dname, fn in DETECTORS.items():
                t0 = time.time()
                ks = fn(pair.source, VOXEL)
                kt = fn(pair.target, VOXEL)
                dt = time.time() - t0
                for e in EPS_LIST:
                    r = metrics.repeatability(ks, kt, pair.T_gt, pair.source,
                                              pair.target, eps=e * VOXEL,
                                              overlap_tau=1.5 * VOXEL)
                    rows.append(dict(mesh=mn, view=vb[1], overlap=pair.overlap,
                                     detector=dname, eps_vox=e,
                                     rel=r["relative"], n_overlap=r["n_overlap"],
                                     n_src=r["n_src"], n_tgt=r["n_tgt"], time=dt))
                got = [f"{e}v:{[x for x in rows if x['detector']==dname and x['eps_vox']==e][-1]['rel']:.2f}"
                       for e in EPS_LIST]
                print(f"  {dname:20s} n={len(ks):4d}/{len(kt):4d} "
                      f"{' '.join(got)}  ({dt:.1f}s)", flush=True)

    with open("experiments/results_repeatability.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("\nguardado -> experiments/results_repeatability.json")


if __name__ == "__main__":
    main()
