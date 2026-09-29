"""
Experimento 6 -- Datos reales: fragmentos RGB-D (Redwood living room).

No hay malla ni objeto aislado: son escenas con ruido, agujeros y superficies
planas grandes. Es el caso donde los metodos basados en keypoints sufren mas.
Como referencia de pose usamos la que produce el registro global + ICP que
converge (verificada por fitness), no un ground truth externo.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(0, str(__file__).rsplit("/experiments/", 1)[0])
from gr import baselines, data, fourpcs, metrics                 # noqa: E402
from gr import pipeline_feature as pf                            # noqa: E402

VOXEL = 0.05          # metros


def prep(pcd):
    p = pcd.voxel_down_sample(VOXEL)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 4,
                                                            max_nn=50))
    p.orient_normals_towards_camera_location(np.zeros(3))
    return p


def main():
    frags = [prep(p) for p in data.load_redwood_fragments()]
    print([len(f.points) for f in frags])
    pairs = [(0, 1), (1, 2), (0, 2)]
    rows = []
    for a, b in pairs:
        src, tgt = frags[a], frags[b]
        # referencia: FPFH+RANSAC con muchas iteraciones + ICP, verificado
        T_ref = baselines.fpfh_ransac(src, tgt, VOXEL, n_iter=4000000)
        rmse, fit = metrics.inlier_rmse_and_fitness(src, tgt, T_ref, VOXEL * 1.5)
        print(f"\n### fragmentos {a}->{b}   referencia fitness={fit:.2f} "
              f"rmse={rmse:.3f}", flush=True)
        cases = {
            "Harris3D+spin+grouping": ("feat", pf.FeatureConfig(
                voxel=VOXEL, detector="harris", descriptor="spin",
                solver="grouping", n_keypoints=400)),
            "Harris3D+spin+RANSAC": ("feat", pf.FeatureConfig(
                voxel=VOXEL, detector="harris", descriptor="spin",
                solver="o3d_ransac", n_keypoints=400)),
            "Harris3D+FPFH+RANSAC": ("feat", pf.FeatureConfig(
                voxel=VOXEL, detector="harris", descriptor="fpfh",
                solver="o3d_ransac", n_keypoints=400)),
            "FPFH+RANSAC (Open3D)": ("fpfh", None),
            "FGR (Open3D)": ("fgr", None),
            "Super4PCS": ("4pcs", fourpcs.FourPCSConfig(
                delta=2 * VOXEL, n_samples=700, n_bases=60,
                variant="super4pcs")),
        }
        for name, (kind, cfg) in cases.items():
            t0 = time.time()
            if kind == "feat":
                T = pf.register(src, tgt, cfg).T
            elif kind == "fpfh":
                T = baselines.fpfh_ransac(src, tgt, VOXEL)
            elif kind == "fgr":
                T = baselines.fgr(src, tgt, VOXEL)
            else:
                _, T = fourpcs.register_with_icp(src, tgt, cfg, 3 * VOXEL)
            dt = time.time() - t0
            rot = metrics.rotation_error_deg(T, T_ref)
            tr = metrics.translation_error(T, T_ref)
            r2, f2 = metrics.inlier_rmse_and_fitness(src, tgt, T, VOXEL * 1.5)
            ok = rot < 5 and tr < 0.3
            rows.append(dict(pair=f"{a}-{b}", method=name, rot_err=rot,
                             trans_err=tr, fitness=f2, inlier_rmse=r2,
                             success=bool(ok), time=dt))
            print(f"  {name:24s} rot={rot:7.2f} trans={tr:.3f} "
                  f"fitness={f2:.2f} {'OK ' if ok else 'FAIL'} {dt:5.1f}s",
                  flush=True)
    with open("experiments/results_real.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
