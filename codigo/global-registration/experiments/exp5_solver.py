"""
Experimento 5 -- Donde se rompe la receta y como arreglarla.

En solapamiento bajo la razon de inliers de las correspondencias cae a ~5%.
Con ~60 correspondencias eso son 3 inliers: el crecimiento voraz de grupos de
Johnson & Hebert no tiene de donde agarrarse. Aqui se compara, sobre EXACTAMENTE
las mismas correspondencias, el paso final:

    agrupamiento geometrico (slides)  vs  clique de distancias  vs  RANSAC

y el efecto de generar mas correspondencias (mutual vs top-k).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/experiments/", 1)[0])
from gr import data, metrics                                     # noqa: E402
from gr import pipeline_feature as pf                            # noqa: E402

VOXEL = 0.008


def main():
    meshes = {n: data.load_mesh(n) for n in ("bunny", "armadillo")}
    combos = []
    for solver in ("grouping", "distance", "ransac", "o3d_ransac"):
        for mm, tk in (("mutual", 1), ("topk", 2), ("topk", 5)):
            combos.append((solver, mm, tk))

    rows = []
    for mn in ("bunny", "armadillo"):
        for dphi in (90, 120, 135, 150):
            for seed in (1, 2, 3):
                pair = data.make_pair(meshes[mn], mn, view_a=(70, 0),
                                      view_b=(70, dphi), voxel=VOXEL,
                                      seed=seed, keep_meshes=False)
                print(f"\n### {mn} dphi={dphi} seed={seed} "
                      f"overlap={pair.overlap:.2f}", flush=True)
                for solver, mm, tk in combos:
                    cfg = pf.FeatureConfig(voxel=VOXEL, detector="harris",
                                           descriptor="spin", solver=solver,
                                           match_mode=mm, top_k=tk,
                                           ransac_iter=20000)
                    t0 = time.time()
                    try:
                        r = pf.register(pair.source, pair.target, cfg, seed=seed)
                        T = r.T
                        ncorr = r.n_correspondences
                    except Exception as e:                       # noqa: BLE001
                        T, ncorr = np.eye(4), 0
                        print("   err", e)
                    dt = time.time() - t0
                    rot = metrics.rotation_error_deg(T, pair.T_gt)
                    tr = metrics.translation_error(T, pair.T_gt)
                    row = dict(mesh=mn, dphi=dphi, seed=seed,
                               overlap=pair.overlap, solver=solver,
                               match=f"{mm}{tk if mm=='topk' else ''}",
                               n_corr=ncorr, rot_err=rot, trans_err=tr,
                               rmse=metrics.rmse_on_gt(pair.source, T, pair.T_gt),
                               success=bool(rot < 5 and tr < 0.05), time=dt)
                    rows.append(row)
                    print(f"  {solver:11s} {row['match']:7s} n={ncorr:4d} "
                          f"rot={rot:7.2f} {'OK ' if row['success'] else 'FAIL'} "
                          f"{dt:5.1f}s", flush=True)
                with open("experiments/results_solver.json", "w") as f:
                    json.dump(rows, f, indent=1)

    print("\n=== RESUMEN por solver x matching (recall) ===")
    for solver in ("grouping", "distance", "ransac", "o3d_ransac"):
        for m in ("mutual", "topk2", "topk5"):
            sub = [r for r in rows if r["solver"] == solver and r["match"] == m]
            if sub:
                print(f"{solver:11s} {m:7s} recall={np.mean([r['success'] for r in sub]):.2f} "
                      f"t={np.median([r['time'] for r in sub]):.1f}s")


if __name__ == "__main__":
    main()
