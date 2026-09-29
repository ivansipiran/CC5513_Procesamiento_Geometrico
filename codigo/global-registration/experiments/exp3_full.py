"""
Experimento 3 -- Comparacion completa.

Metodos con features:
  * receta de las slides  : Harris3D + spin images + agrupamiento geometrico
  * variantes            : cambiando solver (RANSAC) o descriptor (FPFH)
  * baselines Open3D     : FPFH+RANSAC sobre muestreo uniforme, FGR

Metodos sin features:
  * 4PCS
  * Super4PCS

Barrido de solapamiento (angulo entre vistas) y de ruido.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/experiments/", 1)[0])
from gr import baselines, data, fourpcs, metrics                 # noqa: E402
from gr import pipeline_feature as pf                            # noqa: E402

VOXEL = 0.008
ROT_OK, TRANS_OK = 5.0, 0.05


def methods(voxel, overlap_hint):
    F = pf.FeatureConfig
    return {
        "Harris3D+spin+grouping": ("feat", F(voxel=voxel, detector="harris",
                                             descriptor="spin", solver="grouping")),
        "Harris3D+spin+RANSAC": ("feat", F(voxel=voxel, detector="harris",
                                           descriptor="spin", solver="o3d_ransac")),
        "Harris3D+FPFH+RANSAC": ("feat", F(voxel=voxel, detector="harris",
                                           descriptor="fpfh", solver="o3d_ransac")),
        "uniforme+spin+RANSAC": ("feat", F(voxel=voxel, detector="uniform",
                                           descriptor="spin", solver="o3d_ransac")),
        "FPFH+RANSAC (Open3D)": ("fpfh_ransac", None),
        "FGR (Open3D)": ("fgr", None),
        "4PCS": ("4pcs", fourpcs.FourPCSConfig(
            delta=2 * voxel, n_samples=500, n_bases=40, variant="4pcs",
            overlap=overlap_hint)),
        "Super4PCS": ("4pcs", fourpcs.FourPCSConfig(
            delta=2 * voxel, n_samples=500, n_bases=40, variant="super4pcs",
            overlap=overlap_hint)),
    }


def run_method(kind, cfg, pair, seed):
    t0 = time.time()
    if kind == "feat":
        cfg.voxel = pair.voxel
        r = pf.register(pair.source, pair.target, cfg, seed=seed)
        T, extra = r.T, dict(n_corr=r.n_correspondences)
    elif kind == "fpfh_ransac":
        T = baselines.fpfh_ransac(pair.source, pair.target, pair.voxel, seed=seed)
        extra = {}
    elif kind == "fgr":
        T = baselines.fgr(pair.source, pair.target, pair.voxel)
        extra = {}
    elif kind == "4pcs":
        cfg.seed = seed
        cfg.delta = 2 * pair.voxel
        r, T = fourpcs.register_with_icp(pair.source, pair.target, cfg,
                                         3 * pair.voxel)
        extra = dict(lcp=r.lcp, n_cand=r.n_candidates)
    else:
        raise ValueError(kind)
    dt = time.time() - t0
    return T, dt, extra


def evaluate(T, pair, dt, extra):
    rot = metrics.rotation_error_deg(T, pair.T_gt)
    tr = metrics.translation_error(T, pair.T_gt)
    return dict(rot_err=rot, trans_err=tr,
                rmse=metrics.rmse_on_gt(pair.source, T, pair.T_gt),
                success=bool(rot < ROT_OK and tr < TRANS_OK),
                time=dt, **extra)


def main():
    conditions = []
    for mesh_name in ("bunny", "armadillo"):
        for dphi in (45, 90, 135):
            for seed in (1, 2, 3):
                conditions.append((mesh_name, dphi, 0.0, seed))
    for sigma in (0.002, 0.004):
        for mesh_name in ("bunny", "armadillo"):
            for seed in (1, 2, 3):
                conditions.append((mesh_name, 90, sigma, seed))

    meshes = {n: data.load_mesh(n) for n in ("bunny", "armadillo")}
    rows = []
    for (mn, dphi, sigma, seed) in conditions:
        pair = data.make_pair(meshes[mn], f"{mn}_{dphi}", view_a=(70, 0),
                              view_b=(70, dphi), voxel=VOXEL,
                              noise_sigma=sigma, seed=seed, keep_meshes=False)
        print(f"\n### {mn} dphi={dphi} sigma={sigma} seed={seed} "
              f"overlap={pair.overlap:.2f}", flush=True)
        for name, (kind, cfg) in methods(VOXEL, pair.overlap).items():
            try:
                T, dt, extra = run_method(kind, cfg, pair, seed)
                r = evaluate(T, pair, dt, extra)
            except Exception as e:                      # noqa: BLE001
                r = dict(rot_err=180.0, trans_err=9.9, rmse=9.9, success=False,
                         time=0.0, error=repr(e))
            r.update(method=name, mesh=mn, dphi=dphi, sigma=sigma, seed=seed,
                     overlap=pair.overlap)
            rows.append(r)
            print(f"  {name:24s} rot={r['rot_err']:7.2f} "
                  f"trans={r['trans_err']:.4f} rmse={r['rmse']:.4f} "
                  f"{'OK ' if r['success'] else 'FAIL'} {r['time']:5.1f}s",
                  flush=True)
        with open("experiments/results_full.json", "w") as f:
            json.dump(rows, f, indent=1)

    print("\n=== RESUMEN ===")
    names = list(methods(VOXEL, 0.5).keys())
    for name in names:
        sub = [r for r in rows if r["method"] == name]
        s = metrics.summarize(sub)
        print(f"{name:24s} recall={s['recall']:.2f} "
              f"rot_med={s['rot_err_median']:8.2f} t_med={s['time_median']:5.1f}s")


if __name__ == "__main__":
    main()
