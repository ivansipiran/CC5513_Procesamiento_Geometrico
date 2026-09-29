#!/usr/bin/env python3
"""
Demo: global registration de dos scans parciales, por los dos caminos.

Corre los metodos elegidos y abre Polyscope con la configuracion inicial y los
resultados; se cambia entre estados con los radio buttons del panel.

    python3 demo.py --mesh bunny --dphi 90
    python3 demo.py --mesh armadillo --dphi 120          # aqui falla la receta
    python3 demo.py --mesh armadillo --dphi 135 --method super4pcs
    python3 demo.py --dphi 90 --no-view                  # solo numeros

Sin pantalla (contenedor, servidor) Polyscope usa el backend EGL headless y
guarda un PNG por estado en vez de abrir la ventana.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from gr import data, fourpcs, metrics
from gr import pipeline_feature as pf
from gr import psview


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="bunny", choices=["bunny", "armadillo"])
    ap.add_argument("--dphi", type=float, default=90,
                    help="angulo entre las dos vistas (controla el solapamiento)")
    ap.add_argument("--voxel", type=float, default=0.008)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--method", default="ambos",
                    choices=["ambos", "features", "slides", "4pcs", "super4pcs"])
    ap.add_argument("--no-view", dest="view", action="store_false",
                    help="no abrir Polyscope, solo imprimir los numeros")
    ap.add_argument("--shot", default=None,
                    help="prefijo para guardar un PNG por estado (fuerza headless)")
    ap.add_argument("--fig", default=None,
                    help="ademas, guardar una figura estatica de matplotlib")
    args = ap.parse_args()

    mesh = data.load_mesh(args.mesh)
    pair = data.make_pair(mesh, f"{args.mesh} {args.dphi:.0f}deg",
                          view_a=(70, 0), view_b=(70, args.dphi),
                          voxel=args.voxel, noise_sigma=args.noise,
                          seed=args.seed, keep_meshes=False)
    print(pair.summary())

    poses: list[psview.Pose] = [
        psview.Pose("Inicial (pose desconocida)", np.eye(4),
                    "Las dos nubes tal como llegan. En verde y rojo, las "
                    "correspondencias\nque produjeron Harris 3D + spin images.",
                    show_correspondences=True)
    ]
    kp = corr = inl = None
    results = {}

    def report(name, T, dt, extra=""):
        rot = metrics.rotation_error_deg(T, pair.T_gt)
        tr = metrics.translation_error(T, pair.T_gt)
        ok = rot < 5 and tr < 0.05
        line = (f"{name:34s} rot={rot:7.2f} deg  trans={tr:.4f}  "
                f"rmse={metrics.rmse_on_gt(pair.source, T, pair.T_gt):.4f}  "
                f"{'OK' if ok else 'FALLA'}  {dt:5.1f}s {extra}")
        print(line)
        results[name] = T
        poses.append(psview.Pose(
            name, T,
            f"{psview._fmt_error(T, pair.T_gt)}\n{dt:.1f} s  {extra}"))

    # --- con features -------------------------------------------------------
    if args.method in ("ambos", "features"):
        cfg = pf.FeatureConfig(voxel=args.voxel, solver="o3d_ransac",
                               ransac_iter=200000)
        t0 = time.time()
        r = pf.register(pair.source, pair.target, cfg, seed=args.seed)
        report("Harris3D + spin + RANSAC", r.T, time.time() - t0,
               f"({r.n_correspondences} corr)")
        kp = (r.extra["kp_src"], r.extra["kp_tgt"])
        corr = r.extra["pairs"]
        inl = metrics.correspondence_inliers(kp[0], kp[1], corr, pair.T_gt,
                                             3 * args.voxel)

    if args.method in ("ambos", "slides"):
        cfg = pf.FeatureConfig(voxel=args.voxel, solver="grouping")
        t0 = time.time()
        r = pf.register(pair.source, pair.target, cfg, seed=args.seed)
        report("Harris3D + spin + agrupamiento", r.T, time.time() - t0,
               f"({r.n_correspondences} corr)")
        if kp is None:
            kp = (r.extra["kp_src"], r.extra["kp_tgt"])
            corr = r.extra["pairs"]
            inl = metrics.correspondence_inliers(kp[0], kp[1], corr, pair.T_gt,
                                                 3 * args.voxel)

    # --- sin features -------------------------------------------------------
    for v in ("4pcs", "super4pcs"):
        if args.method in ("ambos", v):
            cfg = fourpcs.FourPCSConfig(delta=2 * args.voxel, n_samples=600,
                                        n_bases=50, variant=v, seed=args.seed)
            t0 = time.time()
            rr, T = fourpcs.register_with_icp(pair.source, pair.target, cfg,
                                              3 * args.voxel)
            report(v, T, time.time() - t0, f"(LCP={rr.lcp:.2f})")

    poses.append(psview.Pose("Ground truth", pair.T_gt,
                             "La pose verdadera, para comparar."))

    # --- figura estatica opcional ------------------------------------------
    if args.fig:
        import matplotlib.pyplot as plt
        from gr import viz
        n = len(results) + 1
        fig = plt.figure(figsize=(4.2 * n, 4.2))
        ax = fig.add_subplot(1, n, 1, projection="3d")
        viz.registration(ax, pair.source, pair.target)
        ax.set_title("entrada")
        for i, (name, T) in enumerate(results.items()):
            ax = fig.add_subplot(1, n, i + 2, projection="3d")
            viz.registration(ax, pair.source, pair.target, T)
            ax.set_title(name, fontsize=9)
        print("figura ->", viz.save(fig, args.fig))

    # --- Polyscope ----------------------------------------------------------
    if args.view:
        shots = psview.show(pair, poses, keypoints=kp, correspondences=corr,
                            inliers=inl, screenshot_prefix=args.shot)
        for s in shots:
            print("captura ->", s)


if __name__ == "__main__":
    main()
