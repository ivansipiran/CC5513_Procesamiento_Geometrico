"""Linea de comandos: demo | compare | normals | models."""

from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np

from . import data as D
from . import methods as M
from .normals import ORIENT_METHODS, NormalParams, estimate, pca_normals
from .pipeline import ALL_METHODS, format_header, run_all, run_normals


# --------------------------------------------------------------------------- #
def _scene_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("escena")
    g.add_argument("--model", default="blob", choices=list(D.MODELS))
    g.add_argument("--preset", choices=list(D.PRESETS), default=None,
                   help="regimen predefinido (sobrescribe -n/--noise/--outliers/--hole)")
    g.add_argument("-n", "--n-points", type=int, default=8000)
    g.add_argument("--noise", type=float, default=0.0013, help="sigma, fraccion de la diagonal")
    g.add_argument("--outliers", type=float, default=0.0, help="fraccion de puntos espurios")
    g.add_argument("--hole", type=float, default=0.0, help="radio del hueco, fraccion de la diagonal")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--insecure", action="store_true", help="descargas sin verificar TLS")


def _normal_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("normales")
    g.add_argument("--normals", default="mst", choices=list(ORIENT_METHODS))
    g.add_argument("-k", type=int, default=20, help="vecinos para PCA")
    g.add_argument("--filter", action="store_true", help="filtro estadistico de outliers")
    g.add_argument("--alpha", type=float, default=2.0)


def _recon_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("reconstruccion")
    g.add_argument("--methods", nargs="+", default=None, choices=ALL_METHODS)
    g.add_argument("--res", type=int, default=64, help="celdas en el eje mas largo")
    g.add_argument("--extractor", choices=["mt", "mc"], default="mt")
    g.add_argument("--hoppe-support", type=float, default=3.0)
    g.add_argument("--rbf-centers", type=int, default=700)
    g.add_argument("--rbf-eps", type=float, default=0.01)
    g.add_argument("--poisson-sigma", type=float, default=1.0)
    g.add_argument("--o3d-depth", type=int, default=8)


def _params(a):
    sp = D.SceneParams(model=a.model, n=a.n_points, noise=a.noise, outliers=a.outliers,
                       hole=a.hole, seed=a.seed)
    if getattr(a, "preset", None):
        for k, v in D.PRESETS[a.preset].items():
            setattr(sp, k, v)
    npm = NormalParams(method=a.normals, k=a.k, filter=a.filter, alpha=a.alpha)
    rp = M.ReconParams(res=a.res, extractor=a.extractor, hoppe_support=a.hoppe_support,
                       rbf_centers=a.rbf_centers, rbf_eps=a.rbf_eps,
                       poisson_sigma=a.poisson_sigma, o3d_depth=a.o3d_depth)
    return sp, npm, rp


def _methods(a):
    ms = a.methods or ALL_METHODS
    if "Open3D" in ms and not M.open3d_available():
        if a.methods:
            print("[aviso] open3d no esta instalado: se omite Open3D")
        ms = [m for m in ms if m != "Open3D"]
    return ms


# --------------------------------------------------------------------------- #
def cmd_demo(a) -> int:
    from . import viz
    sp, npm, rp = _params(a)
    app = viz.App(sp, npm, rp, methods=_methods(a))
    app.compute_scene()
    app.compute_normals()
    print(format_header())
    app.compute_methods()
    if a.no_view:
        return 0
    if a.shot:
        files = viz.screenshots(app, a.shot)
        print("capturas:\n  " + "\n  ".join(files))
        return 0
    viz.show(app)
    return 0


def cmd_compare(a) -> int:
    presets = a.presets or list(D.PRESETS)
    _, npm, rp = _params(a)
    ms = _methods(a)
    rows = []
    for name in presets:
        sp = D.SceneParams(model=a.model, seed=a.seed, **D.PRESETS[name])
        sc = D.make_scene(sp)
        nparams = NormalParams(**{**npm.__dict__, "filter": npm.filter or name == "outliers"})
        nr = run_normals(sc, nparams)
        s = nr.summary()
        print(f"\n== {name}: {len(sc.P)} puntos, normales {s['median_deg']:.2f} deg "
              f"(orientadas {100 * s['oriented_ok']:.1f}%)")
        print("  " + format_header())
        res = run_all(sc, nr, rp, methods=ms, verbose=True)
        for m, r in res.items():
            rows.append(dict(regimen=name, metodo=m, **{k: v for k, v in r.summary.items()}))
    print("\nerr = mediana / p95 de la distancia vertice -> superficie real (milesimas de diag)")
    print("compl p95 = distancia superficie real -> malla (detecta huecos sin rellenar)")
    print("cobert. = fraccion de la superficie real a menos de 2% diag de la malla")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("CSV:", a.csv)
    return 0


def cmd_normals(a) -> int:
    """Error angular vs k para varios niveles de ruido (bloque 1.3) y orientacion."""
    noises = a.noise_levels
    ks = a.ks
    print(f"modelo {a.model}, {a.n_points} puntos. Error angular mediano (grados) por k:")
    print("sigma/diag " + "".join(f"{k:>7d}" for k in ks) + "    k*")
    for s in noises:
        sc = D.make_scene(D.SceneParams(model=a.model, n=a.n_points, noise=s, seed=a.seed))
        errs = []
        for k in ks:
            N, _, _ = pca_normals(sc.P, k)
            e = np.degrees(np.arccos(np.clip(np.abs((N * sc.N_true).sum(1)), 0, 1)))
            errs.append(np.median(e))
        kstar = ks[int(np.argmin(errs))]
        print(f"{s:10.4f} " + "".join(f"{e:7.2f}" for e in errs) + f"   {kstar:4d}")
    print("\nOrientacion (k=20, sin ruido):")
    sc = D.make_scene(D.SceneParams(model=a.model, n=a.n_points, noise=0.0, seed=a.seed))
    for key, lab in ORIENT_METHODS.items():
        t0 = time.perf_counter()
        nr = estimate(sc.P, sc.N_true, sc.is_outlier, NormalParams(method=key, k=20))
        print(f"  {lab:40s} bien orientadas {100 * nr.summary()['oriented_ok']:6.1f}%"
              f"   ({1e3 * (time.perf_counter() - t0):.0f} ms)")
    return 0


def cmd_models(a) -> int:
    for name, (fn, desc) in D.MODELS.items():
        print(f"  {name:10s} {desc}")
    if a.fetch:
        names = [m for m in D.MODELS if D.MODELS[m][0]] if a.fetch == "all" else [a.fetch]
        for m in names:
            print("  ->", D.fetch_model(m))
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="run_recon.py",
                                description="Demo de reconstruccion de superficies (SGP, clase 7)")
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("demo", help="reconstruye y abre el visor de Polyscope")
    _scene_args(d), _normal_args(d), _recon_args(d)
    d.add_argument("--no-view", action="store_true", help="solo consola")
    d.add_argument("--shot", default=None, metavar="PREFIJO",
                   help="sin pantalla: guarda PNGs PREFIJO_*.png y termina")
    d.set_defaults(func=cmd_demo)

    c = sub.add_parser("compare", help="tabla de metodos x regimenes (ruido, hueco, ...)")
    _scene_args(c), _normal_args(c), _recon_args(c)
    c.add_argument("--presets", nargs="+", choices=list(D.PRESETS), default=None)
    c.add_argument("--csv", default=None)
    c.set_defaults(func=cmd_compare)

    nm = sub.add_parser("normals", help="error de normales vs k y ruido; orientacion")
    _scene_args(nm)
    nm.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.001, 0.002, 0.004, 0.008])
    nm.add_argument("--ks", nargs="+", type=int, default=[6, 10, 15, 20, 30, 45, 60, 90, 130])
    nm.set_defaults(func=cmd_normals)

    mo = sub.add_parser("models", help="lista / descarga los modelos")
    mo.add_argument("--fetch", default=None, help="nombre de un modelo o 'all'")
    mo.add_argument("--insecure", action="store_true")
    mo.set_defaults(func=cmd_models)

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("demo", "compare", "normals", "models", "-h", "--help"):
        argv = ["demo"] + argv          # sin subcomando = demo
    a = p.parse_args(argv)
    if getattr(a, "insecure", False):
        D.INSECURE_DOWNLOADS = True
    return a.func(a)
