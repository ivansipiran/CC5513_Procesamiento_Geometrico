"""Interfaz de linea de comandos del demo de ICP."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import benchmark as bench
from .data import MODELS, model_path
from .icp import ICPParams, icp
from .scenario import build_scenario


def _scenario_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("escena")
    g.add_argument("--model", default="bunny",
                   help="modelo (nombre conocido o ruta a un .obj)")
    g.add_argument("-n", "--n-points", type=int, default=8000,
                   help="puntos de la nube fuente")
    g.add_argument("--n-target", type=int, default=None,
                   help="puntos de la nube objetivo (por defecto = fuente)")
    g.add_argument("--angle", type=float, default=30.0,
                   help="rotacion inicial de desalineacion en grados")
    g.add_argument("--translation", type=float, default=0.15,
                   help="traslacion inicial (unidades de diagonal del modelo)")
    g.add_argument("--noise", type=float, default=0.002, help="sigma del ruido")
    g.add_argument("--overlap", type=float, default=1.0,
                   help="fraccion de la fuente conservada (solapamiento parcial)")
    g.add_argument("--estimate-normals", action="store_true",
                   help="estimar normales del objetivo por PCA en vez de usar las exactas")
    g.add_argument("--normal-k", type=int, default=20,
                   help="vecinos usados en la estimacion de normales")
    g.add_argument("--seed", type=int, default=0)


def _icp_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("icp")
    g.add_argument("--variant", default="point2point",
                   choices=["point2point", "point2plane"],
                   help="metrica de minimizacion")
    g.add_argument("--nn", default="kdtree",
                   choices=["kdtree", "brute", "brute-loop"],
                   help="backend de vecino mas cercano")
    g.add_argument("--max-iter", type=int, default=50)
    g.add_argument("--sample", type=int, default=0,
                   help="submuestreo de la fuente por iteracion (0 = usar todos)")
    g.add_argument("--reject", type=float, default=3.0,
                   help="umbral de rechazo = factor x mediana de distancias (0 = off)")
    g.add_argument("--trim", type=float, default=1.0,
                   help="fraccion de pares conservados por percentil (trimmed ICP)")
    g.add_argument("--normal-reject", type=float, default=None,
                   help="rechazar pares con normales que difieran mas de X grados")
    g.add_argument("--verbose", action="store_true")


def make_scenario(a) -> "object":
    return build_scenario(model=a.model, n_points=a.n_points, n_target=a.n_target,
                          angle_deg=a.angle, translation=a.translation,
                          noise=a.noise, overlap=a.overlap,
                          estimate_target_normals=a.estimate_normals,
                          normal_k=a.normal_k, seed=a.seed)


def make_params(a) -> ICPParams:
    return ICPParams(variant=a.variant, nn_backend=a.nn, max_iter=a.max_iter,
                     sample_size=a.sample or None,
                     auto_reject_factor=a.reject or None,
                     trim_ratio=a.trim, normal_reject_deg=a.normal_reject,
                     seed=a.seed)


# --------------------------------------------------------------------------- #
def cmd_demo(a) -> None:
    sc = make_scenario(a)
    print(sc.description)
    res = icp(sc.source, sc.target, make_params(a),
              target_normals=sc.target_normals, source_normals=sc.source_normals,
              T_gt=sc.T_gt, verbose=a.verbose)
    print(res.summary())
    if a.no_view:
        return
    from .viz import show
    show(sc, res, max_corr_lines=a.corr_lines)


def cmd_compare(a) -> None:
    sc = make_scenario(a)
    backends = a.backends.split(",")
    variants = a.variants.split(",")
    results = bench.bench_variants(sc, backends=backends, variants=variants,
                                   max_iter=a.max_iter,
                                   sample_size=a.sample or None, verbose=a.verbose)
    if a.plot:
        bench.plot_convergence(results, a.plot)
    if a.no_view:
        return
    from .viz import show_comparison
    show_comparison(sc, results)


def cmd_bench_nn(a) -> None:
    sc = build_scenario(model=a.model, n_points=max(a.sizes), n_target=max(a.sizes),
                        angle_deg=a.angle, seed=a.seed)
    data = bench.bench_neighbors(sc, sizes=a.sizes,
                                 backends=tuple(a.backends.split(",")),
                                 loop_limit=a.loop_limit)
    if a.plot:
        bench.plot_nn_scaling(data, a.plot)


def cmd_models(a) -> None:
    print("Modelos disponibles (se descargan al primer uso):\n")
    for name, (fname, desc) in MODELS.items():
        print(f"  {name:12s} {desc}")
    if a.fetch:
        for name in (a.fetch.split(",") if a.fetch != "all" else MODELS):
            print(model_path(name))


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="icp-demo",
        description="Demo docente del algoritmo ICP con visualizacion en Polyscope.")
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("demo", help="ejecuta ICP y abre el visor con la animacion")
    _scenario_args(d)
    _icp_args(d)
    d.add_argument("--corr-lines", type=int, default=400,
                   help="numero maximo de correspondencias dibujadas")
    d.add_argument("--no-view", action="store_true", help="solo consola, sin Polyscope")
    d.set_defaults(func=cmd_demo)

    c = sub.add_parser("compare", help="compara variantes y backends sobre la misma escena")
    _scenario_args(c)
    c.add_argument("--variants", default="point2point,point2plane")
    c.add_argument("--backends", default="kdtree,brute")
    c.add_argument("--max-iter", type=int, default=60)
    c.add_argument("--sample", type=int, default=0)
    c.add_argument("--plot", default=None, help="ruta del png de convergencia")
    c.add_argument("--verbose", action="store_true")
    c.add_argument("--no-view", action="store_true")
    c.set_defaults(func=cmd_compare)

    b = sub.add_parser("bench-nn", help="mide el costo de la busqueda de vecinos")
    b.add_argument("--model", default="bunny")
    b.add_argument("--sizes", type=int, nargs="+",
                   default=[1000, 2000, 5000, 10000, 20000, 50000])
    b.add_argument("--backends", default="brute-loop,brute,kdtree")
    b.add_argument("--loop-limit", type=int, default=4000,
                   help="tamano maximo para el backend ingenuo (es muy lento)")
    b.add_argument("--angle", type=float, default=30.0)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--plot", default=None, help="ruta del png de escalamiento")
    b.set_defaults(func=cmd_bench_nn)

    m = sub.add_parser("models", help="lista y descarga los modelos disponibles")
    m.add_argument("--fetch", default=None,
                   help="descarga modelos: 'all' o lista separada por comas")
    m.set_defaults(func=cmd_models)

    return p


def main(argv=None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    if not getattr(a, "func", None):
        p.print_help()
        return 1
    np.set_printoptions(precision=5, suppress=True)
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
