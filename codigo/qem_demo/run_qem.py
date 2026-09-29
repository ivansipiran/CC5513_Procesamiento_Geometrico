#!/usr/bin/env python3
"""
Demo de simplificación de mallas con cuádricas de error (Garland & Heckbert 1997).

    python run_qem.py                       # = demo: visor Polyscope con la vaca
    python run_qem.py demo --model fandisk --target 800
    python run_qem.py compare --model cow --plot curvas.png
    python run_qem.py explain               # recorrido numérico paso a paso (consola)
    python run_qem.py models --download     # baja los modelos reales

Opciones del algoritmo (valen para demo y compare):
    --placement optimal|svd|subset|midpoint   cómo elegir v̄
    --weighting area|none                     ponderar planos por área
    --boundary-weight W                       planos ⟂ a los bordes (0 = sin)
    --no-topology / --no-flip-check           desactivar chequeos
    --pair-threshold t                        pares no conectados a distancia < t
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _stream in (sys.stdout, sys.stderr):      # consola de Windows con página de códigos vieja
    try:
        _stream.reconfigure(errors="replace")
    except Exception:                          # noqa: BLE001
        pass

from qem_demo import data, methods  # noqa: E402


def add_algo_args(p):
    p.add_argument("--model", default="cow",
                   help=f"{', '.join(data.ALL_MODELS)} o un archivo .obj/.off/.ply")
    p.add_argument("--placement", default="optimal", choices=["optimal", "svd", "subset", "midpoint"])
    p.add_argument("--weighting", default="area", choices=["area", "none"])
    p.add_argument("--boundary-weight", type=float, default=1000.0)
    p.add_argument("--no-topology", action="store_true", help="no chequear la condición de enlace")
    p.add_argument("--no-flip-check", action="store_true", help="no rechazar inversiones de normales")
    p.add_argument("--pair-threshold", type=float, default=0.0,
                   help="t del paper (fracción de la diagonal); 0 = solo aristas")
    p.add_argument("--insecure", action="store_true", help="descargar sin verificar certificados")


def base_params(a):
    return dict(placement=a.placement, weighting=a.weighting, boundary_weight=a.boundary_weight,
                preserve_topology=not a.no_topology, check_flips=not a.no_flip_check,
                pair_threshold=a.pair_threshold)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"demo", "compare", "explain", "models"}
    if not argv or (argv[0] not in cmds and argv[0] not in ("-h", "--help")):
        argv = ["demo"] + argv

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    d = sub.add_parser("demo", help="visor Polyscope interactivo")
    add_algo_args(d)
    d.add_argument("--method", default="qem",
                   choices=["qem", "length", "clustering", "clustering-qem", "open3d"])
    d.add_argument("--target", type=int, default=None, help="caras objetivo (por defecto 10%%)")
    d.add_argument("--shot", default=None, help="guardar capturas PREFIJO_*.png sin abrir ventana")
    d.add_argument("--no-view", action="store_true", help="calcular e informar sin abrir el visor")

    c = sub.add_parser("compare", help="tabla de error por método y objetivo")
    add_algo_args(c)
    c.add_argument("--targets", type=int, nargs="+", default=None)
    c.add_argument("--methods", default=None,
                   help="lista separada por comas: " + ",".join(methods.METHODS))
    c.add_argument("--plot", default=None, help="PNG con error vs caras")
    c.add_argument("--csv", default=None)
    c.add_argument("--save-obj", default=None, help="carpeta donde guardar las mallas")

    e = sub.add_parser("explain", help="recorrido numérico del algoritmo en una malla chica")
    e.add_argument("--weighting", default="area", choices=["area", "none"])
    e.add_argument("--collapses", type=int, default=12)

    m = sub.add_parser("models", help="lista (y descarga) los modelos")
    m.add_argument("--download", action="store_true")
    m.add_argument("--insecure", action="store_true")

    a = ap.parse_args(argv)
    data.INSECURE = getattr(a, "insecure", False)

    if a.cmd == "explain":
        from qem_demo import explain
        explain.run(a.weighting, a.collapses)
    elif a.cmd == "models":
        for name, (fn, desc) in data.REAL_MODELS.items():
            here = os.path.exists(os.path.join(data.DATA_DIR, fn))
            print(f"  {name:10s} {desc}  [{'descargado' if here else 'se baja al usarlo'}]")
            if a.download and not here:
                data.model_path(name)
        for name, desc in data.SYNTHETIC_MODELS.items():
            print(f"  {name:10s} {desc}  [sintético]")
    elif a.cmd == "compare":
        from qem_demo import compare
        compare.run(a, base_params(a))
    else:
        if a.no_view and not a.shot:
            return summary_only(a)
        from qem_demo import psview
        psview.run(a)


def summary_only(a):
    """--no-view: corre el método elegido y muestra el resultado en consola."""
    from qem_demo import metrics, topology
    V, F = data.load_model(a.model)
    target = a.target or (994 if a.model == "cow" else max(len(F) // 10, 4))
    name = a.method
    if name == "qem":
        name = {"optimal": "qem", "svd": "qem-svd", "subset": "qem-subset", "midpoint": "qem-midpoint"}[a.placement]
    (Vs, Fs, t), = methods.run_at_targets(name, V, F, [target], base_params(a))
    e = metrics.geometric_error((V, F), Vs, Fs)
    s0, s = topology.summary(V, F), topology.summary(Vs, Fs)
    print(f"{a.model}: {s0['F']} → {s['F']} caras con {methods.label(name)} en {t:.2f} s")
    print(f"  Hausdorff {e['hausdorff']:.2f} ‰  medio {e['mean']:.3f} ‰  RMS {e['rms']:.3f} ‰  "
          f"calidad {e['quality']:.2f}  astillas {e['slivers']:.1f}%")
    print(f"  χ {s0['chi']} → {s['chi']}   género {s0['genus']} → {s['genus']}   "
          f"bordes {s0['boundary_loops']} → {s['boundary_loops']}")


if __name__ == "__main__":
    main()
