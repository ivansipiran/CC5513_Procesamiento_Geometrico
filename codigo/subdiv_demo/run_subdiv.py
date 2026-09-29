#!/usr/bin/env python
"""
Demo de subdivisión de mallas: Catmull-Clark, Loop y Doo-Sabin (CC5513).

    python run_subdiv.py                         visor, cubo, Catmull-Clark nivel 2
    python run_subdiv.py demo --model suzanne --scheme doo-sabin --level 3
    python run_subdiv.py demo --model fandisk --scheme loop --crease 40 --level 2
    python run_subdiv.py demo --model L --side-by-side --level 2
    python run_subdiv.py demo --model cube --stencil 30 --level 1
    python run_subdiv.py demo --model torus --basis 0 --delta 0.2 --level 3
    python run_subdiv.py compare --model prism5 --levels 4
    python run_subdiv.py explain
    python run_subdiv.py models [--download]

Opciones comunes: --shot archivo.png (captura), --no-view (sin ventana), --export malla.obj
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from subdiv_demo.models import ALL_MODELS, SYNTHETIC, REMOTE, load_model, download, save_obj  # noqa: E402
from subdiv_demo.viewer import COLOR_MODES  # noqa: E402

SCHEMES = ["catmull-clark", "loop", "doo-sabin"]
ALIASES = {"cc": "catmull-clark", "catmull": "catmull-clark", "ds": "doo-sabin", "doo": "doo-sabin"}


def scheme_arg(s):
    s = ALIASES.get(s.lower(), s.lower())
    if s not in SCHEMES:
        raise argparse.ArgumentTypeError(f"esquema desconocido: {s}")
    return s


def cmd_demo(a):
    from subdiv_demo.viewer import SubdivViewer, run_viewer
    v = SubdivViewer(model=a.model, scheme=a.scheme, variant=a.variant, level=a.level,
                     crease=a.crease, ds_boundary=a.ds_boundary, color=a.color,
                     insecure=a.insecure, cage=not a.no_cage, morph=a.morph, limit=a.limit,
                     stencil=a.stencil, basis=a.basis, delta=a.delta,
                     side_by_side=a.side_by_side)
    H = v.hierarchy()
    k = v.clamp_level(H)
    print(f"[{a.model}] control: {H.meshes[0].summary()}")
    if H.note:
        print("  " + H.note)
    for i in range(1, k + 1):
        print(f"  nivel {i}: {H.meshes[i].summary()}")
        for n in H.steps[i - 1].notes:
            print("     " + n)
    if a.stencil is not None and k >= 1:
        from subdiv_demo.explain import stencil
        st = H.steps[k - 1]
        i = min(a.stencil, H.meshes[k].nV - 1)
        print(f"  estencil del vértice {i} del nivel {k}: {st.describe_vertex(i)}")
        print("     " + stencil(st.S, i))
    if a.export:
        mesh = H.meshes[k].copy_with(v.positions(H, k))
        save_obj(a.export, mesh)
        print(f"  malla exportada a {a.export}")
    if a.no_view and not a.shot:
        return
    run_viewer(v, shot=a.shot, no_view=a.no_view, ui=a.ui)


def cmd_compare(a):
    from subdiv_demo.metrics import run_levels, format_rows
    m = load_model(a.model, insecure=a.insecure)
    if a.crease is not None:
        n = m.mark_creases_by_angle(a.crease)
        print(f"{n} aristas marcadas como pliegue (diedro > {a.crease}°)")
    print(f"[{a.model}] control: {m.summary()}\n")
    summary = []
    for sch in a.schemes:
        variants = a.variants or [None]
        for var in variants:
            try:
                _, rows, ref = run_levels(m, sch, a.levels, variant=var, ds_boundary=a.ds_boundary,
                                          with_dist=not a.no_dist)
            except ValueError as e:
                print(f"{sch} {var}: {e}")
                continue
            label = sch + (f" ({var})" if var else "")
            print(f"== {label}" + ("" if a.no_dist else f"   (límite aprox. por un nivel con {ref.nF} caras)"))
            print(format_rows(rows, sch))
            print()
            summary.append((label, rows))
    if len(summary) > 1:
        k = a.levels
        print(f"== resumen en el nivel {k}")
        print(f"{'esquema':28s} {'F':>8} {'diedro max':>10} {'vol %':>7} {'dist.lim max (o/oo)':>20}")
        for label, rows in summary:
            r = rows[min(k, len(rows) - 1)]
            vol = f"{r['vol']:.1f}" if r["vol"] is not None else "-"
            d = f"{rows[1]['dist'][0]:.2f} -> {r['dist'][0]:.2f}" if r["dist"] else "-"
            print(f"{label:28s} {r['F']:>8} {r['dih_max']:>9.1f}° {vol:>7} {d:>20}")
        print("(dist.lim: nivel 1 -> nivel k)")


def cmd_explain(a):
    from subdiv_demo.explain import main
    main()


def cmd_models(a):
    print("Sintéticos:")
    for k, (_, desc) in SYNTHETIC.items():
        print(f"  {k:10s} {desc}")
    print("Descargables (alecjacobson/common-3d-test-models):")
    for k, (fn, desc) in REMOTE.items():
        print(f"  {k:10s} {desc}")
    if a.download:
        for k in REMOTE:
            try:
                print("  ok:", download(k, insecure=a.insecure))
            except Exception as e:
                print("  error:", k, e)
    print("También puede pasar la ruta de un .obj propio con --model archivo.obj")


def main(argv=None):
    p = argparse.ArgumentParser(description="Demo de subdivisión: Catmull-Clark, Loop, Doo-Sabin")
    sub = p.add_subparsers(dest="cmd")

    def common(q):
        q.add_argument("--model", default="cube", help=f"uno de {ALL_MODELS} o un .obj")
        q.add_argument("--crease", type=float, default=None,
                       help="marca como pliegue las aristas con diedro > ANGULO (CC y Loop)")
        q.add_argument("--ds-boundary", choices=["chaikin", "free"], default="chaikin")
        q.add_argument("--insecure", action="store_true", help="descargar sin verificar SSL")

    d = sub.add_parser("demo", help="visor Polyscope")
    common(d)
    d.add_argument("--scheme", type=scheme_arg, default="catmull-clark")
    d.add_argument("--variant", default=None,
                   help="CC: standard|linear  Loop: loop|warren|linear  DS: doo-sabin|simple")
    d.add_argument("--level", type=int, default=2)
    d.add_argument("--color", default="tipo de vertice", choices=COLOR_MODES)
    d.add_argument("--no-cage", action="store_true")
    d.add_argument("--morph", type=float, default=1.0, help="0 = solo dividir, 1 = promediar")
    d.add_argument("--limit", action="store_true", help="proyectar vértices a su posición límite")
    d.add_argument("--stencil", type=int, default=None, help="mostrar la máscara del vértice i")
    d.add_argument("--basis", type=int, default=None, help="función base del vértice de control j")
    d.add_argument("--delta", type=float, default=0.0, help="desplazar el vértice j en su normal")
    d.add_argument("--side-by-side", action="store_true", help="los tres esquemas juntos")
    d.add_argument("--shot", default=None, help="guardar captura PNG")
    d.add_argument("--no-view", action="store_true", help="no abrir ventana")
    d.add_argument("--ui", action="store_true", help="incluir el panel en la captura")
    d.add_argument("--export", default=None, help="guardar la malla subdividida (.obj)")
    d.set_defaults(fn=cmd_demo)

    c = sub.add_parser("compare", help="tabla de métricas por nivel y esquema")
    common(c)
    c.add_argument("--levels", type=int, default=4)
    c.add_argument("--schemes", type=scheme_arg, nargs="+", default=SCHEMES)
    c.add_argument("--variants", nargs="+", default=None)
    c.add_argument("--no-dist", action="store_true", help="no calcular distancia al límite")
    c.set_defaults(fn=cmd_compare)

    e = sub.add_parser("explain", help="los tres esquemas con números en consola")
    e.set_defaults(fn=cmd_explain)

    mo = sub.add_parser("models", help="listar / descargar modelos")
    mo.add_argument("--download", action="store_true")
    mo.add_argument("--insecure", action="store_true")
    mo.set_defaults(fn=cmd_models)

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0].startswith("-"):
        argv = ["demo"] + list(argv)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
