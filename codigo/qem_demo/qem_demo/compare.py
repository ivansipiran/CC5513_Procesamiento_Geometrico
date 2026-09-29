"""`python run_qem.py compare` — tabla (y curva) de error de cada método."""
import csv
import os

import numpy as np

from . import data, meshio, methods, metrics, topology

COLS = [("method", "método", "{}"), ("F", "caras", "{:d}"), ("hausdorff", "Hausdorff ‰", "{:.2f}"),
        ("mean", "medio ‰", "{:.3f}"), ("rms", "RMS ‰", "{:.3f}"), ("quality", "q", "{:.2f}"),
        ("slivers", "astillas %", "{:.1f}"), ("chi", "χ", "{:d}"), ("genus", "g", "{:.0f}"),
        ("time", "t (s)", "{:.2f}")]


def default_targets(nf, model):
    if model == "cow":
        return [994, 532, 248, 64]
    return sorted({max(4, nf // d) for d in (10, 50, 200)}, reverse=True)


def run(args, base):
    V, F = data.load_model(args.model)
    info = topology.summary(V, F)
    print(f"Modelo {args.model}: V={info['V']} F={info['F']} χ={info['chi']} género={info['genus']:.0f} "
          f"bordes={info['boundary_loops']}")
    targets = args.targets or default_targets(len(F), args.model)
    names = methods.available(args.methods.split(",") if args.methods else methods.DEFAULT_COMPARE)
    od = metrics.SurfaceDistance(V, F)
    rows = []
    for name in names:
        res = methods.run_at_targets(name, V, F, targets, base,
                                     progress=lambda nf: print(f"    {name}: quedan {nf} caras", flush=True))
        for tgt, (Vs, Fs, t) in zip(targets, res):
            e = metrics.geometric_error((V, F), Vs, Fs, orig_dist=od)
            s = topology.summary(Vs, Fs)
            rows.append(dict(method=methods.label(name), key=name, target=tgt, F=len(Fs), time=t,
                             chi=s["chi"], genus=s["genus"], **e))
            if args.save_obj:
                os.makedirs(args.save_obj, exist_ok=True)
                meshio.save_obj(os.path.join(args.save_obj, f"{args.model}_{name}_{len(Fs)}.obj"), Vs, Fs)
        print(f"  {name} listo", flush=True)
    for tgt in targets:
        print(f"\n== objetivo {tgt} caras ==")
        print(methods.fmt_table([r for r in rows if r["target"] == tgt], COLS))
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            keys = ["key", "method", "target", "F", "hausdorff", "mean", "rms", "quality", "slivers",
                    "chi", "genus", "time"]
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print("\n[csv]", args.csv)
    if args.plot:
        plot_curves(V, F, names, base, args.plot, od, args.model)


def plot_curves(V, F, names, base, fn, od, model):
    """Error medio vs número de caras (log-log) para cada método."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nf = len(F)
    targets = np.unique(np.geomspace(max(24, nf // 400), nf // 2, 10).astype(int))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in names:
        res = methods.run_at_targets(name, V, F, list(targets), base)
        xs, mean, haus = [], [], []
        for Vs, Fs, _ in res:
            e = metrics.geometric_error((V, F), Vs, Fs, n=15000, orig_dist=od)
            xs.append(len(Fs))
            mean.append(e["mean"])
            haus.append(e["hausdorff"])
        style = "-" if methods.METHODS[name][0] == "collapse" else "--"
        axes[0].plot(xs, mean, style, marker="o", ms=3, label=methods.label(name))
        axes[1].plot(xs, haus, style, marker="o", ms=3, label=methods.label(name))
        print(f"  curva {name} lista", flush=True)
    for ax, t in zip(axes, ["error medio (‰ de la diagonal)", "Hausdorff (‰ de la diagonal)"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("caras")
        ax.set_title(t)
        ax.grid(True, which="both", alpha=0.3)
        ax.invert_xaxis()
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{model}: error vs número de caras")
    fig.tight_layout()
    fig.savefig(fn, dpi=130)
    print("[plot]", fn)
