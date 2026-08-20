"""Comparativas: costo de la busqueda de vecinos y convergencia de las variantes."""

from __future__ import annotations

import numpy as np

from .icp import ICPParams, ICPResult, icp
from .neighbors import make_backend
from .scenario import Scenario


# --------------------------------------------------------------------------- #
def bench_neighbors(scenario: Scenario,
                    sizes=(1000, 2000, 5000, 10000, 20000),
                    backends=("brute-loop", "brute", "kdtree"),
                    loop_limit: int = 4000,
                    repeats: int = 3) -> dict:
    """Mide una pasada completa de vecino mas cercano (N consultas sobre M puntos).

    Se usa N = M = size para que la comparacion sea directa:
      fuerza bruta -> O(N*M) = O(N^2),  kd-tree -> O(M log M) + O(N log M).
    """
    rng = np.random.default_rng(0)
    out = {b: {"size": [], "build": [], "query": []} for b in backends}
    print(f"\n{'N=M':>8} | " + " | ".join(f"{b:>22}" for b in backends))
    print("-" * (10 + 25 * len(backends)))
    for n in sizes:
        src = scenario.source[rng.permutation(len(scenario.source))[:n]]
        tgt = scenario.target[rng.permutation(len(scenario.target))[:n]]
        row = []
        for b in backends:
            if b == "brute-loop" and n > loop_limit:
                row.append(f"{'(omitido)':>22}")
                continue
            # se repite y se toma el mejor tiempo: la primera pasada incluye
            # calentamiento de asignacion de memoria / hilos de BLAS
            best_b, best_q = np.inf, np.inf
            for _ in range(max(1, repeats)):
                nn = make_backend(b).build(tgt)
                nn.query(src)
                best_b = min(best_b, nn.build_time)
                best_q = min(best_q, nn.query_time)
            out[b]["size"].append(n)
            out[b]["build"].append(best_b)
            out[b]["query"].append(best_q)
            row.append(f"{best_b*1e3:8.2f} +{best_q*1e3:9.2f} ms")
        print(f"{n:>8} | " + " | ".join(row))
    print("(cada celda: tiempo de construccion + tiempo de consulta de N puntos)")

    if "kdtree" in out and "brute" in out and out["kdtree"]["query"]:
        print("\nSpeedup kd-tree vs fuerza bruta vectorizada:")
        for n, qk in zip(out["kdtree"]["size"], out["kdtree"]["query"]):
            if n in out["brute"]["size"]:
                qb = out["brute"]["query"][out["brute"]["size"].index(n)]
                print(f"  N={n:>7}: x{qb/max(qk,1e-12):7.1f}")
    return out


# --------------------------------------------------------------------------- #
def bench_variants(scenario: Scenario,
                   backends=("kdtree", "brute"),
                   variants=("point2point", "point2plane"),
                   max_iter: int = 60,
                   sample_size: int | None = None,
                   verbose: bool = False) -> dict[str, ICPResult]:
    """Ejecuta todas las combinaciones variante x backend sobre el mismo problema."""
    results: dict[str, ICPResult] = {}
    print(f"\n{scenario.description}\n")
    for v in variants:
        for b in backends:
            p = ICPParams(variant=v, nn_backend=b, max_iter=max_iter,
                          sample_size=sample_size)
            r = icp(scenario.source, scenario.target, p,
                    target_normals=scenario.target_normals,
                    source_normals=scenario.source_normals,
                    T_gt=scenario.T_gt, verbose=verbose)
            results[p.label()] = r
            print(r.summary())
    print("\nNota: para una misma metrica el resultado numerico es identico entre "
          "backends;\nlo que cambia es el tiempo. Entre metricas cambia el numero "
          "de iteraciones necesarias.")
    return results


# --------------------------------------------------------------------------- #
def plot_convergence(results: dict[str, ICPResult], path: str = "convergencia.png") -> str:
    """Grafica RMSE y error de pose por iteracion (requiere matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, r in results.items():
        it = np.arange(len(r.history))
        axes[0].semilogy(it, r.rmse_curve(), marker="o", ms=3, label=name)
        rot = [h.rot_err_deg for h in r.history]
        tr = [h.trans_err for h in r.history]
        if all(v is not None for v in rot):
            axes[1].semilogy(it, np.maximum(rot, 1e-8), marker="o", ms=3, label=name)
            axes[2].semilogy(it, np.maximum(tr, 1e-10), marker="o", ms=3, label=name)
    axes[0].set_title("RMSE de correspondencias")
    axes[1].set_title("Error de rotacion (grados)")
    axes[2].set_title("Error de traslacion")
    for ax in axes:
        ax.set_xlabel("iteracion")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plot] guardado en {path}")
    return path


def plot_nn_scaling(data: dict, path: str = "escalamiento_nn.png") -> str:
    """Grafica el costo de la consulta NN en funcion del tamano de la nube."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.4))
    for b, d in data.items():
        if d["size"]:
            ax.loglog(d["size"], np.array(d["query"]) * 1e3, marker="o", label=b)
    ax.set_xlabel("N = M (puntos)")
    ax.set_ylabel("tiempo de consulta [ms]")
    ax.set_title("Costo de una pasada de vecino mas cercano")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plot] guardado en {path}")
    return path
