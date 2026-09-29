"""
Experimento 4 -- 4PCS vs Super4PCS: de donde sale la ganancia.

Se mide, para una misma base, el costo de
  (a) extraer los pares a distancia d  (O(n^2) vs cascara rasterizada)
  (b) extraer los conjuntos congruentes (kd-tree vs join por grilla)
en funcion del numero de puntos.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/experiments/", 1)[0])
from gr import data, fourpcs                                   # noqa: E402


def main():
    mesh = data.load_mesh("armadillo")
    pair = data.make_pair(mesh, "arm", view_a=(70, 0), view_b=(70, 45),
                          voxel=0.003, seed=1, keep_meshes=False)
    Q_all = np.asarray(pair.target.points)
    print("nube completa:", len(Q_all))
    rng = np.random.default_rng(0)
    delta = 0.005
    rows = []
    for n in (500, 1000, 2000, 4000):
        if n > len(Q_all):
            break
        Q = Q_all[rng.choice(len(Q_all), n, replace=False)]
        d1, d2 = 0.35, 0.28
        r1, r2 = 0.4, 0.6

        sg = fourpcs.ShellGrid(Q)
        sg.pairs(d1, delta)                     # calentar cache de offsets
        t0 = time.time()
        p1 = sg.pairs(d1, delta)
        p2 = sg.pairs(d2, delta)
        t_shell = time.time() - t0

        t0 = time.time()
        q1 = fourpcs.pairs_naive(Q, d1, delta)
        q2 = fourpcs.pairs_naive(Q, d2, delta)
        t_naive = time.time() - t0

        t0 = time.time()
        s_grid = fourpcs.find_congruent(Q, p1, r1, p2, r2, delta,
                                        backend="grid", max_sets=2*10**6)
        t_cong_grid = time.time() - t0
        t0 = time.time()
        s_kd = fourpcs.find_congruent(Q, q1, r1, q2, r2, delta,
                                      backend="kdtree", max_sets=2*10**6)
        t_cong_kd = time.time() - t0

        row = dict(n=n, pairs=len(p1) + len(p2),
                   t_pairs_super=t_shell, t_pairs_4pcs=t_naive,
                   t_cong_super=t_cong_grid, t_cong_4pcs=t_cong_kd,
                   n_cong_super=len(s_grid), n_cong_4pcs=len(s_kd))
        rows.append(row)
        print(f"n={n:6d} pares={row['pairs']:8d}  "
              f"pares: super={t_shell:7.3f}s 4pcs={t_naive:7.3f}s "
              f"({t_naive/max(t_shell,1e-9):5.1f}x)  |  "
              f"congruentes: super={t_cong_grid:7.3f}s 4pcs={t_cong_kd:7.3f}s "
              f"({t_cong_kd/max(t_cong_grid,1e-9):4.1f}x)  "
              f"[{len(s_grid)} / {len(s_kd)}]", flush=True)

    with open("experiments/results_scaling.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
