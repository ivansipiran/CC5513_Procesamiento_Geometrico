"""
`python run_qem.py explain` — recorrido numérico del algoritmo sobre una malla chica.

Malla: una pirámide sobre un piso plano (grilla 9 × 9). Tiene los tres casos
que importan: vértices en zona plana, en un pliegue (base de la pirámide) y en
esquinas (cúspide y esquinas de la base).
"""
import numpy as np

from . import quadrics as qd
from .data import grid_faces
from .decimate import PairCollapse

np.set_printoptions(precision=4, suppress=True, linewidth=110)


def pyramid(n=9, r=0.5, h=0.5):
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Z = h * np.maximum(0.0, 1.0 - np.maximum(np.abs(X), np.abs(Y)) / r)
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    return V, grid_faces(n, n)


def title(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


def vid(V, x, y):
    return int(np.argmin(np.abs(V[:, 0] - x) + np.abs(V[:, 1] - y)))


def mat(M, indent="    "):
    return "\n".join(indent + line for line in np.array2string(M).splitlines())


def run(weighting="area", collapses=12):
    V, F = pyramid()
    print(f"Malla: pirámide sobre un piso, {len(V)} vértices, {len(F)} triángulos.")
    print(f"Ponderación de los planos: {weighting}.")

    # ------------------------------------------------------------------ 1
    title("1. Una cara  →  un plano p = (a, b, c, d)  →  K_p = p pᵀ")
    apex = vid(V, 0, 0)
    f = next(i for i, t in enumerate(F) if apex in t)
    P, area = qd.face_planes(V, F[[f]])
    p = P[0]
    print(f"Cara {f} con vértices {F[f].tolist()}:")
    for v in F[f]:
        print(f"    v{v} = {V[v]}")
    print(f"Normal unitaria n = {p[:3]},  d = −n·v₀ = {p[3]:.4f},  área = {area[0]:.4f}")
    K = qd.fundamental_quadric(p)
    print("K_p = p pᵀ =\n" + mat(K))
    x = np.array([0.1, 0.2, 0.3])
    direct = (p[:3] @ x + p[3]) ** 2
    viaK = np.append(x, 1) @ K @ np.append(x, 1)
    print(f"Chequeo con x = {x}:  (n·x + d)² = {direct:.6f}   x̂ᵀ K_p x̂ = {viaK:.6f}   ✓")

    # ------------------------------------------------------------------ 2
    title("2. Cuádrica de un vértice:  Q_v = Σ (cuádricas de sus caras)")
    Q = qd.vertex_quadrics(V, F, weighting, boundary_weight=0.0)
    W = qd.face_weight(V, F, weighting)          # escala de referencia para el rango
    examples = [("zona plana", vid(V, -0.75, 0.75)), ("pliegue (base)", vid(V, 0.5, 0.0)),
                ("esquina de la base", vid(V, 0.5, 0.5)), ("cúspide", apex)]
    for name, v in examples:
        A, b, c = qd.split(Q[v])
        lam, U = np.linalg.eigh(A)
        nf = int((F == v).any(1).sum())
        rank = int(qd.effective_rank(Q[v], ref=W[v]))
        print(f"\nv{v} ({name}), posición {V[v]}, {nf} caras")
        print("  A (3×3) =\n" + mat(A, "      "))
        print(f"  autovalores de A = {lam}   →  rango efectivo {rank} = {qd.RANK_NAMES[rank]}"
              f"  (umbral 5% del área de sus caras, {0.05 * W[v]:.4f})")
        print(f"  Δ(v) en el propio vértice = {qd.quadric_error(Q[v], V[v]):.2e}"
              "  (está sobre todos sus planos)")
    # vértice del borde, sin y con los planos perpendiculares al borde
    Qb = qd.vertex_quadrics(V, F, weighting, boundary_weight=1000.0)
    v = vid(V, -1.0, 0.25)
    for lab, QQ in (("sin", Q), ("con", Qb)):
        lam = np.linalg.eigvalsh(QQ[v][:3, :3])
        rank = int(qd.effective_rank(QQ[v], ref=W[v]))
        print(f"\nv{v} (borde del piso) {lab} restricción de borde: autovalores de A = {lam}"
              f"  →  rango {rank} = {qd.RANK_NAMES[rank]}")
    print("  El plano perpendicular al borde convierte al borde en un 'pliegue' artificial:"
          "\n  el vértice ya solo puede deslizarse A LO LARGO del borde.")
    print("\nLectura: rango 1 = el vértice se puede mover en un PLANO sin costo;"
          "\n         rango 2 = solo a lo largo de una RECTA (el pliegue);"
          "\n         rango 3 = cualquier movimiento cuesta (esquina).")

    # ------------------------------------------------------------------ 3
    title("3. Costo de contraer un par (v₁, v₂):  Q̄ = Q₁ + Q₂,  v̄ = argmin Δ")
    pairs = [("dos vértices del piso", vid(V, -0.75, 0.75), vid(V, -0.5, 0.75)),
             ("piso → pliegue", vid(V, 0.75, 0.0), vid(V, 0.5, 0.0)),
             ("a lo largo del pliegue", vid(V, 0.5, 0.0), vid(V, 0.5, 0.25)),
             ("cúspide → ladera", apex, vid(V, 0.25, 0.0))]
    for name, i, j in pairs:
        Qs = Q[i] + Q[j]
        A, b, c = qd.split(Qs)
        lam = np.linalg.eigvalsh(A)
        print(f"\nPar v{i}–v{j} ({name})")
        print(f"  autovalores de A = {lam}")
        rows = []
        for strat in ("optimal", "subset", "midpoint"):
            vb, cost, how = qd.optimal_placement(Qs[None], V[[i]], V[[j]], strat)
            rows.append((strat, vb[0], cost[0], qd.HOW_NAMES[int(how[0])] if strat == "optimal" else ""))
        for strat, vb, cost, how in rows:
            print(f"  {strat:9s} v̄ = {vb}   Δ(v̄) = {cost:.3e}   {how}")
    print("\nEn el piso todos los candidatos cuestan 0 (el piso sigue siendo piso)."
          "\nCuando A es singular el paper no invierte: busca en el segmento o prueba v₁, v₂ y el medio.")

    # ------------------------------------------------------------------ 4
    title("4. El heap: todos los pares ordenados por costo (con restricción de borde)")
    pc = PairCollapse(V, F, weighting=weighting, boundary_weight=1000.0)
    top = sorted(pc.heap)[:8]
    print(f"{len(pc.heap)} pares (= aristas). Los 8 más baratos:")
    for cost, _, i, j, _, _, vb, how in top:
        print(f"  v{i:<3d}– v{j:<3d} costo {cost:.2e}   v̄ = {vb}")
    costs = np.array([e[0] for e in pc.heap])
    print(f"Pares con costo ≈ 0: {(costs < 1e-12).sum()} de {len(costs)}  (interior del piso y de las laderas)")

    # ------------------------------------------------------------------ 5
    title(f"5. Primeros {collapses} colapsos (sacar el mínimo, contraer, actualizar vecinos)")
    for step in range(collapses):
        before = pc.n_faces
        pc.simplify(before - 1)
        h = pc._log
        if len(h["keep"]) <= step:
            break
        i, j, c = h["keep"][step], h["removed"][step], h["cost"][step]
        print(f"  {step + 1:2d}. v{j:<3d} → v{i:<3d}  costo {c:.2e}  v̄ = {h['newpos'][step]}"
              f"  caras {before} → {pc.n_faces}")
    h = pc.simplify(0)
    first = int(np.argmax(h.cost > 1e-10)) if (h.cost > 1e-10).any() else h.steps
    Vz, Fz, _, _ = h.state(first)
    print(f"\nLos primeros {first} colapsos tienen costo 0. Después de ellos quedan {len(Vz)} vértices"
          f"\ny {len(Fz)} caras, y la malla describe EXACTAMENTE la misma forma (error 0): el piso"
          f"\ncuadrado y la pirámide. Los que quedan están en esquinas o pliegues de la malla:")
    print(mat(Vz[np.lexsort((Vz[:, 1], Vz[:, 0]))]))
    print(f"El colapso {first + 1} ya cuesta {h.cost[first]:.2e}: de ahí en adelante se pierde forma.")
    print(f"\nCorriendo hasta el final: {h.steps} colapsos, quedan {h.nfaces[-1]} caras.")
    s = h.stats
    print(f"Rechazados por topología: {s['rejected_topology']}, por inversión de normales:"
          f" {s['rejected_flip']}; entradas viejas del heap descartadas: {s['stale']}.")
    V2, F2 = pc.current_mesh()
    print(f"Malla final: {len(V2)} vértices, {len(F2)} caras. Vértices finales:")
    print(mat(V2))
