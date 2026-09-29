"""
`python run_subdiv.py explain` : los tres esquemas con números, paso a paso, en consola.
"""
from fractions import Fraction

import numpy as np

from .models import load_model
from .schemes import catmull_clark, loop, doo_sabin, loop_beta, doo_sabin_weights, simple_weights
from .limit import loop_local_matrix, sorted_eigs, limit_positions
from .metrics import run_levels


def fr(x, maxden=512):
    f = Fraction(float(x)).limit_denominator(maxden)
    if abs(float(f) - x) > 1e-9:
        return f"{x:.4f}"
    return str(f)


def vec(p):
    return "(" + ", ".join(f"{c:+.3f}" for c in p) + ")"


def stencil(S, i, names=None):
    row = S.getrow(i)
    parts = []
    for j, w in sorted(zip(row.indices, row.data), key=lambda t: (-t[1], t[0])):
        parts.append(f"{fr(w)}·P{j}")
    return " + ".join(parts)


def title(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def explain_cc():
    title("1. CATMULL-CLARK sobre el cubo (6 quads, 8 vértices de valencia 3)")
    m = load_model("cube")
    st = catmull_clark(m)
    V = m.V
    print(f"control : {m.summary()}")
    print(f"nivel 1 : {st.fine.summary()}")
    print(f"  V' = V + E + F = {m.nV} + {m.nE} + {m.nF} = {st.fine.nV}   "
          f"F' = suma de grados = {m.nH}   (todas quads)")
    f = 0
    fv = m.fidx[m.fptr[f]:m.fptr[f + 1]]
    iF = m.nV + m.nE + f
    print(f"\n  a) punto de cara de la cara {f} = promedio de {fv.tolist()}")
    print(f"     F{f} = {stencil(st.S, iF)}  =  {vec(st.fine.V[iF])}")
    e = 0
    a, b = m.edges[e]
    f1, f2 = m.edge_faces[e]
    iE = m.nV + e
    Fc = m.face_centroids()
    print(f"\n  b) punto de arista de la arista {e} = ({a},{b}), caras vecinas {f1} y {f2}")
    print(f"     E = (P{a} + P{b} + F{f1} + F{f2}) / 4")
    print(f"       = ({vec(V[a])} + {vec(V[b])} + {vec(Fc[f1])} + {vec(Fc[f2])}) / 4")
    print(f"       = {vec(st.fine.V[iE])}")
    print(f"     como fila de S: E = {stencil(st.S, iE)}")
    v = 0
    n = int(m.valence[v])
    print(f"\n  c) punto de vértice del vértice {v} (valencia n = {n})")
    faces_v = m.face_of[m.fidx == v]
    nb = np.concatenate([m.edges[m.edges[:, 0] == v, 1], m.edges[m.edges[:, 1] == v, 0]])
    Q = Fc[faces_v].mean(0)
    R = (0.5 * (V[v] + V[nb])).mean(0)
    print(f"     Q = prom. puntos de cara {faces_v.tolist()} = {vec(Q)}")
    print(f"     R = prom. puntos medios de aristas hacia {nb.tolist()} = {vec(R)}")
    print(f"     P' = Q/n + 2R/n + (n-3)P/n = Q/{n} + 2R/{n} + {n-3}·P/{n} = {vec(st.fine.V[v])}")
    print(f"     como fila de S: P' = {stencil(st.S, v)}")
    print("     (con n = 4 y quads esta máscara es la de la B-spline bicúbica: "
          "36/64 · P, 6/64 aristas, 1/64 diagonales)")

    t = load_model("torus")
    s2 = catmull_clark(t)
    print(f"  toro regular (n=4): P' = {stencil(s2.S, 0)}")

    L = limit_positions(m, "catmull-clark")
    print(f"\n  d) posición límite del vértice 0: {vec(L[0])}  (el control estaba en {vec(V[0])})")
    print("     P_inf = (n² P + 4·Σ vecinos + Σ diagonales) / (n(n+5))  sobre la malla de quads")


def explain_loop():
    title("2. LOOP sobre el octaedro (8 triángulos, valencia 4)")
    m = load_model("octa")
    st = loop(m)
    print(f"control : {m.summary()}")
    print(f"nivel 1 : {st.fine.summary()}")
    print(f"  V' = V + E = {m.nV} + {m.nE} = {st.fine.nV}    F' = 4F = {st.fine.nF}")
    e = 0
    a, b = m.edges[e]
    h = m.edge_he[e]
    c, d = m.fidx[m.prv[h]], m.fidx[m.prv[m.twin[h]]]
    print(f"\n  a) vértice impar de la arista ({a},{b}); opuestos {c} y {d}")
    print(f"     E = 3/8 (P{a} + P{b}) + 1/8 (P{c} + P{d})")
    print(f"     fila de S: {stencil(st.S, m.nV + e)}  =  {vec(st.fine.V[m.nV + e])}")
    print(f"\n  b) vértice par {0} (valencia {int(m.valence[0])})")
    print(f"     fila de S: {stencil(st.S, 0)}  =  {vec(st.fine.V[0])}")

    print("\n  c) beta según la valencia n (peso de cada vecino; el centro recibe 1 - n·beta)")
    print(f"     {'n':>3} {'beta Loop':>11} {'beta Warren':>12} {'l1 = subdominante':>18} {'l3':>8}")
    for n in range(3, 11):
        Lm = loop_local_matrix(n, "loop")
        w, _ = sorted_eigs(Lm)
        bl, bw = float(loop_beta(n, "loop")), float(loop_beta(n, "warren"))
        print(f"     {n:>3} {bl:>11.5f} {bw:>12.5f} {w[1]:>18.5f} {w[3]:>8.4f}")
    print("     Con beta de Loop, l1 = l2 = 3/8 + 1/4 cos(2 pi/n) (autovalor doble,")
    print("     asociado al plano tangente) y l1 > |l3|: por eso la superficie es C1.")
    print("     Para n = 6 ambos dan beta = 1/16 (malla regular = box-spline cuártica, C2).")

    n = 5
    Lm = loop_local_matrix(n, "loop")
    w, U = sorted_eigs(Lm)
    u0 = U[:, 0] / U[:, 0].sum()
    beta = float(loop_beta(n))
    chi = 1 / (3 / (8 * beta) + n)
    print(f"\n  d) máscara de límite = vector propio IZQUIERDO de l0 = 1 (valencia {n}):")
    print(f"     u0 = centro {u0[0]:.5f}, cada vecino {u0[1]:.5f}")
    print(f"     fórmula cerrada: 1 - n·chi = {1 - n * chi:.5f}, chi = {chi:.5f}")
    print(f"     matriz local 1-anillo (n={n}), valores propios: " +
          ", ".join(f"{x:.4f}" for x in w))


def explain_ds():
    title("3. DOO-SABIN sobre el cubo (esquema dual: un vértice nuevo por ESQUINA)")
    m = load_model("cube")
    st = doo_sabin(m)
    print(f"control : {m.summary()}")
    print(f"nivel 1 : {st.fine.summary()}")
    fk = np.bincount(st.fkind, minlength=3)
    print(f"  V' = nº de esquinas = {m.nH}")
    print(f"  F' = F + E + V = {m.nF} + {m.nE} + {m.nV} = {st.fine.nF}   "
          f"(caras-F: {fk[0]}, caras-E: {fk[1]}, caras-V: {fk[2]})")
    print("  Todos los vértices nuevos tienen valencia 4; lo extraordinario pasa a las")
    print("  CARAS: los 8 vértices de valencia 3 se convierten en 8 caras-V triangulares.")
    print("\n  a) pesos de una esquina en una cara de n lados (esquina propia, a 1 paso, a 2...)")
    for n in range(3, 7):
        w = doo_sabin_weights(n)
        s = simple_weights(n)
        print(f"     n={n}  Doo-Sabin: " + ", ".join(fr(x) for x in w[: n // 2 + 1]) +
              "     simple: " + ", ".join(fr(x) for x in s[: n // 2 + 1]))
    print("     En quads ambos dan 9/16, 3/16, 1/16 (B-spline bicuadrática, C1).")
    c = 0
    v, f = int(m.src[c]), int(m.face_of[c])
    print(f"\n  b) esquina {c}: vértice {v} en la cara {f}")
    print(f"     fila de S: {stencil(st.S, c)}  =  {vec(st.fine.V[c])}")
    print(f"     (el vértice original está en {vec(m.V[v])}: la esquina se \"corta\" hacia la cara)")


def explain_counts():
    title("4. Topología: cuánto crece la malla por nivel (malla cerrada)")
    print("  Catmull-Clark : V' = V + E + F     E' = 2E + suma grados   F' = suma grados (= 2E)")
    print("  Loop          : V' = V + E         E' = 2E + 3F            F' = 4F")
    print("  Doo-Sabin     : V' = suma grados   E' = 2E + suma grados   F' = F + E + V")
    print("  En los tres chi = V - E + F se conserva y las caras se multiplican por ~4.")


def explain_convergence():
    title("5. Convergencia: distancia al límite por nivel (‰ de la diagonal, prisma pentagonal)")
    m = load_model("prism5")
    print(f"  {'niv':>3} {'Catmull-Clark':>14} {'Loop':>10} {'Doo-Sabin':>10}")
    res = {}
    for sch in ("catmull-clark", "loop", "doo-sabin"):
        _, rows, _ = run_levels(m, sch, 4, ref_extra=2)
        res[sch] = [r["dist"][0] for r in rows]
    for k in range(4):
        print(f"  {k:>3} {res['catmull-clark'][k]:>14.2f} {res['loop'][k]:>10.2f} "
              f"{res['doo-sabin'][k]:>10.2f}")
    r = [res[s][2] / res[s][3] for s in res]
    print("  cociente nivel2/nivel3: " + ", ".join(f"{x:.1f}" for x in r) +
          "  -> el error baja ~4 veces por nivel (el largo de arista se reduce a la mitad).")


def main():
    np.set_printoptions(precision=4, suppress=True)
    explain_cc()
    explain_loop()
    explain_ds()
    explain_counts()
    explain_convergence()
