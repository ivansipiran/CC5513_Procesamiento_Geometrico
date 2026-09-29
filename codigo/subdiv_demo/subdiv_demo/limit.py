"""
Superficie límite: a qué punto converge cada vértice si se subdivide infinitas veces.

Idea (análisis de la matriz local de subdivisión): alrededor de un vértice, el paso de
subdivisión es una matriz cuadrada L que lleva el 1-anillo (o 2-anillo) del nivel k al del
nivel k+1. Si sus valores propios son  1 = l0 > l1 = l2 > l3 ...  entonces

    P_inf = u0 . P      con u0 el vector propio IZQUIERDO de l0 = 1 (normalizado a suma 1)

Esa es la "máscara de límite". Para Loop y Catmull-Clark tiene forma cerrada:

  Loop, valencia n:     P_inf = (1 - n chi) P + chi * suma(vecinos),  chi = 1/(3/(8 beta) + n)
  CC (malla de quads):  P_inf = (n^2 P + 4 suma(vecinos por arista) + suma(diagonales)) / (n(n+5))
  Curva (pliegue/borde): P_inf = 2/3 P + 1/6 (a + b)          (B-spline cúbica)
  Esquina: P_inf = P

Doo-Sabin es dual (los vértices no sobreviven entre niveles) y no la incluimos aquí.
"""
import numpy as np
from scipy import sparse

from .schemes import classify_vertices, crease_neighbors_matrix, loop_beta, sp, diag
from . import schemes


def loop_limit_matrix(m, variant="loop"):
    nV = m.nV
    smooth, crease, corner = classify_vertices(m)
    n = m.valence.astype(float)
    beta = loop_beta(n, "warren" if variant == "warren" else "loop")
    chi = 1.0 / (3.0 / (8.0 * beta) + n)
    e = m.edges
    r = np.concatenate([e[:, 0], e[:, 1]])
    c = np.concatenate([e[:, 1], e[:, 0]])
    keep = smooth[r]
    return (diag(np.where(smooth, 1 - n * chi, 0.0)) + sp(r[keep], c[keep], chi[r[keep]], (nV, nV))
            + diag(np.where(crease, 2.0 / 3.0, 0.0)) + crease_neighbors_matrix(m, crease, 1.0 / 6.0)
            + diag(corner.astype(float))).tocsr()


def cc_limit_matrix(m):
    """Requiere malla de cuadriláteros (después de un paso de CC siempre lo es)."""
    assert m.is_quad_mesh()
    nV = m.nV
    smooth, crease, corner = classify_vertices(m)
    n = m.valence.astype(float)
    den = n * (n + 5)
    e = m.edges
    r = np.concatenate([e[:, 0], e[:, 1]])
    c = np.concatenate([e[:, 1], e[:, 0]])
    keep = smooth[r]
    diag_v = m.fidx[m.nxt[m.nxt]]            # vértice opuesto a la esquina en su quad
    hk = smooth[m.src]
    return (diag(np.where(smooth, n * n / den, 0.0))
            + sp(r[keep], c[keep], 4.0 / den[r[keep]], (nV, nV))
            + sp(m.src[hk], diag_v[hk], 1.0 / den[m.src[hk]], (nV, nV))
            + diag(np.where(crease, 2.0 / 3.0, 0.0)) + crease_neighbors_matrix(m, crease, 1.0 / 6.0)
            + diag(corner.astype(float))).tocsr()


def limit_positions(m, scheme, variant=None):
    """Posición límite de cada vértice de m (None si el esquema no la tiene)."""
    if scheme == "loop":
        if variant == "linear":
            return None
        return loop_limit_matrix(m, variant) @ m.V
    if scheme == "catmull-clark":
        if variant == "linear":
            return None
        if m.is_quad_mesh():
            return cc_limit_matrix(m) @ m.V
        st = schemes.catmull_clark(m)          # un paso: todo quads; los puntos de vértice
        return (cc_limit_matrix(st.fine) @ st.fine.V)[:m.nV]   # tienen el mismo límite
    return None


# --------------------------------------------------------------------------------
#  Matriz local alrededor de un vértice interior de valencia n (para `explain`)
# --------------------------------------------------------------------------------
def loop_local_matrix(n, variant="loop"):
    """
    1-anillo de Loop: [centro, v_1..v_n]  ->  [centro', impares de las n aristas].
        centro' = (1 - n b) c + b sum v_i
        impar_i = 3/8 (c + v_i) + 1/8 (v_{i-1} + v_{i+1})
    """
    b = float(loop_beta(n, variant))
    L = np.zeros((n + 1, n + 1))
    L[0, 0] = 1 - n * b
    L[0, 1:] = b
    for i in range(n):
        L[1 + i, 0] = 3 / 8
        L[1 + i, 1 + i] = 3 / 8
        L[1 + i, 1 + (i - 1) % n] = 1 / 8
        L[1 + i, 1 + (i + 1) % n] = 1 / 8
    return L


def sorted_eigs(L):
    w, U = np.linalg.eig(L.T)                 # vectores propios izquierdos de L
    order = np.argsort(-np.abs(w))
    return w[order].real, U[:, order].real
