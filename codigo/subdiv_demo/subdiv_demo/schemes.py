"""
Los tres esquemas de subdivisión de la clase, escritos de la misma forma:

    1. TOPOLOGÍA  – qué vértices nuevos hay y cómo se conectan (las caras nuevas).
    2. GEOMETRÍA  – dónde va cada vértice nuevo: una combinación afín de vértices viejos.

La geometría se guarda como una matriz dispersa S (n_nuevos x n_viejos):

    P_nuevo = S @ P_viejo          cada fila de S es la "máscara" (estencil) de un vértice

Así la matriz ES el esquema: una fila muestra los pesos de un vértice nuevo, una columna
muestra a qué vértices nuevos afecta un vértice viejo (control local), y el producto
S_k ... S_2 S_1 lleva la malla de control al nivel k.

Además de S devolvemos S_lin: las posiciones "solo topología" (dividir sin promediar).
Interpolar entre S_lin y S muestra la separación split / average del esquema.

Esquemas
--------
catmull_clark(m)   primal, cuadriláteros; generaliza B-splines bicúbicas (C2, C1 en extraord.)
loop(m)            primal, triángulos;    generaliza box-splines cuárticas (C2, C1 en extraord.)
doo_sabin(m)       dual, corte de esquinas; generaliza B-splines bicuadráticas (C1)
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from .mesh import PolyMesh


def sp(rows, cols, vals, shape):
    """Matriz dispersa a partir de tripletas; las entradas repetidas se suman."""
    vals = np.broadcast_to(np.asarray(vals, dtype=np.float64), np.shape(rows))
    return sparse.csr_matrix((vals, (rows, cols)), shape=shape)


def diag(x):
    return sparse.diags(np.asarray(x, dtype=np.float64))


@dataclass
class Step:
    """Resultado de un paso de subdivisión (nivel k-1 -> nivel k)."""
    scheme: str
    coarse: PolyMesh          # malla del nivel k-1
    fine: PolyMesh            # malla del nivel k (posiciones = S @ coarse.V)
    S: sparse.csr_matrix      # reglas geométricas
    S_lin: sparse.csr_matrix  # solo topología (sin promediar)
    vkind: np.ndarray         # tipo de cada vértice nuevo (índice en vkind_names)
    vkind_names: list
    vparent: np.ndarray       # elemento viejo del que nace (vértice, arista, cara o esquina)
    fkind: np.ndarray         # tipo de cada cara nueva
    fkind_names: list
    notes: list = field(default_factory=list)

    def describe_vertex(self, i):
        """Frase corta que dice de dónde sale el vértice nuevo i."""
        k, p = int(self.vkind[i]), int(self.vparent[i])
        name = self.vkind_names[k]
        m = self.coarse
        if self.scheme == "doo-sabin":
            v, f = int(m.src[p]), int(m.face_of[p])
            return f"{name}: esquina del vertice {v} en la cara {f} ({int(m.deg[f])}-gono)"
        if name.startswith("punto de cara"):
            return f"{name} {p} ({int(m.deg[p])}-gono)"
        if name.startswith("punto de arista") or name.startswith("impar"):
            a, b = m.edges[p]
            tag = " [pliegue/borde]" if m.sharp[p] else ""
            return f"{name}: arista {p} = ({a},{b}){tag}"
        return f"{name}: vertice {p}, valencia {int(m.valence[p])}"


# --------------------------------------------------------------------------------
#  Clasificación de vértices para bordes y pliegues (Hoppe et al. 1994, simplificado)
# --------------------------------------------------------------------------------
def classify_vertices(m):
    """
    ns = nº de aristas afiladas (borde o pliegue) que llegan al vértice.
        ns < 2           -> 'suave'   (ns = 1 es un "dardo": se usa la regla suave)
        ns = 2           -> 'pliegue' (regla de curva: B-spline sobre la línea afilada)
        ns > 2, o vértice de borde con una sola cara -> 'esquina' (queda fijo)
    """
    ns = np.bincount(m.edges[m.sharp].ravel(), minlength=m.nV)
    corner = (ns > 2) | (m.boundary_vertex & (m.nfaces_v == 1))
    crease = (ns == 2) & ~corner
    smooth = ~(corner | crease)
    return smooth, crease, corner


def crease_neighbors_matrix(m, rows_mask, w):
    """Para cada vértice en rows_mask, peso w a sus dos vecinos por aristas afiladas."""
    e = m.edges[m.sharp]
    r = np.concatenate([e[:, 0], e[:, 1]])
    c = np.concatenate([e[:, 1], e[:, 0]])
    keep = rows_mask[r]
    return sp(r[keep], c[keep], w, (m.nV, m.nV))


# ================================================================================
#  CATMULL-CLARK (1978)
# ================================================================================
def catmull_clark(m, variant="standard"):
    """
    Topología: cada cara de grado n se parte en n cuadriláteros
        [vértice, punto de arista siguiente, punto de cara, punto de arista anterior]
    Numeración nueva:  [0, nV) puntos de vértice | [nV, nV+nE) de arista | resto de cara

    Geometría (variant="standard"):
        punto de cara    F = promedio de los vértices de la cara
        punto de arista  E = (a + b + F1 + F2) / 4          (arista afilada: (a+b)/2)
        punto de vértice P'= Q/n + 2R/n + (n-3)P/n          Q = prom. puntos de cara
                                                             R = prom. puntos medios de arista
                         pliegue: 3/4 P + 1/8 (a + b)       esquina: P
    variant="linear": solo topología (F, puntos medios, P igual).
    """
    nV, nE, nF, H = m.nV, m.nE, m.nF, m.nH

    # ---------------- 1. topología (vectorizada: un quad por esquina) ----------------
    e_out = nV + m.edge_of            # punto de la arista que SALE de la esquina
    e_in = nV + m.edge_of[m.prv]      # punto de la arista que ENTRA a la esquina
    f_pt = nV + nE + m.face_of        # punto de la cara
    quads = np.stack([m.src, e_out, f_pt, e_in], axis=1)

    # ---------------- 2. geometría ----------------
    Fmat = sp(m.face_of, m.fidx, 1.0 / m.deg[m.face_of], (nF, nV))
    Mid = sp(np.repeat(np.arange(nE), 2), m.edges.ravel(), 0.5, (nE, nV))
    I = sparse.identity(nV, format="csr")
    S_lin = sparse.vstack([I, Mid, Fmat]).tocsr()

    sharp = m.sharp
    # puntos de arista: suave -> 1/2 Mid + 1/4 (F1 + F2);  afilada -> Mid
    hs = np.nonzero(~sharp[m.edge_of])[0]
    EF = sp(m.edge_of[hs], m.face_of[hs], 0.25, (nE, nF))
    E = diag(np.where(sharp, 1.0, 0.5)) @ Mid + EF @ Fmat

    # puntos de vértice
    smooth, crease, corner = classify_vertices(m)
    n = m.valence.astype(float)
    VF = sp(m.fidx, m.face_of, 1.0 / m.nfaces_v[m.fidx], (nV, nF))             # prom. caras
    VE = sp(m.edges.ravel(), np.repeat(np.arange(nE), 2),
            1.0 / n[m.edges.ravel()], (nV, nE))                                   # prom. aristas
    Q = VF @ Fmat
    R = VE @ Mid
    s = smooth.astype(float)
    Vp = (diag(s / n) @ Q + diag(2 * s / n) @ R + diag(s * (n - 3) / n)
          + diag(np.where(crease, 0.75, 0.0)) + crease_neighbors_matrix(m, crease, 0.125)
          + diag(corner.astype(float)))
    S = sparse.vstack([Vp, E, Fmat]).tocsr()
    if variant == "linear":
        S = S_lin

    newV = S @ m.V
    cr = m.edges[m.crease]
    ce = nV + np.nonzero(m.crease)[0]
    crease_pairs = np.concatenate([np.stack([cr[:, 0], ce], 1), np.stack([ce, cr[:, 1]], 1)])
    fine = PolyMesh(newV, quads, crease_pairs=crease_pairs, name=m.name)

    vkind = np.concatenate([np.zeros(nV, int), np.ones(nE, int), np.full(nF, 2)])
    vparent = np.concatenate([np.arange(nV), np.arange(nE), np.arange(nF)])
    fkind = np.minimum(m.deg[m.face_of], 6) - 3        # grado de la cara madre (3,4,5,6+)
    notes = []
    if not m.is_quad_mesh():
        notes.append(f"{int((m.deg != 4).sum())} caras no-cuadriláteras -> sus puntos de cara "
                     "quedan como vértices extraordinarios (valencia != 4)")
    return Step("catmull-clark", m, fine, S, S_lin, vkind,
                ["punto de vertice", "punto de arista", "punto de cara"], vparent,
                fkind, ["de triangulo", "de cuadrilatero", "de pentagono", "de 6+-gono"], notes)


# ================================================================================
#  LOOP (1987)
# ================================================================================
def loop_beta(n, kind="loop"):
    """Peso de cada vecino en la regla de vértice par: P' = (1 - n beta) P + beta * suma."""
    n = np.asarray(n, dtype=float)
    if kind == "warren":
        return np.where(n == 3, 3.0 / 16.0, 3.0 / (8.0 * np.maximum(n, 1)))
    c = 3.0 / 8.0 + 0.25 * np.cos(2 * np.pi / np.maximum(n, 1))
    return (5.0 / 8.0 - c * c) / np.maximum(n, 1)


def loop(m, variant="loop"):
    """
    Topología: cada triángulo se parte en 4 (un vértice nuevo por arista).
    Numeración nueva:  [0, nV) vértices pares (viejos) | [nV, nV+nE) impares (de arista)

    Geometría:
        impar (arista a-b, opuestos c y d):  3/8 (a+b) + 1/8 (c+d)     afilada: (a+b)/2
        par   (valencia n):                 (1 - n beta) P + beta * suma vecinos
                                            pliegue: 3/4 P + 1/8 (a+b)   esquina: P
        variant = "loop"   beta = 1/n (5/8 - (3/8 + 1/4 cos(2 pi/n))^2)   (el de Loop)
                  "warren" beta = 3/(8n)  (3/16 si n=3)                  (más simple)
                  "linear" solo topología (puntos medios)
    """
    if not m.is_triangle_mesh():
        raise ValueError("Loop necesita una malla de triángulos (use triangulated())")
    nV, nE, H = m.nV, m.nE, m.nH

    # ---------------- 1. topología ----------------
    corner_tris = np.stack([m.src, nV + m.edge_of, nV + m.edge_of[m.prv]], axis=1)
    center = (nV + m.edge_of).reshape(-1, 3)          # las 3 aristas de cada triángulo
    tris = np.concatenate([corner_tris, center])

    # ---------------- 2. geometría ----------------
    Mid = sp(np.repeat(np.arange(nE), 2), m.edges.ravel(), 0.5, (nE, nV))
    I = sparse.identity(nV, format="csr")
    S_lin = sparse.vstack([I, Mid]).tocsr()

    sharp = m.sharp
    hs = np.nonzero(~sharp[m.edge_of])[0]              # half-edges de aristas suaves
    opp = m.fidx[m.prv[hs]]                            # vértice opuesto a la arista en la cara
    Odd_smooth = (sp(m.edge_of[hs], m.src[hs], 3.0 / 16.0, (nE, nV))
                  + sp(m.edge_of[hs], m.dst[hs], 3.0 / 16.0, (nE, nV))
                  + sp(m.edge_of[hs], opp, 1.0 / 8.0, (nE, nV)))
    Odd = Odd_smooth + diag(sharp.astype(float)) @ Mid

    smooth, crease, corner = classify_vertices(m)
    n = m.valence.astype(float)
    beta = loop_beta(n, "warren" if variant == "warren" else "loop")
    e = m.edges
    r = np.concatenate([e[:, 0], e[:, 1]])
    c = np.concatenate([e[:, 1], e[:, 0]])
    keep = smooth[r]
    Even = (diag(np.where(smooth, 1 - n * beta, 0.0)) + sp(r[keep], c[keep], beta[r[keep]], (nV, nV))
            + diag(np.where(crease, 0.75, 0.0)) + crease_neighbors_matrix(m, crease, 0.125)
            + diag(corner.astype(float)))
    S = sparse.vstack([Even, Odd]).tocsr()
    if variant == "linear":
        S = S_lin

    newV = S @ m.V
    cr = m.edges[m.crease]
    ce = nV + np.nonzero(m.crease)[0]
    crease_pairs = np.concatenate([np.stack([cr[:, 0], ce], 1), np.stack([ce, cr[:, 1]], 1)])
    fine = PolyMesh(newV, tris, crease_pairs=crease_pairs, name=m.name)

    vkind = np.concatenate([np.zeros(nV, int), np.ones(nE, int)])
    vparent = np.concatenate([np.arange(nV), np.arange(nE)])
    fkind = np.concatenate([np.zeros(H, int), np.ones(m.nF, int)])
    return Step("loop", m, fine, S, S_lin, vkind, ["par (vertice viejo)", "impar (de arista)"],
                vparent, fkind, ["triangulo de esquina", "triangulo central"], [])


# ================================================================================
#  DOO-SABIN (1978)
# ================================================================================
def doo_sabin_weights(n):
    """Pesos de Doo-Sabin para una cara de n lados: alpha_0 (propio) y alpha_k (a k pasos)."""
    k = np.arange(n)
    w = (3.0 + 2.0 * np.cos(2 * np.pi * k / n)) / (4.0 * n)
    w[0] = (n + 5.0) / (4.0 * n)
    return w


def simple_weights(n):
    """Variante simple: (P + F + M_ant + M_sig) / 4  = 1/2 P + 1/8 (vecinos) + 1/4 F."""
    w = np.full(n, 0.25 / n)
    w[0] += 0.5
    w[1] += 0.125
    w[-1] += 0.125
    return w


def doo_sabin(m, variant="doo-sabin", boundary="chaikin"):
    """
    Esquema DUAL: un vértice nuevo por cada ESQUINA (vértice v dentro de la cara f).
    Numeración nueva: el vértice nuevo c es la esquina c de la malla vieja (0..nH-1).

    Topología (corte de esquinas), tres tipos de caras nuevas:
        cara-F  por cada cara f:          sus esquinas, en el mismo orden (n-gono)
        cara-E  por cada arista interior: las 4 esquinas de sus extremos (cuadrilátero)
        cara-V  por cada vértice:         las esquinas a su alrededor (k-gono, k = valencia)

    Geometría (cara de n lados, esquina i):
        "doo-sabin":  P_i' = sum_j alpha_{|i-j|} P_j,  alpha_0 = (n+5)/4n,
                                                   alpha_k = (3 + 2 cos(2 pi k/n)) / 4n
        "simple":     P_i' = (P_i + F + M_{i-1,i} + M_{i,i+1}) / 4   (centroide y puntos medios)
        En cuadriláteros ambos dan 9/16, 3/16, 1/16, 3/16 (B-spline bicuadrática).

    Borde:
        "chaikin": la esquina junto a una arista de borde a-b va a 3/4 a + 1/4 b, así el
                   borde es un corte de esquinas de Chaikin (curva B-spline cuadrática).
        "free":    se usa la regla interior también en el borde (el borde se encoge).
    """
    nV, H = m.nV, m.nH

    # ---------------- 1. topología ----------------
    faces = []
    fkind = []
    # caras-F: mismas esquinas, mismo orden
    faces += [np.arange(m.fptr[f], m.fptr[f + 1]) for f in range(m.nF)]
    fkind += [0] * m.nF
    # caras-E: arista a->b (half-edge h en f), twin t (b->a en g)
    inner = np.nonzero(~m.boundary_edge)[0]
    h = m.edge_he[inner]
    t = m.twin[h]
    Efaces = np.stack([m.nxt[h], h, m.nxt[t], t], axis=1)
    faces += list(Efaces)
    fkind += [1] * len(inner)
    # caras-V: abanico de esquinas alrededor de cada vértice
    n_vfaces_open = 0
    for v, fan, closed in m.fans():
        if len(fan) >= 3:
            faces.append(np.array(fan))
            fkind.append(2)
            n_vfaces_open += (not closed)

    # ---------------- 2. geometría ----------------
    rows, cols, vals = [], [], []
    wfun = doo_sabin_weights if variant == "doo-sabin" else simple_weights
    for n in np.unique(m.deg):
        fs = np.nonzero(m.deg == n)[0]
        w = wfun(int(n))
        base = m.fptr[fs]                                   # (k,)
        for i in range(n):                                  # esquina i de cada cara
            for j in range(n):                              # vértice j de la cara
                rows.append(base + i)
                cols.append(m.fidx[base + j])
                vals.append(np.full(len(fs), w[(j - i) % n]))
    rows, cols, vals = map(np.concatenate, (rows, cols, vals))

    if boundary == "chaikin" and not m.is_closed():
        out_b = m.twin < 0                                   # arista que sale de la esquina es borde
        in_b = m.twin[m.prv] < 0                             # arista que entra es borde
        special = out_b | in_b
        keep = ~special[rows]
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
        c_both = np.nonzero(out_b & in_b)[0]                 # esquina de una sola cara
        c_out = np.nonzero(out_b & ~in_b)[0]
        c_in = np.nonzero(in_b & ~out_b)[0]
        rows = np.concatenate([rows, c_both, c_out, c_out, c_in, c_in])
        cols = np.concatenate([cols, m.src[c_both], m.src[c_out], m.dst[c_out],
                               m.src[c_in], m.src[m.prv[c_in]]])
        vals = np.concatenate([vals, np.ones(len(c_both)), np.full(len(c_out), .75),
                               np.full(len(c_out), .25), np.full(len(c_in), .75),
                               np.full(len(c_in), .25)])
    S = sp(rows, cols, vals, (H, nV))
    S_lin = sp(np.arange(H), m.src, 1.0, (H, nV))            # esquinas sobre su vértice

    newV = S @ m.V
    deg = np.array([len(f) for f in faces])
    fptr = np.concatenate([[0], np.cumsum(deg)])
    fidx = np.concatenate(faces)
    fine = PolyMesh(newV, (fptr, fidx), name=m.name)

    notes = []
    if m.crease.any():
        notes.append("Doo-Sabin no usa pliegues en este demo (solo CC y Loop)")
    if n_vfaces_open:
        notes.append(f"{n_vfaces_open} caras-V abiertas cerradas en el borde")
    vkind = np.minimum(m.deg[m.face_of], 6) - 3
    return Step("doo-sabin", m, fine, S, S_lin, vkind,
                ["esquina de triangulo", "esquina de cuadrilatero", "esquina de pentagono",
                 "esquina de 6+-gono"],
                np.arange(H), np.array(fkind), ["cara-F", "cara-E", "cara-V"], notes)


# ================================================================================
SCHEMES = {
    "catmull-clark": dict(fn=catmull_clark, variants=["standard", "linear"],
                          label="Catmull-Clark"),
    "loop": dict(fn=loop, variants=["loop", "warren", "linear"], label="Loop"),
    "doo-sabin": dict(fn=doo_sabin, variants=["doo-sabin", "simple"], label="Doo-Sabin"),
}


def subdivide_step(m, scheme, variant=None, ds_boundary="chaikin"):
    if scheme == "catmull-clark":
        return catmull_clark(m, variant or "standard")
    if scheme == "loop":
        return loop(m, variant or "loop")
    if scheme == "doo-sabin":
        return doo_sabin(m, variant or "doo-sabin", boundary=ds_boundary)
    raise ValueError(scheme)


def prepare_control(m, scheme):
    """Loop necesita triángulos: si la malla de control tiene polígonos, la triangulamos."""
    if scheme == "loop" and not m.is_triangle_mesh():
        return m.triangulated(), "malla de control triangulada para Loop"
    return m, None


def subdivide(m, scheme, levels, variant=None, ds_boundary="chaikin"):
    """Devuelve (lista de mallas [nivel 0..levels], lista de Steps)."""
    m0, note = prepare_control(m, scheme)
    meshes, steps = [m0], []
    for _ in range(levels):
        st = subdivide_step(meshes[-1], scheme, variant, ds_boundary)
        steps.append(st)
        meshes.append(st.fine)
    return meshes, steps
