"""Conteos topológicos de una malla triangular: χ, componentes, bordes, género."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def edges(F):
    """Aristas únicas (u < w) y cuántas caras tiene cada una."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(E, axis=0, return_counts=True)
    return uniq, counts


def corner_fans(F):
    """Agrupa las esquinas (cara, vértice) en "abanicos": dos esquinas del mismo
    vértice están en el mismo abanico si sus caras comparten una arista que pasa
    por ese vértice. Un vértice variedad tiene un solo abanico.

    Devuelve (label, nfans): label[3f + c] = abanico de la esquina c de la cara f,
    nfans[v] = número de abanicos del vértice v."""
    m = len(F)
    corner = np.arange(3 * m).reshape(m, 3)
    rows = []
    for c in range(3):
        v = F[:, c]
        for o in (1, 2):
            w = F[:, (c + o) % 3]
            rows.append(np.stack([np.minimum(v, w), np.maximum(v, w), v, corner[:, c]], 1))
    R = np.concatenate(rows)
    R = R[np.lexsort((R[:, 2], R[:, 1], R[:, 0]))]
    same = np.all(R[1:, :3] == R[:-1, :3], axis=1)
    a, b = R[:-1, 3][same], R[1:, 3][same]
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(3 * m, 3 * m))
    _, label = connected_components(g, directed=False)
    vert = F.ravel()
    pairs = np.unique(np.stack([vert, label], 1), axis=0)
    nfans = np.bincount(pairs[:, 0], minlength=int(F.max()) + 1 if m else 0)
    return label, nfans


def summary(V, F):
    """Diccionario con V, E, F, χ, componentes, lazos de borde, género y no-variedad."""
    used = np.unique(F)
    nv, nf = len(used), len(F)
    E, cnt = edges(F)
    ne = len(E)
    chi = nv - ne + nf
    n = len(V)
    # componentes conexas (sobre los vértices usados)
    g = coo_matrix((np.ones(ne), (E[:, 0], E[:, 1])), shape=(n, n))
    ncomp_all, lab = connected_components(g, directed=False)
    ncomp = len(np.unique(lab[used]))
    # lazos de borde = componentes del grafo formado solo por aristas de borde
    B = E[cnt == 1]
    if len(B):
        gb = coo_matrix((np.ones(len(B)), (B[:, 0], B[:, 1])), shape=(n, n))
        _, lb = connected_components(gb, directed=False)
        nloops = len(np.unique(lb[np.unique(B)]))
    else:
        nloops = 0
    nonmanifold_edges = int((cnt > 2).sum())
    nonmanifold_verts = int((corner_fans(F)[1] > 1).sum()) if nf else 0
    # χ = Σ_c (2 − 2g_c − b_c)   ⇒   g = (2C − χ − B) / 2   (válido si es variedad)
    manifold = nonmanifold_edges == 0 and nonmanifold_verts == 0
    genus = (2 * ncomp - chi - nloops) / 2 if manifold else float("nan")
    return dict(V=nv, E=ne, F=nf, chi=chi, components=ncomp, boundary_loops=nloops,
                boundary_edges=int(len(B)), nonmanifold_edges=nonmanifold_edges,
                nonmanifold_verts=nonmanifold_verts, genus=genus)


def triangle_quality(V, F):
    """q = 4√3·área / (l₁² + l₂² + l₃²) ∈ [0, 1];  1 = equilátero, → 0 = astilla."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    s = ((b - a) ** 2).sum(1) + ((c - b) ** 2).sum(1) + ((a - c) ** 2).sum(1)
    return 4 * np.sqrt(3) * area / np.maximum(s, 1e-300)


def typical_edge_length(V, F):
    """Largo de arista típico a partir del área media (barato: sin armar aristas)."""
    if len(F) == 0:
        return 0.0
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).mean()
    return float(np.sqrt(4 * area / np.sqrt(3)))          # lado del equilátero de esa área


def mean_edge_length(V, F):
    E, _ = edges(F)
    return float(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).mean())
