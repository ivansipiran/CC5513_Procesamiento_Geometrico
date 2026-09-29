"""Error geométrico entre dos mallas (estilo Metro) y calidad de triángulos.

La distancia punto–superficie es exacta punto–triángulo; lo aproximado es la
elección de triángulos candidatos (los de las k muestras más cercanas).
Como las mallas están normalizadas a diagonal 1, los errores se reportan en
milésimas de la diagonal (‰)."""
import numpy as np
from scipy.spatial import cKDTree

from .topology import triangle_quality


def closest_point_on_triangles(P, A, B, C):
    """Punto más cercano a P[k] en el triángulo (A[k], B[k], C[k]) — Ericson,
    "Real-Time Collision Detection", §5.1.5, vectorizado.

    Se evalúan las 7 regiones de Voronoi del triángulo (3 vértices, 3 aristas,
    interior); cada máscara sobrescribe a las de menor prioridad."""
    dot = lambda x, y: np.einsum("ij,ij->i", x, y)
    AB, AC = B - A, C - A
    AP, BP, CP = P - A, P - B, P - C
    d1, d2 = dot(AB, AP), dot(AC, AP)
    d3, d4 = dot(AB, BP), dot(AC, BP)
    d5, d6 = dot(AB, CP), dot(AC, CP)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        v = np.where(denom != 0, vb / denom, 0.0)
        w = np.where(denom != 0, vc / denom, 0.0)
        R = A + AB * v[:, None] + AC * w[:, None]                     # interior
        m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)                # arista BC
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        R[m] = B[m] + (C - B)[m] * t[m, None]
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)                           # arista AC
        t = d2 / (d2 - d6)
        R[m] = A[m] + AC[m] * t[m, None]
        m = (d6 >= 0) & (d5 <= d6)                                      # vértice C
        R[m] = C[m]
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)                           # arista AB
        t = d1 / (d1 - d3)
        R[m] = A[m] + AB[m] * t[m, None]
        m = (d3 >= 0) & (d4 <= d3)                                      # vértice B
        R[m] = B[m]
        m = (d1 <= 0) & (d2 <= 0)                                       # vértice A
        R[m] = A[m]
    return R


def sample_surface(V, F, n, rng):
    """n puntos uniformes sobre la superficie (proporcional al área) + su triángulo."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    p = area / area.sum() if area.sum() > 0 else None
    tri = rng.choice(len(F), size=n, p=p)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    u, v, w = 1 - s, s * (1 - r2), s * r2
    pts = a[tri] * u[:, None] + b[tri] * v[:, None] + c[tri] * w[:, None]
    return pts, tri


def _bary_template(m):
    """Centroides de los m² subtriángulos de una subdivisión regular (coords. baricéntricas)."""
    pts = []
    for i in range(m):
        for j in range(m - i):
            pts.append(((i + 1 / 3) / m, (j + 1 / 3) / m))          # subtriángulo "hacia arriba"
            if i + j < m - 1:
                pts.append(((i + 2 / 3) / m, (j + 2 / 3) / m))      # "hacia abajo"
    B = np.array(pts)
    return np.column_stack([1 - B.sum(1), B[:, 0], B[:, 1]])


def cover_samples(V, F, delta, max_m=48):
    """Muestras que CUBREN cada triángulo: todo punto del triángulo queda a menos de
    ~delta de una muestra de ese mismo triángulo. Devuelve (puntos, triángulo)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    L = np.max([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                np.linalg.norm(a - c, axis=1)], axis=0)
    m = np.clip(np.ceil(L / delta).astype(int), 1, max_m)
    pts, tri = [], []
    for mm in np.unique(m):
        sel = np.nonzero(m == mm)[0]
        B = _bary_template(int(mm))                                   # (s, 3)
        P = np.einsum("sk,tkd->tsd", B, np.stack([a[sel], b[sel], c[sel]], 1))
        pts.append(P.reshape(-1, 3))
        tri.append(np.repeat(sel, len(B)))
    return np.concatenate(pts), np.concatenate(tri)


class SurfaceDistance:
    """Distancia de puntos cualesquiera a una malla fija (se construye una vez).

    Para cada punto: se buscan las k muestras más cercanas de la malla y se calcula
    la distancia EXACTA a sus triángulos. Como las muestras cubren cada triángulo con
    resolución delta, el resultado es exacto salvo en ranuras más angostas que ~2·delta."""

    def __init__(self, V, F, k=8, delta=None):
        self.V, self.F, self.k = V, F, k
        if delta is None:
            from .topology import mean_edge_length
            delta = float(np.clip(0.5 * mean_edge_length(V, F), 0.0015, 0.004))
        self.pts, self.tri = cover_samples(V, F, delta)
        self.tree = cKDTree(self.pts)

    def __call__(self, P, chunk=20000):
        out = np.empty(len(P))
        k = min(self.k, len(self.pts))
        for s in range(0, len(P), chunk):
            Pc = P[s:s + chunk]
            _, idx = self.tree.query(Pc, k=k)
            idx = idx.reshape(len(Pc), -1)
            T = self.F[self.tri[idx]].reshape(-1, 3)
            Pr = np.repeat(Pc, idx.shape[1], axis=0)
            R = closest_point_on_triangles(Pr, self.V[T[:, 0]], self.V[T[:, 1]], self.V[T[:, 2]])
            d = np.linalg.norm(R - Pr, axis=1).reshape(len(Pc), -1)
            out[s:s + chunk] = d.min(1)
        return out


def geometric_error(orig, V, F, n=30000, seed=1, orig_dist=None):
    """Error simétrico entre la original y (V, F).

    orig      : (Vo, Fo) de la malla original
    orig_dist : SurfaceDistance de la original (para no reconstruirla cada vez)
    Devuelve dict en ‰ de la diagonal: hausdorff (máx), mean, rms."""
    Vo, Fo = orig
    rng = np.random.default_rng(seed)
    if orig_dist is None:
        orig_dist = SurfaceDistance(Vo, Fo)
    simp_dist = SurfaceDistance(V, F)
    po, _ = sample_surface(Vo, Fo, n, rng)
    ps, _ = sample_surface(V, F, n, rng)
    d_os = np.concatenate([simp_dist(po), simp_dist(Vo)])     # original → simplificada
    d_so = np.concatenate([orig_dist(ps), orig_dist(V)])      # simplificada → original
    both = np.concatenate([d_os, d_so])
    q = triangle_quality(V, F)
    return dict(hausdorff=1000 * both.max(), mean=1000 * both.mean(),
                rms=1000 * np.sqrt((both ** 2).mean()),
                d_os_max=1000 * d_os.max(), d_so_max=1000 * d_so.max(),
                quality=float(q.mean()), slivers=100.0 * float((q < 0.1).mean()))
