"""
Métricas para comparar niveles y esquemas.

    V, E, F, chi              chi = V - E + F se conserva: la subdivisión no cambia topología
    irregulares               vértices extraordinarios (valencia != 4 en quads, != 6 en tri.)
                              y caras extraordinarias (grado != 4) en CC/Doo-Sabin
    diedro máx.               ángulo máximo entre normales de caras vecinas: tiende a 0 si
                              la superficie límite es suave (C1)
    volumen %                 respecto de la malla de control: los esquemas aproximantes
                              "encogen" las zonas convexas
    dist. al límite           distancia de la malla de nivel k a la superficie límite
                              (aproximada por un nivel mucho más fino), en ‰ de la diagonal
"""
import time
import numpy as np
from scipy.spatial import cKDTree

from .schemes import subdivide_step, prepare_control


# ------------------------------------------------------------ punto - triángulo
def closest_point_triangle(p, a, b, c):
    """Punto más cercano de cada triángulo (a,b,c) a p (Ericson, Real-Time Collision Det.)."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) < 1e-300, 1e-300, va + vb + vc)
    v = vb / den
    w = vc / den
    res = a + ab * v[:, None] + ac * w[:, None]                      # interior
    eps = 1e-300
    # regiones de vértices y aristas
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.where(np.abs(d1 - d3) < eps, eps, d1 - d3)
    res[m] = (a + ab * t[:, None])[m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.where(np.abs(d2 - d6) < eps, eps, d2 - d6)
    res[m] = (a + ac * t[:, None])[m]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < eps, eps, (d4 - d3) + (d5 - d6))
    res[m] = (b + (c - b) * t[:, None])[m]
    m = (d1 <= 0) & (d2 <= 0)
    res[m] = a[m]
    m = (d3 >= 0) & (d4 <= d3)
    res[m] = b[m]
    m = (d6 >= 0) & (d5 <= d6)
    res[m] = c[m]
    return res


def point_to_mesh_distance(P, mesh, k=12):
    """Distancia de cada punto P a la malla (candidatos: k triángulos de centroide más cercano)."""
    T = mesh.triangles()
    A, B, C = mesh.V[T[:, 0]], mesh.V[T[:, 1]], mesh.V[T[:, 2]]
    tree = cKDTree((A + B + C) / 3)
    k = min(k, len(T))
    _, idx = tree.query(P, k=k)
    idx = idx.reshape(len(P), k)
    best = np.full(len(P), np.inf)
    for j in range(k):
        t = idx[:, j]
        q = closest_point_triangle(P, A[t], B[t], C[t])
        best = np.minimum(best, np.linalg.norm(P - q, axis=1))
    return best


def surface_samples(mesh):
    """Vértices + puntos medios de aristas + centroides de caras."""
    mids = 0.5 * (mesh.V[mesh.edges[:, 0]] + mesh.V[mesh.edges[:, 1]])
    return np.concatenate([mesh.V, mids, mesh.face_centroids()])


# ------------------------------------------------------------ irregularidad
def irregular_counts(mesh, scheme):
    interior = ~mesh.boundary_vertex
    regular_val = 6 if scheme == "loop" else 4
    ev = int(((mesh.valence != regular_val) & interior).sum())
    ef = int((mesh.deg != (3 if scheme == "loop" else 4)).sum())
    return ev, ef


def level_row(mesh, scheme, control_vol, diag, t=0.0, dist=None):
    ev, ef = irregular_counts(mesh, scheme)
    vol = mesh.volume()
    dih = mesh.dihedral_angles()
    return dict(V=mesh.nV, E=mesh.nE, F=mesh.nF, chi=mesh.euler(), ev=ev, ef=ef,
                dih_max=float(dih.max()) if len(dih) else 0.0,
                dih_mean=float(dih.mean()) if len(dih) else 0.0,
                vol=(100 * vol / control_vol) if (vol is not None and control_vol) else None,
                t=t, dist=dist)


def run_levels(m, scheme, levels, variant=None, ds_boundary="chaikin", ref_extra=2,
               ref_max_faces=400_000, with_dist=True):
    """Subdivide y calcula las métricas de cada nivel (incluye distancia al límite)."""
    m0, _ = prepare_control(m, scheme)
    meshes, times = [m0], [0.0]
    for _ in range(levels):
        t0 = time.perf_counter()
        st = subdivide_step(meshes[-1], scheme, variant, ds_boundary)
        times.append(time.perf_counter() - t0)
        meshes.append(st.fine)
    ref = meshes[-1]
    if with_dist:
        for _ in range(ref_extra):
            if ref.nF * 4 > ref_max_faces:
                break
            ref = subdivide_step(ref, scheme, variant, ds_boundary).fine
    diag = m0.bbox_diag()
    cvol = m0.volume()
    rows = []
    for k, mk in enumerate(meshes):
        dist = None
        if with_dist and mk is not ref:
            d = point_to_mesh_distance(surface_samples(mk), ref)
            dist = (1000 * d.max() / diag, 1000 * d.mean() / diag)
        elif with_dist:
            dist = (0.0, 0.0)
        rows.append(level_row(mk, scheme, cvol, diag, times[k], dist))
    return meshes, rows, ref


def format_rows(rows, scheme):
    irr = "irr.V/irr.F" if scheme != "loop" else "irr.V"
    head = f"{'niv':>3} {'V':>8} {'F':>8} {'chi':>4} {irr:>11} {'diedro max':>10} {'vol %':>7} " \
           f"{'dist.lim max/med (o/oo)':>24} {'t [s]':>7}"
    out = [head, "-" * len(head)]
    for k, r in enumerate(rows):
        irrs = f"{r['ev']}/{r['ef']}" if scheme != "loop" else f"{r['ev']}"
        vol = f"{r['vol']:.1f}" if r["vol"] is not None else "-"
        dist = f"{r['dist'][0]:.2f} / {r['dist'][1]:.2f}" if r["dist"] is not None else "-"
        out.append(f"{k:>3} {r['V']:>8} {r['F']:>8} {r['chi']:>4} {irrs:>11} {r['dih_max']:>9.1f}° "
                   f"{vol:>7} {dist:>24} {r['t']:>7.3f}")
    return "\n".join(out)
