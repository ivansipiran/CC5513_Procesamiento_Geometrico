"""Métodos de comparación.

· clustering   Rossignac & Borrel (1993): grilla uniforme, todos los vértices de una
               celda se funden en uno (el promedio). Muy rápido, sin control de
               error ni de topología.
· clustering-qem  lo mismo, pero el representante de cada celda es el mínimo de la
               suma de las cuádricas de la celda (Lindstrom 2000): misma grilla,
               mejor posición.
· open3d       simplify_quadric_decimation de Open3D (también Garland–Heckbert),
               como referencia externa si está instalado.

"arista más corta" no está aquí: es decimate.PairCollapse con cost="length".
"""
import numpy as np

from . import quadrics as qd


def vertex_clustering(V, F, resolution, representative="mean", weighting="area"):
    """Agrupa vértices en una grilla de `resolution` celdas a lo largo del eje más largo."""
    lo, hi = V.min(0), V.max(0)
    h = (hi - lo).max() / resolution
    cell = np.floor((V - lo) / h).astype(np.int64)
    cell = np.minimum(cell, np.ceil((hi - lo) / h).astype(np.int64))
    _, label = np.unique(cell, axis=0, return_inverse=True)
    label = label.ravel()
    nc = label.max() + 1
    if representative == "qem":
        Q = qd.vertex_quadrics(V, F, weighting)
        Qc = np.zeros((nc, 4, 4))
        np.add.at(Qc, label, Q)
        cnt = np.bincount(label, minlength=nc)
        mean = np.zeros((nc, 3))
        np.add.at(mean, label, V)
        mean /= cnt[:, None]
        # mínimo de la cuádrica, lo más cerca posible del promedio (pseudo-inversa)
        P, _, _ = qd.optimal_placement(Qc, mean, mean, "svd", rcond=1e-3)
        # si el mínimo se escapa de la celda, usar el promedio
        far = np.linalg.norm(P - mean, axis=1) > 1.5 * h
        P[far] = mean[far]
    else:
        P = np.zeros((nc, 3))
        np.add.at(P, label, V)
        P /= np.bincount(label, minlength=nc)[:, None]
    G = label[F]
    G = G[(G[:, 0] != G[:, 1]) & (G[:, 1] != G[:, 2]) & (G[:, 2] != G[:, 0])]
    _, keep = np.unique(np.sort(G, axis=1), axis=0, return_index=True)
    G = G[np.sort(keep)]
    used = np.unique(G)
    remap = np.full(nc, -1)
    remap[used] = np.arange(len(used))
    return P[used], remap[G]


def clustering_to_target(V, F, target, representative="mean"):
    """Busca (bisección) la resolución de grilla cuyo número de caras se acerca más a target."""
    lo, hi = 1, 2048
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        Vc, Fc = vertex_clustering(V, F, mid, representative)
        if best is None or abs(len(Fc) - target) < abs(len(best[1]) - target):
            best = (Vc, Fc, mid)
        if len(Fc) < target:
            lo = mid + 1
        elif len(Fc) > target:
            hi = mid - 1
        else:
            break
    return best


_HAS_O3D = None


def has_open3d():
    global _HAS_O3D
    if _HAS_O3D is None:
        try:
            import open3d  # noqa: F401
            _HAS_O3D = True
        except Exception:          # noqa: BLE001
            _HAS_O3D = False
    return _HAS_O3D


def open3d_quadric(V, F, target, scale=1000.0):
    """Open3D escalando la malla por `scale` (y deshaciendo la escala al final).

    Ojo: Open3D decide si A es invertible con un umbral ABSOLUTO sobre det(A). Con
    la malla normalizada a diagonal 1 y cuádricas ponderadas por área, det(A) es
    diminuto, nunca pasa el umbral y Open3D termina usando siempre "el mejor de
    v₁, v₂, punto medio". Escalando ×1000 se comporta como el paper. (Nuestro
    chequeo usa λ_min/λ_max, que no depende de la escala.)"""
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V * scale),
                                  o3d.utility.Vector3iVector(F))
    s = m.simplify_quadric_decimation(int(target))
    s.remove_unreferenced_vertices()
    return np.asarray(s.vertices) / scale, np.asarray(s.triangles, dtype=np.int64)
