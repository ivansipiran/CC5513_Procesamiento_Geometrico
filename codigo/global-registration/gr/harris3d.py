"""
Harris 3D  --  Sipiran & Bustos, "Harris 3D: a robust extension of the Harris
operator for interest point detection on 3D meshes", The Visual Computer, 2011.

Pipeline para cada punto v:
  1. vecindad  N(v)          (anillos adaptativos en mallas / kNN en nubes)
  2. trasladar N(v) al origen
  3. PCA: la direccion de menor varianza pasa a ser el eje z
  4. ajustar  z = f(x,y) = p1/2 x^2 + p2 xy + p3/2 y^2 + p4 x + p5 y + p6
  5. matriz de auto-correlacion
         A = p4^2/s^2 + p1^2 + p2^2
         B = p5^2/s^2 + p2^2 + p3^2
         C = p4 p5/s^2 + p1 p2 + p2 p3
         E = [[A, C], [C, B]]
  6. respuesta de Harris  h = det(E) - k tr(E)^2
  7. seleccion de keypoints (maximos locales / non-max suppression)

Nota sobre `s` (sigma): sale de la gaussiana de suavizamiento y tiene unidades
de longitud, asi que debe escalar con el tamano de la vecindad. Aqui se define
como  sigma = sigma_factor * r,  con r el radio efectivo de la vecindad.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import scipy.sparse as sp


# --------------------------------------------------------------------------- #
# nucleo comun: respuesta de Harris para una vecindad
# --------------------------------------------------------------------------- #
def _harris_response_batch(nb: np.ndarray,
                           valid: np.ndarray,
                           k: float,
                           sigma_factor: float,
                           response: str = "harris") -> np.ndarray:
    """Respuesta de Harris para un lote de vecindades.

    nb    : (M, K, 3) coordenadas de los vecinos (con relleno)
    valid : (M, K) mascara booleana de vecinos reales
    """
    M, K, _ = nb.shape
    w = valid.astype(np.float64)
    cnt = w.sum(axis=1, keepdims=True)                       # (M,1)

    # 2. trasladar el centroide de la vecindad al origen
    centroid = (nb * w[..., None]).sum(axis=1) / np.maximum(cnt, 1)
    X = (nb - centroid[:, None, :]) * w[..., None]           # (M,K,3)

    # 3. PCA -> el autovector de menor autovalor define el eje z
    cov = np.einsum("mki,mkj->mij", X, X) / np.maximum(cnt[..., None], 1)
    evals, evecs = np.linalg.eigh(cov)                       # ascendente
    # base (e1, e2, n) con n = direccion de menor varianza
    R = np.stack([evecs[:, :, 2], evecs[:, :, 1], evecs[:, :, 0]], axis=1)  # (M,3,3)
    # orientar: el punto de interes es el primero de la vecindad
    Y = np.einsum("mij,mkj->mki", R, nb - centroid[:, None, :])             # (M,K,3)

    # el punto de interes debe quedar en el origen del plano tangente
    Y = Y - Y[:, 0:1, :]

    x, y, z = Y[..., 0], Y[..., 1], Y[..., 2]
    r = np.maximum(np.sqrt(x**2 + y**2).max(axis=1), 1e-12)  # radio efectivo
    sigma = np.maximum(sigma_factor * r, 1e-9)

    # 4. ajuste por minimos cuadrados de la cuadratica (ecuaciones normales)
    #    columnas:  x^2/2, xy, y^2/2, x, y, 1
    A_ = np.stack([0.5 * x**2, x * y, 0.5 * y**2, x, y, np.ones_like(x)], axis=2)
    A_ = A_ * w[..., None]
    zz = z * w
    AtA = np.einsum("mki,mkj->mij", A_, A_)
    Atz = np.einsum("mki,mk->mi", A_, zz)
    ridge = np.maximum(np.trace(AtA, axis1=1, axis2=2), 1e-30) * 1e-10
    AtA = AtA + ridge[:, None, None] * np.eye(6)[None, :, :]
    try:
        p = np.linalg.solve(AtA, Atz[..., None])[..., 0]     # (M,6)
    except np.linalg.LinAlgError:
        p = np.einsum("mij,mj->mi", np.linalg.pinv(AtA), Atz)
    p1, p2, p3, p4, p5, _ = [p[:, i] for i in range(6)]

    # 5. matriz de auto-correlacion
    s2 = sigma**2
    Aa = p4**2 / s2 + p1**2 + p2**2
    Bb = p5**2 / s2 + p2**2 + p3**2
    Cc = p4 * p5 / s2 + p1 * p2 + p2 * p3

    det = Aa * Bb - Cc**2
    tr = Aa + Bb
    if response == "harris":
        h = det - k * tr**2
    elif response == "noble":                                # media armonica
        h = det / (tr + 1e-12)
    elif response == "min_eig":
        disc = np.sqrt(np.maximum((Aa - Bb)**2 + 4 * Cc**2, 0.0))
        h = 0.5 * (tr - disc)
    else:
        raise ValueError(response)
    h[~np.isfinite(h)] = 0.0
    return h


# --------------------------------------------------------------------------- #
# version para nubes de puntos (kNN o radio)  --  vectorizada
# --------------------------------------------------------------------------- #
def harris_response_pointcloud(points: np.ndarray,
                               k_ring: int = 60,
                               harris_k: float = 0.04,
                               sigma_factor: float = 1.0,
                               response: str = "harris") -> np.ndarray:
    """Respuesta de Harris 3D en cada punto de la nube (vecindad = kNN)."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    tree = o3d.geometry.KDTreeFlann(pcd)
    n = len(points)
    K = min(k_ring, n)
    idx = np.zeros((n, K), dtype=np.int64)
    for i in range(n):
        _, ii, _ = tree.search_knn_vector_3d(points[i], K)
        ii = np.asarray(ii)
        idx[i, :len(ii)] = ii
        if len(ii) < K:
            idx[i, len(ii):] = i
    nb = points[idx]
    valid = np.ones((n, K), dtype=bool)
    return _harris_response_batch(nb, valid, harris_k, sigma_factor, response)


# --------------------------------------------------------------------------- #
# version para mallas con vecindades adaptativas por anillos
# --------------------------------------------------------------------------- #
def _adjacency(mesh: o3d.geometry.TriangleMesh) -> sp.csr_matrix:
    tris = np.asarray(mesh.triangles)
    n = len(mesh.vertices)
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    data = np.ones(len(e), dtype=bool)
    A = sp.csr_matrix((data, (e[:, 0], e[:, 1])), shape=(n, n))
    A.data[:] = True
    return A


def adaptive_ring_neighborhoods(mesh: o3d.geometry.TriangleMesh,
                                delta: float,
                                max_rings: int = 8):
    """Vecindades adaptativas segun las slides:

        ring_k(v)   = vertices a distancia de grafo exactamente k
        d_ring      = max_{w in ring_k(v)} ||v - w||
        radius_v    = menor k tal que d_ring(v, ring_k(v)) >= delta

    Devuelve una lista de arreglos con los indices de N(v) (anillos 0..radius_v).
    """
    V = np.asarray(mesh.vertices)
    n = len(V)
    A = _adjacency(mesh)
    I = sp.identity(n, dtype=bool, format="csr")

    reach = I.copy()                      # vertices a <= k saltos
    done = np.zeros(n, dtype=bool)
    nbs: list[np.ndarray | None] = [None] * n

    for _ in range(1, max_rings + 1):
        reach = ((reach @ A) + reach).astype(bool)
        indptr, indices = reach.indptr, reach.indices
        for v in np.flatnonzero(~done):
            nb = indices[indptr[v]:indptr[v + 1]]
            d = np.linalg.norm(V[nb] - V[v], axis=1).max() if len(nb) else 0.0
            if d >= delta:
                nbs[v] = nb
                done[v] = True
        if done.all():
            break
    for v in np.flatnonzero(~done):       # no alcanzo delta: usar lo que hay
        nbs[v] = reach.indices[reach.indptr[v]:reach.indptr[v + 1]]
    return nbs


def harris_response_mesh(mesh: o3d.geometry.TriangleMesh,
                         delta_frac: float = 0.01,
                         harris_k: float = 0.04,
                         sigma_factor: float = 1.0,
                         response: str = "harris",
                         max_rings: int = 8) -> np.ndarray:
    """Respuesta de Harris 3D en cada vertice de la malla (anillos adaptativos).

    delta_frac : delta como fraccion de la diagonal del bounding box.
    """
    V = np.asarray(mesh.vertices)
    diag = np.linalg.norm(V.max(axis=0) - V.min(axis=0))
    nbs = adaptive_ring_neighborhoods(mesh, delta_frac * diag, max_rings)

    # agrupar por tamano para poder vectorizar por lotes
    sizes = np.array([len(x) for x in nbs])
    h = np.zeros(len(V))
    Kmax = int(sizes.max())
    order = np.argsort(sizes)
    # lotes de tamano homogeneo con relleno
    B = 4096
    for start in range(0, len(order), B):
        chunk = order[start:start + B]
        K = int(sizes[chunk].max())
        nb = np.zeros((len(chunk), K, 3))
        valid = np.zeros((len(chunk), K), dtype=bool)
        for j, v in enumerate(chunk):
            ids = np.asarray(nbs[v])
            ids = np.concatenate([[v], ids[ids != v]])       # v primero
            m = min(len(ids), K)
            nb[j, :m] = V[ids[:m]]
            valid[j, :m] = True
            nb[j, m:] = V[v]
        h[chunk] = _harris_response_batch(nb, valid, harris_k, sigma_factor, response)
    del Kmax
    return h


# --------------------------------------------------------------------------- #
# seleccion de keypoints
# --------------------------------------------------------------------------- #
def select_keypoints(points: np.ndarray,
                     resp: np.ndarray,
                     n_keypoints: int = 200,
                     nms_radius: float | None = None,
                     mode: str = "clustering") -> np.ndarray:
    """Selecciona indices de keypoints a partir de la respuesta de Harris.

    mode="clustering": los mas altos primero, descartando los que caen dentro de
                       `nms_radius` de uno ya aceptado (supresion de no maximos).
    mode="fraction"  : simplemente los `n_keypoints` con respuesta mas alta.
    """
    order = np.argsort(-resp)
    if mode == "fraction" or nms_radius is None:
        return order[:n_keypoints]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    tree = o3d.geometry.KDTreeFlann(pcd)
    taken = np.zeros(len(points), dtype=bool)
    out = []
    for i in order:
        if taken[i]:
            continue
        out.append(i)
        _, nb, _ = tree.search_radius_vector_3d(points[i], nms_radius)
        taken[np.asarray(nb)] = True
        if len(out) >= n_keypoints:
            break
    return np.asarray(out, dtype=np.int64)


@dataclass
class Keypoints:
    indices: np.ndarray
    points: np.ndarray
    response: np.ndarray


def detect_pointcloud(pcd: o3d.geometry.PointCloud,
                      n_keypoints: int = 200,
                      k_ring: int = 60,
                      harris_k: float = 0.04,
                      sigma_factor: float = 1.0,
                      nms_radius: float | None = None,
                      response: str = "harris") -> Keypoints:
    pts = np.asarray(pcd.points)
    h = harris_response_pointcloud(pts, k_ring, harris_k, sigma_factor, response)
    idx = select_keypoints(pts, h, n_keypoints, nms_radius)
    return Keypoints(idx, pts[idx], h)


def detect_mesh(mesh: o3d.geometry.TriangleMesh,
                n_keypoints: int = 200,
                delta_frac: float = 0.01,
                harris_k: float = 0.04,
                sigma_factor: float = 1.0,
                nms_radius: float | None = None,
                response: str = "harris") -> Keypoints:
    V = np.asarray(mesh.vertices)
    h = harris_response_mesh(mesh, delta_frac, harris_k, sigma_factor, response)
    idx = select_keypoints(V, h, n_keypoints, nms_radius)
    return Keypoints(idx, V[idx], h)
