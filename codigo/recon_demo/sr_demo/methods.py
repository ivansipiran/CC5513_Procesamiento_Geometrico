"""Metodos de reconstruccion implicita (Bloques 2.3 - 2.5 del tutorial).

Todos devuelven un campo escalar en la grilla con la MISMA convencion de signo
(F < 0 dentro, F > 0 fuera), de modo que el mismo extractor sirve para los tres:

* Hoppe 1992   -- distancia con signo al plano tangente del centroide mas
                  cercano; NaN donde el punto proyectado queda lejos de los datos.
* RBF (Carr)   -- interpolante biharmonico phi(r) = r con restricciones
                  fuera de la superficie (p +- eps n) y bloque polinomial lineal.
* Poisson      -- se "salpican" las normales en la grilla (campo V), se toma la
                  divergencia y se resuelve  Lap(phi) = div V  por FFT. El
                  iso-valor es el promedio de phi en las muestras.
* Open3D       -- Poisson screened (Kazhdan & Hoppe 2013) sobre octree, como
                  referencia de una implementacion de produccion (opcional).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .grid import Grid


@dataclass
class ReconParams:
    res: int = 64                 # celdas en el eje mas largo
    hoppe_k: int = 20             # vecinos del plano tangente (centroide)
    hoppe_support: float = 3.0    # rho + delta, en multiplos del espaciado medio
    rbf_centers: int = 700
    rbf_eps: float = 0.01         # desplazamiento fuera de la superficie (frac. diag)
    poisson_sigma: float = 1.0    # suavizado gaussiano del campo V (en celdas)
    poisson_adaptive: bool = True # pesa cada normal por el area local que representa
    o3d_depth: int = 8
    o3d_trim: float = 0.0         # recorta los vertices de menor densidad (cuantil)
    extractor: str = "mt"         # mt (propio) | mc (scikit-image)


@dataclass
class FieldResult:
    name: str
    grid: Grid | None
    vals: np.ndarray | None                 # (nx, ny, nz); NaN = no definido
    t_field: float = 0.0
    extras: dict = field(default_factory=dict)
    mesh: tuple | None = None               # solo para metodos sin campo (Open3D)


def spacing(P: np.ndarray) -> float:
    d, _ = cKDTree(P).query(P, k=2, workers=-1)
    return float(np.median(d[:, 1]))


# --------------------------------------------------------------------------- #
# Hoppe
# --------------------------------------------------------------------------- #
def hoppe(P, N, grid: Grid, prm: ReconParams, noise_abs: float = 0.0) -> FieldResult:
    t0 = time.perf_counter()
    k = min(prm.hoppe_k, len(P))
    _, idx = cKDTree(P).query(P, k=k, workers=-1)
    O = P[idx].mean(1)                          # centros de los planos tangentes
    X = grid.points()
    _, j = cKDTree(O).query(X, workers=-1)
    f = ((X - O[j]) * N[j]).sum(1)
    Z = X - f[:, None] * N[j]                   # proyeccion sobre el plano
    rho_delta = prm.hoppe_support * spacing(P) + noise_abs
    dz, _ = cKDTree(P).query(Z, workers=-1)
    f[dz > rho_delta] = np.nan
    vals = f.reshape(grid.shape)
    return FieldResult("Hoppe", grid, vals, time.perf_counter() - t0,
                       extras=dict(centers=O, rho_delta=rho_delta,
                                   undefined=float(np.isnan(f).mean())))


# --------------------------------------------------------------------------- #
# RBF
# --------------------------------------------------------------------------- #
def farthest_point_sampling(P: np.ndarray, m: int, seed: int = 0) -> np.ndarray:
    m = min(m, len(P))
    rng = np.random.default_rng(seed)
    sel = np.empty(m, int)
    sel[0] = rng.integers(len(P))
    d = np.linalg.norm(P - P[sel[0]], axis=1)
    for i in range(1, m):
        sel[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(P - P[sel[i]], axis=1))
    return sel


def rbf(P, N, grid: Grid, prm: ReconParams, diag: float) -> FieldResult:
    t0 = time.perf_counter()
    sel = farthest_point_sampling(P, prm.rbf_centers)
    C0, N0 = P[sel], N[sel]
    tree = cKDTree(P)
    eps = np.full(len(C0), prm.rbf_eps * diag)
    # Carr et al.: el punto p + eps n no debe quedar cerca de OTRA parte de la
    # superficie (zonas delgadas). Si la muestra mas cercana esta a menos de
    # eps/2, eps se reduce a la mitad (hasta 4 veces). Se usa eps/2 y no
    # "el mas cercano debe ser p" porque con ruido esa regla encoge eps hasta
    # el nivel del ruido y las restricciones se vuelven inutiles.
    for _ in range(4):
        for sgn in (+1, -1):
            d, _ = tree.query(C0 + sgn * eps[:, None] * N0, workers=-1)
            bad = d < 0.5 * eps
            eps[bad] *= 0.5
    C = np.vstack([C0, C0 + eps[:, None] * N0, C0 - eps[:, None] * N0])
    fval = np.r_[np.zeros(len(C0)), eps, -eps]
    m = len(C)
    A = np.linalg.norm(C[:, None] - C[None], axis=-1)          # phi(r) = r
    Pm = np.c_[np.ones(m), C]
    M = np.block([[A, Pm], [Pm.T, np.zeros((4, 4))]])
    sol = np.linalg.solve(M, np.r_[fval, np.zeros(4)])
    w, c = sol[:m], sol[m:]
    t_solve = time.perf_counter() - t0
    X = grid.points()
    f = np.empty(len(X))
    B = max(1, 4_000_000 // m)
    for a in range(0, len(X), B):
        Xa = X[a:a + B]
        D = np.sqrt(np.maximum((Xa ** 2).sum(1)[:, None] - 2 * Xa @ C.T + (C ** 2).sum(1)[None], 0))
        f[a:a + B] = D @ w + c[0] + Xa @ c[1:]
    vals = f.reshape(grid.shape)
    return FieldResult("RBF", grid, vals, time.perf_counter() - t0,
                       extras=dict(constraints=C, constraint_vals=fval, t_solve=t_solve,
                                   system=M.shape[0], eps_mean=float(eps.mean() / diag)))


# --------------------------------------------------------------------------- #
# Poisson por FFT
# --------------------------------------------------------------------------- #
def splat(P, W, grid: Grid) -> np.ndarray:
    """Reparte W (n, 3) en los 8 vertices de la celda con pesos trilineales."""
    V = np.zeros(grid.shape + (3,))
    g = grid.to_index(P)
    i0 = np.floor(g).astype(int)
    fr = g - i0
    for o in np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]):
        wgt = np.prod(np.where(o, fr, 1 - fr), axis=1)
        ii = i0 + o
        for a in range(3):
            np.add.at(V[..., a], (ii[:, 0], ii[:, 1], ii[:, 2]), wgt * W[:, a])
    return V


def poisson_fft_solve(rhs: np.ndarray, h: float) -> np.ndarray:
    """Resuelve Lap(phi) = rhs (laplaciano de 7 puntos, frontera periodica)."""
    lam = 0.0
    for a, n in enumerate(rhs.shape):
        k = np.arange(n)
        s = (2 * np.cos(2 * np.pi * k / n) - 2) / h ** 2
        shp = [1, 1, 1]
        shp[a] = n
        lam = lam + s.reshape(shp)
    R = np.fft.fftn(rhs)
    lam = np.where(lam == 0, 1.0, lam)
    Phi = R / lam
    Phi.flat[0] = 0.0
    return np.real(np.fft.ifftn(Phi))


def poisson(P, N, grid: Grid, prm: ReconParams) -> FieldResult:
    t0 = time.perf_counter()
    if prm.poisson_adaptive:   # area que representa cada muestra ~ r_k^2
        d, _ = cKDTree(P).query(P, k=9, workers=-1)
        a = d[:, -1] ** 2
        a = a / a.mean()
    else:
        a = np.ones(len(P))
    V = splat(P, N * a[:, None], grid)
    if prm.poisson_sigma > 0:
        V = np.stack([gaussian_filter(V[..., c], prm.poisson_sigma, mode="constant")
                      for c in range(3)], -1)
    div = sum(np.gradient(V[..., c], grid.h, axis=c) for c in range(3))
    phi = poisson_fft_solve(div, grid.h)
    iso = float(grid.sample(phi, P).mean())
    vals = phi - iso
    return FieldResult("Poisson", grid, vals, time.perf_counter() - t0,
                       extras=dict(V=V, div=div, iso=iso))


# --------------------------------------------------------------------------- #
# Open3D (opcional)
# --------------------------------------------------------------------------- #
def open3d_available() -> bool:
    try:
        import open3d  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def open3d_poisson(P, N, prm: ReconParams) -> FieldResult | None:
    try:
        import open3d as o3d
    except Exception:  # noqa: BLE001
        return None
    t0 = time.perf_counter()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd.normals = o3d.utility.Vector3dVector(N)
    try:
        vl = o3d.utility.VerbosityLevel.Error
        o3d.utility.set_verbosity_level(vl)
    except Exception:  # noqa: BLE001
        pass
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=prm.o3d_depth)
    dens = np.asarray(dens)
    if prm.o3d_trim > 0 and len(dens):
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, prm.o3d_trim))
    V = np.asarray(mesh.vertices).copy()
    T = np.asarray(mesh.triangles).astype(np.int64).copy()
    return FieldResult("Open3D", None, None, time.perf_counter() - t0, mesh=(V, T))
