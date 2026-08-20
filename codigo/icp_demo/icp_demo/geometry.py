"""Utilidades geometricas: transformaciones rigidas, normales y metricas de error."""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Transformaciones rigidas (matrices homogeneas 4x4)
# --------------------------------------------------------------------------- #
def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def apply_transform(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Aplica la transformacion rigida T a la nube P (n,3)."""
    return P @ T[:3, :3].T + T[:3, 3]


def transform_normals(T: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Para transformaciones rigidas basta rotar (R es ortogonal)."""
    return N @ T[:3, :3].T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return make_transform(R.T, -R.T @ t)


def rodrigues(omega: np.ndarray) -> np.ndarray:
    """Exponencial de so(3): vector de rotacion -> matriz de rotacion."""
    theta = float(np.linalg.norm(omega))
    if theta < 1e-14:
        return np.eye(3)
    k = omega / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def rotation_angle(R: np.ndarray) -> float:
    """Angulo (en radianes) de la rotacion R."""
    c = (np.trace(R[:3, :3]) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def project_to_rotation(M: np.ndarray) -> np.ndarray:
    """Proyecta una matriz al grupo SO(3) (util tras linealizar)."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def random_rotation(rng: np.random.Generator, angle_deg: float,
                    axis: np.ndarray | None = None) -> np.ndarray:
    """Rotacion de magnitud fija `angle_deg` alrededor de un eje (aleatorio si None)."""
    if axis is None:
        axis = rng.normal(size=3)
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-20)
    return rodrigues(axis * np.deg2rad(angle_deg))


# --------------------------------------------------------------------------- #
# Metricas de error respecto al ground truth
# --------------------------------------------------------------------------- #
def pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    """Error de pose: (angulo en grados, norma de traslacion).

    Se mide la transformacion residual T_gt^-1 * T_est: si el registro es
    perfecto la residual es la identidad.
    """
    D = invert_transform(T_gt) @ T_est
    return np.rad2deg(rotation_angle(D)), float(np.linalg.norm(D[:3, 3]))


def rmse_ground_truth(P_src: np.ndarray, T_est: np.ndarray, T_gt: np.ndarray) -> float:
    """RMSE punto a punto usando las correspondencias verdaderas (no las estimadas).

    Es la metrica honesta: el RMSE que reporta ICP internamente usa sus propias
    correspondencias y puede ser bajo aunque el registro este mal.
    """
    A = apply_transform(T_est, P_src)
    B = apply_transform(T_gt, P_src)
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


# --------------------------------------------------------------------------- #
# Normales estimadas desde la nube (PCA local)
# --------------------------------------------------------------------------- #
def estimate_normals(P: np.ndarray, k: int = 20,
                     orient_reference: np.ndarray | None = None) -> np.ndarray:
    """Estima normales por PCA sobre los k vecinos mas cercanos de cada punto.

    Asi es como se obtienen normales para datos de escaner reales (una nube
    cruda no trae normales). El vector propio de menor valor propio de la
    matriz de covarianza local es la normal del plano tangente ajustado.

    orient_reference: si se entrega (n,3), se usa para fijar el signo de cada
    normal; si no, se orientan hacia afuera del centroide de la nube.
    """
    from scipy.spatial import cKDTree

    k = int(max(3, min(k, len(P))))
    tree = cKDTree(P)
    _, idx = tree.query(P, k=k, workers=-1)
    nbrs = P[idx]                                  # (n, k, 3)
    nbrs = nbrs - nbrs.mean(axis=1, keepdims=True)
    # covarianzas locales (n,3,3) y vector propio menor
    C = np.einsum("nki,nkj->nij", nbrs, nbrs) / k
    _, eigvec = np.linalg.eigh(C)                  # ascendente
    N = eigvec[:, :, 0]

    if orient_reference is not None:
        flip = np.sum(N * orient_reference, axis=1) < 0
    else:
        outward = P - P.mean(axis=0)
        flip = np.sum(N * outward, axis=1) < 0
    N[flip] *= -1
    return N
