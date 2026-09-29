"""
Spin Images  --  Johnson & Hebert, "Using spin images for efficient object
recognition in cluttered 3D scenes", TPAMI 1999.  (Slides 17-24)

Base local en un punto orientado (p, n):
    alpha = sqrt(||q-p||^2 - (n . (q-p))^2)      distancia radial al eje
    beta  = n . (q-p)                            altura sobre el plano tangente

El punto q cae en el bin
    i = floor((W*bin/2 - beta) / bin)            fila
    j = floor(alpha / bin)                       columna

Se usa acumulacion bilineal (como en el paper original) para reducir el ruido
de discretizacion, y un angulo de soporte para descartar puntos cuya normal
difiere demasiado de n (evita mezclar caras opuestas de una superficie fina).

Similitud entre dos spin images (slide 22):
    R(P,Q) = correlacion de Pearson sobre los N bins con datos en ambas
    C(P,Q) = atanh(R)^2 - lambda / (N - 3)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class SpinImageParams:
    bin_size: float = 0.016        # tamano de bin (unidades del objeto)
    image_width: int = 16          # W: numero de bins por lado
    support_angle_deg: float = 60.0
    bilinear: bool = True
    normalize: bool = True         # normalizar a suma 1 (robustez a densidad)

    @property
    def support_radius(self) -> float:
        return self.bin_size * self.image_width * 1.05


def compute_spin_images(points: np.ndarray,
                        normals: np.ndarray,
                        keypoint_idx: np.ndarray,
                        params: SpinImageParams,
                        cloud_points: np.ndarray | None = None,
                        cloud_normals: np.ndarray | None = None) -> np.ndarray:
    """Spin images para los puntos `keypoint_idx` de la nube.

    Devuelve un arreglo (K, W*W) con las imagenes aplanadas (fila 0 = beta max).
    """
    cloud_points = points if cloud_points is None else cloud_points
    cloud_normals = normals if cloud_normals is None else cloud_normals

    W = params.image_width
    b = params.bin_size
    cos_thr = np.cos(np.radians(params.support_angle_deg))

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_points))
    tree = o3d.geometry.KDTreeFlann(pcd)

    out = np.zeros((len(keypoint_idx), W, W), dtype=np.float64)
    for s, i in enumerate(keypoint_idx):
        p = points[i]
        n = normals[i]
        n = n / (np.linalg.norm(n) + 1e-12)

        k, nb, _ = tree.search_radius_vector_3d(p, params.support_radius)
        if k < 3:
            continue
        nb = np.asarray(nb)
        d = cloud_points[nb] - p
        # angulo de soporte
        cs = cloud_normals[nb] @ n
        keep = np.abs(cs) >= cos_thr        # |.| : la orientacion global puede
        d = d[keep]                          #       estar invertida en un scan
        if len(d) < 3:
            continue

        beta = d @ n
        alpha = np.sqrt(np.maximum((d * d).sum(axis=1) - beta**2, 0.0))

        # coordenadas continuas en la grilla
        col = alpha / b                                  # j
        row = (W * b / 2.0 - beta) / b                    # i
        ok = (col >= 0) & (col < W) & (row >= 0) & (row < W)
        col, row = col[ok], row[ok]
        if len(col) == 0:
            continue

        img = out[s]
        if params.bilinear:
            j0 = np.floor(col).astype(int)
            i0 = np.floor(row).astype(int)
            a = col - j0
            c = row - i0
            for di, dj, wgt in ((0, 0, (1 - a) * (1 - c)),
                                (0, 1, a * (1 - c)),
                                (1, 0, (1 - a) * c),
                                (1, 1, a * c)):
                ii, jj = i0 + di, j0 + dj
                m = (ii >= 0) & (ii < W) & (jj >= 0) & (jj < W)
                np.add.at(img, (ii[m], jj[m]), wgt[m])
        else:
            j0 = np.floor(col).astype(int)
            i0 = np.floor(row).astype(int)
            np.add.at(img, (i0, j0), 1.0)

        if params.normalize:
            ssum = img.sum()
            if ssum > 0:
                img /= ssum

    return out.reshape(len(keypoint_idx), -1)


# --------------------------------------------------------------------------- #
# similitud (slide 22)
# --------------------------------------------------------------------------- #
def spin_similarity_matrix(P: np.ndarray, Q: np.ndarray,
                           lam: float = 3.0,
                           min_overlap_bins: int = 8) -> np.ndarray:
    """C(P,Q) = atanh(R)^2 - lambda/(N-3) para cada par (fila de P, fila de Q).

    La correlacion se calcula solo sobre los bins con datos en AMBAS imagenes
    (definicion de Johnson & Hebert); N es ese numero de bins.
    """
    Pm = P > 0
    Qm = Q > 0
    # N: bins solapados  (K1, K2)
    N = Pm.astype(np.float64) @ Qm.astype(np.float64).T

    sp = P @ Qm.T.astype(np.float64)          # sum p_i sobre bins comunes
    sq = Pm.astype(np.float64) @ Q.T          # sum q_i
    spq = P @ Q.T
    sp2 = (P * P) @ Qm.T.astype(np.float64)
    sq2 = Pm.astype(np.float64) @ (Q * Q).T

    num = N * spq - sp * sq
    den = np.sqrt(np.maximum((N * sp2 - sp**2) * (N * sq2 - sq**2), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(den > 1e-15, num / den, 0.0)
    R = np.clip(R, -0.999999, 0.999999)

    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.arctanh(R) ** 2 - lam * np.where(N > 3, 1.0 / (N - 3), np.inf)
    C[N < min_overlap_bins] = -np.inf
    C[~np.isfinite(C)] = -np.inf
    return C


# --------------------------------------------------------------------------- #
# correspondencias
# --------------------------------------------------------------------------- #
def match(C: np.ndarray,
          mode: str = "mutual",
          top_k: int = 1,
          ratio: float | None = None) -> np.ndarray:
    """Extrae correspondencias (i_source, j_target) de la matriz de similitud.

    mode="mutual" : mejor mutuo (mas preciso, menos correspondencias)
    mode="topk"   : los top_k mejores destinos por cada origen
    mode="global" : las mejores correspondencias globalmente
    ratio         : test de Lowe adaptado (mejor vs segundo mejor)
    """
    K1, K2 = C.shape
    best_j = np.argmax(C, axis=1)
    best_v = C[np.arange(K1), best_j]

    if ratio is not None:
        C2 = C.copy()
        C2[np.arange(K1), best_j] = -np.inf
        second = C2.max(axis=1)
        # C mayor = mejor, asi que exigimos margen aditivo relativo
        keep = best_v > second + ratio * np.abs(second + 1e-12)
    else:
        keep = np.isfinite(best_v)

    if mode == "mutual":
        best_i = np.argmax(C, axis=0)
        mutual = best_i[best_j] == np.arange(K1)
        sel = np.flatnonzero(keep & mutual & np.isfinite(best_v))
        pairs = np.stack([sel, best_j[sel]], axis=1)
        order = np.argsort(-best_v[sel])
        return pairs[order]
    if mode == "topk":
        idx = np.argsort(-C, axis=1)[:, :top_k]
        rows = np.repeat(np.arange(K1), top_k)
        pairs = np.stack([rows, idx.ravel()], axis=1)
        vals = C[pairs[:, 0], pairs[:, 1]]
        pairs = pairs[np.isfinite(vals)]
        vals = vals[np.isfinite(vals)]
        return pairs[np.argsort(-vals)]
    if mode == "global":
        sel = np.flatnonzero(keep & np.isfinite(best_v))
        pairs = np.stack([sel, best_j[sel]], axis=1)
        return pairs[np.argsort(-best_v[sel])]
    raise ValueError(mode)


def describe(pcd: o3d.geometry.PointCloud,
             keypoint_idx: np.ndarray,
             params: SpinImageParams) -> np.ndarray:
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    return compute_spin_images(pts, nrm, keypoint_idx, params)
