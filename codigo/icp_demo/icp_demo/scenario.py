"""Construccion del problema de registro a partir de un modelo real.

A partir de una malla se generan dos nubes de puntos:

  * OBJETIVO (target): nube fija, con normales (exactas de la malla o estimadas
    por PCA como ocurriria con un escaner real).
  * FUENTE (source): la misma superficie remuestreada, desplazada por una
    transformacion rigida conocida (ground truth), opcionalmente con ruido
    gaussiano y con solapamiento parcial (se recorta media nube).

Remuestrear la fuente de forma independiente es importante: si ambas nubes
tuvieran exactamente los mismos puntos el problema seria artificialmente facil
(existiria la correspondencia perfecta).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import data as data_mod
from .geometry import (apply_transform, estimate_normals, make_transform,
                       random_rotation, transform_normals)


@dataclass
class Scenario:
    source: np.ndarray
    target: np.ndarray
    source_normals: np.ndarray
    target_normals: np.ndarray
    T_gt: np.ndarray            # lleva la fuente sobre el objetivo
    model: str
    description: str = ""


def build_scenario(model: str = "bunny",
                   n_points: int = 8000,
                   n_target: int | None = None,
                   angle_deg: float = 30.0,
                   axis: np.ndarray | None = None,
                   translation: float = 0.15,
                   noise: float = 0.002,
                   overlap: float = 1.0,
                   estimate_target_normals: bool = False,
                   normal_k: int = 20,
                   seed: int = 0,
                   cache_dir: str | None = None) -> Scenario:
    """Genera el par de nubes a registrar.

    n_points   : puntos de la nube fuente
    n_target   : puntos del objetivo (por defecto igual que la fuente)
    angle_deg  : magnitud de la rotacion inicial de desalineacion
    translation: magnitud de la traslacion (en unidades de diagonal del modelo)
    noise      : sigma del ruido gaussiano agregado a la fuente
    overlap    : fraccion de la fuente conservada (recorte por un plano) para
                 simular solapamiento parcial entre vistas
    """
    rng = np.random.default_rng(seed)
    mesh = data_mod.load_mesh(model, cache_dir)
    mesh.V = data_mod.normalize_unit(mesh.V)

    n_target = n_target or n_points
    tgt, tgt_n = data_mod.sample_surface(mesh, n_target, rng)
    src, src_n = data_mod.sample_surface(mesh, n_points, rng)

    # solapamiento parcial: se corta la fuente por un plano aleatorio
    if overlap < 1.0:
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        proj = src @ d
        thr = np.quantile(proj, overlap)
        keep = proj <= thr
        src, src_n = src[keep], src_n[keep]

    if noise > 0:
        src = src + rng.normal(scale=noise, size=src.shape)

    # transformacion verdadera aplicada a la fuente (lo que ICP debe deshacer
    # es su inversa; aqui guardamos la que lleva la fuente perturbada al objetivo)
    R = random_rotation(rng, angle_deg, axis)
    t = rng.normal(size=3)
    t = t / max(np.linalg.norm(t), 1e-20) * translation
    T_perturb = make_transform(R, t)

    src_moved = apply_transform(T_perturb, src)
    src_n_moved = transform_normals(T_perturb, src_n)
    T_gt = np.linalg.inv(T_perturb)     # la solucion que ICP debe encontrar

    if estimate_target_normals:
        tgt_n = estimate_normals(tgt, k=normal_k, orient_reference=tgt_n)

    desc = (f"{model}: fuente {len(src_moved)} pts / objetivo {len(tgt)} pts, "
            f"rot {angle_deg:.0f} deg, trans {translation:.3f}, "
            f"ruido {noise:g}, solapamiento {overlap:.2f}")

    return Scenario(source=src_moved, target=tgt,
                    source_normals=src_n_moved, target_normals=tgt_n,
                    T_gt=T_gt, model=model, description=desc)
