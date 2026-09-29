"""Orquesta: escena -> normales -> campo implicito -> malla -> metricas."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import methods as M
from .data import Scene
from .extract import marching_cubes, marching_tetrahedra
from .grid import Grid
from .metrics import accuracy, completeness, mesh_stats, topology_label
from .normals import NormalParams, NormalResult, estimate

ALL_METHODS = ["Hoppe", "RBF", "Poisson", "Open3D"]
COVER_TAU = 0.02     # una muestra real esta "cubierta" si hay malla a < 2 % diag


@dataclass
class MethodResult:
    name: str
    field: M.FieldResult
    V: np.ndarray
    T: np.ndarray
    stats: dict
    err: np.ndarray               # por vertice, fraccion de la diagonal
    comp: np.ndarray              # por muestra real, fraccion de la diagonal
    t_extract: float = 0.0
    summary: dict = field(default_factory=dict)


def run_normals(scene: Scene, prm: NormalParams) -> NormalResult:
    return estimate(scene.P, scene.N_true, scene.is_outlier, prm, seed=scene.params.seed)


def build_grid(P: np.ndarray, res: int) -> Grid:
    # margen amplio: Poisson por FFT es periodico y necesita "aire" alrededor
    return Grid.from_bounds(P.min(0), P.max(0), res, pad=0.10)


def _finish(name, fr, V, T, t_ext, scene) -> MethodResult:
    st = mesh_stats(V, T)
    err = accuracy(scene.surface, V[np.unique(T)] if len(T) else V[:0], scene.diag)
    comp = completeness(scene.gt_samples, V, T, scene.diag)
    s = dict(
        err_med=float(np.median(err)) if len(err) else np.nan,
        err_p95=float(np.percentile(err, 95)) if len(err) else np.nan,
        comp_p95=float(np.percentile(comp, 95)),
        cover=float((comp < COVER_TAU).mean()),
        topo=topology_label(st), genus=st["genus"], comps=st["comps"],
        closed=st["closed"], chi=st["chi"], tris=st["F"],
        t_field=fr.t_field, t_extract=t_ext,
    )
    # error por vertice (en el indice original de V, para colorear)
    e_full = np.zeros(len(V))
    if len(T):
        e_full[np.unique(T)] = err
    return MethodResult(name, fr, V, T, st, e_full, comp, t_ext, s)


def run_method(name: str, scene: Scene, nr: NormalResult, prm: M.ReconParams,
               grid: Grid | None = None) -> MethodResult | None:
    P, N = nr.P, nr.N
    grid = grid or build_grid(P, prm.res)
    if name == "Hoppe":
        fr = M.hoppe(P, N, grid, prm, noise_abs=2 * scene.params.noise * scene.diag)
    elif name == "RBF":
        fr = M.rbf(P, N, grid, prm, scene.diag)
    elif name == "Poisson":
        fr = M.poisson(P, N, grid, prm)
    elif name == "Open3D":
        fr = M.open3d_poisson(P, N, prm)
        if fr is None:
            return None
        V, T = fr.mesh
        return _finish(name, fr, V, T, 0.0, scene)
    else:
        raise ValueError(name)
    t0 = time.perf_counter()
    out = None
    if prm.extractor == "mc":
        out = marching_cubes(fr.vals, grid)
    if out is None:
        out = marching_tetrahedra(fr.vals, grid)
    V, T = out
    return _finish(name, fr, V, T, time.perf_counter() - t0, scene)


def run_all(scene: Scene, nr: NormalResult, prm: M.ReconParams,
            methods=None, verbose: bool = False) -> dict:
    methods = methods or [m for m in ALL_METHODS if m != "Open3D" or M.open3d_available()]
    grid = build_grid(nr.P, prm.res)
    res = {}
    for m in methods:
        r = run_method(m, scene, nr, prm, grid)
        if r is not None:
            res[m] = r
            if verbose:
                print("  " + format_row(r), flush=True)
    return res


def format_header() -> str:
    return (f"{'metodo':8s} {'err med':>8s} {'err p95':>8s} {'compl p95':>9s} "
            f"{'cobert.':>7s} {'tiempo':>7s}  topologia")


def format_row(r: MethodResult) -> str:
    s = r.summary
    return (f"{r.name:8s} {1e3 * s['err_med']:8.2f} {1e3 * s['err_p95']:8.2f} "
            f"{1e3 * s['comp_p95']:9.2f} {100 * s['cover']:6.1f}% "
            f"{s['t_field'] + s['t_extract']:6.2f}s  {s['topo']}")
