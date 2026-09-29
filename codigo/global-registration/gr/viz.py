"""Figuras (matplotlib, sin ventana) para inspeccionar cada etapa."""
from __future__ import annotations

import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import open3d as o3d                     # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection   # noqa: E402


def _equal_axes(ax, P):
    lo, hi = P.min(axis=0), P.max(axis=0)
    c = (lo + hi) / 2
    r = (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def mesh_scalar(ax, mesh, scalar, cmap="jet", keypoints=None, view=(15, -60)):
    """Malla coloreada por un campo escalar por vertice (rango percentilizado)."""
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    r = np.argsort(np.argsort(scalar)) / max(1, len(scalar) - 1)
    fc = plt.get_cmap(cmap)(r[F].mean(axis=1))
    pc = Poly3DCollection(V[F], facecolors=fc, edgecolors="none",
                          linewidths=0, shade=False)
    ax.add_collection3d(pc)
    if keypoints is not None and len(keypoints):
        ax.scatter(*V[keypoints].T, c="k", s=14, depthshade=False)
    ax.view_init(*view)
    _equal_axes(ax, V)


def cloud(ax, pcd, color=None, s=1.5, view=(15, -60), scalar=None, cmap="jet"):
    P = np.asarray(pcd.points) if hasattr(pcd, "points") else np.asarray(pcd)
    if scalar is not None:
        r = np.argsort(np.argsort(scalar)) / max(1, len(scalar) - 1)
        ax.scatter(*P.T, c=r, cmap=cmap, s=s, depthshade=False, linewidths=0)
    else:
        ax.scatter(*P.T, c=color, s=s, depthshade=False, linewidths=0)
    ax.view_init(*view)
    _equal_axes(ax, P)


def registration(ax, source, target, T=np.eye(4), view=(15, -60), s=1.2):
    src = copy.deepcopy(source).transform(T)
    S, Tt = np.asarray(src.points), np.asarray(target.points)
    ax.scatter(*S.T, c="#e07b39", s=s, depthshade=False, linewidths=0, label="source")
    ax.scatter(*Tt.T, c="#3a7ca5", s=s, depthshade=False, linewidths=0, label="target")
    ax.view_init(*view)
    _equal_axes(ax, np.vstack([S, Tt]))


def correspondences(ax, src_pts, tgt_pts, pairs, inlier_mask=None,
                    source=None, target=None, view=(15, -60), max_lines=120):
    """Dibuja las dos nubes separadas y las lineas de correspondencia."""
    if source is not None:
        ax.scatter(*np.asarray(source.points).T, c="#dddddd", s=0.6,
                   depthshade=False, linewidths=0)
    if target is not None:
        ax.scatter(*np.asarray(target.points).T, c="#bbbbbb", s=0.6,
                   depthshade=False, linewidths=0)
    pairs = np.asarray(pairs)
    sel = np.arange(len(pairs))
    if len(sel) > max_lines:
        sel = np.linspace(0, len(pairs) - 1, max_lines).astype(int)
    for j in sel:
        i, k = pairs[j]
        ok = True if inlier_mask is None else bool(inlier_mask[j])
        ax.plot(*np.array([src_pts[i], tgt_pts[k]]).T,
                c="#2ca02c" if ok else "#d62728", lw=0.7, alpha=0.85)
    ax.view_init(*view)
    pts = np.vstack([src_pts, tgt_pts])
    _equal_axes(ax, pts)


def save(fig, path, dpi=120):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
