"""
Banco de datos con ground-truth para global registration.

Idea: a partir de una malla completa simulamos dos "range scans" por
hidden-point-removal desde dos puntos de vista distintos. Eso produce
    - solapamiento parcial controlado por el angulo entre vistas,
    - auto-oclusion y bordes realistas,
    - y una transformacion rigida exacta como ground truth.

Todo se normaliza para que la diagonal del bounding box de la malla completa
sea 1.0, de modo que los umbrales (voxel, radios, ruido) se expresan como
fraccion del diametro del objeto y son comparables entre modelos.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d


# --------------------------------------------------------------------------- #
# utilidades
# --------------------------------------------------------------------------- #
def normalize_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Centra en el origen y escala para diagonal del bbox = 1."""
    mesh = copy.deepcopy(mesh)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    v = np.asarray(mesh.vertices)
    c = v.mean(axis=0)
    v = v - c
    diag = np.linalg.norm(v.max(axis=0) - v.min(axis=0))
    v = v / diag
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.compute_vertex_normals()
    return mesh


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Rotacion uniforme en SO(3) (via cuaternion aleatorio)."""
    u1, u2, u3 = rng.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def sph2cart(theta_deg: float, phi_deg: float, r: float = 1.0) -> np.ndarray:
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    return r * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


# --------------------------------------------------------------------------- #
# simulacion de range scan
# --------------------------------------------------------------------------- #
def raycast_scan(mesh: o3d.geometry.TriangleMesh,
                 theta: float,
                 phi: float,
                 distance: float = 2.0,
                 width_px: int = 640,
                 fov_deg: float = 45.0) -> np.ndarray:
    """Simula un range scan: lanza rayos desde una camara pinhole y devuelve
    los puntos de impacto (auto-oclusion incluida)."""
    tm = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tm)
    eye = sph2cart(theta, phi, distance)
    up = [0.0, 0.0, 1.0] if abs(np.dot(eye / np.linalg.norm(eye), [0, 0, 1])) < 0.95 \
        else [0.0, 1.0, 0.0]
    rays = scene.create_rays_pinhole(fov_deg=fov_deg, center=[0.0, 0.0, 0.0],
                                     eye=eye, up=up,
                                     width_px=width_px, height_px=width_px)
    ans = scene.cast_rays(rays)
    hit = ans["t_hit"].isfinite()
    pts = rays[hit][:, :3] + rays[hit][:, 3:] * ans["t_hit"][hit].reshape((-1, 1))
    return pts.numpy().astype(np.float64)


def visible_vertices(mesh: o3d.geometry.TriangleMesh,
                     camera: np.ndarray,
                     radius_factor: float = 1000.0) -> np.ndarray:
    """Indices de vertices visibles desde `camera` (Katz et al., HPR)."""
    pcd = o3d.geometry.PointCloud(mesh.vertices)
    diameter = np.linalg.norm(
        np.asarray(pcd.points).max(axis=0) - np.asarray(pcd.points).min(axis=0))
    _, idx = pcd.hidden_point_removal(camera, diameter * radius_factor)
    return np.asarray(idx, dtype=np.int64)


def crop_mesh_to_vertices(mesh: o3d.geometry.TriangleMesh,
                          keep: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Submalla con los triangulos cuyos 3 vertices estan en `keep`."""
    m = copy.deepcopy(mesh)
    mask = np.zeros(len(m.vertices), dtype=bool)
    mask[keep] = True
    tris = np.asarray(m.triangles)
    keep_tri = mask[tris].all(axis=1)
    m.triangles = o3d.utility.Vector3iVector(tris[keep_tri])
    m.triangle_normals = o3d.utility.Vector3dVector(np.empty((0, 3)))
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    return m


def scan_from_view(mesh: o3d.geometry.TriangleMesh,
                   theta: float,
                   phi: float,
                   distance: float = 3.0):
    """Devuelve (submalla visible, camara) desde la direccion (theta, phi)."""
    cam = sph2cart(theta, phi, distance)
    keep = visible_vertices(mesh, cam)
    return crop_mesh_to_vertices(mesh, keep), cam


# --------------------------------------------------------------------------- #
# par de registro
# --------------------------------------------------------------------------- #
@dataclass
class RegistrationPair:
    """Un problema de global registration con ground truth.

    Convencion: T_gt lleva `source` sobre `target`  ->  target ~ T_gt * source
    """
    name: str
    source: o3d.geometry.PointCloud
    target: o3d.geometry.PointCloud
    T_gt: np.ndarray
    overlap: float                      # fraccion de puntos de source con
                                        # correspondencia en target bajo T_gt
    voxel: float                        # resolucion usada (fraccion del diametro)
    source_mesh: o3d.geometry.TriangleMesh | None = None
    target_mesh: o3d.geometry.TriangleMesh | None = None
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"{self.name}: |S|={len(self.source.points)} "
                f"|T|={len(self.target.points)} overlap={self.overlap:.2f}")


def compute_overlap(source: o3d.geometry.PointCloud,
                    target: o3d.geometry.PointCloud,
                    T_gt: np.ndarray,
                    tau: float) -> float:
    s = copy.deepcopy(source).transform(T_gt)
    tree = o3d.geometry.KDTreeFlann(target)
    n_ok = 0
    for p in np.asarray(s.points):
        k, _, d2 = tree.search_radius_vector_3d(p, tau)
        n_ok += int(k > 0)
    return n_ok / max(1, len(s.points))


def perturb(pcd: o3d.geometry.PointCloud,
            rng: np.random.Generator,
            noise_sigma: float = 0.0,
            outlier_ratio: float = 0.0) -> o3d.geometry.PointCloud:
    """Ruido gaussiano isotropico + outliers uniformes dentro del bbox inflado."""
    out = copy.deepcopy(pcd)
    pts = np.asarray(out.points).copy()
    if noise_sigma > 0:
        pts = pts + rng.normal(scale=noise_sigma, size=pts.shape)
    if outlier_ratio > 0:
        n_out = int(round(outlier_ratio * len(pts)))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        pad = 0.05 * (hi - lo)
        extra = rng.uniform(lo - pad, hi + pad, size=(n_out, 3))
        pts = np.vstack([pts, extra])
    out.points = o3d.utility.Vector3dVector(pts)
    return out


def make_pair(mesh: o3d.geometry.TriangleMesh,
              name: str,
              view_a=(70.0, 0.0),
              view_b=(70.0, 60.0),
              voxel: float = 0.01,
              noise_sigma: float = 0.0,
              outlier_ratio: float = 0.0,
              seed: int = 0,
              keep_meshes: bool = True) -> RegistrationPair:
    """Construye un par (source, target) con ground truth exacto.

    `source` = scan desde view_a, transformado por una pose rigida aleatoria.
    `target` = scan desde view_b, en el marco original de la malla.
    """
    rng = np.random.default_rng(seed)

    # nubes: range scan por raycasting (muestreo realista + auto-oclusion)
    pcd_a = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        raycast_scan(mesh, *view_a))).voxel_down_sample(voxel)
    pcd_b = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        raycast_scan(mesh, *view_b))).voxel_down_sample(voxel)

    # mallas parciales (para la version de Harris3D sobre mallas)
    mesh_a, cam_a = scan_from_view(mesh, *view_a) if keep_meshes else (None, None)
    mesh_b, cam_b = scan_from_view(mesh, *view_b) if keep_meshes else (None, None)

    # pose aleatoria aplicada al scan A: lo sacamos del marco comun
    R = random_rotation(rng)
    t = rng.uniform(-0.5, 0.5, size=3)
    T_rand = make_transform(R, t)          # lleva A original -> A observado
    T_gt = np.linalg.inv(T_rand)           # lleva A observado -> marco comun (B)

    src = copy.deepcopy(pcd_a).transform(T_rand)
    tgt = pcd_b

    src = perturb(src, rng, noise_sigma, outlier_ratio)
    tgt = perturb(tgt, rng, noise_sigma, outlier_ratio)

    # normales orientadas hacia el sensor (es lo que entrega un escaner real y
    # evita ambiguedades de signo artificiales entre los dos scans)
    cam_a = sph2cart(view_a[0], view_a[1], 2.0)
    cam_b = sph2cart(view_b[0], view_b[1], 2.0)
    cam_a_obs = (T_rand[:3, :3] @ cam_a) + T_rand[:3, 3]
    for p, cam in ((src, cam_a_obs), (tgt, cam_b)):
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * 4, max_nn=50))
        p.orient_normals_towards_camera_location(cam)

    ov = compute_overlap(src, tgt, T_gt, tau=voxel * 1.5)

    sm = copy.deepcopy(mesh_a).transform(T_rand) if keep_meshes else None
    tm = mesh_b if keep_meshes else None

    return RegistrationPair(
        name=name, source=src, target=tgt, T_gt=T_gt, overlap=ov, voxel=voxel,
        source_mesh=sm, target_mesh=tm,
        meta=dict(view_a=view_a, view_b=view_b, noise_sigma=noise_sigma,
                  outlier_ratio=outlier_ratio, seed=seed))


# --------------------------------------------------------------------------- #
# mallas de referencia
# --------------------------------------------------------------------------- #
_MESH_LOADERS = {
    "bunny": lambda: o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path),
    "armadillo": lambda: o3d.io.read_triangle_mesh(o3d.data.ArmadilloMesh().path),
}


def load_mesh(name: str) -> o3d.geometry.TriangleMesh:
    return normalize_mesh(_MESH_LOADERS[name]())


def load_redwood_fragments():
    """Fragmentos reales RGB-D (Redwood living room) que trae Open3D."""
    d = o3d.data.DemoICPPointClouds()
    return [o3d.io.read_point_cloud(p) for p in d.paths]
