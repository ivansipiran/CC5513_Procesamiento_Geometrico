"""Datos: superficie implicita de referencia, modelos reales y escenarios.

Dos tipos de "verdad" (ground truth):

* ``blob`` -- la misma idea del tutorial: F(x) = t - sum_i w_i exp(-|x-c_i|^2/s_i^2)
  con 6 centros en anillo + 2 fuera de el. Superficie de GENERO 1.
  Convencion en todo el demo: F < 0 dentro, F > 0 fuera (grad F apunta afuera).
  Da normales exactas (gradiente analitico) y distancia a la superficie de
  cualquier punto, |F|/|grad F|.

* modelos de malla (Stanford, via alecjacobson/common-3d-test-models) -- se
  muestrean puntos sobre la malla con la normal de la cara; la distancia a la
  superficie se aproxima con una nube densa + distancia al plano tangente.

Todas las magnitudes del escenario (ruido, radio del hueco) se dan como
FRACCION DE LA DIAGONAL de la caja envolvente, asi los mismos numeros sirven
para cualquier modelo.
"""

from __future__ import annotations

import os
import ssl
import urllib.request
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(os.path.dirname(HERE), "data")
BASE_URL = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/"

MODELS = {
    # nombre      archivo               descripcion
    "blob":      (None,                 "superficie implicita de genero 1 (la del tutorial); F exacta"),
    "bunny":     ("stanford-bunny.obj", "conejo de Stanford (malla con agujeros en la base)"),
    "armadillo": ("armadillo.obj",      "armadillo de Stanford (cerrado, mucho detalle)"),
    "planck":    ("max-planck.obj",     "busto de Max Planck (detalle fino, base plana)"),
    "fandisk":   ("fandisk.obj",        "pieza CAD con aristas vivas (Hoppe 1994)"),
}

INSECURE_DOWNLOADS = False  # se activa con --insecure


# --------------------------------------------------------------------------- #
# Superficie implicita de referencia
# --------------------------------------------------------------------------- #
class BlobSurface:
    """Metaballs gaussianas: anillo de 6 centros + 2 lobulos. Genero 1."""

    def __init__(self) -> None:
        ang = np.deg2rad(np.arange(6) * 60.0 + 7.0)
        ring = np.c_[np.cos(ang), 0.92 * np.sin(ang), 0.10 * np.sin(2 * ang)]
        extra = np.array([[1.62, 0.55, 0.22],     # lobulo exterior
                          [-0.35, -1.35, 0.38]])  # lobulo inferior, fuera del plano
        self.c = np.vstack([ring, extra])
        self.w = np.array([1.00, 0.92, 1.08, 0.95, 1.05, 0.90, 0.80, 0.75])
        self.s = np.array([0.56, 0.52, 0.58, 0.54, 0.57, 0.53, 0.42, 0.38])
        self.t = 0.50
        self.genus = 1
        # el hueco se abre sobre el tubo, en el lado opuesto a los lobulos
        a = np.deg2rad(187.0)
        self.hole_hint = np.array([np.cos(a), 0.92 * np.sin(a), 1.5])

    def F(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        d2 = ((X[:, None, :] - self.c[None]) ** 2).sum(-1)
        return self.t - (self.w * np.exp(-d2 / self.s ** 2)).sum(1)

    def grad(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        D = X[:, None, :] - self.c[None]                    # (n, m, 3)
        e = self.w * np.exp(-(D ** 2).sum(-1) / self.s ** 2)  # (n, m)
        return (e[:, :, None] * 2.0 * D / (self.s ** 2)[None, :, None]).sum(1)

    def distance(self, X: np.ndarray) -> np.ndarray:
        """Distancia (aprox. de primer orden) a la superficie F = 0."""
        out = np.empty(len(X))
        for a in range(0, len(X), 20000):
            Xa = X[a:a + 20000]
            out[a:a + 20000] = np.abs(self.F(Xa)) / np.maximum(
                np.linalg.norm(self.grad(Xa), axis=1), 1e-12)
        return out

    def project(self, X: np.ndarray, iters: int = 8) -> np.ndarray:
        """Proyeccion de Newton sobre F = 0."""
        X = X.copy()
        for _ in range(iters):
            f = self.F(X)
            g = self.grad(X)
            X -= (f / np.maximum((g * g).sum(1), 1e-12))[:, None] * g
        return X

    def normals(self, X: np.ndarray) -> np.ndarray:
        g = self.grad(X)
        return g / np.linalg.norm(g, axis=1, keepdims=True)

    def sample(self, n: int, rng: np.random.Generator, relax: int = 12) -> np.ndarray:
        """Muestras casi uniformes: proyeccion de Newton + relajacion tangencial."""
        lo, hi = self.c.min(0) - 0.9, self.c.max(0) + 0.9
        pts = np.empty((0, 3))
        while len(pts) < 3 * n:  # candidatos cerca de la superficie
            X = rng.uniform(lo, hi, size=(12 * n, 3))
            f = self.F(X)
            pts = np.vstack([pts, X[np.abs(f) < 0.12]])
        X = self.project(pts[rng.choice(len(pts), n, replace=False)])
        # relajacion: cada punto se aleja de sus vecinos en el plano tangente
        for _ in range(relax):
            tree = cKDTree(X)
            d, idx = tree.query(X, k=7)
            h = np.median(d[:, 1])
            D = X[:, None, :] - X[idx[:, 1:]]                       # (n,6,3)
            wgt = np.exp(-(d[:, 1:] / h) ** 2)[:, :, None]
            push = (wgt * D / np.maximum(d[:, 1:, None], 1e-12)).sum(1)
            Nn = self.normals(X)
            push -= (push * Nn).sum(1, keepdims=True) * Nn
            X = self.project(X + 0.25 * h * push, iters=3)
        return X

    def reference_mesh(self, res: int = 110):
        """Malla de referencia (marching tetrahedra sobre la F exacta)."""
        from .extract import marching_tetrahedra
        from .grid import Grid
        lo, hi = self.c.min(0) - 0.8, self.c.max(0) + 0.8
        g = Grid.from_bounds(lo, hi, res)
        vals = self.F(g.points()).reshape(g.shape)
        return marching_tetrahedra(vals, g)


# --------------------------------------------------------------------------- #
# Mallas
# --------------------------------------------------------------------------- #
def _urlopen_chain(url: str, timeout: float = 60.0) -> bytes:
    """Descarga con varias estrategias (evita el bug SSL de Windows/conda)."""
    errors = []
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=timeout, context=ctx) as r:
            return r.read()
    except Exception as e:  # noqa: BLE001
        errors.append(f"certifi: {e}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except Exception as e:  # noqa: BLE001
        errors.append(f"sistema: {e}")
    try:
        import requests
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception as e:  # noqa: BLE001
        errors.append(f"requests: {e}")
    if INSECURE_DOWNLOADS:
        ctx = ssl._create_unverified_context()  # noqa: S323
        with urllib.request.urlopen(url, timeout=timeout, context=ctx) as r:
            return r.read()
    raise RuntimeError("No se pudo descargar " + url + "\n  " + "\n  ".join(errors)
                       + "\n  (pruebe con --insecure o descargue el archivo a mano en "
                       + CACHE_DIR + ")")


def fetch_model(name: str) -> str:
    fname = MODELS[name][0]
    path = os.path.join(CACHE_DIR, fname)
    if not os.path.exists(path):
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"[datos] descargando {fname} ...", flush=True)
        data = _urlopen_chain(BASE_URL + fname)
        with open(path, "wb") as f:
            f.write(data)
    return path


def load_obj(path: str):
    V, F = [], []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                ids = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                ids = [i - 1 if i > 0 else len(V) + i for i in ids]
                for j in range(1, len(ids) - 1):
                    F.append([ids[0], ids[j], ids[j + 1]])
    return np.asarray(V, float), np.asarray(F, np.int64)


class MeshSurface:
    """Superficie de referencia dada por una malla (normalizada: diag = 1)."""

    def __init__(self, V: np.ndarray, F: np.ndarray, rng: np.random.Generator,
                 n_dense: int = 400_000) -> None:
        lo, hi = V.min(0), V.max(0)
        V = (V - 0.5 * (lo + hi)) / np.linalg.norm(hi - lo)
        self.V, self.F = V, F
        self.genus = None
        tri = V[F]
        cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.area = 0.5 * np.linalg.norm(cr, axis=1)
        keep = self.area > 1e-14
        self.F, self.area, cr = self.F[keep], self.area[keep], cr[keep]
        self.fn = cr / np.linalg.norm(cr, axis=1, keepdims=True)
        self.dense, self.dense_n = self._sample(n_dense, rng)
        self.tree = cKDTree(self.dense)
        self.h_dense = np.sqrt(self.area.sum() / n_dense)

    def _sample(self, n: int, rng: np.random.Generator):
        fi = rng.choice(len(self.F), n, p=self.area / self.area.sum())
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        T = self.V[self.F[fi]]
        X = T[:, 0] + u[:, None] * (T[:, 1] - T[:, 0]) + v[:, None] * (T[:, 2] - T[:, 0])
        return X, self.fn[fi]

    def sample(self, n: int, rng: np.random.Generator, relax: int = 0):
        return self._sample(n, rng)

    def distance(self, X: np.ndarray) -> np.ndarray:
        d, j = self.tree.query(X, workers=-1)
        D = X - self.dense[j]
        normal = np.abs((D * self.dense_n[j]).sum(1))
        tang = np.sqrt(np.maximum(d ** 2 - normal ** 2, 0))
        # cerca de la superficie, la distancia al plano tangente es mucho mas
        # precisa que la distancia al punto denso mas cercano
        return np.where(tang < 2 * self.h_dense, normal, d)

    def reference_mesh(self, res: int = 0):
        return self.V, self.F


# --------------------------------------------------------------------------- #
# Escenario (la nube de entrada)
# --------------------------------------------------------------------------- #
@dataclass
class SceneParams:
    model: str = "blob"
    n: int = 8000
    noise: float = 0.0013      # sigma del ruido gaussiano, fraccion de la diagonal
    outliers: float = 0.0      # fraccion de puntos espurios (uniformes en la caja)
    hole: float = 0.0          # radio del hueco, fraccion de la diagonal
    seed: int = 0


@dataclass
class Scene:
    params: SceneParams
    surface: object
    P: np.ndarray                  # nube de entrada (n,3)
    N_true: np.ndarray             # normal verdadera (NaN en outliers)
    is_outlier: np.ndarray
    diag: float
    n_hole_removed: int = 0
    gt_samples: np.ndarray = field(default=None)   # muestras densas de la verdad
    ref_mesh: tuple = None

    @property
    def lo(self):
        return self.P.min(0)

    @property
    def hi(self):
        return self.P.max(0)


_SURF_CACHE: dict = {}


def get_surface(model: str, seed: int = 0):
    if model not in _SURF_CACHE:
        if model == "blob":
            _SURF_CACHE[model] = BlobSurface()
        else:
            V, F = load_obj(fetch_model(model))
            _SURF_CACHE[model] = MeshSurface(V, F, np.random.default_rng(1234))
    return _SURF_CACHE[model]


def hole_direction():
    d = np.array([0.08, 0.05, 1.0])
    return d / np.linalg.norm(d)


def make_scene(p: SceneParams) -> Scene:
    rng = np.random.default_rng(p.seed)
    surf = get_surface(p.model)
    if p.model == "blob":
        X = surf.sample(p.n, rng)
        N = surf.normals(X)
    else:
        X, N = surf.sample(p.n, rng)
    lo, hi = X.min(0), X.max(0)
    diag = float(np.linalg.norm(hi - lo))

    n_removed = 0
    if p.hole > 0:
        # "tapa" que el escaner no vio: puntos a menos de r del punto mas alto
        # Y en el mismo lado de la superficie (normal parecida), asi la
        # superficie de abajo sigue ahi y el metodo tiene que tender un puente
        hint = getattr(surf, "hole_hint", None)
        i = (np.argmin(np.linalg.norm(X - hint, axis=1)) if hint is not None
             else np.argmax(X @ hole_direction()))
        c, nc = X[i], N[i]
        keep = (np.linalg.norm(X - c, axis=1) > p.hole * diag) | (N @ nc < 0.2)
        n_removed = int((~keep).sum())
        X, N = X[keep], N[keep]

    if p.noise > 0:
        X = X + rng.normal(scale=p.noise * diag, size=X.shape)

    n_out = int(round(p.outliers * len(X)))
    is_out = np.zeros(len(X), bool)
    if n_out:
        pad = 0.05 * (hi - lo)
        O = rng.uniform(lo - pad, hi + pad, size=(n_out, 3))
        X = np.vstack([X, O])
        N = np.vstack([N, np.full((n_out, 3), np.nan)])
        is_out = np.r_[is_out, np.ones(n_out, bool)]
        perm = rng.permutation(len(X))
        X, N, is_out = X[perm], N[perm], is_out[perm]

    # muestras densas de la verdad (para medir cobertura / completitud)
    gt = (surf.sample(30000, np.random.default_rng(99), relax=2) if p.model == "blob"
          else surf.sample(30000, np.random.default_rng(99)))
    gt = gt[0] if isinstance(gt, tuple) else gt

    return Scene(params=p, surface=surf, P=X, N_true=N, is_outlier=is_out,
                 diag=diag, n_hole_removed=n_removed, gt_samples=gt)


# Regimenes predefinidos (los del bloque 2.6 del tutorial)
PRESETS = {
    "limpio":        dict(noise=0.0013, outliers=0.0,  hole=0.0,  n=8000),
    "ruido":         dict(noise=0.0053, outliers=0.0,  hole=0.0,  n=8000),
    "hueco":         dict(noise=0.0013, outliers=0.0,  hole=0.24, n=8000),
    "densidad baja": dict(noise=0.0013, outliers=0.0,  hole=0.0,  n=1200),
    "outliers":      dict(noise=0.0013, outliers=0.06, hole=0.0,  n=8000),
}
