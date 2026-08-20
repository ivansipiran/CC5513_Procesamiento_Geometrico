"""Descarga y carga de modelos 3D reales, y muestreo de nubes de puntos.

Los modelos provienen del repositorio publico de mallas de prueba
"common-3d-test-models" (Alec Jacobson), que reune clasicos del Stanford 3D
Scanning Repository (bunny, armadillo, happy buddha) y otros.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from dataclasses import dataclass

import numpy as np

_BASE = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data"

# nombre -> (archivo remoto, descripcion)
MODELS = {
    "bunny":     ("stanford-bunny.obj", "Stanford Bunny (~35k vertices) - el clasico de registro"),
    "armadillo": ("armadillo.obj",      "Stanford Armadillo (~50k vertices) - geometria con mucho detalle"),
    "happy":     ("happy.obj",          "Happy Buddha (~50k vertices) - superficie con partes finas"),
    "max-planck": ("max-planck.obj",    "Busto de Max Planck (~50k vertices)"),
    "nefertiti": ("nefertiti.obj",      "Busto de Nefertiti (~50k vertices) - escaneo real"),
    "cow":       ("cow.obj",            "Vaca (~3k vertices) - modelo pequeno para pruebas rapidas"),
    "fandisk":   ("fandisk.obj",        "Fandisk (~6k vertices) - CAD, muchas zonas planas"),
}

DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def default_cache_dir() -> str:
    return os.environ.get("ICP_DEMO_DATA", DEFAULT_CACHE)


def model_path(name: str, cache_dir: str | None = None, download: bool = True) -> str:
    """Devuelve la ruta local del modelo, descargandolo si hace falta."""
    if name not in MODELS:
        raise KeyError(
            f"Modelo desconocido '{name}'. Disponibles: {', '.join(sorted(MODELS))}"
        )
    cache_dir = cache_dir or default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    fname = MODELS[name][0]
    path = os.path.join(cache_dir, fname)
    if not os.path.exists(path):
        if not download:
            raise FileNotFoundError(path)
        url = f"{_BASE}/{fname}"
        print(f"[data] descargando {name} desde {url}", file=sys.stderr)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
        print(f"[data] guardado en {path}", file=sys.stderr)
    return path


# --------------------------------------------------------------------------- #
# Lectura de mallas
# --------------------------------------------------------------------------- #
@dataclass
class Mesh:
    V: np.ndarray  # (n, 3) vertices
    F: np.ndarray  # (m, 3) triangulos

    @property
    def n_vertices(self) -> int:
        return len(self.V)

    @property
    def n_faces(self) -> int:
        return len(self.F)


def read_obj(path: str) -> Mesh:
    """Lector minimo de OBJ (v / f). Triangula poligonos por abanico."""
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                # formatos: "i", "i/j", "i//k", "i/j/k"
                idx = [int(p.split("/")[0]) for p in parts]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):  # abanico
                    faces.append((idx[0], idx[k], idx[k + 1]))
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), np.int64)
    if len(V) == 0:
        raise ValueError(f"No se leyeron vertices de {path}")
    return Mesh(V=V, F=F)


def load_mesh(name_or_path: str, cache_dir: str | None = None) -> Mesh:
    """Acepta un nombre de MODELS o una ruta a un .obj propio."""
    if name_or_path in MODELS:
        path = model_path(name_or_path, cache_dir)
    else:
        path = name_or_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{name_or_path}' no es un modelo conocido ni una ruta existente."
            )
    return read_obj(path)


# --------------------------------------------------------------------------- #
# Normalizacion y muestreo
# --------------------------------------------------------------------------- #
def normalize_unit(V: np.ndarray) -> np.ndarray:
    """Centra en el origen y escala a diagonal de bounding box = 1.

    Trabajar en escala unitaria hace que los umbrales (ruido, distancia maxima
    de correspondencia) sean interpretables y comparables entre modelos.
    """
    V = V - V.mean(axis=0)
    diag = np.linalg.norm(V.max(axis=0) - V.min(axis=0))
    return V / max(diag, 1e-12)


def face_normals(mesh: Mesh) -> np.ndarray:
    v0 = mesh.V[mesh.F[:, 0]]
    v1 = mesh.V[mesh.F[:, 1]]
    v2 = mesh.V[mesh.F[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-20)


def sample_surface(mesh: Mesh, n: int, rng: np.random.Generator):
    """Muestreo uniforme por area sobre los triangulos.

    Devuelve (puntos (n,3), normales exactas (n,3)).
    Si la malla no trae caras, se submuestrean los vertices.
    """
    if mesh.n_faces == 0:
        idx = rng.choice(mesh.n_vertices, size=min(n, mesh.n_vertices), replace=False)
        P = mesh.V[idx]
        return P, np.zeros_like(P)

    v0 = mesh.V[mesh.F[:, 0]]
    v1 = mesh.V[mesh.F[:, 1]]
    v2 = mesh.V[mesh.F[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    total = area.sum()
    if total <= 0:
        raise ValueError("Malla degenerada (area total nula)")
    prob = area / total

    fidx = rng.choice(len(area), size=n, p=prob)
    # coordenadas baricentricas uniformes en el triangulo
    r1 = np.sqrt(rng.random(n))[:, None]
    r2 = rng.random(n)[:, None]
    P = (1 - r1) * v0[fidx] + r1 * (1 - r2) * v1[fidx] + r1 * r2 * v2[fidx]

    N = cross[fidx]
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-20)
    return P, N
