"""
Mallas de control: sintéticas (pequeñas, pensadas para ver las reglas) y modelos reales
descargados de alecjacobson/common-3d-test-models.

Todas se centran y escalan a diagonal de caja = 1, así los errores se leen en ‰ de la
diagonal como en los demos anteriores.
"""
import os
import numpy as np

from .mesh import PolyMesh, MeshError, remove_unused_vertices, split_nonmanifold_vertices

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")
BASE_URL = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/"

REMOTE = {
    "suzanne": ("suzanne.obj", "mono de Blender: 468 quads + 32 triangulos, con bordes (ojos)"),
    "fandisk": ("fandisk.obj", "pieza CAD con aristas vivas (pruebe --crease 40)"),
    "cow": ("cow.obj", "vaca de Garland-Heckbert, 5804 triangulos"),
    "spot": ("spot.obj", "vaca Spot de K. Crane, 5856 triangulos"),
}


# ------------------------------------------------------------------ sintéticos
def cube():
    V = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    # índice = 4x + 2y + z ; caras orientadas hacia afuera
    F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return V, F


def tetrahedron():
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    F = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]
    return V, F


def octahedron():
    V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    F = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
    return V, F


def icosahedron():
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
         [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
         [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    return V, F


def prism(n=5, h=1.0):
    """Prisma de base n-gonal: dos n-gonos + n quads (caras extraordinarias)."""
    a = 2 * np.pi * np.arange(n) / n
    V = np.concatenate([np.stack([np.cos(a), np.sin(a), np.zeros(n)], 1),
                        np.stack([np.cos(a), np.sin(a), np.full(n, h)], 1)])
    F = [list(range(n))[::-1], list(range(n, 2 * n))]
    F += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return V, F


def torus(n=8, m=4, R=1.0, r=0.45):
    """Toro de n x m cuadriláteros (todos los vértices regulares, valencia 4)."""
    u = 2 * np.pi * np.arange(n) / n
    v = 2 * np.pi * np.arange(m) / m
    U, W = np.meshgrid(u, v, indexing="ij")
    X = (R + r * np.cos(W)) * np.cos(U)
    Y = (R + r * np.cos(W)) * np.sin(U)
    Z = r * np.sin(W)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    idx = lambda i, j: (i % n) * m + (j % m)
    F = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
         for i in range(n) for j in range(m)]
    return V, F


def voxels(cells):
    """
    Malla de cuadriláteros del borde de un conjunto de cubos unitarios (polycubo).
    Da vértices de valencia 3, 4, 5 y 6: ideal para ver vértices extraordinarios.
    """
    cells = set(map(tuple, cells))
    quads = []
    dirs = [((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
            ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
            ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
            ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
            ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
            ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)])]
    for c in cells:
        for d, corners in dirs:
            if (c[0] + d[0], c[1] + d[1], c[2] + d[2]) not in cells:
                quads.append([(c[0] + a, c[1] + b, c[2] + e) for a, b, e in corners])
    keys = {}
    F = [[keys.setdefault(p, len(keys)) for p in q] for q in quads]
    V = np.array(sorted(keys, key=keys.get), float)
    return V, F


def frame_cells():
    """Marco cuadrado 3x3 con el centro vacío: género 1."""
    return [(i, j, 0) for i in range(3) for j in range(3) if (i, j) != (1, 1)]


def l_cells():
    return [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 2, 0)]


def patch(n=6, seed=0):
    """Parche abierto n x n de quads con relieve: muestra las reglas de borde."""
    x = np.linspace(-1, 1, n + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Z = 0.35 * np.exp(-((X - 0.2) ** 2 + (Y + 0.1) ** 2) * 3) \
        - 0.25 * np.exp(-((X + 0.5) ** 2 + (Y - 0.5) ** 2) * 6)
    rng = np.random.default_rng(seed)
    Z = Z + 0.03 * rng.standard_normal(Z.shape)
    V = np.stack([X, Z, Y], -1).reshape(-1, 3)          # altura en y (eje "arriba")
    idx = lambda i, j: i * (n + 1) + j
    F = [[idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)]
         for i in range(n) for j in range(n)]
    return V, F


def spiky():
    """Cubo con una cara empujada hacia afuera y con una tapa pentagonal y un triángulo."""
    V, F = prism(5, 1.2)
    V = V.copy()
    V[5:, :2] *= 0.55                     # tapa superior más chica: forma de "bote"
    return V, F


SYNTHETIC = {
    "cube": (cube, "cubo: 6 quads, 8 vertices de valencia 3"),
    "tetra": (tetrahedron, "tetraedro: 4 triangulos, valencia 3"),
    "octa": (octahedron, "octaedro: 8 triangulos, valencia 4"),
    "icosa": (icosahedron, "icosaedro: 20 triangulos, valencia 5"),
    "prism5": (lambda: prism(5), "prisma pentagonal: 2 pentagonos + 5 quads"),
    "can": (lambda: prism(8, 1.3), "prisma octogonal (lata): pruebe --crease 60 (bordes vivos)"),
    "boat": (spiky, "prisma pentagonal ahusado (tapa chica)"),
    "torus": (torus, "toro 8x4 quads: todo regular (B-spline pura)"),
    "L": (lambda: voxels(l_cells()), "polycubo en L: valencias 3, 4, 5"),
    "frame": (lambda: voxels(frame_cells()), "marco 3x3 hueco: genero 1, valencias 3-6"),
    "patch": (patch, "parche abierto 6x6 con relieve: reglas de borde"),
}


# ------------------------------------------------------------------ OBJ
def load_obj(path):
    V, F = [], []
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                ids = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    ids.append(i - 1 if i > 0 else len(V) + i)
                # quitar repeticiones consecutivas
                clean = [v for k, v in enumerate(ids) if v != ids[k - 1]]
                if len(clean) >= 3:
                    F.append(clean)
    return np.array(V, float), F


def save_obj(path, mesh):
    with open(path, "w") as fh:
        for p in mesh.V:
            fh.write(f"v {p[0]:.7f} {p[1]:.7f} {p[2]:.7f}\n")
        for f in mesh.face_list():
            fh.write("f " + " ".join(str(i + 1) for i in f) + "\n")


def download(name, insecure=False, verbose=True):
    from .download import fetch
    fname = REMOTE[name][0]
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        if verbose:
            print(f"descargando {fname} ...")
        fetch(BASE_URL + fname, path, insecure=insecure)
    return path


# ------------------------------------------------------------------ interfaz
def normalize(V):
    V = V - 0.5 * (V.min(0) + V.max(0))
    return V / np.linalg.norm(V.max(0) - V.min(0))


def build_mesh(V, F, name):
    V, F = remove_unused_vertices(np.asarray(V, float), F)
    m = PolyMesh(normalize(V), F, name=name)
    m, n_split = split_nonmanifold_vertices(m)
    if n_split:
        print(f"[{name}] {n_split} vertices no-variedad separados")
    return m


def load_model(name, insecure=False):
    if name in SYNTHETIC:
        V, F = SYNTHETIC[name][0]()
        return build_mesh(V, F, name)
    if name in REMOTE:
        path = download(name, insecure=insecure)
        V, F = load_obj(path)
        return build_mesh(V, F, name)
    if os.path.exists(name):
        V, F = load_obj(name)
        return build_mesh(V, F, os.path.splitext(os.path.basename(name))[0])
    raise ValueError(f"modelo desconocido: {name}")


ALL_MODELS = list(SYNTHETIC) + list(REMOTE)
