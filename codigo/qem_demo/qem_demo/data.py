"""Modelos: mallas reales (descarga bajo demanda) y mallas sintéticas para clase."""
import os
import ssl
import sys
import urllib.request

import numpy as np

from . import meshio

BASE_URL = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# nombre → (archivo en el repositorio, descripción)
REAL_MODELS = {
    "cow": ("cow.obj", "vaca de Garland & Heckbert (5804 caras, la del paper)"),
    "fandisk": ("fandisk.obj", "pieza CAD: caras planas y aristas vivas (12946 caras)"),
    "bunny": ("stanford-bunny.obj", "conejo de Stanford escaneado, con hoyos en la base (69k caras)"),
    "armadillo": ("armadillo.obj", "armadillo de Stanford (100k caras, ~15 s de cálculo)"),
}
SYNTHETIC_MODELS = {
    "terreno": "campo de alturas abierto (borde cuadrado) — restricciones de borde",
    "toro": "toro de género 1 — preservación de topología",
    "piezas": "16 bloques separados por una ranura — pares no conectados (umbral t)",
}
ALL_MODELS = list(REAL_MODELS) + list(SYNTHETIC_MODELS)

INSECURE = False          # --insecure en la línea de comandos


def _download(url, dst):
    """Descarga con varios intentos: certifi → contexto del sistema → requests → sin verificar.

    (En Windows + conda, el contexto SSL por defecto a veces falla al leer el
    almacén de certificados del sistema — por eso el primer intento usa certifi.)"""
    errors = []
    attempts = []
    try:
        import certifi
        attempts.append(("certifi", lambda: ssl.create_default_context(cafile=certifi.where())))
    except ImportError:
        pass
    attempts.append(("sistema", ssl.create_default_context))
    for name, make_ctx in attempts:
        try:
            with urllib.request.urlopen(url, context=make_ctx(), timeout=60) as r:
                data = r.read()
            break
        except Exception as e:          # noqa: BLE001
            errors.append(f"{name}: {e}")
    else:
        data = None
        try:
            import requests
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.content
        except Exception as e:          # noqa: BLE001
            errors.append(f"requests: {e}")
        if data is None and INSECURE:
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=ctx, timeout=60) as r:
                data = r.read()
        if data is None:
            raise RuntimeError("no se pudo descargar " + url + "\n  " + "\n  ".join(errors) +
                               "\n  (pruebe con --insecure o descargue el archivo a " + dst + ")")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(data)


def model_path(name):
    fname = REAL_MODELS[name][0]
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"  descargando {fname} ...", flush=True)
        _download(BASE_URL + fname, path)
    return path


def load_model(name):
    """Devuelve (V, F) limpia y normalizada (diagonal de la caja = 1)."""
    if name in REAL_MODELS:
        V, F = meshio.load_mesh(model_path(name))
    elif name in SYNTHETIC_MODELS:
        V, F = {"terreno": terrain, "toro": torus, "piezas": pieces}[name]()
    elif os.path.exists(name):
        V, F = meshio.load_mesh(name)
    else:
        sys.exit(f"modelo desconocido: {name}. Opciones: {', '.join(ALL_MODELS)} o un archivo .obj/.off/.ply")
    V, F = meshio.clean(V, F)
    return meshio.normalize(V), F


# ---------------------------------------------------------------------------
# Mallas sintéticas
# ---------------------------------------------------------------------------
def grid_faces(nu, nv, wrap_u=False, wrap_v=False):
    """Triangulación de una grilla nu × nv (índice = i*nv + j)."""
    iu = np.arange(nu if wrap_u else nu - 1)
    jv = np.arange(nv if wrap_v else nv - 1)
    I, J = np.meshgrid(iu, jv, indexing="ij")
    I, J = I.ravel(), J.ravel()
    a = I * nv + J
    b = ((I + 1) % nu) * nv + J
    c = ((I + 1) % nu) * nv + (J + 1) % nv
    d = I * nv + (J + 1) % nv
    return np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])


def terrain(n=70, seed=3):
    """Campo de alturas z = f(x, y) sobre [-1,1]², con cerros, un valle y una meseta.

    Es abierto (tiene borde): sirve para ver qué pasa con y sin las
    restricciones de borde de la sección 5 del paper."""
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    rng = np.random.default_rng(seed)
    Z = np.zeros_like(X)
    for _ in range(6):
        cx, cy = rng.uniform(-0.8, 0.8, 2)
        s = rng.uniform(0.15, 0.35)
        Z += rng.uniform(0.1, 0.35) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))
    Z -= 0.25 * np.exp(-((X + 0.2 * Y - 0.1) ** 2) / 0.01)          # valle
    Z = np.maximum(Z, 0.05) + 0.25 * ((np.abs(X - 0.5) < 0.25) & (np.abs(Y + 0.5) < 0.25))  # meseta
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)[:, [0, 2, 1]]   # y hacia arriba
    return V, grid_faces(n, n)


def torus(nu=64, nv=24, R=1.0, r=0.35):
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, W = np.meshgrid(u, v, indexing="ij")
    V = np.stack([(R + r * np.cos(W)) * np.cos(U), r * np.sin(W),
                  (R + r * np.cos(W)) * np.sin(U)], -1).reshape(-1, 3)
    return V, grid_faces(nu, nv, True, True)


def subdivided_cube(n=3):
    """Cubo [-1,1]³ con cada cara dividida en n × n cuadrados (2n² triángulos)."""
    Vl, Fl = [], []
    g = np.linspace(-1, 1, n + 1)
    U, W = np.meshgrid(g, g, indexing="ij")
    U, W = U.ravel(), W.ravel()
    one = np.ones_like(U)
    for axis in range(3):
        for sign in (-1, 1):
            P = np.zeros((len(U), 3))
            a, b = [ax for ax in range(3) if ax != axis]
            P[:, axis], P[:, a], P[:, b] = sign * one, U, W
            Fq = grid_faces(n + 1, n + 1)
            if (sign < 0) != (axis == 1):            # orientar hacia afuera
                Fq = Fq[:, ::-1]
            Fl.append(Fq + len(Vl) * len(U))
            Vl.append(P)
    from .meshio import clean
    return clean(np.concatenate(Vl), np.concatenate(Fl))


def pieces(k=4, gap=0.12, n=3):
    """k × k bloques cúbicos separados por una ranura de ancho `gap` (en semilados).

    Sin pares no conectados cada bloque es una componente aparte y a lo más se
    reduce a un tetraedro (4 caras): el mínimo es 4·k² caras. Con el umbral t del
    paper (t > ranura) los bloques se pueden fundir en un solo sólido."""
    Vc, Fc = subdivided_cube(n)
    Vc = Vc * np.array([1.0, 0.6, 1.0])
    Vl, Fl = [], []
    step = 2 + gap
    for a in range(k):
        for b in range(k):
            Fl.append(Fc + len(Vl) * len(Vc))
            Vl.append(Vc + np.array([a * step, 0, b * step]))
    return np.concatenate(Vl), np.concatenate(Fl)
