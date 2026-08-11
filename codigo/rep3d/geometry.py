"""
geometry.py
===========
Núcleo geométrico de la app de representaciones 3D.

Aquí viven los ALGORITMOS DE CONVERSIÓN entre representaciones:

    Malla triangular  ->  Nube de puntos      (muestreo de la superficie)
    Malla triangular  ->  Representación por voxels (voxelización)

El código está escrito para ser LEÍDO en clase: cada función es
autocontenida y la app muestra su código fuente tal cual se ejecuta
(usando inspect.getsource). Por eso se evitan atajos de librerías en
las funciones de conversión y se implementa el algoritmo "a mano".

Autor: material de apoyo para el curso
"Procesamiento Geométrico y Análisis de Formas".
"""

import numpy as np
import trimesh


# ----------------------------------------------------------------------
# Utilidades de mallas
# ----------------------------------------------------------------------

def normalize_mesh(vertices, faces):
    """Centra la malla en el origen y la escala para que quepa en un cubo
    unitario [-0.5, 0.5]^3. Así todas las mallas se ven a una escala
    comparable, sin importar las unidades del archivo original."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    vertices = vertices - center
    scale = np.abs(vertices).max()
    if scale > 0:
        vertices = vertices / (2.0 * scale)   # radio máximo -> 0.5
    return vertices, faces


def load_mesh(path):
    """Carga una malla desde un archivo (.obj, .ply, .stl, ...) y la
    devuelve como (vertices, faces) ya normalizada."""
    mesh = trimesh.load(path, force='mesh')
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError("El archivo no contiene una malla triangular válida.")
    return normalize_mesh(mesh.vertices, mesh.faces)


# ----------------------------------------------------------------------
# Formas de ejemplo generadas por código (no requieren archivos)
# ----------------------------------------------------------------------

def generate_shapes():
    """Devuelve un diccionario {nombre: (vertices, faces)} con varias
    mallas generadas proceduralmente para tener un demo inmediato."""
    shapes = {}

    # Toro: forma con "agujero" (género 1), buena para mostrar topología.
    toro = trimesh.creation.torus(major_radius=1.0, minor_radius=0.35,
                                  major_sections=48, minor_sections=24)
    shapes["Toro"] = normalize_mesh(toro.vertices, toro.faces)

    # Esfera (icosfera): superficie cerrada suave y uniforme.
    esfera = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    shapes["Esfera"] = normalize_mesh(esfera.vertices, esfera.faces)

    # Cubo: caras planas grandes, útil para ver el muestreo por área.
    cubo = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    shapes["Cubo"] = normalize_mesh(cubo.vertices, cubo.faces)

    # Cápsula: cilindro con tapas esféricas.
    capsula = trimesh.creation.capsule(height=1.2, radius=0.5, count=(24, 24))
    shapes["Capsula"] = normalize_mesh(capsula.vertices, capsula.faces)

    # "Blob" orgánico: esfera deformada con ruido sinusoidal, para tener
    # algo menos regular que las primitivas.
    base = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    v = base.vertices.copy()
    r = np.linalg.norm(v, axis=1, keepdims=True)
    dirs = v / r
    disp = 1.0 + 0.18 * np.sin(4.0 * v[:, 0]) * np.cos(4.0 * v[:, 1]) \
               + 0.12 * np.sin(5.0 * v[:, 2])
    v = dirs * disp[:, None]
    shapes["Blob"] = normalize_mesh(v, base.faces)

    return shapes


# ----------------------------------------------------------------------
# CONVERSIÓN 1:  Malla  ->  Nube de puntos
# ----------------------------------------------------------------------

def mesh_to_pointcloud(vertices, faces, n_samples=20000, method="area"):
    """Convierte una malla triangular en una nube de puntos.

    method="vertices": usa directamente los vértices de la malla.
        Es lo más simple, pero la densidad depende de cómo se malló el
        objeto (zonas con triángulos pequeños quedan sobre-representadas).

    method="area": MUESTREO UNIFORME POR ÁREA (el más usado).
        Idea del algoritmo:
          1. El área de cada triángulo determina su probabilidad de ser
             elegido -> los triángulos grandes reciben más puntos.
          2. Para cada punto: se elige un triángulo al azar (ponderado por
             área) y se toma un punto aleatorio DENTRO de él usando
             coordenadas baricéntricas (u, v).
        Resultado: puntos repartidos de forma pareja sobre la superficie,
        independientemente de la resolución de la malla.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    if method == "vertices":
        return vertices.copy()

    # --- Muestreo uniforme por área ---
    # Vértices de cada triángulo: A, B, C   (shape: n_tri x 3)
    A = vertices[faces[:, 0]]
    B = vertices[faces[:, 1]]
    C = vertices[faces[:, 2]]

    # Área de cada triángulo = 1/2 |(B-A) x (C-A)|
    areas = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    probs = areas / areas.sum()

    # 1) Elegir qué triángulo aporta cada punto (ponderado por área)
    tri_idx = np.random.choice(len(faces), size=n_samples, p=probs)

    # 2) Punto aleatorio dentro del triángulo (coordenadas baricéntricas).
    #    El "reflejo" u+v>1 -> (1-u, 1-v) garantiza distribución uniforme
    #    dentro del triángulo.
    u = np.random.rand(n_samples, 1)
    v = np.random.rand(n_samples, 1)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]

    a = A[tri_idx]
    b = B[tri_idx]
    c = C[tri_idx]
    points = a + u * (b - a) + v * (c - a)
    return points


def pointcloud_normals(vertices, faces, points, method="area"):
    """(Opcional) Normales aproximadas por punto para colorear la nube.
    Para 'vertices' usa la normal de vértice; en otro caso devuelve None
    y la app colorea por posición."""
    return None


# ----------------------------------------------------------------------
# CONVERSIÓN 2:  Malla  ->  Voxels
# ----------------------------------------------------------------------

def mesh_to_voxels(vertices, faces, resolution=48, solid=True):
    """Convierte una malla en una grilla de ocupación (voxelización).

    Devuelve:
        occ    : arreglo booleano (R x R x R), True = voxel ocupado
        origin : esquina inferior del bounding box (x,y,z)
        pitch  : tamaño de cada voxel (lado del cubo)

    Algoritmo (voxelización por muestreo, fácil de explicar):
      1. Se define una grilla regular de RxRxR que cubre el bounding box
         del objeto. Cada celda es un voxel de lado 'pitch'.
      2. Se muestrea densamente la SUPERFICIE de la malla (puntos).
      3. Cada punto "cae" en una celda: se marca esa celda como ocupada.
         -> Esto produce una voxelización de la CÁSCARA (superficie).
      4. Si solid=True, se RELLENA el interior: toda celda encerrada por
         la cáscara se marca como ocupada (relleno tipo "flood fill" desde
         afuera; lo no alcanzado desde el exterior es interior).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    # 1) Grilla que cubre el bounding box (con un pequeño margen).
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extent = (vmax - vmin).max()
    pitch = extent / resolution
    origin = vmin - pitch * 0.5                      # margen de medio voxel
    dims = np.ceil((vmax - vmin + pitch) / pitch).astype(int)
    dims = np.maximum(dims, 1)

    occ = np.zeros(tuple(dims), dtype=bool)

    # 2) Muestreo denso de la superficie (reutiliza el muestreo por área).
    n = max(200000, resolution ** 2 * 60)
    pts = mesh_to_pointcloud(vertices, faces, n_samples=n, method="area")

    # 3) Cada punto -> índice de celda -> marcar ocupado.
    idx = np.floor((pts - origin) / pitch).astype(int)
    idx = np.clip(idx, 0, np.array(dims) - 1)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    # 4) Relleno sólido opcional: lo alcanzable desde el borde exterior
    #    es "afuera"; lo demás (no cáscara) es interior -> ocupado.
    if solid:
        occ = _fill_interior(occ)

    return occ, origin, pitch


def _fill_interior(occ):
    """Marca como ocupado el interior encerrado por la cáscara de voxels.
    Estrategia: se hace un 'flood fill' del ESPACIO VACÍO empezando desde
    el borde de la grilla. Todo voxel vacío que NO se alcanza desde el
    borde está encerrado por la cáscara => es interior => se ocupa."""
    from scipy import ndimage

    empty = ~occ
    # Etiqueta regiones conectadas de espacio vacío (conectividad de caras).
    structure = ndimage.generate_binary_structure(3, 1)
    labels, _ = ndimage.label(empty, structure=structure)

    # Etiquetas que tocan el borde de la grilla = "exterior".
    border = set()
    border.update(labels[0, :, :].ravel())
    border.update(labels[-1, :, :].ravel())
    border.update(labels[:, 0, :].ravel())
    border.update(labels[:, -1, :].ravel())
    border.update(labels[:, :, 0].ravel())
    border.update(labels[:, :, -1].ravel())
    border.discard(0)

    outside = np.isin(labels, list(border)) & empty
    interior = empty & ~outside          # vacío no alcanzado desde afuera
    return occ | interior


def voxels_to_cube_mesh(occ, origin, pitch):
    """Construye una MALLA de cubos para visualizar los voxels ocupados
    (estilo 'Minecraft'). Por eficiencia sólo se generan las caras que
    quedan en la FRONTERA (entre un voxel ocupado y uno vacío); las caras
    internas ocultas no se dibujan. Implementación vectorizada para que
    sea fluida incluso con resoluciones altas.

    Devuelve (vertices, faces) listos para registrar como superficie.
    """
    occ = np.asarray(occ)
    dims = np.array(occ.shape)

    # Padding para consultar vecinos fuera de rango como "vacío".
    padded = np.zeros(dims + 2, dtype=bool)
    padded[1:-1, 1:-1, 1:-1] = occ

    # Plantilla de las 8 esquinas de un cubo unitario.
    corner = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    # Cada cara: (esquinas que la forman, desplazamiento del vecino)
    cube_faces = [
        ((0, 3, 2, 1), (0, 0, -1)),   # cara -Z
        ((4, 5, 6, 7), (0, 0, 1)),    # cara +Z
        ((0, 1, 5, 4), (0, -1, 0)),   # cara -Y
        ((3, 7, 6, 2), (0, 1, 0)),    # cara +Y
        ((0, 4, 7, 3), (-1, 0, 0)),   # cara -X
        ((1, 2, 6, 5), (1, 0, 0)),    # cara +X
    ]

    occ_idx = np.argwhere(occ)              # (N,3) voxels ocupados
    if len(occ_idx) == 0:
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=int)
    bases = origin + occ_idx * pitch        # esquina de cada voxel
    i, j, k = occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]

    verts_blocks = []
    for cface, (di, dj, dk) in cube_faces:
        # ¿el vecino en esa dirección está vacío? -> cara visible
        visible = ~padded[i + 1 + di, j + 1 + dj, k + 1 + dk]
        if not visible.any():
            continue
        b = bases[visible]                                  # (n,3)
        offs = corner[list(cface)] * pitch                  # (4,3)
        quad = b[:, None, :] + offs[None, :, :]             # (n,4,3)
        verts_blocks.append(quad.reshape(-1, 3))

    if not verts_blocks:
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=int)
    verts = np.concatenate(verts_blocks, axis=0)
    faces = np.arange(len(verts)).reshape(-1, 4)
    return verts, faces.astype(int)


# ----------------------------------------------------------------------
# Textos explicativos (se muestran en el panel de la app, junto al código)
# ----------------------------------------------------------------------

EXPLICACION_NUBE = (
    "MALLA -> NUBE DE PUNTOS\n"
    "Una nube de puntos descarta la conectividad (las caras) y conserva\n"
    "solo posiciones. La forma correcta de obtenerla es MUESTREAR la\n"
    "superficie de manera uniforme por AREA: cada triangulo aporta puntos\n"
    "en proporcion a su tamano, usando coordenadas baricentricas para\n"
    "ubicar cada punto dentro del triangulo. Asi la densidad no depende\n"
    "de como se mallo el objeto. (Alternativa mas cara: Poisson-disk, que\n"
    "ademas separa los puntos a distancia minima.)"
)

EXPLICACION_VOXEL = (
    "MALLA -> VOXELS\n"
    "Se divide el espacio en una grilla regular RxRxR. Cada celda (voxel)\n"
    "guarda si hay geometria o no (ocupacion 0/1). Aqui se voxeliza\n"
    "muestreando densamente la superficie y marcando la celda que contiene\n"
    "cada punto (cascara). El relleno solido se logra con un flood-fill:\n"
    "el vacio alcanzable desde el borde es 'exterior'; el vacio encerrado\n"
    "por la cascara es 'interior' y se marca ocupado. La RESOLUCION define\n"
    "el detalle y el costo: 128^3 ~2M, 256^3 ~16M, 512^3 ~134M voxels."
)
