"""Lectura/escritura de mallas (OBJ, OFF; PLY vía trimesh si está) y limpieza."""
import os
import numpy as np


def load_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        V, F = _load_obj(path)
    elif ext == ".off":
        V, F = _load_off(path)
    else:
        try:
            import trimesh
        except ImportError as e:
            raise RuntimeError(f"formato {ext} requiere trimesh (pip install trimesh)") from e
        m = trimesh.load(path, force="mesh", process=False)
        V, F = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
    return V, F


def _load_obj(path):
    V, F = [], []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(V) + i for i in idx]
                for t in range(1, len(idx) - 1):          # polígonos → abanico de triángulos
                    F.append([idx[0], idx[t], idx[t + 1]])
    return np.asarray(V, float), np.asarray(F, np.int64).reshape(-1, 3)


def _load_off(path):
    with open(path) as fh:
        toks = [ln for ln in (l.split("#")[0].strip() for l in fh) if ln]
    head = toks[0]
    start = 1
    if head.upper().startswith("OFF") and len(head.split()) > 1:
        counts = head.split()[1:]
    else:
        counts = toks[1].split()
        start = 2
    nv, nf = int(counts[0]), int(counts[1])
    V = np.array([[float(x) for x in toks[start + i].split()[:3]] for i in range(nv)])
    F = []
    for i in range(nf):
        vals = [int(x) for x in toks[start + nv + i].split()]
        k, idx = vals[0], vals[1:1 + vals[0]]
        for t in range(1, k - 1):
            F.append([idx[0], idx[t], idx[t + 1]])
    return V, np.asarray(F, np.int64).reshape(-1, 3)


def save_obj(path, V, F):
    with open(path, "w") as fh:
        fh.write(f"# {len(V)} vertices, {len(F)} faces\n")
        np.savetxt(fh, V, fmt="v %.7f %.7f %.7f")
        np.savetxt(fh, F + 1, fmt="f %d %d %d")


def normalize(V):
    """Centra en la caja envolvente y escala para que su diagonal mida 1.

    Así los errores se leen directamente como fracción de la diagonal
    (en el demo se reportan en milésimas, ‰)."""
    lo, hi = V.min(0), V.max(0)
    return (V - 0.5 * (lo + hi)) / np.linalg.norm(hi - lo)


def clean(V, F, merge_tol=1e-9):
    """Une vértices duplicados, quita caras degeneradas/duplicadas y vértices sueltos."""
    if merge_tol > 0:
        key = np.round(V / merge_tol).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        V, F = V[first], inv.ravel()[F]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])]
    _, keep = np.unique(np.sort(F, axis=1), axis=0, return_index=True)
    F = F[np.sort(keep)]
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    return split_nonmanifold_vertices(V[used], remap[F])


def split_nonmanifold_vertices(V, F):
    """Duplica los vértices "pellizcados" (dos abanicos de caras que solo se tocan
    en un punto). La vaca del paper tiene uno: sin separarlo, χ = 1 y el género
    calculado sale 0.5."""
    from .topology import corner_fans
    label, nfans = corner_fans(F)
    if len(nfans) == 0 or nfans.max() <= 1:
        return V, F
    vert = F.ravel()
    fans, inv = np.unique(label, return_inverse=True)
    first_vertex = np.zeros(len(fans), np.int64)
    first_vertex[inv] = vert                     # vértice original de cada abanico
    return V[first_vertex], inv.reshape(-1, 3)
