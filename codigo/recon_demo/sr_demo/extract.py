"""Extraccion de iso-superficies e iso-lineas (Bloque 2.1 / 2.2 del tutorial).

* marching_tetrahedra -- implementacion propia y vectorizada. Cada cubo se
  divide en 6 tetraedros (descomposicion de Kuhn, todos comparten la diagonal
  000-111). En un tetraedro hay solo 16 casos de signo, que se reducen a dos:
  1 vertice distinto -> 1 triangulo; 2 contra 2 -> un cuadrilatero (2
  triangulos). No hay tabla de 256 casos ni ambiguedad de cara.
  Los vertices de la malla se identifican por la ARISTA de la grilla en que
  caen, asi la malla sale indexada (no una sopa de triangulos) y se puede
  calcular chi = V - E + F.
  Valores NaN = funcion no definida (Hoppe lejos de los datos): los
  tetraedros que tocan un NaN se descartan, lo que deja bordes en la malla.

* marching_cubes -- scikit-image (opcional) para comparar teselados.

* marching_squares -- la version 2D, usada para dibujar la iso-linea en el
  corte del campo. La silla (caso ambiguo) se resuelve con el valor del centro.
"""

from __future__ import annotations

import numpy as np

# esquinas del cubo como offsets (i, j, k)
_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
_E = np.eye(3, dtype=int)
# 6 tetraedros de Kuhn: camino 000 -> e_a -> e_a+e_b -> 111 para cada permutacion
_TETS = []
for a, b, c in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
    path = [np.zeros(3, int), _E[a], _E[a] + _E[b], np.ones(3, int)]
    _TETS.append([int(np.flatnonzero((_CORNERS == p).all(1))[0]) for p in path])
_TETS = np.array(_TETS)  # (6, 4) indices en _CORNERS


def marching_tetrahedra(vals: np.ndarray, grid, iso: float = 0.0):
    """Iso-superficie {F = iso} de un campo en grilla. Devuelve (V, F).

    Convencion: F < iso es "dentro". Los triangulos se orientan con la normal
    hacia afuera (hacia F creciente).
    """
    F = np.asarray(vals, float) - iso
    nx, ny, nz = F.shape
    inside = F < 0
    defined = np.isfinite(F)

    # 1) celdas activas: 8 esquinas definidas y signo mixto
    def corner(arr, off):
        i, j, k = off
        return arr[i:nx - 1 + i, j:ny - 1 + j, k:nz - 1 + k]

    cnt_in = sum(corner(inside & defined, o).astype(np.int8) for o in _CORNERS)
    all_def = np.logical_and.reduce([corner(defined, o) for o in _CORNERS])
    active = all_def & (cnt_in > 0) & (cnt_in < 8)
    ci, cj, ck = np.nonzero(active)
    if len(ci) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)

    # 2) indice global de las 8 esquinas de cada celda activa
    gidx = np.stack([np.ravel_multi_index((ci + o[0], cj + o[1], ck + o[2]), F.shape)
                     for o in _CORNERS], axis=1)                    # (m, 8)
    tv = gidx[:, _TETS].reshape(-1, 4)                               # (6m, 4)
    Ff = F.ravel()
    fv = Ff[tv]
    ins = fv < 0
    nin = ins.sum(1)
    sel = (nin > 0) & (nin < 4)
    tv, fv, ins, nin = tv[sel], fv[sel], ins[sel], nin[sel]

    pts_of = lambda g: grid.origin + grid.h * np.c_[np.unravel_index(g, F.shape)]  # noqa: E731

    tris_a, tris_b = [], []   # pares de vertices de la grilla por arista

    # 3a) un vertice distinto de los otros tres -> 1 triangulo
    one = (nin == 1) | (nin == 3)
    if one.any():
        t, s = tv[one], ins[one]
        odd_mask = np.where((nin[one] == 1)[:, None], s, ~s)
        odd = np.argmax(odd_mask, 1)
        others = np.argsort(odd_mask, axis=1, kind="stable")[:, :3]   # los 3 restantes
        a = t[np.arange(len(t)), odd]
        tris_a.append(np.repeat(a[:, None], 3, 1))
        tris_b.append(np.take_along_axis(t, others, 1))
    # 3b) dos contra dos -> cuadrilatero (a,c)(a,d)(b,d)(b,c) -> 2 triangulos
    two = nin == 2
    if two.any():
        t, s = tv[two], ins[two]
        order = np.argsort(~s, axis=1, kind="stable")   # primero los 2 de adentro
        a, b, c, d = (np.take_along_axis(t, order[:, [q]], 1)[:, 0] for q in range(4))
        tris_a += [np.c_[a, a, b], np.c_[a, b, b]]
        tris_b += [np.c_[c, d, d], np.c_[c, d, c]]
    EA = np.vstack(tris_a)                   # (T, 3) extremo 1 de cada arista
    EB = np.vstack(tris_b)                   # (T, 3) extremo 2

    # 4) vertices unicos por arista de grilla
    lo, hi = np.minimum(EA, EB), np.maximum(EA, EB)
    key = lo.astype(np.int64) * Ff.size + hi
    ukey, inv = np.unique(key.ravel(), return_inverse=True)
    ul, uh = ukey // Ff.size, ukey % Ff.size
    fl, fh = Ff[ul], Ff[uh]
    tpar = fl / (fl - fh)
    V = pts_of(ul) + tpar[:, None] * (pts_of(uh) - pts_of(ul))
    T = inv.reshape(-1, 3).astype(np.int64)

    # 5) orientacion: la normal debe apuntar de "adentro" a "afuera"
    #    (de los extremos con F<0 a los extremos con F>0)
    pin = np.where((Ff[EA] < 0)[..., None], pts_of(EA.ravel()).reshape(-1, 3, 3),
                   pts_of(EB.ravel()).reshape(-1, 3, 3)).mean(1)
    pout = np.where((Ff[EA] < 0)[..., None], pts_of(EB.ravel()).reshape(-1, 3, 3),
                    pts_of(EA.ravel()).reshape(-1, 3, 3)).mean(1)
    nrm = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    flip = (nrm * (pout - pin)).sum(1) < 0
    T[flip] = T[flip][:, [0, 2, 1]]
    return V, T


def marching_cubes(vals: np.ndarray, grid, iso: float = 0.0):
    """Marching cubes de scikit-image (Lewiner). Devuelve (V, F) o None."""
    try:
        from skimage.measure import marching_cubes as mc
    except Exception:  # noqa: BLE001
        return None
    F = np.asarray(vals, float) - iso
    mask = np.isfinite(F)
    Fz = np.where(mask, F, np.nanmax(np.abs(F[mask])) if mask.any() else 1.0)
    try:
        V, T, _, _ = mc(Fz, level=0.0, spacing=(grid.h,) * 3,
                        mask=mask if not mask.all() else None,
                        gradient_direction="ascent")
    except (ValueError, RuntimeError):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    V = V + grid.origin
    return V, T.astype(np.int64)


def marching_squares(G: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Iso-linea G = 0 de un campo 2D (G[i, j] en (x[i], y[j])).

    Devuelve (nodos (m,2), aristas (s,2)). La silla se desambigua con el
    promedio de las 4 esquinas.
    """
    nx, ny = G.shape
    v00, v10, v11, v01 = G[:-1, :-1], G[1:, :-1], G[1:, 1:], G[:-1, 1:]
    ok = np.isfinite(v00) & np.isfinite(v10) & np.isfinite(v11) & np.isfinite(v01)
    X0, Y0 = np.meshgrid(x[:-1], y[:-1], indexing="ij")
    X1, Y1 = np.meshgrid(x[1:], y[1:], indexing="ij")

    def cross(va, vb, pa, pb):
        t = va / (va - vb)
        return pa + t[..., None] * (pb - pa)

    # 4 aristas de cada celda: abajo (00-10), derecha (10-11), arriba (01-11), izq (00-01)
    P00, P10 = np.stack([X0, Y0], -1), np.stack([X1, Y0], -1)
    P11, P01 = np.stack([X1, Y1], -1), np.stack([X0, Y1], -1)
    edges = [(v00, v10, P00, P10), (v10, v11, P10, P11),
             (v01, v11, P01, P11), (v00, v01, P00, P01)]
    has = [ok & ((va < 0) != (vb < 0)) for va, vb, _, _ in edges]
    with np.errstate(invalid="ignore", divide="ignore"):
        pts = [cross(va, vb, pa, pb) for va, vb, pa, pb in edges]
    ncross = sum(h.astype(int) for h in has)

    segs = []
    # caso normal: exactamente 2 cruces -> un segmento entre ellos
    two = ncross == 2
    idx = np.stack(has, -1)[two]                       # (m, 4) booleano
    P = np.stack(pts, -2)[two]                         # (m, 4, 2)
    first = np.argmax(idx, 1)
    second = 3 - np.argmax(idx[:, ::-1], 1)
    r = np.arange(len(P))
    segs.append(np.stack([P[r, first], P[r, second]], 1))
    # silla: 4 cruces. Si el centro tiene el signo de v00 (v00 y v11 unidos por
    # el centro), quedan aisladas v10 -> (abajo, derecha) y v01 -> (arriba, izq).
    # Si no, quedan aisladas v00 -> (abajo, izq) y v11 -> (derecha, arriba).
    four = ncross == 4
    if four.any():
        P = np.stack(pts, -2)[four]
        center = 0.25 * (v00 + v10 + v11 + v01)[four]
        same = (center < 0) == (v00[four] < 0)
        pa = np.where(same[:, None], P[:, 0], P[:, 0])
        pb = np.where(same[:, None], P[:, 1], P[:, 3])
        pc = np.where(same[:, None], P[:, 2], P[:, 2])
        pd = np.where(same[:, None], P[:, 3], P[:, 1])
        segs += [np.stack([pa, pb], 1), np.stack([pc, pd], 1)]
    S = np.vstack(segs) if segs else np.zeros((0, 2, 2))
    nodes = S.reshape(-1, 2)
    E = np.arange(len(nodes)).reshape(-1, 2)
    return nodes, E
