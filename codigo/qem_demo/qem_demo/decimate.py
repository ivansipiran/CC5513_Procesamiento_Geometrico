"""
Simplificación por colapso de pares — algoritmo de Garland & Heckbert (1997).

El resumen del algoritmo en el paper (sección 4) tiene 5 pasos; la clase
`PairCollapse` los implementa con esos mismos nombres:

    Paso 1  calcular la cuádrica Q de cada vértice inicial
    Paso 2  seleccionar los pares válidos (aristas + pares cercanos opcionales)
    Paso 3  para cada par, calcular el destino óptimo v̄ y su costo v̄ᵀ(Q₁+Q₂)v̄
    Paso 4  meter todos los pares en un heap ordenado por costo
    Paso 5  repetir: sacar el par más barato, contraerlo, y recalcular el costo
            de los pares que tocan al vértice nuevo

La maquinaria (heap, contracción, chequeos) NO depende del costo. Con
cost="length" el mismo código colapsa siempre la arista más corta en su punto
medio: es el experimento de control que muestra que lo que importa es la métrica
de error, no el mecanismo de colapso.

Estructuras (listas de conjuntos de Python, a propósito: se leen fácil):
    V[v]        posición actual del vértice v            (n × 3, se modifica)
    F[f]        los 3 vértices de la cara f               (m × 3, se modifica)
    vfaces[v]   conjunto de caras vivas que tocan a v
    pairs[v]    conjunto de vértices con los que v forma par
    Q[v]        cuádrica 4 × 4 del vértice v
    version[v]  cuántas veces cambió v (para descartar entradas viejas del heap)
"""
import heapq
import itertools
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from . import quadrics as qd


# ---------------------------------------------------------------------------
# Historial: la secuencia de colapsos permite reconstruir la malla en cualquier
# punto sin volver a correr el algoritmo (el slider del visor lo usa).
# ---------------------------------------------------------------------------
@dataclass
class History:
    V0: np.ndarray                     # malla original
    F0: np.ndarray
    Q0: np.ndarray                     # cuádricas iniciales (n × 4 × 4)
    W0: np.ndarray = None              # peso de las caras de cada vértice (escala de A)
    keep: np.ndarray = None            # colapso t:  removed[t] → keep[t]
    removed: np.ndarray = None
    newpos: np.ndarray = None          # posición v̄ del vértice resultante
    cost: np.ndarray = None            # costo del colapso (lo que salió del heap)
    how: np.ndarray = None             # cómo se eligió v̄ (quadrics.HOW_*)
    virtual: np.ndarray = None         # True si el par no era arista
    nfaces: np.ndarray = None          # caras vivas DESPUÉS de cada colapso
    params: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    @property
    def steps(self):
        return len(self.keep)

    def max_step(self):
        """Último paso que todavía deja al menos una cara (sin topología se puede llegar a 0)."""
        if self.steps == 0:
            return 0
        alive = np.nonzero(self.nfaces > 0)[0]
        return int(alive[-1]) + 1 if len(alive) else 0

    def faces_at(self, k):
        return int(self.nfaces[k - 1]) if k > 0 else len(self.F0)

    def step_for_faces(self, target):
        """Primer paso k con #caras ≤ target (o el último si no se llega)."""
        k = int(np.searchsorted(-self.nfaces, -target, side="left")) + 1
        return min(k, self.max_step()) if target < len(self.F0) else 0

    def parent_at(self, k):
        """parent[v] = vértice (original) en el que terminó v después de k colapsos."""
        parent = np.arange(len(self.V0))
        parent[self.removed[:k]] = self.keep[:k]
        while True:                                  # "pointer jumping" hasta la raíz
            nxt = parent[parent]
            if np.array_equal(nxt, parent):
                return parent
            parent = nxt

    def state(self, k):
        """Malla después de k colapsos, compactada.

        Devuelve (V, F, ids, parent): `ids[i]` es el índice original del vértice i."""
        parent = self.parent_at(k)
        pos = self.V0.copy()
        if k > 0:
            # posición final de cada vértice que sobrevivió = última v̄ que recibió
            rev = self.keep[:k][::-1]
            uniq, idx = np.unique(rev, return_index=True)
            pos[uniq] = self.newpos[:k][::-1][idx]
        F = parent[self.F0]
        ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])
        F = F[ok]
        if not self.params.get("preserve_topology", True):
            # caras coincidentes se eliminan de a pares (ver PairCollapse.contract)
            _, first, inv, cnt = np.unique(np.sort(F, axis=1), axis=0, return_index=True,
                                           return_inverse=True, return_counts=True)
            keep = np.zeros(len(F), bool)
            keep[first[cnt % 2 == 1]] = True
            F = F[keep]
        ids = np.unique(F)
        remap = np.full(len(self.V0), -1)
        remap[ids] = np.arange(len(ids))
        return pos[ids], remap[F], ids, parent

    def quadrics_at(self, k, parent=None, ids=None):
        """Q de cada vértice vivo = suma de las Q iniciales de todos los que se le unieron."""
        if parent is None:
            parent = self.parent_at(k)
        n = len(self.Q0)
        flat = self.Q0.reshape(n, 16)
        Q = np.stack([np.bincount(parent, weights=flat[:, c], minlength=n) for c in range(16)], 1)
        Q = Q.reshape(n, 4, 4)
        return Q if ids is None else Q[ids]

    def weights_at(self, k, parent=None, ids=None):
        if parent is None:
            parent = self.parent_at(k)
        W = np.bincount(parent, weights=self.W0, minlength=len(self.W0))
        return W if ids is None else W[ids]


# ---------------------------------------------------------------------------
# El algoritmo
# ---------------------------------------------------------------------------
class PairCollapse:
    """Simplificación de Garland & Heckbert.

    Parámetros
    ----------
    cost              "qem" (cuádricas) o "length" (arista más corta, punto medio)
    placement         cómo elegir v̄ con "qem": optimal | svd | subset | midpoint
    weighting         "area" (ponderar planos por área) o "none" (paper 1997)
    boundary_weight   peso de los planos perpendiculares a los bordes (0 = sin)
    preserve_topology rechazar colapsos que cambian la topología (condición de enlace)
    check_flips       rechazar colapsos que dan vuelta la normal de alguna cara
    pair_threshold    t del paper: también son pares los vértices a distancia < t
                      aunque no compartan arista (permite unir piezas separadas)
    """

    def __init__(self, V, F, cost="qem", placement="optimal", weighting="area",
                 boundary_weight=1000.0, preserve_topology=True, check_flips=True,
                 pair_threshold=0.0, rcond=1e-5, flip_min_cos=0.0):
        self.V = np.array(V, dtype=float)
        self.F = np.array(F, dtype=np.int64)
        self.cost_kind = cost
        self.placement = placement if cost == "qem" else "midpoint"
        self.weighting = weighting
        self.boundary_weight = boundary_weight
        self.preserve_topology = preserve_topology
        self.check_flips = check_flips
        self.pair_threshold = pair_threshold
        self.rcond = rcond
        self.flip_min_cos = flip_min_cos

        n, m = len(self.V), len(self.F)
        self.alive = np.ones(n, bool)
        self.face_alive = np.ones(m, bool)
        self.n_faces = m
        self.version = np.zeros(n, np.int64)
        self.vfaces = [set() for _ in range(n)]
        for f, (a, b, c) in enumerate(self.F):
            self.vfaces[a].add(f)
            self.vfaces[b].add(f)
            self.vfaces[c].add(f)
        self.boundary = self._boundary_vertices()
        self.stats = dict(collapses=0, stale=0, rejected_topology=0, rejected_flip=0,
                          virtual_pairs=0)
        self._tiebreak = itertools.count()

        self.step1_compute_quadrics()
        I, J = self.step2_select_pairs()
        self.step3_4_build_heap(I, J)
        self.history = History(V0=np.array(V, float), F0=np.array(F, np.int64),
                               Q0=self.Q.copy(), params=self.params(),
                               W0=qd.face_weight(self.V, self.F, self.weighting))
        self._log = dict(keep=[], removed=[], newpos=[], cost=[], how=[], virtual=[], nfaces=[])

    def params(self):
        return dict(cost=self.cost_kind, placement=self.placement, weighting=self.weighting,
                    boundary_weight=self.boundary_weight, preserve_topology=self.preserve_topology,
                    check_flips=self.check_flips, pair_threshold=self.pair_threshold)

    # ----------------------------------------------------------- Paso 1
    def step1_compute_quadrics(self):
        """Q_v = Σ (cuádricas de las caras que tocan a v) + restricciones de borde."""
        self.Q = qd.vertex_quadrics(self.V, self.F, self.weighting, self.boundary_weight)

    # ----------------------------------------------------------- Paso 2
    def step2_select_pairs(self):
        """Pares válidos: (1) toda arista de la malla; (2) si t > 0, todo par de
        vértices a distancia < t, aunque no estén conectados."""
        F = self.F
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        E = np.unique(E, axis=0)
        if self.pair_threshold > 0:
            close = cKDTree(self.V).query_pairs(self.pair_threshold, output_type="ndarray")
            close = np.sort(close, axis=1)
            allp = np.unique(np.concatenate([E, close]), axis=0)
            self.stats["virtual_pairs"] = len(allp) - len(E)
            E = allp
        self.pairs = [set() for _ in range(len(self.V))]
        for a, b in E:
            self.pairs[a].add(b)
            self.pairs[b].add(a)
        return E[:, 0], E[:, 1]

    # ----------------------------------------------------------- Pasos 3 y 4
    def pair_cost(self, I, J):
        """Destino v̄ y costo para los pares (I[k], J[k]) — vectorizado."""
        v1, v2 = self.V[I], self.V[J]
        if self.cost_kind == "length":
            vbar = 0.5 * (v1 + v2)
            return vbar, ((v1 - v2) ** 2).sum(1), np.full(len(I), qd.HOW_SUBSET, np.int8)
        Qs = self.Q[I] + self.Q[J]                         # la cuádrica del par es la SUMA
        return qd.optimal_placement(Qs, v1, v2, self.placement, self.rcond)

    def step3_4_build_heap(self, I, J):
        vbar, cost, how = self.pair_cost(I, J)
        self.heap = [(float(c), next(self._tiebreak), int(i), int(j), 0, 0, vb, int(h))
                     for c, i, j, vb, h in zip(cost, I, J, vbar, how)]
        heapq.heapify(self.heap)

    def _push(self, i, partners):
        if not partners:
            return
        J = np.fromiter(partners, np.int64)
        I = np.full(len(J), i)
        vbar, cost, how = self.pair_cost(I, J)
        vi = self.version[i]
        for c, j, vb, h in zip(cost, J, vbar, how):
            heapq.heappush(self.heap, (float(c), next(self._tiebreak), i, int(j),
                                       vi, self.version[j], vb, int(h)))

    # ----------------------------------------------------------- Paso 5
    def simplify(self, target_faces=0, progress=None):
        """Colapsa pares, del más barato al más caro, hasta tener ≤ target_faces
        caras (0 = hasta que no quede ningún colapso válido)."""
        t0 = time.perf_counter()
        last = t0
        while self.heap and self.n_faces > target_faces:
            cost, _, i, j, vi, vj, vbar, how = heapq.heappop(self.heap)
            # Entradas viejas: uno de los vértices murió o cambió después de
            # calcular este costo. En vez de borrar del heap (caro), las ignoramos.
            if not (self.alive[i] and self.alive[j]) or \
               self.version[i] != vi or self.version[j] != vj:
                self.stats["stale"] += 1
                continue
            shared = self.vfaces[i] & self.vfaces[j]           # caras que desaparecen
            if not self.is_valid_contraction(i, j, shared):
                self.stats["rejected_topology"] += 1
                continue
            if self.check_flips and self.flips_normal(i, j, vbar, shared):
                self.stats["rejected_flip"] += 1
                continue
            self.contract(i, j, vbar, shared)
            self._record(keep=i, removed=j, newpos=vbar, cost=cost, how=how,
                         virtual=not shared, nfaces=self.n_faces)
            if progress and time.perf_counter() - last > 0.5:
                last = time.perf_counter()
                progress(self.n_faces)
        self.stats["collapses"] = len(self._log["keep"])
        self.stats["time"] = time.perf_counter() - t0
        return self.finish_history()

    def _record(self, **entry):
        for key, value in entry.items():
            self._log[key].append(value)

    def neighbors(self, v):
        """Vértices unidos a v por una arista (1-anillo)."""
        if not self.vfaces[v]:
            return set()
        ring = set(self.F[list(self.vfaces[v])].ravel().tolist())
        ring.discard(v)
        return ring

    def is_valid_contraction(self, i, j, shared):
        """Chequeos topológicos antes de unir j con i.

        Condición de enlace (Dey et al. 1999): el colapso de la arista ij preserva la
        topología si los vecinos comunes de i y j son EXACTAMENTE los vértices opuestos
        a la arista (uno por cada cara que la contiene). Un vecino común extra significa
        que existe un triángulo i-j-k "hueco": colapsarlo pellizca la superficie.
        """
        if shared and self.preserve_topology:
            if len(shared) > 2:                                  # arista no-variedad
                return False
            opposite = {int(v) for f in shared for v in self.F[f]} - {i, j}
            if self.neighbors(i) & self.neighbors(j) != opposite:
                return False
            # Dos vértices de borde unidos por una arista interior: colapsarla
            # tocaría el borde consigo mismo (la "condición de enlace" con el
            # vértice ficticio que cierra los bordes).
            if self.boundary[i] and self.boundary[j] and len(shared) == 2:
                return False
            # Triángulo "suelto" (sus 3 aristas son de borde): colapsarlo lo borra.
            if len(shared) == 1:
                (k,) = opposite
                if len(self.vfaces[i] & self.vfaces[k]) == 1 and \
                   len(self.vfaces[j] & self.vfaces[k]) == 1 and self.boundary[k]:
                    return False
        # ¿Quedarían dos caras con los mismos 3 vértices?
        #  · preservando topología: se rechaza (pasa, p.ej., al aplastar un tetraedro)
        #  · sin preservarla: las dos caras coincidentes son una "pared interna"
        #    (típico al fundir dos piezas con pares no conectados) y se borran ambas.
        self._dups = []
        faces_i = {tuple(sorted(self.F[f])): f for f in self.vfaces[i] - shared}
        for f in self.vfaces[j] - shared:
            tri = tuple(sorted(i if v == j else int(v) for v in self.F[f]))
            if tri in faces_i:
                if self.preserve_topology:
                    return False
                self._dups.append((faces_i.pop(tri), f))
        return True

    def flips_normal(self, i, j, vbar, shared):
        """¿Alguna cara vecina se da vuelta (o se degenera) al mover i y j a v̄?"""
        faces = list((self.vfaces[i] | self.vfaces[j]) - shared)
        if not faces:
            return False
        T = self.F[faces]
        P_old = self.V[T]                                        # (k, 3, 3)
        P_new = P_old.copy()
        P_new[(T == i) | (T == j)] = vbar
        n_old = np.cross(P_old[:, 1] - P_old[:, 0], P_old[:, 2] - P_old[:, 0])
        n_new = np.cross(P_new[:, 1] - P_new[:, 0], P_new[:, 2] - P_new[:, 0])
        l_old = np.linalg.norm(n_old, axis=1)
        l_new = np.linalg.norm(n_new, axis=1)
        if np.any(l_new <= 1e-9 * np.maximum(l_old, 1e-300)):   # cara degenerada
            return True
        cos = (n_old * n_new).sum(1) / np.maximum(l_old * l_new, 1e-300)
        return bool(np.any(cos < self.flip_min_cos))

    def contract(self, i, j, vbar, shared):
        """(v₁, v₂) → v̄ :  i sobrevive en la posición v̄, j desaparece."""
        # 1) las caras que contenían la arista ij quedan degeneradas: se eliminan
        for f in shared:
            self.face_alive[f] = False
            for v in self.F[f]:
                self.vfaces[v].discard(f)
        self.n_faces -= len(shared)
        # 1b) caras que quedarían duplicadas (solo sin preservar topología)
        for pair in self._dups:
            for f in pair:
                self.face_alive[f] = False
                for v in self.F[f]:
                    self.vfaces[v].discard(f)
            self.n_faces -= 2
        # 2) el resto de las caras de j ahora apuntan a i
        for f in self.vfaces[j]:
            self.F[f][self.F[f] == j] = i
            self.vfaces[i].add(f)
        self.vfaces[j] = set()
        # 3) posición y cuádrica del vértice nuevo:  Q̄ = Q₁ + Q₂
        self.V[i] = vbar
        self.Q[i] += self.Q[j]
        self.boundary[i] |= self.boundary[j]
        self.alive[j] = False
        self.version[i] += 1
        # 4) los pares de j pasan a ser pares de i
        for k in self.pairs[j]:
            self.pairs[k].discard(j)
            if k != i:
                self.pairs[k].add(i)
                self.pairs[i].add(k)
        self.pairs[i].discard(j)
        self.pairs[j] = set()
        # 5) recalcular el costo de todos los pares que tocan a i
        self._push(i, self.pairs[i])

    def _boundary_vertices(self):
        F = self.F
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        uniq, cnt = np.unique(E, axis=0, return_counts=True)
        b = np.zeros(len(self.V), bool)
        b[uniq[cnt == 1].ravel()] = True
        return b

    def finish_history(self):
        h, L = self.history, self._log
        h.keep = np.array(L["keep"], np.int64)
        h.removed = np.array(L["removed"], np.int64)
        h.newpos = np.array(L["newpos"], float).reshape(-1, 3)
        h.cost = np.array(L["cost"], float)
        h.how = np.array(L["how"], np.int8)
        h.virtual = np.array(L["virtual"], bool)
        h.nfaces = np.array(L["nfaces"], np.int64)
        h.stats = dict(self.stats)
        return h

    def current_mesh(self):
        """Malla actual compactada (sin vértices ni caras muertas)."""
        F = self.F[self.face_alive]
        ids = np.unique(F)
        remap = np.full(len(self.V), -1)
        remap[ids] = np.arange(len(ids))
        return self.V[ids], remap[F]


def simplify(V, F, target_faces, **kw):
    """Atajo: devuelve (V, F) simplificada a ≤ target_faces caras."""
    pc = PairCollapse(V, F, **kw)
    pc.simplify(target_faces)
    return pc.current_mesh()
