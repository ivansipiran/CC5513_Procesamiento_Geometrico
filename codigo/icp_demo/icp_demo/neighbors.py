"""Backends de busqueda del vecino mas cercano.

Este es el cuello de botella de ICP: en cada iteracion hay que encontrar, para
cada uno de los N puntos de la nube fuente, el punto mas cercano entre los M de
la nube objetivo.

Se implementan tres estrategias para poder compararlas en clase:

  brute-loop : version "de libro", un bucle por consulta. O(N*M) con overhead
               de Python. Sirve para ver el costo real del enfoque ingenuo.
  brute      : misma complejidad O(N*M) pero vectorizada con la identidad
               ||p-q||^2 = ||p||^2 - 2 p.q + ||q||^2, que delega el trabajo a
               BLAS. Rapida en constantes, pero sigue siendo cuadratica.
  kdtree     : arbol k-d (scipy.spatial.cKDTree). Construccion O(M log M) una
               sola vez, consultas ~O(log M) amortizado. Es la version
               optimizada que se usa en la practica.

Todas exponen la misma interfaz: build(target) y query(source) -> (dist, idx).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import cKDTree


class NNBackend(ABC):
    name = "base"

    def __init__(self) -> None:
        self.build_time = 0.0
        self.query_time = 0.0
        self.n_queries = 0

    def build(self, target: np.ndarray) -> "NNBackend":
        t0 = time.perf_counter()
        self._build(np.ascontiguousarray(target, dtype=np.float64))
        self.build_time = time.perf_counter() - t0
        return self

    def query(self, source: np.ndarray):
        """Devuelve (distancias (n,), indices (n,)) del vecino mas cercano."""
        t0 = time.perf_counter()
        dist, idx = self._query(np.ascontiguousarray(source, dtype=np.float64))
        dt = time.perf_counter() - t0
        self.query_time += dt
        self.n_queries += len(source)
        self.last_query_time = dt
        return dist, idx

    @abstractmethod
    def _build(self, target: np.ndarray) -> None: ...

    @abstractmethod
    def _query(self, source: np.ndarray): ...


class BruteForceLoop(NNBackend):
    """Version ingenua: un bucle Python por punto consultado. O(N*M)."""

    name = "brute-loop"

    def _build(self, target: np.ndarray) -> None:
        self.target = target

    def _query(self, source: np.ndarray):
        n = len(source)
        dist = np.empty(n)
        idx = np.empty(n, dtype=np.int64)
        Q = self.target
        for i in range(n):
            d2 = np.sum((Q - source[i]) ** 2, axis=1)   # recorre TODO el objetivo
            j = int(np.argmin(d2))
            idx[i] = j
            dist[i] = np.sqrt(d2[j])
        return dist, idx


class BruteForceVectorized(NNBackend):
    """Fuerza bruta vectorizada por bloques. Sigue siendo O(N*M)."""

    name = "brute"

    def __init__(self, block_entries: int = 4_000_000) -> None:
        super().__init__()
        # el bloque se dimensiona por numero de distancias calculadas a la vez,
        # para que la matriz temporal quepa en cache/RAM sin penalizar el tiempo
        self.block_entries = block_entries
        self.chunk = 1024

    def _build(self, target: np.ndarray) -> None:
        self.target = target
        self.target_sqnorm = np.einsum("ij,ij->i", target, target)
        self.chunk = int(max(64, min(8192, self.block_entries // max(len(target), 1))))

    def _query(self, source: np.ndarray):
        n = len(source)
        dist = np.empty(n)
        idx = np.empty(n, dtype=np.int64)
        Q, qn = self.target, self.target_sqnorm
        for s in range(0, n, self.chunk):
            e = min(s + self.chunk, n)
            blk = source[s:e]
            # ||p||^2 - 2 p.q + ||q||^2 ; el termino ||p||^2 no afecta el argmin
            d2 = qn[None, :] - 2.0 * (blk @ Q.T)
            j = np.argmin(d2, axis=1)
            idx[s:e] = j
            dist[s:e] = np.sqrt(np.maximum(
                d2[np.arange(e - s), j] + np.einsum("ij,ij->i", blk, blk), 0.0))
        return dist, idx


class KDTreeBackend(NNBackend):
    """Arbol k-d de scipy: construccion O(M log M), consulta ~O(log M)."""

    name = "kdtree"

    def __init__(self, leafsize: int = 16, workers: int = -1) -> None:
        super().__init__()
        self.leafsize = leafsize
        self.workers = workers

    def _build(self, target: np.ndarray) -> None:
        self.tree = cKDTree(target, leafsize=self.leafsize)

    def _query(self, source: np.ndarray):
        dist, idx = self.tree.query(source, k=1, workers=self.workers)
        return np.asarray(dist), np.asarray(idx, dtype=np.int64)


BACKENDS = {
    "brute-loop": BruteForceLoop,
    "brute": BruteForceVectorized,
    "kdtree": KDTreeBackend,
}


def make_backend(name: str) -> NNBackend:
    if name not in BACKENDS:
        raise KeyError(f"Backend '{name}' desconocido. Opciones: {', '.join(BACKENDS)}")
    return BACKENDS[name]()
