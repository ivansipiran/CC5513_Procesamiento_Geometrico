"""Grilla regular de celdas cubicas donde se evaluan las funciones implicitas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    origin: np.ndarray      # esquina (x0, y0, z0)
    h: float                # lado de la celda
    shape: tuple            # numero de VERTICES por eje (nx, ny, nz)

    @staticmethod
    def from_bounds(lo, hi, res: int, pad: float = 0.0) -> "Grid":
        """`res` = numero de celdas a lo largo del eje mas largo de la caja."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        ext = hi - lo
        h = ext.max() / res
        margin = max(pad * ext.max(), 2 * h)
        lo, hi = lo - margin, hi + margin
        shape = tuple(int(np.ceil((hi[a] - lo[a]) / h)) + 1 for a in range(3))
        return Grid(origin=lo, h=float(h), shape=shape)

    def axis(self, a: int) -> np.ndarray:
        return self.origin[a] + self.h * np.arange(self.shape[a])

    def points(self) -> np.ndarray:
        x, y, z = (self.axis(a) for a in range(3))
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        return np.c_[X.ravel(), Y.ravel(), Z.ravel()]

    def to_index(self, X: np.ndarray) -> np.ndarray:
        """Coordenadas continuas de indice (para interpolacion trilineal)."""
        return (X - self.origin) / self.h

    def sample(self, vals: np.ndarray, X: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates
        return map_coordinates(vals, self.to_index(X).T, order=1, mode="nearest")

    @property
    def n(self) -> int:
        return int(np.prod(self.shape))
