"""
Malla poligonal con estructura de half-edges, en forma vectorizada (numpy).

Una malla de subdivisión tiene caras de cualquier grado (triángulos, cuadriláteros,
pentágonos...). Las guardamos en formato CSR, igual que una matriz dispersa:

    fptr = [0, 4, 8, 11, ...]   inicio de cada cara en fidx
    fidx = [v0 v1 v2 v3 | v4 v5 v6 v7 | v8 v9 v10 | ...]

Cada posición de fidx es una *esquina* (corner) de una cara y, a la vez, el half-edge
que sale de ese vértice dentro de la cara:

    half-edge h  va de  src[h] = fidx[h]   a   dst[h] = fidx[nxt[h]]

Con eso tenemos, sin punteros ni objetos por elemento:
    nxt[h], prv[h]   siguiente/anterior half-edge en la misma cara
    face_of[h]       cara a la que pertenece h
    edge_of[h]       arista (no orientada) a la que pertenece h
    twin[h]          half-edge opuesto (en la cara vecina), -1 si h está en el borde

Aristas "afiladas" (sharp): bordes de la malla + pliegues marcados por el usuario.
Los esquemas CC y Loop usan reglas de curva sobre ellas.
"""
import numpy as np


class MeshError(ValueError):
    pass


class PolyMesh:
    def __init__(self, V, faces, crease_pairs=None, name=""):
        self.V = np.ascontiguousarray(V, dtype=np.float64)
        self.name = name
        if isinstance(faces, tuple):                       # (fptr, fidx)
            fptr, fidx = faces
            self.fptr = np.asarray(fptr, dtype=np.int64)
            self.fidx = np.asarray(fidx, dtype=np.int64)
        elif isinstance(faces, np.ndarray) and faces.ndim == 2:
            F, k = faces.shape
            self.fptr = np.arange(F + 1, dtype=np.int64) * k
            self.fidx = faces.astype(np.int64).ravel()
        else:                                              # lista de listas
            deg = np.array([len(f) for f in faces], dtype=np.int64)
            self.fptr = np.concatenate([[0], np.cumsum(deg)])
            self.fidx = np.fromiter((v for f in faces for v in f), dtype=np.int64,
                                    count=int(deg.sum()))
        self._build_topology()
        self.crease = np.zeros(self.nE, dtype=bool)        # pliegues marcados (no bordes)
        if crease_pairs is not None and len(crease_pairs):
            ids = self.edge_ids(np.asarray(crease_pairs))
            self.crease[ids[ids >= 0]] = True

    # ------------------------------------------------------------------ topología
    def _build_topology(self):
        nV = len(self.V)
        fptr, fidx = self.fptr, self.fidx
        self.nV, self.nF, self.nH = nV, len(fptr) - 1, len(fidx)
        self.deg = np.diff(fptr)
        if self.nF == 0:
            raise MeshError("la malla no tiene caras")
        if self.deg.min() < 3:
            raise MeshError("hay caras con menos de 3 vértices")
        H = self.nH
        self.face_of = np.repeat(np.arange(self.nF), self.deg)
        start = fptr[self.face_of]
        pos = np.arange(H) - start
        d = self.deg[self.face_of]
        self.nxt = start + (pos + 1) % d
        self.prv = start + (pos - 1) % d
        self.src = fidx
        self.dst = fidx[self.nxt]
        if np.any(self.src == self.dst):
            raise MeshError("hay caras con vértices repetidos consecutivos")

        # aristas no orientadas: clave única min*nV + max
        a, b = np.minimum(self.src, self.dst), np.maximum(self.src, self.dst)
        key = a * nV + b
        ukey, self.edge_of = np.unique(key, return_inverse=True)
        self.edge_of = self.edge_of.ravel()
        self.edges = np.stack([ukey // nV, ukey % nV], axis=1)
        self.nE = len(ukey)
        cnt = np.bincount(self.edge_of, minlength=self.nE)
        if cnt.max() > 2:
            raise MeshError(f"malla no-variedad: {int((cnt > 2).sum())} aristas con más de 2 caras")
        # orientación consistente: el mismo half-edge dirigido no puede aparecer 2 veces
        dkey = self.src * nV + self.dst
        if len(np.unique(dkey)) != H:
            raise MeshError("orientación inconsistente (caras vecinas recorren la arista "
                            "en el mismo sentido)")

        # twin: los dos half-edges de cada arista interior quedan contiguos al ordenar
        order = np.argsort(self.edge_of, kind="stable")
        e_sorted = self.edge_of[order]
        self.twin = np.full(H, -1, dtype=np.int64)
        same = np.nonzero(e_sorted[1:] == e_sorted[:-1])[0]
        h0, h1 = order[same], order[same + 1]
        self.twin[h0], self.twin[h1] = h1, h0

        self.boundary_edge = cnt == 1
        # un half-edge representante por arista y las (hasta) 2 caras de cada arista
        self.edge_he = np.full(self.nE, -1, dtype=np.int64)
        self.edge_he[self.edge_of[::-1]] = np.arange(H)[::-1]
        self.edge_faces = np.full((self.nE, 2), -1, dtype=np.int64)
        self.edge_faces[:, 0] = self.face_of[self.edge_he]
        tw = self.twin[self.edge_he]
        self.edge_faces[tw >= 0, 1] = self.face_of[tw[tw >= 0]]

        self.valence = np.bincount(self.edges.ravel(), minlength=nV)   # nº de aristas
        self.nfaces_v = np.bincount(fidx, minlength=nV)                # nº de caras
        bv = np.zeros(nV, dtype=bool)
        bv[self.edges[self.boundary_edge].ravel()] = True
        self.boundary_vertex = bv

    # ------------------------------------------------------------------ consultas
    def edge_ids(self, pairs):
        """Índice de arista para cada par (a, b); -1 si no existe."""
        pairs = np.atleast_2d(pairs)
        k = np.minimum(pairs[:, 0], pairs[:, 1]) * self.nV + np.maximum(pairs[:, 0], pairs[:, 1])
        ukey = self.edges[:, 0] * self.nV + self.edges[:, 1]
        i = np.searchsorted(ukey, k)
        i = np.clip(i, 0, len(ukey) - 1)
        return np.where(ukey[i] == k, i, -1)

    @property
    def sharp(self):
        """Aristas donde se usan reglas de curva: bordes y pliegues."""
        return self.boundary_edge | self.crease

    def face_list(self):
        fp, fi = self.fptr, self.fidx
        return [fi[fp[i]:fp[i + 1]].tolist() for i in range(self.nF)]

    def faces_for_display(self):
        """ndarray (F,k) si todas las caras tienen el mismo grado; si no, lista de listas."""
        if self.deg.min() == self.deg.max():
            return self.fidx.reshape(self.nF, int(self.deg[0]))
        return self.face_list()

    def is_triangle_mesh(self):
        return bool(np.all(self.deg == 3))

    def is_quad_mesh(self):
        return bool(np.all(self.deg == 4))

    def is_closed(self):
        return not self.boundary_edge.any()

    def euler(self):
        return self.nV - self.nE + self.nF

    def n_components(self):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        e = self.edges
        A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(self.nV, self.nV))
        return connected_components(A, directed=False)[0]

    def genus(self):
        """Solo tiene sentido en mallas cerradas: chi = 2c - 2g."""
        if not self.is_closed():
            return None
        return (2 * self.n_components() - self.euler()) / 2

    # --------------------------------------------------------- abanicos de vértice
    def fans(self):
        """
        Recorre las esquinas alrededor de cada vértice.
            succ(c) = twin[prv[c]]   esquina del mismo vértice en la cara siguiente
        Devuelve lista de (vértice, [esquinas en orden], cerrado?).
        En una variedad cada vértice tiene exactamente un abanico.
        """
        H = self.nH
        succ = np.where(self.twin[self.prv] >= 0, self.twin[self.prv], -1)
        visited = np.zeros(H, dtype=bool)
        out = []
        # abanicos abiertos: empiezan en esquinas sin predecesor (twin[c] == -1)
        succ_l = succ.tolist()
        for c in np.nonzero(self.twin < 0)[0].tolist():
            fan = []
            while c >= 0 and not visited[c]:
                visited[c] = True
                fan.append(c)
                c = succ_l[c]
            out.append((int(self.src[fan[0]]), fan, False))
        for c0 in np.nonzero(~visited)[0].tolist():
            if visited[c0]:
                continue
            fan, c = [], c0
            while not visited[c]:
                visited[c] = True
                fan.append(c)
                c = succ_l[c]
            out.append((int(self.src[c0]), fan, True))
        return out

    # ------------------------------------------------------------------ geometría
    def face_centroids(self):
        s = np.zeros((self.nF, 3))
        np.add.at(s, self.face_of, self.V[self.fidx])
        return s / self.deg[:, None]

    def face_normals(self, unit=True):
        """Normal de Newell (sirve para polígonos no planos). Magnitud = 2*área."""
        p, q = self.V[self.src], self.V[self.dst]
        n = np.zeros((self.nF, 3))
        np.add.at(n, self.face_of, np.cross(p, q))
        if unit:
            L = np.linalg.norm(n, axis=1, keepdims=True)
            n = n / np.maximum(L, 1e-300)
        return n

    def vertex_normals(self):
        fn = self.face_normals(unit=False)
        vn = np.zeros((self.nV, 3))
        np.add.at(vn, self.fidx, fn[self.face_of])
        return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-300)

    def triangles(self):
        """Triangulación en abanico (para volumen, muestreo, etc.)."""
        start = self.fptr[:-1]
        tris = []
        for k in np.unique(self.deg):
            fs = np.nonzero(self.deg == k)[0]
            s = start[fs]
            for j in range(1, k - 1):
                tris.append(np.stack([self.fidx[s], self.fidx[s + j], self.fidx[s + j + 1]], 1))
        return np.concatenate(tris)

    def area(self):
        return 0.5 * np.linalg.norm(self.face_normals(unit=False), axis=1).sum()

    def volume(self):
        if not self.is_closed():
            return None
        T = self.triangles()
        a, b, c = self.V[T[:, 0]], self.V[T[:, 1]], self.V[T[:, 2]]
        return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0

    def dihedral_angles(self):
        """Ángulo (grados) entre normales de caras vecinas, por arista interior."""
        inner = ~self.boundary_edge
        n = self.face_normals()
        f0, f1 = self.edge_faces[inner, 0], self.edge_faces[inner, 1]
        c = np.clip(np.einsum("ij,ij->i", n[f0], n[f1]), -1, 1)
        return np.degrees(np.arccos(c))

    def angle_defect(self):
        """Curvatura gaussiana discreta: 2pi - suma de ángulos (pi - suma en el borde)."""
        T = self.triangles()
        P = self.V[T]
        ang = np.zeros(self.nV)
        for i in range(3):
            u = P[:, (i + 1) % 3] - P[:, i]
            w = P[:, (i + 2) % 3] - P[:, i]
            cs = np.einsum("ij,ij->i", u, w) / np.maximum(
                np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-300)
            np.add.at(ang, T[:, i], np.arccos(np.clip(cs, -1, 1)))
        full = np.where(self.boundary_vertex, np.pi, 2 * np.pi)
        return full - ang

    def bbox_diag(self):
        return float(np.linalg.norm(self.V.max(0) - self.V.min(0)))

    # ------------------------------------------------------------------ utilidades
    def copy_with(self, V):
        m = object.__new__(PolyMesh)
        m.__dict__.update(self.__dict__)
        m.V = np.ascontiguousarray(V, dtype=np.float64)
        return m

    def triangulated(self):
        """
        Divide cada polígono en triángulos. Para no crear aristas repetidas (que harían la
        malla no-variedad), cada polígono se abanica desde el primer vértice cuyas
        diagonales todavía no existen. Los pliegues se conservan.
        """
        existing = set(map(tuple, self.edges.tolist()))
        tris = []
        for f in self.face_list():
            n = len(f)
            if n == 3:
                tris.append(f)
                continue
            best = None
            for s in range(n):
                g = f[s:] + f[:s]
                diags = [tuple(sorted((g[0], g[j]))) for j in range(2, n - 1)]
                if not any(d in existing for d in diags):
                    best = (g, diags)
                    break
            g, diags = best if best else (f, [])
            existing.update(diags)
            tris += [[g[0], g[j], g[j + 1]] for j in range(1, n - 1)]
        cp = self.edges[self.crease]
        return PolyMesh(self.V, np.array(tris), crease_pairs=cp, name=self.name)

    def mark_creases_by_angle(self, angle_deg):
        """Marca como pliegue toda arista interior con ángulo diedro > angle_deg."""
        self.crease = np.zeros(self.nE, dtype=bool)
        if angle_deg is None or angle_deg >= 180:
            return 0
        inner = np.nonzero(~self.boundary_edge)[0]
        self.crease[inner[self.dihedral_angles() > angle_deg]] = True
        return int(self.crease.sum())

    def summary(self):
        d = np.bincount(self.deg)
        degs = ", ".join(f"{k}-gonos: {int(c)}" for k, c in enumerate(d) if c)
        return (f"V={self.nV}  E={self.nE}  F={self.nF}  chi={self.euler()}  "
                f"bordes={int(self.boundary_edge.sum())}  pliegues={int(self.crease.sum())}  [{degs}]")


def remove_unused_vertices(V, faces_list):
    used = np.unique(np.concatenate([np.asarray(f) for f in faces_list]))
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], [remap[np.asarray(f)].tolist() for f in faces_list]


def split_nonmanifold_vertices(mesh):
    """
    Un vértice cuyo entorno está formado por más de un abanico de caras (dos conos que
    se tocan en un punto) no es variedad. Lo duplicamos: un vértice por abanico.
    """
    fans = mesh.fans()
    seen = {}
    V = list(mesh.V)
    fidx = mesh.fidx.copy()
    n_split = 0
    for v, fan, _ in fans:
        if v in seen:
            V.append(mesh.V[v])
            fidx[fan] = len(V) - 1
            n_split += 1
        else:
            seen[v] = True
    if n_split == 0:
        return mesh, 0
    m = PolyMesh(np.array(V), (mesh.fptr, fidx), crease_pairs=None, name=mesh.name)
    return m, n_split
