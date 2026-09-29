"""
Cuádricas de error — la matemática de Garland & Heckbert (SIGGRAPH 1997).

Todo lo que el algoritmo necesita saber de la geometría está en este archivo.
El resto (decimate.py) es la maquinaria de colapsos, que es la misma para
cualquier costo.

1. Un plano como vector de 4 números
------------------------------------
Un triángulo define un plano  a·x + b·y + c·z + d = 0  con  (a, b, c) = n
unitario.  Guardamos el plano como  p = (a, b, c, d).  Para un punto v, con
coordenadas homogéneas  v̂ = (x, y, z, 1):

        distancia(v, plano) = n·v + d = pᵀ v̂           (con signo)
        distancia²          = (pᵀ v̂)² = v̂ᵀ (p pᵀ) v̂

La matriz 4×4  K_p = p pᵀ  es la *cuádrica fundamental* del plano.

2. Muchos planos = una sola matriz
----------------------------------
La suma de distancias al cuadrado a varios planos es

        Δ(v) = Σ_p (pᵀ v̂)² = v̂ᵀ ( Σ_p K_p ) v̂ = v̂ᵀ Q v̂

así que basta guardar Q = Σ K_p (simétrica: 10 números) y olvidar los planos.
Al colapsar dos vértices, la cuádrica del nuevo vértice es Q₁ + Q₂: sigue
midiendo la distancia a TODOS los planos originales que "representa".

3. El mejor punto para colapsar
-------------------------------
Escribimos  Q = [[A, b], [bᵀ, c]]  (A es 3×3, b ∈ R³, c escalar). Entonces

        Δ(v) = vᵀ A v + 2 bᵀ v + c
        ∇Δ(v) = 2 (A v + b) = 0     ⇒     v̄ = −A⁻¹ b      (si A es invertible)
        Δ(v̄) = c + bᵀ v̄

A = Σ n nᵀ (ponderada) resume las normales de los planos:
  · todas las normales iguales (zona plana)   → rango 1: un plano de mínimos
  · dos familias de normales (arista viva)     → rango 2: una recta de mínimos
  · tres o más direcciones (esquina)           → rango 3: un único mínimo
Las superficies de nivel Δ(v) = cte son elipsoides (discos, cigarros o
esferas en esos tres casos). Ver `quadric_ellipsoids`.
"""
import numpy as np


# ---------------------------------------------------------------------------
# 1. Planos de las caras y cuádricas fundamentales
# ---------------------------------------------------------------------------
def face_planes(V, F):
    """Plano p = (a, b, c, d) de cada triángulo, con |(a,b,c)| = 1, y su área.

    Devuelve  P (m, 4)  y  area (m,).  Caras degeneradas quedan con p = 0.
    """
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)          # normal no normalizada, |cross| = 2·área
    twice_area = np.linalg.norm(cross, axis=1)
    n = np.zeros_like(cross)
    ok = twice_area > 1e-20
    n[ok] = cross[ok] / twice_area[ok, None]
    d = -np.einsum("ij,ij->i", n, v0)           # el plano pasa por v0:  n·v0 + d = 0
    P = np.concatenate([n, d[:, None]], axis=1)
    return P, 0.5 * twice_area


def fundamental_quadric(p):
    """K_p = p pᵀ para un plano p = (a, b, c, d) (o un arreglo (m, 4) de planos)."""
    p = np.asarray(p, dtype=float)
    return p[..., :, None] * p[..., None, :]


def face_quadrics(V, F, weighting="area"):
    """Una cuádrica por cara:  w_f · p_f p_fᵀ.

    weighting = "area": w_f = área del triángulo (Garland, tesis 1999). La suma
                        de cuádricas aproxima una integral sobre la superficie y no
                        depende de cuán fina esté la teselación.
    weighting = "none": w_f = 1 (paper original de 1997). Zonas con muchos
                        triángulos chicos "pesan" más que zonas con pocos grandes.
    """
    P, area = face_planes(V, F)
    K = fundamental_quadric(P)
    if weighting == "area":
        K = K * area[:, None, None]
    elif weighting != "none":
        raise ValueError(f"weighting desconocido: {weighting}")
    return K


# ---------------------------------------------------------------------------
# 2. Cuádricas de vértice (Paso 1 del algoritmo)
# ---------------------------------------------------------------------------
def vertex_quadrics(V, F, weighting="area", boundary_weight=0.0):
    """Q_v = suma de las cuádricas de las caras que tocan a v (+ restricciones de borde).

    Con boundary_weight > 0 se agregan las restricciones de borde del paper
    (sección 5): por cada arista de borde se construye un plano que contiene la
    arista y es PERPENDICULAR a su cara. Alejarse de ese plano significa mover el
    borde hacia adentro o hacia afuera, así que el borde se conserva.
    """
    n = len(V)
    Q = np.zeros((n, 4, 4))
    Kf = face_quadrics(V, F, weighting)
    for corner in range(3):                      # cada cara suma su K a sus 3 vértices
        np.add.at(Q, F[:, corner], Kf)
    if boundary_weight > 0:
        Qb = boundary_quadrics(V, F, boundary_weight, weighting)
        Q += Qb
    return Q


def boundary_edges_with_faces(F):
    """Aristas que pertenecen a una sola cara: (u, w, cara), con el sentido de la cara."""
    m = len(F)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    face_of = np.tile(np.arange(m), 3)
    key = np.sort(E, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    on_border = counts[inv] == 1
    return E[on_border], face_of[on_border]


def boundary_quadrics(V, F, weight, weighting="area"):
    """Cuádricas de los planos perpendiculares a las aristas de borde."""
    Q = np.zeros((len(V), 4, 4))
    E, fid = boundary_edges_with_faces(F)
    if len(E) == 0:
        return Q
    P, _ = face_planes(V, F[fid])
    nf = P[:, :3]                                 # normal de la cara que contiene la arista
    u, w = V[E[:, 0]], V[E[:, 1]]
    e = w - u
    m = np.cross(e, nf)                           # ⟂ a la arista y ⟂ a la normal de la cara
    mlen = np.linalg.norm(m, axis=1)
    ok = mlen > 1e-20
    m[ok] /= mlen[ok, None]
    d = -np.einsum("ij,ij->i", m, u)
    K = fundamental_quadric(np.concatenate([m, d[:, None]], axis=1))
    wgt = weight * (np.einsum("ij,ij->i", e, e) if weighting == "area" else 1.0)
    K = K * np.broadcast_to(wgt, (len(E),))[:, None, None]
    np.add.at(Q, E[:, 0], K)
    np.add.at(Q, E[:, 1], K)
    return Q


# ---------------------------------------------------------------------------
# 3. Evaluar y minimizar Δ(v) = v̂ᵀ Q v̂
# ---------------------------------------------------------------------------
def split(Q):
    """Q = [[A, b], [bᵀ, c]]  →  (A, b, c)."""
    return Q[..., :3, :3], Q[..., :3, 3], Q[..., 3, 3]


def quadric_error(Q, v):
    """Δ(v) = vᵀ A v + 2 bᵀ v + c   (vectorizado sobre la primera dimensión)."""
    A, b, c = split(Q)
    Av = np.einsum("...ij,...j->...i", A, v)
    err = np.einsum("...i,...i->...", v, Av) + 2.0 * np.einsum("...i,...i->...", b, v) + c
    return np.maximum(err, 0.0)                   # es ≥ 0; el redondeo puede dar -1e-18


# Cómo se eligió v̄ en cada colapso (para estadísticas y colores).
HOW_SOLVE, HOW_SEGMENT, HOW_SUBSET, HOW_PINV = 0, 1, 2, 3
HOW_NAMES = {HOW_SOLVE: "A⁻¹b (A invertible)",
             HOW_SEGMENT: "mejor punto del segmento",
             HOW_SUBSET: "mejor de v₁, v₂, punto medio",
             HOW_PINV: "pseudo-inversa (A singular)"}

PLACEMENTS = ("optimal", "svd", "subset", "midpoint")
PLACEMENT_HELP = {
    "optimal": "paper: v̄ = −A⁻¹b si A es invertible;\nsi no, el mejor punto del segmento v₁v₂;\nsi tampoco, el mejor de v₁, v₂ y el punto medio",
    "svd": "v̄ = medio + A⁺(−A·medio − b)  (pseudo-inversa):\nel mínimo de Δ más cercano al punto medio",
    "subset": "el mejor de {v₁, v₂, punto medio}\n(sin resolver ningún sistema)",
    "midpoint": "siempre el punto medio\n(Δ solo decide el ORDEN de los colapsos)",
}


def _best_of_subset(Qs, v1, v2):
    """Evalúa Δ en v₁, v₂ y el punto medio; devuelve el mejor de los tres."""
    cands = np.stack([v1, v2, 0.5 * (v1 + v2)], axis=1)          # (k, 3, 3)
    errs = quadric_error(Qs[:, None], cands)                      # (k, 3)
    best = np.argmin(errs, axis=1)
    idx = np.arange(len(v1))
    return cands[idx, best], errs[idx, best]


def optimal_placement(Qs, v1, v2, strategy="optimal", rcond=1e-5):
    """Posición v̄ y costo Δ(v̄) para colapsar pares (v₁, v₂) con cuádrica Qs = Q₁ + Q₂.

    Vectorizado: Qs (k,4,4), v1, v2 (k,3). Devuelve vbar (k,3), cost (k,), how (k,).

    strategy = "optimal" (paper, sección 4):
        1) si A es invertible:  v̄ = −A⁻¹ b
        2) si no, el mínimo de Δ sobre el segmento v₁v₂
        3) si tampoco, el mejor de {v₁, v₂, punto medio}
       "A invertible" = λ_min(A) > rcond · λ_max(A)  (número de condición acotado).
    """
    k = len(v1)
    how = np.full(k, HOW_SUBSET, dtype=np.int8)
    vbar = np.empty((k, 3))
    A, b, _ = split(Qs)
    mid = 0.5 * (v1 + v2)

    if strategy == "midpoint":
        return mid, quadric_error(Qs, mid), how
    if strategy == "subset":
        vb, cost = _best_of_subset(Qs, v1, v2)
        return vb, cost, how
    if strategy == "svd":
        # Pseudo-inversa truncada: resuelve A x = −(A·mid + b) con x ⟂ núcleo de A.
        # En una zona plana (rango 1) mueve el punto medio SOLO en la dirección normal.
        lam, U = np.linalg.eigh(A)                                  # A simétrica
        r = -(np.einsum("kij,kj->ki", A, mid) + b)
        coef = np.einsum("kji,kj->ki", U, r)                        # Uᵀ r
        keep = lam > rcond * np.maximum(lam[:, -1:], 1e-300)
        inv = np.where(keep, 1.0 / np.where(keep, lam, 1.0), 0.0)
        vb = mid + np.einsum("kij,kj->ki", U, coef * inv)
        how[:] = np.where(keep.all(axis=1), HOW_SOLVE, HOW_PINV)
        return vb, quadric_error(Qs, vb), how
    if strategy != "optimal":
        raise ValueError(f"estrategia desconocida: {strategy}")

    # 1) ¿A invertible?  (eigvalsh devuelve los autovalores en orden creciente)
    lam = np.linalg.eigvalsh(A)
    invertible = lam[:, 0] > rcond * np.maximum(lam[:, 2], 1e-300)
    if invertible.any():
        vbar[invertible] = np.linalg.solve(A[invertible], -b[invertible][..., None])[..., 0]
        how[invertible] = HOW_SOLVE

    # 2) Segmento v(t) = v₁ + t·e,  e = v₂ − v₁,  t ∈ [0, 1]
    #    Δ(t) = cuadrática en t;  dΔ/dt = 2 eᵀ(A v₁ + b) + 2 t eᵀ A e = 0
    rest = ~invertible
    if rest.any():
        e = v2[rest] - v1[rest]
        Ar, br = A[rest], b[rest]
        eAe = np.einsum("ki,kij,kj->k", e, Ar, e)
        slope = np.einsum("ki,ki->k", e, np.einsum("kij,kj->ki", Ar, v1[rest]) + br)
        seg_ok = eAe > rcond * np.maximum(lam[rest, 2], 1e-300) * np.einsum("ki,ki->k", e, e)
        t = np.clip(-slope / np.where(seg_ok, eAe, 1.0), 0.0, 1.0)
        vseg = v1[rest] + t[:, None] * e
        # 3) Donde el segmento tampoco sirve (Δ constante sobre él): mejor de tres
        vsub, _ = _best_of_subset(Qs[rest], v1[rest], v2[rest])
        vbar[rest] = np.where(seg_ok[:, None], vseg, vsub)
        how_rest = np.where(seg_ok, HOW_SEGMENT, HOW_SUBSET)
        how[rest] = how_rest

    return vbar, quadric_error(Qs, vbar), how


# ---------------------------------------------------------------------------
# 4. Interpretación geométrica: rango de A y elipsoides de error
# ---------------------------------------------------------------------------
RANK_NAMES = {1: "plano", 2: "pliegue", 3: "esquina"}


def face_weight(V, F, weighting="area"):
    """Peso total de las caras de cada vértice (= traza de A sin restricciones de borde).

    Sirve de escala de referencia: un autovalor de A es "grande" si es comparable
    a este peso. (No se puede usar λ_max: con restricciones de borde, λ_max es el
    peso de borde, que es enorme y achicaría todo lo demás.)"""
    _, area = face_planes(V, F)
    w = area if weighting == "area" else np.ones(len(F))
    W = np.zeros(len(V))
    for corner in range(3):
        np.add.at(W, F[:, corner], w)
    return W


def effective_rank(Q, tol=0.05, ref=None):
    """Rango "efectivo" de A: cuántos autovalores superan tol·ref.

    ref = peso de las caras del vértice (face_weight); si no se da, λ_max.
    1 = zona plana, 2 = pliegue / arista viva / borde, 3 = esquina."""
    lam = np.linalg.eigvalsh(Q[..., :3, :3])
    if ref is None:
        ref = lam[..., 2]
    ref = np.maximum(np.asarray(ref, float), 1e-300)
    return np.clip((lam > tol * ref[..., None]).sum(axis=-1), 1, 3)


def quadric_ellipsoids(Q, centers, size, ref=None, max_ratio=4.0, min_ratio=0.12, template=None):
    """Malla con un elipsoide {v : (v−x)ᵀ A (v−x) = ε} por cuádrica.

    El nivel es ε = ref·size², con ref = peso de las caras del vértice (o λ_max).
    Un eje con autovalor λ mide  size·√(ref/λ):
      · dirección normal (λ ≈ ref)          → size
      · direcciones "gratis" (λ ≈ 0)        → se recortan a max_ratio·size
      · dirección con restricción de borde  → muy corta (min_ratio·size)
    Así lo que se ve es la FORMA:
      disco   → zona plana: el vértice se puede mover libremente en el plano
      cigarro → pliegue o borde: solo se puede mover a lo largo de la arista
      esfera  → esquina: cualquier movimiento cuesta
    """
    if template is None:
        template = unit_sphere(2)
    Vt, Ft = template
    A = Q[:, :3, :3]
    lam, U = np.linalg.eigh(A)                    # columnas de U = ejes del elipsoide
    lam = np.maximum(lam, 1e-300)
    if ref is None:
        ref = lam[:, 2]
    radii = size * np.sqrt(np.asarray(ref, float)[:, None] / lam)
    radii = np.clip(radii, min_ratio * size, max_ratio * size)
    # vértice del elipsoide = centro + U · diag(r) · (punto de la esfera unitaria)
    pts = np.einsum("kij,kj,tj->kti", U, radii, Vt) + centers[:, None, :]
    nt = len(Vt)
    faces = (Ft[None, :, :] + (np.arange(len(Q)) * nt)[:, None, None]).reshape(-1, 3)
    return pts.reshape(-1, 3), faces


def unit_sphere(subdiv=2):
    """Icosfera unitaria (para dibujar elipsoides)."""
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    for _ in range(subdiv):
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        uniq, inv = np.unique(E, axis=0, return_inverse=True)
        inv = inv.ravel()
        mids = V[uniq[:, 0]] + V[uniq[:, 1]]
        mids /= np.linalg.norm(mids, axis=1, keepdims=True)
        m = len(F)
        a, b, c = (inv[:m] + len(V), inv[m:2 * m] + len(V), inv[2 * m:] + len(V))
        V = np.vstack([V, mids])
        F = np.concatenate([np.stack([F[:, 0], a, c], 1), np.stack([F[:, 1], b, a], 1),
                            np.stack([F[:, 2], c, b], 1), np.stack([a, b, c], 1)])
    return V, F
