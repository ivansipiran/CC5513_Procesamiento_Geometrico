"""
Visor Polyscope del demo de subdivisión.

Panel:
  Malla de control   modelo, pliegues por ángulo, jaula
  Esquema            Catmull-Clark / Loop / Doo-Sabin + variante, nivel, split->average
  Colores            tipo de vértice, tipo de cara, valencia, curvatura, función base
  Estencil           fila de S: de qué vértices (y con qué pesos) sale un vértice nuevo
  Control local      columna de S_k...S_1: mover UN vértice de control y ver qué cambia
  Resultados         tabla por nivel (V, F, chi, irregulares, diedro máx., volumen)
"""
import time

import numpy as np

from . import _compat as C
from .models import load_model, SYNTHETIC, REMOTE
from .schemes import subdivide_step, prepare_control
from .limit import limit_positions
from .metrics import level_row

SCHEME_KEYS = ["catmull-clark", "loop", "doo-sabin"]
SCHEME_LABEL = {"catmull-clark": "Catmull-Clark", "loop": "Loop", "doo-sabin": "Doo-Sabin"}
VARIANTS = {
    "catmull-clark": [("standard", "reglas de Catmull-Clark"),
                      ("linear", "solo topologia (sin promediar)")],
    "loop": [("loop", "beta de Loop (1987)"),
             ("warren", "beta de Warren = 3/(8n)"),
             ("linear", "solo topologia (puntos medios)")],
    "doo-sabin": [("doo-sabin", "pesos de Doo-Sabin"),
                  ("simple", "simple: (P + F + M1 + M2)/4")],
}
COLOR_MODES = ["ninguno", "tipo de vertice", "tipo de cara", "valencia",
               "curvatura (defecto angular)"]

PAL = np.array([[0.93, 0.42, 0.18],   # naranjo
                [0.18, 0.52, 0.88],   # azul
                [0.30, 0.72, 0.35],   # verde
                [0.86, 0.72, 0.16],   # amarillo
                [0.62, 0.40, 0.82],   # morado
                [0.55, 0.55, 0.55]])  # gris
MESH_COLOR = (0.80, 0.82, 0.88)
MAX_FACES = 700_000


class Hierarchy:
    """Niveles 0..k de un (modelo, esquema, variante, pliegues, borde DS)."""

    def __init__(self, m, scheme, variant, ds_boundary):
        self.scheme, self.variant, self.ds_boundary = scheme, variant, ds_boundary
        m0, self.note = prepare_control(m, scheme)
        self.meshes = [m0]
        self.steps = []
        self.rows = {}
        self.times = [0.0]

    def can_refine(self):
        return self.meshes[-1].nF * 4 <= MAX_FACES

    def ensure(self, k):
        while len(self.meshes) <= k:
            if not self.can_refine():
                return False
            t0 = time.perf_counter()
            st = subdivide_step(self.meshes[-1], self.scheme, self.variant, self.ds_boundary)
            self.times.append(time.perf_counter() - t0)
            self.steps.append(st)
            self.meshes.append(st.fine)
        return True

    def row(self, k):
        if k not in self.rows:
            m0 = self.meshes[0]
            self.rows[k] = level_row(self.meshes[k], self.scheme, m0.volume(), m0.bbox_diag(),
                                     self.times[k])
        return self.rows[k]


class SubdivViewer:
    def __init__(self, model="cube", scheme="catmull-clark", variant=None, level=2,
                 crease=None, ds_boundary="chaikin", color="tipo de vertice", insecure=False,
                 cage=True, morph=1.0, limit=False, stencil=None, basis=None, delta=0.0,
                 side_by_side=False, max_level=6):
        self.models = list(SYNTHETIC) + list(REMOTE)
        if model not in self.models:
            self.models.append(model)
        self.model = model
        self.scheme = scheme
        self.variant = {s: VARIANTS[s][0][0] for s in SCHEME_KEYS}
        if variant:
            self.variant[scheme] = variant
        self.level = level
        self.max_level = max_level
        self.crease_on = crease is not None
        self.crease_angle = float(crease) if crease is not None else 40.0
        self.ds_boundary = ds_boundary
        self.color_mode = COLOR_MODES.index(color) if color in COLOR_MODES else 1
        self.insecure = insecure
        self.show_cage = cage
        self.cage_prev = False            # jaula = nivel 0 o nivel anterior
        self.morph = morph
        self.playing = False
        self.limit = limit
        self.stencil_on = stencil is not None
        self.stencil_i = int(stencil or 0)
        self.basis_on = basis is not None
        self.basis_j = int(basis or 0)
        self.delta = float(delta)
        self.side = side_by_side
        self.smooth = True
        self.show_edges = True
        self._base = {}
        self._hier = {}
        self._basis_cache = {}
        self.message = ""
        self.registered = set()

    # ------------------------------------------------------------------ datos
    def base_mesh(self, name):
        key = (name, self.crease_on, self.crease_angle)
        if key not in self._base:
            if name not in self._base:
                self._base[name] = load_model(name, insecure=self.insecure)
            m = self._base[name]
            mm = m.copy_with(m.V)
            mm.mark_creases_by_angle(self.crease_angle if self.crease_on else None)
            self._base[key] = mm
        return self._base[key]

    def hierarchy(self, scheme=None):
        scheme = scheme or self.scheme
        var = self.variant[scheme]
        key = (self.model, scheme, var, self.crease_on, self.crease_angle, self.ds_boundary)
        if key not in self._hier:
            self._hier[key] = Hierarchy(self.base_mesh(self.model), scheme, var, self.ds_boundary)
        return self._hier[key]

    def clamp_level(self, H):
        ok = H.ensure(self.level)
        if not ok:
            self.level = len(H.meshes) - 1
            self.message = f"nivel limitado a {self.level} (mas de {MAX_FACES} caras)"
        return self.level

    def basis_column(self, H, k):
        """x_k = S_k ... S_1 e_j : cuánto se mueve cada vértice del nivel k al mover P_j."""
        key = (id(H), self.basis_j)
        cols = self._basis_cache.setdefault(key, [])
        if not cols:
            x = np.zeros(H.meshes[0].nV)
            x[self.basis_j] = 1.0
            cols.append(x)
        while len(cols) <= k:
            cols.append(H.steps[len(cols) - 1].S @ cols[-1])
        return cols[k]

    def displaced(self, H, k):
        """Posiciones del nivel k con el vértice de control j desplazado delta en su normal."""
        P = H.meshes[k].V
        if self.basis_on and self.delta != 0.0:
            n = H.meshes[0].vertex_normals()[self.basis_j]
            P = P + self.delta * np.outer(self.basis_column(H, k), n)
        return P

    def positions(self, H, k):
        P = self.displaced(H, k)
        if k >= 1 and self.morph < 1.0:
            st = H.steps[k - 1]
            Pp = self.displaced(H, k - 1)
            P = (1 - self.morph) * (st.S_lin @ Pp) + self.morph * (st.S @ Pp)
        if self.limit:
            L = limit_positions(H.meshes[k].copy_with(P), H.scheme, H.variant)
            if L is not None:
                P = L
        return P

    # ------------------------------------------------------------------ dibujo
    def _remove(self, keep=()):
        import polyscope as ps
        for name in list(self.registered):
            if name in keep:
                continue
            try:
                if name.startswith("pc:"):
                    ps.remove_point_cloud(name[3:])
                elif name.startswith("cn:"):
                    ps.remove_curve_network(name[3:])
                else:
                    ps.remove_surface_mesh(name[3:])
            except Exception:
                pass
            self.registered.discard(name)

    def _surface(self, name, P, mesh):
        import polyscope as ps
        s = ps.register_surface_mesh(name, P, mesh.faces_for_display(), color=MESH_COLOR,
                                     smooth_shade=self.smooth,
                                     edge_width=1.0 if (self.show_edges and mesh.nF <= 60000) else 0.0)
        try:
            s.set_edge_color((0.15, 0.15, 0.2))
        except Exception:
            pass
        self.registered.add("sm:" + name)
        return s

    def _cage(self, name, P, mesh, color=(0.9, 0.3, 0.2), radius=0.0012):
        import polyscope as ps
        c = ps.register_curve_network(name, P, mesh.edges, color=color)
        c.set_radius(radius, relative=True)
        self.registered.add("cn:" + name)
        return c

    def rebuild(self):
        import polyscope as ps
        self._remove()
        self.message = ""
        if self.side:
            return self._rebuild_side()
        H = self.hierarchy()
        k = self.clamp_level(H)
        mesh = H.meshes[k]
        P = self.positions(H, k)
        self.surf = self._surface("subdivision", P, mesh)
        self._apply_colors(H, k, self.surf, P)
        if self.show_cage:
            kc = k - 1 if (self.cage_prev or self.stencil_on) and k >= 1 else 0
            if H.meshes[kc].nE > 6000:
                self.message = "jaula oculta: la malla de control es muy grande"
            elif kc != k or k == 0:
                self._cage("jaula", self.displaced(H, kc), H.meshes[kc])
        if mesh.crease.any():
            # solo los nodos usados (si no, Polyscope dibuja una esfera por cada vértice)
            used, ce = np.unique(mesh.edges[mesh.crease], return_inverse=True)
            self._crease_nodes = used
            cn = ps.register_curve_network("pliegues", P[used], ce.reshape(-1, 2),
                                           color=(0.95, 0.75, 0.1))
            cn.set_radius(0.0018, relative=True)
            self.registered.add("cn:pliegues")
        if self.stencil_on and k >= 1:
            self._draw_stencil(H, k, P)
        if self.basis_on:
            pj = self.displaced(H, 0)[self.basis_j][None]
            pc = ps.register_point_cloud("vertice de control j", pj, color=(0.95, 0.1, 0.5))
            pc.set_radius(0.012, relative=True)
            self.registered.add("pc:vertice de control j")

    def update_positions(self):
        """Solo mueve vértices (morph / delta / límite) sin re-registrar la malla."""
        if self.side or self.stencil_on:
            return self.rebuild()
        H = self.hierarchy()
        k = self.level
        P = self.positions(H, k)
        try:
            self.surf.update_vertex_positions(P)
        except Exception:
            return self.rebuild()
        if self.color_mode == 4 or "pc:vertices subdivision" in self.registered:
            self._apply_colors(H, k, self.surf, P)
        if "cn:pliegues" in self.registered:
            import polyscope as ps
            ps.get_curve_network("pliegues").update_node_positions(P[self._crease_nodes])
        if self.show_cage:
            import polyscope as ps
            kc = k - 1 if self.cage_prev and k >= 1 else 0
            if "cn:jaula" in self.registered:
                ps.get_curve_network("jaula").update_node_positions(self.displaced(H, kc))
        if self.basis_on and "pc:vertice de control j" in self.registered:
            import polyscope as ps
            ps.get_point_cloud("vertice de control j").update_point_positions(
                self.displaced(H, 0)[self.basis_j][None])

    def _rebuild_side(self):
        """Los tres esquemas lado a lado, mismo nivel, misma malla de control."""
        m = self.base_mesh(self.model)
        w = (m.V[:, 0].max() - m.V[:, 0].min()) * 1.25 + 0.05
        lv = self.level
        for i, sch in enumerate(SCHEME_KEYS):
            H = self.hierarchy(sch)
            ok = H.ensure(lv)
            k = lv if ok else len(H.meshes) - 1
            off = np.array([(i - 1) * w, 0, 0])
            P = self.positions(H, k) + off
            s = self._surface(SCHEME_LABEL[sch], P, H.meshes[k])
            self._apply_colors(H, k, s, P)
            if self.show_cage:
                self._cage("jaula " + SCHEME_LABEL[sch], H.meshes[0].V + off, H.meshes[0])

    def _apply_colors(self, H, k, surf, P):
        mode = COLOR_MODES[self.color_mode]
        mesh = H.meshes[k]
        if self.basis_on and not self.side:
            x = self.basis_column(H, k)
            C.add_scalar(surf, "funcion base", x, "vertices", cmap="reds", vminmax=(0.0, max(x.max(), 1e-9)))
            return
        if mode == "tipo de vertice" and k >= 1:
            st = H.steps[k - 1]
            self._points(surf, P, PAL[st.vkind % len(PAL)])
        elif mode == "tipo de cara" and k >= 1:
            st = H.steps[k - 1]
            C.add_colors(surf, "tipo de cara", PAL[st.fkind % len(PAL)], "faces")
        elif mode == "valencia":
            col, big = self._valence_colors(mesh, H.scheme)
            self._points(surf, P, col, big)
        elif mode.startswith("curvatura"):
            K = mesh.copy_with(P).angle_defect()
            lim = np.quantile(np.abs(K), 0.98) + 1e-12
            C.add_scalar(surf, "defecto angular", K, "vertices", cmap="coolwarm", vminmax=(-lim, lim))

    def _points(self, surf, P, colors, big=None):
        """Vértices como esferas de color (se leen mejor que colores interpolados)."""
        import polyscope as ps
        name = "vertices " + surf.get_name() if hasattr(surf, "get_name") else "vertices"
        n = len(P)
        pc = ps.register_point_cloud(name, P)
        C.add_colors(pc, "color", colors)
        r = 0.0045 if n < 3000 else (0.0025 if n < 30000 else 0.0012)
        pc.set_radius(r, relative=True)
        if big is not None:
            try:
                pc.add_scalar_quantity("radio", np.where(big, 2.0, 0.6), enabled=False)
                pc.set_point_radius_quantity("radio")
            except Exception:
                pass
        self.registered.add("pc:" + name)

    @staticmethod
    def _valence_colors(mesh, scheme):
        reg = 6 if scheme == "loop" else 4
        col = np.tile(np.array([0.78, 0.78, 0.80]), (mesh.nV, 1))
        v = mesh.valence
        col[v < reg] = PAL[0]
        col[v == reg - 1] = PAL[3] if reg == 6 else PAL[0]
        col[v > reg] = PAL[1]
        col[v > reg + 1] = PAL[4]
        col[mesh.boundary_vertex] = [0.25, 0.25, 0.25]
        big = (v != reg) & ~mesh.boundary_vertex
        return col, big

    def _draw_stencil(self, H, k, P):
        import polyscope as ps
        st = H.steps[k - 1]
        i = int(np.clip(self.stencil_i, 0, H.meshes[k].nV - 1))
        self.stencil_i = i
        row = st.S.getrow(i)
        Pp = self.displaced(H, k - 1)
        pts = Pp[row.indices]
        pc = ps.register_point_cloud("estencil (nivel k-1)", pts)
        C.add_scalar(pc, "peso", row.data, cmap="reds", vminmax=(0, row.data.max()))
        diag = H.meshes[0].bbox_diag()
        pc.set_radius(0.014, relative=True)
        try:
            pc.add_scalar_quantity("radio", 0.4 + row.data / row.data.max(), enabled=False)
            pc.set_point_radius_quantity("radio")
        except Exception:
            pass
        self.registered.add("pc:estencil (nivel k-1)")
        q = ps.register_point_cloud("vertice nuevo", P[i][None], color=(0.1, 0.8, 0.2))
        q.set_radius(0.012, relative=True)
        try:  # malla semitransparente para ver el estencil detrás
            ps.set_transparency_mode("pretty")
            self.surf.set_transparency(0.55)
        except Exception:
            pass
        self.registered.add("pc:vertice nuevo")
        nodes = np.concatenate([P[i][None], pts])
        edges = np.stack([np.zeros(len(pts), int), np.arange(1, len(pts) + 1)], 1)
        cn = ps.register_curve_network("mascara", nodes, edges, color=(0.2, 0.7, 0.3))
        cn.set_radius(0.0015, relative=True)
        self.registered.add("cn:mascara")

    # ------------------------------------------------------------------ panel
    def callback(self):
        import polyscope.imgui as psim
        changed = moved = False
        H = self.hierarchy()

        # ---------------- malla de control
        if C.collapsing(psim, "Malla de control"):
            psim.PushItemWidth(160)
            for i, name in enumerate(self.models):
                if i % 4:
                    psim.SameLine()
                if C.radio(psim, name, self.model == name):
                    if name != self.model:
                        try:
                            self.base_mesh(name)
                            self.model = name
                            self._basis_cache.clear()
                            self.basis_j = 0
                            self.stencil_i = 0
                            changed = True
                        except Exception as e:
                            self.message = f"no se pudo cargar {name}: {e}"
            m0 = H.meshes[0]
            psim.TextUnformatted(C.T(m0.summary()))
            if H.note:
                C.text_colored(psim, PAL[3], H.note)
            ch, self.crease_on = psim.Checkbox("pliegues por angulo diedro", self.crease_on)
            changed |= ch
            if self.crease_on:
                psim.SameLine()
                ch, v = psim.SliderFloat("umbral [grados]", self.crease_angle, 10.0, 120.0)
                if ch:
                    self.crease_angle = round(v)
                    changed = True
            ch, self.show_cage = psim.Checkbox("mostrar jaula", self.show_cage)
            changed |= ch
            psim.SameLine()
            ch, self.cage_prev = psim.Checkbox("jaula = nivel anterior", self.cage_prev)
            changed |= ch
            psim.PopItemWidth()

        # ---------------- esquema
        if C.collapsing(psim, "Esquema"):
            for i, s in enumerate(SCHEME_KEYS):
                if i:
                    psim.SameLine()
                if C.radio(psim, SCHEME_LABEL[s], self.scheme == s):
                    if s != self.scheme:
                        self.scheme = s
                        self.stencil_i = 0
                        changed = True
            ch, self.side = psim.Checkbox("los tres lado a lado", self.side)
            changed |= ch
            for key, label in VARIANTS[self.scheme]:
                if C.radio(psim, label, self.variant[self.scheme] == key):
                    if self.variant[self.scheme] != key:
                        self.variant[self.scheme] = key
                        changed = True
            if self.scheme == "doo-sabin":
                psim.TextUnformatted("borde:")
                psim.SameLine()
                for key, label in (("chaikin", "Chaikin (3/4, 1/4)"), ("free", "regla interior")):
                    if C.radio(psim, label, self.ds_boundary == key):
                        if self.ds_boundary != key:
                            self.ds_boundary = key
                            changed = True
                    psim.SameLine()
                psim.NewLine()
            psim.PushItemWidth(200)
            ch, lv = psim.SliderInt("nivel", self.level, 0, self.max_level)
            if ch:
                self.level = lv
                changed = True
            psim.SameLine()
            if psim.Button("-") and self.level > 0:
                self.level -= 1
                changed = True
            psim.SameLine()
            if psim.Button("+") and self.level < self.max_level:
                self.level += 1
                changed = True
            ch, t = psim.SliderFloat("dividir -> promediar", self.morph, 0.0, 1.0)
            if ch:
                self.morph = t
                moved = True
            psim.SameLine()
            if psim.Button("Play" if not self.playing else "Pausa"):
                self.playing = not self.playing
                if self.playing:
                    self.morph = 0.0
                    self._t0 = time.perf_counter()
            psim.PopItemWidth()
            if self.scheme != "doo-sabin":
                ch, self.limit = psim.Checkbox("proyectar al limite (mascara de limite)", self.limit)
                moved |= ch
            else:
                self.limit = False
            ch, self.smooth = psim.Checkbox("sombreado suave", self.smooth)
            changed |= ch
            psim.SameLine()
            ch, self.show_edges = psim.Checkbox("aristas", self.show_edges)
            changed |= ch
            for n in (H.steps[self.level - 1].notes if 0 < self.level <= len(H.steps) else []):
                C.text_colored(psim, PAL[3], n)

        # ---------------- colores
        if C.collapsing(psim, "Colores"):
            for i, name in enumerate(COLOR_MODES):
                if i % 3:
                    psim.SameLine()
                if C.radio(psim, name, self.color_mode == i):
                    if self.color_mode != i:
                        self.color_mode = i
                        changed = True
            self._legend(H)

        # ---------------- estencil
        if not self.side and C.collapsing(psim, "Estencil (fila de S)", default_open=False):
            ch, self.stencil_on = psim.Checkbox("mostrar estencil del vertice nuevo i", self.stencil_on)
            changed |= ch
            if self.stencil_on:
                if self.level == 0:
                    psim.TextUnformatted("suba al nivel >= 1")
                else:
                    changed |= self._stencil_panel(H, psim)

        # ---------------- control local / función base
        if not self.side and C.collapsing(psim, "Control local (columna de S)", default_open=False):
            ch, self.basis_on = psim.Checkbox("funcion base del vertice de control j", self.basis_on)
            changed |= ch
            if self.basis_on:
                changed |= self._basis_panel(H, psim)
                ch, d = psim.SliderFloat("desplazamiento (normal)", self.delta, -0.3, 0.3)
                if ch:
                    self.delta = d
                    moved = True

        # ---------------- resultados
        if C.collapsing(psim, "Resultados por nivel"):
            self._results(psim)
        if self.message:
            C.text_colored(psim, PAL[0], self.message)

        # animación split -> average
        if self.playing:
            self.morph = min(1.0, (time.perf_counter() - self._t0) / 2.0)
            moved = True
            if self.morph >= 1.0:
                self.playing = False

        if changed:
            self.rebuild()
        elif moved:
            self.update_positions()

    def _legend(self, H):
        import polyscope.imgui as psim
        mode = COLOR_MODES[self.color_mode]
        if self.basis_on and not self.side:
            psim.TextUnformatted("(la funcion base manda sobre el color)")
            return
        if mode in ("tipo de vertice", "tipo de cara"):
            if self.level == 0:
                psim.TextUnformatted("(nivel 0: sin tipos)")
                return
            st = H.steps[self.level - 1] if self.level <= len(H.steps) else None
            if st is None:
                return
            names = st.vkind_names if mode == "tipo de vertice" else st.fkind_names
            kinds = st.vkind if mode == "tipo de vertice" else st.fkind
            cnt = np.bincount(kinds, minlength=len(names))
            for i, nm in enumerate(names):
                if cnt[i]:
                    C.text_colored(psim, PAL[i % len(PAL)], f"  {nm}: {cnt[i]}")
        elif mode == "valencia":
            reg = 6 if self.scheme == "loop" else 4
            C.text_colored(psim, PAL[0], f"  valencia < {reg}")
            psim.TextUnformatted(f"  gris: {reg} (regular)")
            C.text_colored(psim, PAL[1], f"  valencia {reg + 1}")
            C.text_colored(psim, PAL[4], f"  valencia > {reg + 1}")
            psim.TextUnformatted("  gris oscuro: borde")
        elif mode.startswith("curvatura"):
            psim.TextUnformatted(C.T("  2π - Σ ángulos: rojo > 0 (elíptico), azul < 0 (silla)"))

    def _stencil_panel(self, H, psim):
        changed = False
        k = self.level
        st = H.steps[k - 1]
        n = H.meshes[k].nV
        ch, i = C.input_int(psim, "i (nivel k)", self.stencil_i)
        if ch:
            self.stencil_i = int(np.clip(i, 0, n - 1))
            changed = True
        for kind, nm in enumerate(st.vkind_names):
            idx = np.nonzero(st.vkind == kind)[0]
            if len(idx) == 0:
                continue
            if psim.Button(C.T(f"sig. {nm}")):
                nxt = idx[idx > self.stencil_i]
                self.stencil_i = int(nxt[0] if len(nxt) else idx[0])
                changed = True
            psim.SameLine()
        psim.NewLine()
        # vértices cuyo estencil toca un vértice extraordinario del nivel anterior
        reg = 6 if self.scheme == "loop" else 4
        mc = H.meshes[k - 1]
        irr = (mc.valence != reg) & ~mc.boundary_vertex
        if self.scheme == "doo-sabin":
            irr_new = np.nonzero(mc.deg[mc.face_of] != 4)[0]
        else:
            irr_new = np.nonzero((st.vkind == 0) & irr[np.minimum(st.vparent, mc.nV - 1)])[0]
        if len(irr_new) and psim.Button("sig. extraordinario"):
            nxt = irr_new[irr_new > self.stencil_i]
            self.stencil_i = int(nxt[0] if len(nxt) else irr_new[0])
            changed = True
        i = int(np.clip(self.stencil_i, 0, n - 1))
        psim.TextUnformatted(C.T(st.describe_vertex(i)))
        row = st.S.getrow(i)
        order = np.argsort(-row.data)
        from .explain import fr
        parts = [f"{fr(row.data[o])}*P{row.indices[o]}" for o in order]
        lines, cur = [], ""
        for p in parts:
            if len(cur) + len(p) > 60:
                lines.append(cur)
                cur = ""
            cur += ("" if not cur else " + ") + p
        lines.append(cur)
        for ln in lines:
            psim.TextUnformatted(C.T("  " + ln))
        psim.TextUnformatted(C.T(f"  {len(row.data)} vértices, suma de pesos = {row.data.sum():.6f}, "
                                 f"todos >= 0 -> combinación convexa"))
        return changed

    def _basis_panel(self, H, psim):
        changed = False
        m0 = H.meshes[0]
        ch, j = C.input_int(psim, "j (nivel 0)", self.basis_j)
        if ch:
            self.basis_j = int(np.clip(j, 0, m0.nV - 1))
            changed = True
        reg = 6 if self.scheme == "loop" else 4
        irr = np.nonzero((m0.valence != reg) & ~m0.boundary_vertex)[0]
        if len(irr):
            psim.SameLine()
            if psim.Button("sig. extraordinario##b"):
                nxt = irr[irr > self.basis_j]
                self.basis_j = int(nxt[0] if len(nxt) else irr[0])
                changed = True
        reg_v = np.nonzero((m0.valence == reg) & ~m0.boundary_vertex)[0]
        if len(reg_v):
            psim.SameLine()
            if psim.Button("sig. regular##b"):
                nxt = reg_v[reg_v > self.basis_j]
                self.basis_j = int(nxt[0] if len(nxt) else reg_v[0])
                changed = True
        k = self.level
        x = self.basis_column(H, k)
        supp = int((x > 1e-12).sum())
        psim.TextUnformatted(C.T(f"valencia de j: {int(m0.valence[self.basis_j])}   "
                                 f"soporte en nivel {k}: {supp} de {len(x)} vértices "
                                 f"({100 * supp / len(x):.1f} %)"))
        psim.TextUnformatted(C.T(f"máximo de la función base: {x.max():.3f} "
                                 f"(el vértice de control NO se interpola)"))
        return changed

    def _results(self, psim):
        H = self.hierarchy()
        rows = []
        for k in range(min(self.level, len(H.meshes) - 1) + 1):
            r = H.row(k)
            irr = f"{r['ev']}/{r['ef']}" if self.scheme != "loop" else str(r["ev"])
            vol = f"{r['vol']:.1f}" if r["vol"] is not None else "-"
            rows.append([k, r["V"], r["F"], r["chi"], irr, f"{r['dih_max']:.1f}", vol,
                         f"{1000 * r['t']:.0f}"])
        hdr = ["niv", "V", "F", "chi", "irr V/F" if self.scheme != "loop" else "irr V",
               "diedro max", "vol %", "ms"]
        C.table(psim, "res", hdr, rows)
        psim.TextUnformatted(C.T("chi = V - E + F no cambia: subdividir no cambia la topología"))


def set_camera(ps, v):
    """Vista 3/4 desde arriba; más lejos si están los tres esquemas lado a lado."""
    m = v.base_mesh(v.model)
    ext = m.V.max(0) - m.V.min(0)
    c = 0.5 * (m.V.max(0) + m.V.min(0))
    r = float(np.linalg.norm(ext))
    if v.side:
        r *= 1.9
    d = np.array([0.55, 0.45, 1.0])
    d = d / np.linalg.norm(d)
    try:
        ps.look_at(tuple(c + 1.35 * r * d), tuple(c))
    except Exception:
        pass


def run_viewer(v, shot=None, no_view=False, ui=False):
    import polyscope as ps
    C.init_polyscope(ps, headless=shot is not None and no_view)
    ps.set_program_name("Subdivision de mallas - CC5513")
    try:
        ps.set_up_dir("y_up")
    except Exception:
        pass
    C.set_ground(ps, "shadow_only")
    v.rebuild()
    ps.reset_camera_to_home_view()
    set_camera(ps, v)
    ps.set_user_callback(v.callback)
    if shot:
        for _ in range(4):
            C.frame_tick(ps)
        C.screenshot(ps, shot, ui)
        print(f"captura guardada en {shot}")
        if no_view:
            return
    if not no_view:
        ps.show()
