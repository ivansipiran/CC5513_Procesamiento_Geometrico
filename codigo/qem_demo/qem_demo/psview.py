"""
Visor Polyscope del demo de simplificación.

Mecánica (igual que los demos anteriores): se calcula una vez y se explora en el
panel. Para los métodos de colapso se guarda la secuencia COMPLETA de colapsos
(decimate.History), así que el slider de caras, los botones ±1 y Play recorren
el historial sin volver a correr el algoritmo.

Estructuras en la escena
  original            la malla de entrada (a la izquierda)
  simplificada        el resultado actual
  próximo colapso     la arista que saldrá del heap a continuación (v₁ azul,
                      v₂ rojo, destino v̄ amarillo)
  elipsoides          superficies de nivel de las cuádricas de los vértices
"""
import os
import time

import numpy as np
import polyscope as ps

from . import _compat as C
from ._compat import T, psim
from . import baselines, data, meshio, metrics, methods, topology
from . import quadrics as qd

RANK_COLORS = {1: (0.78, 0.82, 0.90), 2: (1.00, 0.58, 0.10), 3: (0.85, 0.10, 0.12)}
COL_V1, COL_V2, COL_VBAR = (0.15, 0.45, 1.0), (0.95, 0.15, 0.15), (1.0, 0.85, 0.0)
MESH_COLOR = (0.93, 0.88, 0.80)

VIEW_METHODS = [("qem", "QEM — Garland & Heckbert"), ("length", "arista más corta (punto medio)"),
                ("clustering", "clustering de vértices (promedio)"),
                ("clustering-qem", "clustering + cuádricas"), ("open3d", "Open3D (quadric)")]
COLOR_MODES = [("none", "sin color"), ("rank", "tipo de vértice (rango de A)"),
               ("qerr", "error de la cuádrica √(Δ/w)  [‰]"),
               ("dist", "distancia a la original  [‰]"), ("quality", "calidad de triángulos")]
PAPER_COW = [994, 532, 248, 64]          # Garland & Heckbert 1997, figura 5


def _remove(kind, name):
    has, rm = {"mesh": (ps.has_surface_mesh, ps.remove_surface_mesh),
               "points": (ps.has_point_cloud, ps.remove_point_cloud),
               "curve": (ps.has_curve_network, ps.remove_curve_network)}[kind]
    if has(name):
        rm(name)


class Viewer:
    def __init__(self, args):
        self.args = args
        self.model = args.model
        self.method = args.method
        self.placement = args.placement
        self.area_weight = args.weighting == "area"
        self.border = args.boundary_weight > 0
        self.border_w = args.boundary_weight if args.boundary_weight > 0 else 1000.0
        self.topology = not args.no_topology
        self.flips = not args.no_flip_check
        self.pair_t = args.pair_threshold
        self.color_mode = "rank"
        self.rank_tol = 0.05
        self.show_edges = True
        self.show_original = True
        self.show_next = True
        self.show_pair_ellipsoid = False
        self.show_ellipsoids = False
        self.ell_scale = 1.0
        self.ell_max = 2000
        self.playing = False
        self.speed = 5
        self.auto_metrics = True
        self.hcache, self.ocache = {}, {}
        self.rows = []
        self.metrics = None
        self.metrics_dirty = True
        self.last_change = 0.0
        self.message = ""
        self.pending_recompute = False
        self.load_model(self.model, args.target)

    # ------------------------------------------------------------ datos
    def load_model(self, name, target=None):
        print(f"[modelo] {name}", flush=True)
        self.model = name
        self.V0, self.F0 = data.load_model(name)
        self.info0 = topology.summary(self.V0, self.F0)
        self.orig_dist = None
        self.metrics = None
        self.hcache = {}
        self.ocache = {}
        nf = len(self.F0)
        if not target:
            target = 994 if name == "cow" else max(nf // 10, 4)     # 994 = figura 5 del paper
        self.target = int(target)
        self.target = int(np.clip(self.target, 1, nf))
        w = np.ptp(self.V0[:, 0])
        self.offset = np.array([-1.15 * w - 0.05, 0, 0])
        self.edge_len0 = topology.mean_edge_length(self.V0, self.F0)
        self.register_original()
        self.compute()
        self.set_camera()

    def params(self):
        return dict(placement=self.placement, weighting="area" if self.area_weight else "none",
                    boundary_weight=self.border_w if self.border else 0.0,
                    preserve_topology=self.topology, check_flips=self.flips,
                    pair_threshold=self.pair_t)

    def is_collapse(self):
        return methods.METHODS[self.method][0] == "collapse"

    def compute(self):
        """Obtiene el historial (métodos de colapso) y actualiza la escena."""
        self.hist = None
        if self.is_collapse():
            kw = self.params()
            kw.update(methods.METHODS[self.method][1])
            if self.method != "qem":
                kw.pop("placement")
            key = tuple(sorted(kw.items()))
            if key not in self.hcache:
                t0 = time.perf_counter()
                print(f"[{self.method}] simplificando {len(self.F0)} caras hasta el final ...", flush=True)
                self.hcache[key] = methods.collapse_history(
                    self.V0, self.F0, {}, kw,
                    progress=lambda nf: print(f"    quedan {nf} caras", flush=True))
                h = self.hcache[key]
                print(f"    {h.steps} colapsos en {time.perf_counter() - t0:.1f} s; mínimo {h.nfaces[-1]} caras",
                      flush=True)
            self.hist = self.hcache[key]
            self.k = self.hist.step_for_faces(self.target)
        self.update_mesh()

    def simplified_other(self):
        key = (self.method, self.target)
        if key not in self.ocache:
            t0 = time.perf_counter()
            if self.method.startswith("clustering"):
                rep = "qem" if self.method == "clustering-qem" else "mean"
                V, F, res = baselines.clustering_to_target(self.V0, self.F0, self.target, rep)
                note = f"grilla de {res} celdas en el eje más largo"
            else:
                V, F = baselines.open3d_quadric(self.V0, self.F0, self.target)
                note = ""
            self.ocache[key] = (V, F, time.perf_counter() - t0, note)
        return self.ocache[key]

    # ------------------------------------------------------------ escena
    def register_original(self):
        m = ps.register_surface_mesh("original", self.V0 + self.offset, self.F0, color=MESH_COLOR,
                                     edge_width=1.0 if len(self.F0) < 40000 else 0.0,
                                     smooth_shade=False)
        m.set_enabled(self.show_original)

    def update_mesh(self):
        h = self.hist
        if h is not None:
            V, F, ids, parent = h.state(self.k)
            self.cur = dict(V=V, F=F, ids=ids, Q=h.quadrics_at(self.k, parent, ids),
                            W=h.weights_at(self.k, parent, ids), time=None, note="")
            self.target_faces = len(F)
        else:
            if self.method == "open3d" and not baselines.has_open3d():
                self.message = "Open3D no está instalado (pip install open3d)"
                self.method = "qem"
                return self.compute()
            V, F, t, note = self.simplified_other()
            self.cur = dict(V=V, F=F, ids=None, Q=None, W=None, time=t, note=note)
        # conteos baratos ahora; χ, género, etc. cuando el slider se queda quieto
        self.cur["info"] = dict(V=len(np.unique(F)), F=len(F))
        self.cur["full_info"] = False
        m = ps.register_surface_mesh("simplificada", V, F, color=MESH_COLOR,
                                     edge_width=1.0 if self.show_edges else 0.0, smooth_shade=False)
        self.apply_colors(m)
        self.update_next_collapse()
        self.update_ellipsoids()
        self.metrics_dirty = True
        self.last_change = time.perf_counter()

    def vertex_ranks(self):
        c = self.cur
        return qd.effective_rank(c["Q"], self.rank_tol, ref=c["W"])

    def apply_colors(self, m=None):
        m = m or ps.get_surface_mesh("simplificada")
        c = self.cur
        mode = self.color_mode
        if mode in ("rank", "qerr") and c["Q"] is None:
            mode = "dist"
        name_pts = T("tipo de vértice")
        if mode != "rank":
            _remove("points", name_pts)
        if mode == "rank":
            # Colores en los VÉRTICES (esferitas): interpolados sobre triángulos grandes
            # pintarían caras enteras y confundirían.
            r = self.vertex_ranks()
            col = np.array([RANK_COLORS[int(x)] for x in (1, 2, 3)])[r - 1]
            m.set_color(MESH_COLOR)
            try:
                m.remove_all_quantities()
            except Exception:          # noqa: BLE001
                pass
            el = topology.typical_edge_length(c["V"], c["F"])
            pc = ps.register_point_cloud(name_pts, c["V"], radius=0.0, color=(0.5, 0.5, 0.5))
            pc.set_radius(0.12 * el, relative=False)
            pc.add_color_quantity("tipo", col, enabled=True)
        elif mode == "qerr":
            err = qd.quadric_error(c["Q"], c["V"]) / np.maximum(c["W"], 1e-300)
            val = 1000 * np.sqrt(err)
            m.add_scalar_quantity(T("error de la cuádrica (‰)"), val, enabled=True, cmap="viridis",
                                  vminmax=(0.0, float(np.percentile(val, 99)) + 1e-9))
        elif mode == "dist":
            d = 1000 * self.get_orig_dist()(c["V"])
            m.add_scalar_quantity(T("distancia a la original (‰)"), d, enabled=True, cmap="reds",
                                  vminmax=(0.0, float(d.max()) + 1e-9))
        elif mode == "quality":
            q = topology.triangle_quality(c["V"], c["F"])
            m.add_scalar_quantity("calidad q", q, defined_on="faces", enabled=True, cmap="viridis",
                                  vminmax=(0.0, 1.0))

    def get_orig_dist(self):
        if self.orig_dist is None:
            self.orig_dist = metrics.SurfaceDistance(self.V0, self.F0)
        return self.orig_dist

    def next_collapse(self):
        """(i, j, pos_i, pos_j, v̄, costo, cómo, virtual) del colapso k, o None."""
        h, c = self.hist, self.cur
        if h is None or self.k >= h.steps:
            return None
        i, j = int(h.keep[self.k]), int(h.removed[self.k])
        ii, jj = np.searchsorted(c["ids"], [i, j])
        return dict(i=i, j=j, ii=ii, jj=jj, pi=c["V"][ii], pj=c["V"][jj], vbar=h.newpos[self.k],
                    cost=h.cost[self.k], how=int(h.how[self.k]), virtual=bool(h.virtual[self.k]))

    def update_next_collapse(self):
        nc = self.next_collapse() if self.show_next else None
        names = [T(n) for n in ("próximo colapso", "v1 (sobrevive)", "v2 (desaparece)", "destino v̄", "elipsoide del par")]
        if nc is None:
            _remove("curve", names[0])
            for n in names[1:4]:
                _remove("points", n)
            _remove("mesh", names[4])
            return
        el = max(np.linalg.norm(nc["pi"] - nc["pj"]), 1e-6)
        r = 0.1 * el + 0.01 * self.edge_len0
        ps.register_curve_network(names[0], np.array([nc["pi"], nc["pj"]]), np.array([[0, 1]]),
                                  radius=0.35 * r, color=(0.1, 0.1, 0.1))
        ps.register_point_cloud(names[1], nc["pi"][None], radius=r, color=COL_V1)
        ps.register_point_cloud(names[2], nc["pj"][None], radius=r, color=COL_V2)
        ps.register_point_cloud(names[3], nc["vbar"][None], radius=1.1 * r, color=COL_VBAR)
        if self.show_pair_ellipsoid:
            c = self.cur
            Qs = c["Q"][nc["ii"]] + c["Q"][nc["jj"]]
            W = c["W"][nc["ii"]] + c["W"][nc["jj"]]
            Pe, Te = qd.quadric_ellipsoids(Qs[None], nc["vbar"][None], 0.35 * el, ref=np.array([W]),
                                         max_ratio=3.0, template=qd.unit_sphere(3))
            e = ps.register_surface_mesh(names[4], Pe, Te, color=COL_VBAR, smooth_shade=True)
            e.set_transparency(0.55)
        else:
            _remove("mesh", names[4])

    def update_ellipsoids(self):
        name = "elipsoides"
        c = self.cur
        pts = T("tipo de vértice")
        if ps.has_point_cloud(pts):          # los elipsoides ya vienen coloreados por tipo
            ps.get_point_cloud(pts).set_enabled(not (self.show_ellipsoids and c["Q"] is not None))
        if not self.show_ellipsoids or c["Q"] is None:
            _remove("mesh", name)
            return
        n = len(c["V"])
        sel = np.arange(n)
        if n > self.ell_max:
            sel = np.sort(np.random.default_rng(0).permutation(n)[:self.ell_max])
        el = topology.typical_edge_length(c["V"], c["F"])
        tmpl = qd.unit_sphere(1)
        Pe, Te = qd.quadric_ellipsoids(c["Q"][sel], c["V"][sel], 0.1 * el * self.ell_scale,
                                     ref=c["W"][sel], max_ratio=4.0, template=tmpl)
        r = qd.effective_rank(c["Q"][sel], self.rank_tol, ref=c["W"][sel])
        col = np.repeat(np.array([RANK_COLORS[x] for x in (1, 2, 3)])[r - 1], len(tmpl[0]), axis=0)
        e = ps.register_surface_mesh(name, Pe, Te, smooth_shade=True)
        e.add_color_quantity("tipo", col, enabled=True)

    def set_camera(self):
        lo = np.minimum(self.V0.min(0) + self.offset, self.V0.min(0))
        hi = np.maximum(self.V0.max(0) + self.offset, self.V0.max(0))
        if not self.show_original:
            lo, hi = self.V0.min(0), self.V0.max(0)
        center = 0.5 * (lo + hi)
        ext = hi - lo
        view = {"terreno": np.array([0.0, 0.8, 1.0]), "fandisk": np.array([0.25, 0.35, 1.0])}
        d = view.get(self.model, np.array([0.0, 0.2, 1.0]))
        d = d / np.linalg.norm(d)
        # los paneles tapan los costados de la ventana: dejar la escena en el tercio central
        dist = max(1.45 * ext[0], 1.8 * ext[1]) + 0.6 * ext[2]
        center = center + np.array([0.16 * ext[0], 0.0, 0.0])
        ps.look_at(tuple(center + dist * d), tuple(center))

    def focus_next(self):
        """Acerca la cámara al próximo colapso."""
        nc = self.next_collapse()
        if nc is None:
            return
        c = 0.5 * (nc["pi"] + nc["pj"])
        # dirección de vista: la normal promedio de las caras alrededor
        F, V = self.cur["F"], self.cur["V"]
        around = np.any((F == nc["ii"]) | (F == nc["jj"]), axis=1)
        tri = V[F[around]]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).sum(0)
        n = n / max(np.linalg.norm(n), 1e-12)
        el = np.linalg.norm(nc["pi"] - nc["pj"])
        dist = max(12 * el, 8 * self.edge_len0)
        ps.look_at(tuple(c + dist * (0.8 * n + 0.2 * np.array([0, 1.0, 0]))), tuple(c))

    # ------------------------------------------------------------ métricas
    def ensure_info(self):
        if not self.cur.get("full_info"):
            self.cur["info"] = topology.summary(self.cur["V"], self.cur["F"])
            self.cur["full_info"] = True

    def compute_metrics(self):
        self.ensure_info()
        c = self.cur
        t0 = time.perf_counter()
        e = metrics.geometric_error((self.V0, self.F0), c["V"], c["F"], n=20000,
                                    orig_dist=self.get_orig_dist())
        e["eval_time"] = time.perf_counter() - t0
        self.metrics = e
        self.metrics_dirty = False

    SHORT = {"qem": "QEM óptimo", "qem-svd": "QEM pseudo-inv.", "qem-subset": "QEM v1/v2/medio",
             "qem-midpoint": "QEM medio", "length": "arista corta", "clustering": "clustering",
             "clustering-qem": "clustering+Q", "open3d": "Open3D"}
    TABLE_HEAD = ["método", "caras", "H ‰", "medio ‰", "q", "χ", "t (s)"]

    def current_row(self, label=None):
        c, e, i = self.cur, self.metrics, self.cur["info"]
        if self.hist is not None:
            h = self.hist
            t = h.stats["time_init"] + h.stats["time"] * self.k / max(h.steps, 1)
        else:
            t = c["time"]
        key = self.method
        if key == "qem":
            key = {"optimal": "qem", "svd": "qem-svd", "subset": "qem-subset",
                   "midpoint": "qem-midpoint"}[self.placement]
        return [label or self.SHORT[key], i["F"], f"{e['hausdorff']:.1f}", f"{e['mean']:.2f}",
                f"{e['quality']:.2f}", i["chi"], f"{t:.2f}"]

    def compare_all(self):
        """Corre todos los métodos al número de caras actual y llena la tabla."""
        saved = (self.method, self.placement, getattr(self, "k", 0))
        target = self.cur["info"]["F"]
        self.target = target
        for name, placement in [("qem", "optimal"), ("qem", "subset"), ("qem", "midpoint"),
                                ("length", None), ("clustering", None), ("clustering-qem", None),
                                ("open3d", None)]:
            if name == "open3d" and not baselines.has_open3d():
                continue
            self.method = name
            if placement:
                self.placement = placement
            self.compute()
            self.compute_metrics()
            self.rows.append(self.current_row())
        self.method, self.placement = saved[0], saved[1]
        self.compute()

    def save_obj(self):
        out = os.path.join(os.path.dirname(data.DATA_DIR), "salidas")
        os.makedirs(out, exist_ok=True)
        fn = os.path.join(out, f"{self.model}_{self.method}_{self.cur['info']['F']}.obj")
        meshio.save_obj(fn, self.cur["V"], self.cur["F"])
        self.message = f"guardado {fn}"
        print("[obj]", fn)

    # ------------------------------------------------------------ panel
    def gui(self):
        changed_params = False
        psim.PushItemWidth(170)

        if C.header("Modelo"):
            for name in data.ALL_MODELS:
                if C.radio(name, self.model == name):
                    if name != self.model:
                        self.load_model(name)
                        return
                if name in ("armadillo", "piezas"):
                    pass
                else:
                    psim.SameLine()
            i = self.info0
            psim.TextUnformatted(f"V={i['V']}  F={i['F']}  χ={i['chi']}  género={i['genus']:.0f}  "
                                 f"bordes={i['boundary_loops']}  componentes={i['components']}")
            desc = data.REAL_MODELS.get(self.model, (None, data.SYNTHETIC_MODELS.get(self.model, "")))[1]
            psim.TextDisabled(desc)

        if C.header("Método"):
            for key, lab in VIEW_METHODS:
                if key == "open3d" and not baselines.has_open3d():
                    continue
                if C.radio(lab, self.method == key):
                    if key != self.method:
                        self.method = key
                        changed_params = True
            if self.is_collapse():
                C.separator_text("Parámetros del colapso")
                if self.method == "qem":
                    psim.TextUnformatted("Posición del vértice nuevo v̄:")
                    for p in qd.PLACEMENTS:
                        if C.radio(p, self.placement == p):
                            if p != self.placement:
                                self.placement = p
                                changed_params = True
                        psim.SameLine()
                    psim.NewLine()
                    psim.TextDisabled(qd.PLACEMENT_HELP[self.placement])
                    ch, self.area_weight = psim.Checkbox("ponderar planos por área", self.area_weight)
                    changed_params |= ch
                    ch, self.border = psim.Checkbox("restricción de borde (planos ⟂)", self.border)
                    changed_params |= ch
                ch, self.topology = psim.Checkbox("preservar topología (condición de enlace)", self.topology)
                changed_params |= ch
                ch, self.flips = psim.Checkbox("rechazar inversión de normales", self.flips)
                changed_params |= ch
                ch, t = C.slider_float("umbral t de pares", self.pair_t, 0.0, 0.03, "%.4f")
                if ch:
                    self.pair_t = t
                    self.pending_recompute = True
                psim.TextDisabled("t > 0: también son pares vértices a distancia < t (fracción de la diagonal)")
                if self.pending_recompute:
                    if psim.Button("aplicar t"):
                        self.pending_recompute = False
                        changed_params = True
            elif self.cur.get("note"):
                psim.TextDisabled(self.cur["note"])

        if changed_params:
            self.playing = False
            self.compute()
            return

        if C.header("Simplificación"):
            nf0 = len(self.F0)
            cur_faces = self.cur["info"]["F"]
            ch, v = C.slider_int("caras", cur_faces, 1, nf0)
            if ch:
                self.set_target(v)
            if self.hist is not None:
                h = self.hist
                if psim.Button("<< inicio"):
                    self.set_k(0)
                psim.SameLine()
                if psim.Button("< -1"):
                    self.set_k(self.k - 1)
                psim.SameLine()
                if psim.Button("+1 >"):
                    self.set_k(self.k + 1)
                psim.SameLine()
                if psim.Button("Pausa" if self.playing else "Play"):
                    self.playing = not self.playing
                psim.SameLine()
                psim.PushItemWidth(90)
                _, self.speed = C.slider_int("colapsos/cuadro", self.speed, 1, 200)
                psim.PopItemWidth()
                psim.TextUnformatted(f"colapso {self.k} de {h.max_step()}   ·   {self.cur['info']['V']} vértices, "
                                     f"{cur_faces} caras (mínimo alcanzable {h.faces_at(h.max_step())})")
            psim.TextUnformatted("objetivos:")
            presets = [(f"{p}%", max(1, nf0 * p // 100)) for p in (50, 20, 5, 1)]
            if self.model == "cow":
                presets += [(str(p), p) for p in PAPER_COW]
            for lab, v in presets:
                psim.SameLine()
                if psim.Button(lab):
                    self.set_target(v)
            if self.model == "cow":
                psim.TextDisabled("994 / 532 / 248 / 64 = las caras de la figura 5 del paper")
            if self.hist is not None and self.hist.steps:
                h = self.hist
                C.separator_text("Costo de cada colapso (log10)")
                cost = np.log10(np.maximum(h.cost, 1e-16))
                step = max(1, len(cost) // 400)
                C.plot_lines("##costo", cost[::step], overlay=f"colapso {self.k}: "
                             f"{h.cost[min(self.k, h.steps - 1)]:.2e}", size=(0, 80))
                hw = np.bincount(h.how[:self.k], minlength=4)
                s = h.stats
                if self.method == "qem":
                    psim.TextUnformatted(f"cómo se eligió v̄ hasta aquí:  A⁻¹b {hw[0]} · segmento {hw[1]}")
                    psim.TextUnformatted(f"      v₁/v₂/medio {hw[2]} · pseudo-inversa {hw[3]}")
                psim.TextDisabled(f"rechazados: topología {s['rejected_topology']}, normales "
                                  f"{s['rejected_flip']} · entradas viejas del heap {s['stale']}"
                                  + (f" · pares virtuales {s['virtual_pairs']}" if s['virtual_pairs'] else ""))

        if self.hist is not None and C.header("Próximo colapso"):
            ch, self.show_next = psim.Checkbox("mostrar", self.show_next)
            psim.SameLine()
            ch2, self.show_pair_ellipsoid = psim.Checkbox("elipsoide de Q₁+Q₂", self.show_pair_ellipsoid)
            if ch or ch2:
                self.update_next_collapse()
            nc = self.next_collapse()
            if nc:
                if psim.Button("acercar cámara"):
                    self.focus_next()
                psim.SameLine()
                if psim.Button("vista completa"):
                    self.set_camera()
                psim.TextColored((*COL_V1, 1), f"v{nc['i']} (sobrevive)")
                psim.SameLine()
                psim.TextColored((*COL_V2, 1), f"v{nc['j']} (desaparece)")
                psim.SameLine()
                psim.TextColored((*COL_VBAR, 1), "→ v̄")
                Qs = self.cur["Q"][nc["ii"]] + self.cur["Q"][nc["jj"]]
                lam = np.linalg.eigvalsh(Qs[:3, :3])
                wsum = self.cur["W"][nc["ii"]] + self.cur["W"][nc["jj"]]
                rk = int(qd.effective_rank(Qs, self.rank_tol, ref=wsum))
                psim.TextUnformatted(f"costo Δ(v̄) = {nc['cost']:.3e}" +
                                     ("   (par NO conectado)" if nc["virtual"] else ""))
                if self.method == "qem":
                    psim.TextUnformatted(f"v̄ por: {qd.HOW_NAMES[nc['how']]}")
                    psim.TextUnformatted(f"λ(A)/w = {np.array2string(lam / max(wsum, 1e-300), precision=3)}"
                                         f"  → {qd.RANK_NAMES[rk]}")
            else:
                psim.TextDisabled("no quedan colapsos válidos")

        if C.header("Visualización"):
            for key, lab in COLOR_MODES:
                if C.radio(lab, self.color_mode == key):
                    self.color_mode = key
                    self.apply_colors()
            if self.color_mode == "rank":
                for r in (1, 2, 3):
                    psim.TextColored((*RANK_COLORS[r], 1), f"  ■ {qd.RANK_NAMES[r]}")
                    if r < 3:
                        psim.SameLine()
                ch, tol = C.slider_float("umbral de rango", self.rank_tol, 0.005, 0.3)
                if ch:
                    self.rank_tol = tol
                    self.apply_colors()
                    self.update_ellipsoids()
            ch, self.show_edges = psim.Checkbox("aristas", self.show_edges)
            if ch:
                ps.get_surface_mesh("simplificada").set_edge_width(1.0 if self.show_edges else 0.0)
            psim.SameLine()
            ch, self.show_original = psim.Checkbox("original al lado", self.show_original)
            if ch:
                ps.get_surface_mesh("original").set_enabled(self.show_original)
            ch, self.show_ellipsoids = psim.Checkbox("elipsoides de error de los vértices", self.show_ellipsoids)
            if ch:
                self.update_ellipsoids()
            if self.show_ellipsoids:
                ch1, self.ell_scale = C.slider_float("tamaño", self.ell_scale, 0.2, 3.0, "%.2f")
                ch2, self.ell_max = C.slider_int("máx. elipsoides", self.ell_max, 100, 10000)
                if ch1 or ch2:
                    self.update_ellipsoids()
                psim.TextDisabled("disco = plano · cigarro = pliegue/borde · esfera = esquina")

        if C.header("Resultados"):
            _, self.auto_metrics = psim.Checkbox("medir automáticamente", self.auto_metrics)
            psim.SameLine()
            if psim.Button("medir"):
                self.compute_metrics()
            i = self.cur["info"]
            if not self.cur.get("full_info"):
                i = dict(i, chi="…", components="…", boundary_loops="…", genus=float("nan"),
                         nonmanifold_edges=0, nonmanifold_verts=0)
            psim.TextUnformatted(f"V={i['V']}  F={i['F']}  χ={i['chi']}  componentes={i['components']}  "
                                 f"bordes={i['boundary_loops']}  género="
                                 + ("—" if np.isnan(i['genus']) else f"{i['genus']:.0f}")
                                 + (f"  no-variedad: {i['nonmanifold_edges']} aristas, "
                                    f"{i['nonmanifold_verts']} vértices"
                                    if i['nonmanifold_edges'] or i['nonmanifold_verts'] else ""))
            e = self.metrics
            if e and not self.metrics_dirty:
                psim.TextUnformatted(f"Hausdorff {e['hausdorff']:.2f} ‰   medio {e['mean']:.3f} ‰   "
                                     f"RMS {e['rms']:.3f} ‰   (‰ de la diagonal)")
                psim.TextUnformatted(f"calidad media {e['quality']:.2f}   astillas (q<0.1) {e['slivers']:.1f}%")
                if psim.Button("agregar a la tabla"):
                    self.rows.append(self.current_row())
                psim.SameLine()
            else:
                psim.TextDisabled("(midiendo …)" if self.auto_metrics else "(sin medir)")
            if psim.Button("comparar todos los métodos"):
                self.compare_all()
            psim.SameLine()
            if psim.Button("limpiar tabla"):
                self.rows = []
            psim.SameLine()
            if psim.Button("guardar OBJ"):
                self.save_obj()
            if self.rows:
                C.table(self.rows, self.TABLE_HEAD, key="resultados")
        if self.message:
            psim.TextColored((1, 0.6, 0.2, 1), self.message)
        psim.PopItemWidth()

    # ------------------------------------------------------------ navegación
    def set_k(self, k):
        h = self.hist
        k = int(np.clip(k, 0, h.max_step()))
        if k != self.k:
            self.k = k
            self.target = h.faces_at(k)
            self.update_mesh()

    def set_target(self, faces):
        self.target = int(faces)
        if self.hist is not None:
            self.set_k(self.hist.step_for_faces(self.target))
        else:
            self.update_mesh()

    def callback(self):
        if self.playing and self.hist is not None:
            if self.k >= self.hist.max_step():
                self.playing = False
            else:
                self.set_k(self.k + self.speed)
        self.gui()
        quiet = not self.playing and time.perf_counter() - self.last_change > 0.35
        if quiet and not self.cur.get("full_info"):
            self.ensure_info()
        if self.auto_metrics and self.metrics_dirty and quiet:
            self.compute_metrics()


def run(args):
    ps.set_program_name("Simplificación de mallas — cuádricas de error")
    if args.shot:
        C.allow_headless()
    ps.init()
    ps.set_up_dir("y_up")
    try:
        ps.set_ground_plane_mode("shadow_only")
    except Exception:          # noqa: BLE001
        pass
    viewer = Viewer(args)
    ps.set_user_callback(viewer.callback)
    if args.shot:
        shots(viewer, args.shot)
    if not args.no_view and not args.shot:
        ps.show()
    return viewer


def shots(v, prefix):
    """Capturas sin ventana: una por estado interesante (como --shot en los otros demos)."""
    d = os.path.dirname(prefix)
    if d:
        os.makedirs(d, exist_ok=True)

    def snap(tag, ui=False):
        v.compute_metrics()
        C.frame_tick(4)
        fn = f"{prefix}_{tag}.png"
        C.screenshot(fn, ui=ui)
        print("[shot]", fn, flush=True)

    snap("1_rango")
    v.show_ellipsoids = True
    v.update_ellipsoids()
    snap("2_elipsoides")
    v.show_ellipsoids = False
    v.update_ellipsoids()
    v.color_mode = "dist"
    v.apply_colors()
    snap("3_distancia")
    v.color_mode = "rank"
    v.apply_colors()
    if v.hist is not None:
        v.show_pair_ellipsoid = True
        v.update_next_collapse()
        v.focus_next()
        snap("4_proximo")
        v.set_camera()
    snap("0_panel", ui=True)
    for name in ("length", "clustering"):
        v.method = name
        v.compute()
        snap(f"5_{name}")
