"""Visor interactivo en Polyscope.

Estructuras:
  nube de entrada ........ puntos + normales (estimadas / verdaderas), coloreables
                           por error angular, orientacion, variacion de superficie
  descartados (filtro) ... puntos eliminados por el filtro estadistico
  superficie real ........ referencia (desactivada al inicio)
  Hoppe / RBF / Poisson / Open3D ... mallas reconstruidas, coloreadas por error
  corte del campo ........ plano de la grilla coloreado por el valor de F
                           (azul = dentro, rojo = fuera, gris = no definido)
  iso-linea .............. F = 0 en el corte, por marching squares
  campo V (Poisson) ...... normales salpicadas en la grilla, sobre el corte
  restricciones RBF ...... centros sobre la superficie y a +-eps

El panel lateral permite regenerar la nube, re-estimar normales y volver a
reconstruir sin salir del visor. Solo se recalcula lo que cambio.
"""

from __future__ import annotations

import time

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from . import methods as M
from .data import MODELS, PRESETS, SceneParams, make_scene
from .extract import marching_squares
from .normals import ORIENT_METHODS, NormalParams
from .pipeline import ALL_METHODS, format_header, format_row, run_method, run_normals, build_grid

METHOD_COLORS = {"Hoppe": (0.93, 0.62, 0.25), "RBF": (0.45, 0.75, 0.45),
                 "Poisson": (0.40, 0.60, 0.90), "Open3D": (0.75, 0.55, 0.85)}
CLOUD_MODES = ["uniforme", "error angular", "orientacion", "variacion", "outliers reales"]


# --------------------------------------------------------------------------- #
# compatibilidad con versiones antiguas de Polyscope / imgui
# --------------------------------------------------------------------------- #
def _separator_text(label: str) -> None:
    if hasattr(psim, "SeparatorText"):
        psim.SeparatorText(label)
    else:
        psim.Separator()
        psim.Text(label)


def _header(label: str, default_open: bool = True) -> bool:
    flag = getattr(psim, "ImGuiTreeNodeFlags_DefaultOpen", 32) if default_open else 0
    return psim.CollapsingHeader(label, flag)


def _remove(name: str) -> None:
    for fn in ("remove_surface_mesh", "remove_point_cloud", "remove_curve_network"):
        f = getattr(ps, fn, None)
        if f is None:
            continue
        try:
            f(name, error_if_absent=False)
        except TypeError:
            try:
                f(name)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass


def _smooth(mesh) -> None:
    try:
        mesh.set_smooth_shade(True)
    except Exception:  # noqa: BLE001
        pass


def diverging(vals: np.ndarray, scale: float, bands: bool = True) -> np.ndarray:
    """Azul (F<0, dentro) - blanco (F=0) - rojo (F>0, fuera); gris = NaN.

    Las bandas (curvas de nivel) muestran si el campo se parece a una
    distancia (bandas equiespaciadas) o no (Poisson, RBF lejos de los datos).
    """
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    if np.ndim(scale) == 0:
        scale = (scale, scale)
    s_neg, s_pos = max(scale[0], 1e-12), max(scale[1], 1e-12)
    sc = np.where(v < 0, s_neg, s_pos)
    t = np.clip(np.where(ok, v, 0) / sc, -1, 1)
    blue, white, red = np.array([.23, .30, .75]), np.array([.97, .97, .97]), np.array([.80, .15, .15])
    a = np.abs(t)[:, None]
    col = np.where((t < 0)[:, None], (1 - a) * white + a * blue, (1 - a) * white + a * red)
    if bands:
        shade = 0.88 + 0.12 * np.cos(2 * np.pi * np.where(ok, v, 0) / (0.25 * sc))
        col = col * shade[:, None]
    col[~ok] = (0.55, 0.55, 0.55)
    return np.clip(col, 0, 1)


# --------------------------------------------------------------------------- #
class App:
    def __init__(self, sp: SceneParams, npm: NormalParams, rp: M.ReconParams,
                 methods=None) -> None:
        self.sp, self.npm, self.rp = sp, npm, rp
        avail = [m for m in ALL_METHODS if m != "Open3D" or M.open3d_available()]
        self.avail = avail
        self.enabled = {m: (methods is None or m in methods) for m in avail}
        self.scene = None
        self.nr = None
        self.res: dict = {}
        self.cache_keys: dict = {}
        self.normals_version = 0
        self.active = next((m for m in ("Poisson", "RBF", "Hoppe", "Open3D") if self.enabled.get(m)), None)
        self.side_by_side = False
        self.color_by_error = True
        self.err_max = 0.005
        self.show_edges = False
        self.cloud_mode = 2
        self.show_normals = False
        self.slice_on = False
        self.slice_axis = 1
        self.slice_pos = 0.5
        self.show_iso = True
        self.show_V = False
        self.show_constraints = False
        self.pending: list = []
        self.pending_wait = 0
        self.log: list = []

    # ------------------------------------------------------------------ calculo
    def say(self, msg: str) -> None:
        print(msg, flush=True)
        self.log = (self.log + [msg])[-4:]

    def compute_scene(self) -> None:
        t0 = time.perf_counter()
        self.scene = make_scene(self.sp)
        self.res.clear()
        self.cache_keys.clear()
        self.say(f"[escena] {self.sp.model}: {len(self.scene.P)} puntos "
                 f"(hueco -{self.scene.n_hole_removed}, outliers {int(self.scene.is_outlier.sum())}) "
                 f"en {time.perf_counter() - t0:.1f}s")

    def compute_normals(self) -> None:
        self.nr = run_normals(self.scene, self.npm)
        self.normals_version += 1
        s = self.nr.summary()
        self.say(f"[normales] {self.npm.method} k={self.npm.k}: error mediano "
                 f"{s['median_deg']:.2f} deg, bien orientadas {100 * s['oriented_ok']:.1f}%")

    def _key(self, m: str):
        r = self.rp
        common = (self.normals_version, r.res, r.extractor)
        spec = {"Hoppe": (r.hoppe_k, r.hoppe_support),
                "RBF": (r.rbf_centers, r.rbf_eps),
                "Poisson": (r.poisson_sigma, r.poisson_adaptive),
                "Open3D": (self.normals_version, r.o3d_depth, r.o3d_trim)}[m]
        return spec if m == "Open3D" else common + spec

    def compute_methods(self) -> None:
        grid = build_grid(self.nr.P, self.rp.res)
        for m in self.avail:
            if not self.enabled[m]:
                self.res.pop(m, None)
                continue
            key = self._key(m)
            if self.cache_keys.get(m) == key and m in self.res:
                continue
            print(f"[{m}] reconstruyendo ...", flush=True)
            r = run_method(m, self.scene, self.nr, self.rp, grid)
            if r is not None:
                self.res[m] = r
                self.cache_keys[m] = key
                print("  " + format_row(r), flush=True)
        if self.active not in self.res and self.res:
            self.active = next(iter(self.res))

    def run_pending(self) -> None:
        stages = set(self.pending)
        self.pending = []
        if "scene" in stages:
            self.compute_scene()
        if "scene" in stages or "normals" in stages:
            self.compute_normals()
        self.compute_methods()
        self.draw_all()
        if "scene" in stages:
            _set_camera(self)

    def request(self, stage: str) -> None:
        self.pending.append(stage)
        self.pending_wait = 1   # deja pasar un cuadro para mostrar "calculando..."

    # ------------------------------------------------------------------ dibujo
    def draw_all(self) -> None:
        self.draw_reference()
        self.draw_cloud()
        self.draw_meshes()
        self.draw_slice()

    def draw_reference(self) -> None:
        V, T = self.scene.surface.reference_mesh()
        m = ps.register_surface_mesh("superficie real", V, T, color=(0.8, 0.8, 0.8),
                                     enabled=False)
        _smooth(m)

    def draw_cloud(self) -> None:
        nr, sc = self.nr, self.scene
        diag = sc.diag
        pc = ps.register_point_cloud("nube de entrada", nr.P, radius=0.0022,
                                     color=(0.20, 0.20, 0.20))
        mode = CLOUD_MODES[self.cloud_mode]
        if mode == "error angular":
            pc.add_scalar_quantity("error angular (grados)", np.nan_to_num(nr.ang_err, nan=0.0),
                                   enabled=True, cmap="viridis", vminmax=(0.0, 20.0))
        elif mode == "orientacion":
            col = np.tile([0.25, 0.70, 0.30], (len(nr.P), 1))
            col[nr.flipped] = (0.85, 0.15, 0.15)
            col[nr.is_outlier] = (0.55, 0.55, 0.55)
            pc.add_color_quantity("orientacion (verde ok / rojo invertida)", col, enabled=True)
        elif mode == "variacion":
            pc.add_scalar_quantity("variacion de superficie", nr.variation, enabled=True,
                                   cmap="viridis", vminmax=(0.0, 0.05))
        elif mode == "outliers reales":
            col = np.tile([0.25, 0.25, 0.25], (len(nr.P), 1))
            col[nr.is_outlier] = (0.95, 0.10, 0.10)
            pc.add_color_quantity("outliers reales (rojo)", col, enabled=True)
        L = 0.012 * diag
        try:
            pc.add_vector_quantity("normales estimadas", nr.N * L, enabled=self.show_normals,
                                   vectortype="ambient", color=(0.15, 0.35, 0.85), radius=0.0012)
            Nt = np.nan_to_num(nr.N_true)
            pc.add_vector_quantity("normales verdaderas", Nt * L, enabled=False,
                                   vectortype="ambient", color=(0.1, 0.6, 0.2), radius=0.0012)
        except TypeError:
            pc.add_vector_quantity("normales estimadas", nr.N * L, enabled=self.show_normals)
        # puntos eliminados por el filtro
        _remove("descartados por el filtro")
        if (~nr.keep).any():
            Pd = sc.P[~nr.keep]
            out = sc.is_outlier[~nr.keep]
            col = np.where(out[:, None], [[0.1, 0.7, 0.2]], [[0.95, 0.55, 0.1]])
            d = ps.register_point_cloud("descartados por el filtro", Pd, radius=0.003)
            d.add_color_quantity("verde = outlier real, naranja = inlier perdido", col, enabled=True)

    def draw_meshes(self) -> None:
        for m in ALL_METHODS:
            _remove(m)
        if not self.res:
            return
        names = list(self.res)
        ext = self.nr.P.max(0) - self.nr.P.min(0)
        for i, m in enumerate(names):
            r = self.res[m]
            if len(r.T) == 0:
                continue
            V = r.V.copy()
            show = self.side_by_side or m == self.active
            if self.side_by_side:   # grilla de 2 columnas
                V[:, 0] += (i % 2 - 0.5) * 1.15 * ext[0]
                V[:, 1] -= (i // 2) * 1.15 * ext[1]
            mesh = ps.register_surface_mesh(m, V, r.T, color=METHOD_COLORS[m], enabled=show)
            _smooth(mesh)
            if self.show_edges:
                try:
                    mesh.set_edge_width(1.0)
                except Exception:  # noqa: BLE001
                    pass
            mesh.add_scalar_quantity("error (fraccion de la diagonal)", r.err,
                                     enabled=self.color_by_error, cmap="reds",
                                     vminmax=(0.0, self.err_max))
        # en modo "lado a lado" la nube estorba
        try:
            ps.get_point_cloud("nube de entrada").set_enabled(not self.side_by_side)
        except Exception:  # noqa: BLE001
            pass

    def draw_slice(self) -> None:
        for nm in ("corte del campo", "iso-linea F=0 (marching squares)",
                   "campo V (Poisson)", "restricciones RBF", "centros de planos (Hoppe)"):
            _remove(nm)
        r = self.res.get(self.active)
        # con el corte visible, la malla activa se vuelve semitransparente
        for m in self.res:
            try:
                ps.get_surface_mesh(m).set_transparency(
                    0.45 if (self.slice_on and m == self.active and not self.side_by_side) else 1.0)
            except Exception:  # noqa: BLE001
                pass
        if not self.slice_on or self.side_by_side or r is None or r.field.vals is None:
            return
        g, vals = r.field.grid, r.field.vals
        a = self.slice_axis
        b, c = [x for x in range(3) if x != a]
        i = int(round(self.slice_pos * (g.shape[a] - 1)))
        G = np.take(vals, i, axis=a)                        # (n_b, n_c)
        xb, xc = g.axis(b), g.axis(c)
        Bb, Cc = np.meshgrid(xb, xc, indexing="ij")
        P = np.zeros((Bb.size, 3))
        P[:, a] = g.axis(a)[i]
        P[:, b], P[:, c] = Bb.ravel(), Cc.ravel()
        nb, nc = G.shape
        idx = np.arange(nb * nc).reshape(nb, nc)
        q = np.c_[idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()]
        T = np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        # escala separada para cada lado: asi dentro y fuera se ven aunque el
        # campo no sea simetrico (Poisson, RBF)
        neg, pos = G[np.isfinite(G) & (G < 0)], G[np.isfinite(G) & (G > 0)]
        scale = (np.percentile(-neg, 70) if len(neg) else 1.0,
                 np.percentile(pos, 35) if len(pos) else 1.0)
        sl = ps.register_surface_mesh("corte del campo", P, T)
        sl.add_color_quantity("F (azul dentro / rojo fuera / gris indefinido)",
                              diverging(G.ravel(), scale), enabled=True)
        for fn, arg in (("set_back_face_policy", "identical"), ("set_material", "flat")):
            try:
                getattr(sl, fn)(arg)
            except Exception:  # noqa: BLE001
                pass
        if self.show_iso:
            nodes2, E = marching_squares(G, xb, xc)
            if len(E):
                nodes = np.zeros((len(nodes2), 3))
                nodes[:, a] = g.axis(a)[i]
                nodes[:, b], nodes[:, c] = nodes2[:, 0], nodes2[:, 1]
                ps.register_curve_network("iso-linea F=0 (marching squares)", nodes, E,
                                          radius=0.0015, color=(0.05, 0.05, 0.05))
        if self.show_V and "V" in r.field.extras:
            Vf = np.take(r.field.extras["V"], i, axis=a).reshape(-1, 3)
            mag = np.linalg.norm(Vf, axis=1)
            keep = mag > 0.02 * mag.max()
            if keep.any():
                pcv = ps.register_point_cloud("campo V (Poisson)", P[keep], radius=0.0008,
                                              color=(0.1, 0.1, 0.1))
                vv = Vf[keep] / mag.max() * 1.5 * g.h
                try:
                    pcv.add_vector_quantity("V", vv, enabled=True, vectortype="ambient",
                                            color=(0.1, 0.5, 0.1))
                except TypeError:
                    pcv.add_vector_quantity("V", vv, enabled=True)
        if self.show_constraints and "constraints" in r.field.extras:
            C = r.field.extras["constraints"]
            f = r.field.extras["constraint_vals"]
            pcc = ps.register_point_cloud("restricciones RBF", C, radius=0.003)
            pcc.add_color_quantity("azul -eps / blanco 0 / rojo +eps",
                                   diverging(f, np.abs(f).max(), bands=False), enabled=True)
        if self.show_constraints and "centers" in r.field.extras:
            ps.register_point_cloud("centros de planos (Hoppe)", r.field.extras["centers"],
                                    radius=0.0018, color=(0.9, 0.5, 0.1))

    def results_table(self) -> None:
        cols = ["metodo", "err med", "err p95", "compl p95", "cobert.", "tiempo"]
        rows = []
        for m, r in self.res.items():
            s = r.summary
            rows.append((m, [m, f"{1e3 * s['err_med']:.2f}", f"{1e3 * s['err_p95']:.2f}",
                             f"{1e3 * s['comp_p95']:.2f}", f"{100 * s['cover']:.1f}%",
                             f"{s['t_field'] + s['t_extract']:.2f}s"]))
        hl = (1.0, 0.85, 0.3, 1.0)
        if hasattr(psim, "BeginTable"):
            flags = getattr(psim, "ImGuiTableFlags_Borders", 0) | getattr(psim, "ImGuiTableFlags_RowBg", 0)
            if psim.BeginTable("tabla_resultados", len(cols), flags):
                for c in cols:
                    psim.TableSetupColumn(c)
                psim.TableHeadersRow()
                for m, vals in rows:
                    psim.TableNextRow()
                    for j, v in enumerate(vals):
                        psim.TableSetColumnIndex(j)
                        if m == self.active and not self.side_by_side:
                            psim.TextColored(hl, v)
                        else:
                            psim.TextUnformatted(v)
                psim.EndTable()
            for m, r in self.res.items():
                psim.TextUnformatted(f"{m}: {r.summary['topo']}")
            return
        psim.TextUnformatted(format_header())      # imgui antiguo: texto plano
        for m, r in self.res.items():
            txt = format_row(r)
            if m == self.active and not self.side_by_side:
                psim.TextColored(hl, txt)
            else:
                psim.TextUnformatted(txt)

    def frame_camera(self) -> None:
        lo, hi = self.nr.P.min(0), self.nr.P.max(0)
        if self.side_by_side and self.res:
            ext = hi - lo
            k = len(self.res)
            lo = lo + np.array([-0.575 * ext[0], -((k - 1) // 2) * 1.15 * ext[1], 0])
            hi = hi + np.array([0.575 * ext[0], 0, 0])
        _set_camera(self, lo, hi)

    # ------------------------------------------------------------------ panel
    def callback(self) -> None:
        if self.pending:
            if self.pending_wait > 0:
                self.pending_wait -= 1
                psim.TextColored((1.0, 0.8, 0.2, 1.0), "Calculando... (ver consola)")
                return
            self.run_pending()

        psim.PushItemWidth(170)
        sc, sp = self.scene, self.sp
        psim.TextUnformatted(f"Modelo: {sp.model}   puntos: {len(self.nr.P)}   diag: {sc.diag:.3f}")

        # ---------------------------------------------------------- escena
        if _header("1. Nube de entrada"):
            for name in MODELS:
                if psim.RadioButton(name, sp.model == name):
                    sp.model = name
                psim.SameLine()
            psim.NewLine()
            _separator_text("Regimenes (bloque 2.6)")
            for j, (name, pr) in enumerate(PRESETS.items()):
                if psim.Button(name):
                    for k, v in pr.items():
                        setattr(sp, k, v)
                    self.request("scene")
                if j % 3 != 2:
                    psim.SameLine()
            psim.NewLine()
            _, sp.n = psim.InputInt("puntos", sp.n, 500)
            sp.n = int(np.clip(sp.n, 200, 200000))
            _, sp.noise = psim.SliderFloat("ruido (frac. diag)", sp.noise, 0.0, 0.015, "%.4f")
            _, sp.outliers = psim.SliderFloat("outliers (fraccion)", sp.outliers, 0.0, 0.2, "%.3f")
            _, sp.hole = psim.SliderFloat("radio hueco (frac. diag)", sp.hole, 0.0, 0.4, "%.3f")
            _, sp.seed = psim.InputInt("semilla", sp.seed, 1)
            if psim.Button("Generar nube"):
                self.request("scene")

        # ---------------------------------------------------------- normales
        npm = self.npm
        if _header("2. Normales"):
            for key, lab in ORIENT_METHODS.items():
                if psim.RadioButton(lab, npm.method == key):
                    npm.method = key
                    self.request("normals")
            ch, npm.k = psim.SliderInt("k vecinos (PCA)", npm.k, 4, 120)
            _, npm.k_graph = psim.SliderInt("k grafo (MST)", npm.k_graph, 3, 30)
            ch2, npm.filter = psim.Checkbox("filtro estadistico de outliers", npm.filter)
            if ch2:
                self.request("normals")
            if npm.filter:
                _, npm.alpha = psim.SliderFloat("alpha (media + a*desv)", npm.alpha, 0.5, 4.0, "%.2f")
            if psim.Button("Estimar normales"):
                self.request("normals")
            s = self.nr.summary()
            psim.TextUnformatted(f"error angular: mediana {s['median_deg']:.2f} deg, p95 {s['p95_deg']:.2f} deg")
            psim.TextUnformatted(f"bien orientadas: {100 * s['oriented_ok']:.1f} %")
            if npm.filter:
                psim.TextUnformatted(f"filtro: quedan {s['n']} pts, precision {self.nr.filter_precision:.3f},"
                                     f" recall {self.nr.filter_recall:.3f}")
            psim.TextUnformatted(f"tiempo: PCA {1e3 * self.nr.t_pca:.0f} ms, orientacion {1e3 * self.nr.t_orient:.0f} ms")
            _separator_text("Colorear la nube por")
            changed = False
            for j, lab in enumerate(CLOUD_MODES):
                if psim.RadioButton(lab + "##cm", self.cloud_mode == j):
                    self.cloud_mode = j
                    changed = True
                if j < len(CLOUD_MODES) - 1:
                    psim.SameLine()
            ch, self.show_normals = psim.Checkbox("mostrar normales", self.show_normals)
            if changed or ch:
                self.draw_cloud()

        # ---------------------------------------------------------- reconstruccion
        rp = self.rp
        if _header("3. Reconstruccion"):
            for m in self.avail:
                _, self.enabled[m] = psim.Checkbox(m + "##en", self.enabled[m])
                psim.SameLine()
            psim.NewLine()
            _, rp.res = psim.SliderInt("resolucion grilla", rp.res, 16, 128)
            psim.TextUnformatted("extractor:")
            psim.SameLine()
            if psim.RadioButton("marching tetrahedra (propio)", rp.extractor == "mt"):
                rp.extractor = "mt"
            psim.SameLine()
            if psim.RadioButton("marching cubes (skimage)", rp.extractor == "mc"):
                rp.extractor = "mc"
            if psim.TreeNode("Hoppe 1992"):
                _, rp.hoppe_k = psim.SliderInt("k (centroide del plano)", rp.hoppe_k, 3, 60)
                _, rp.hoppe_support = psim.SliderFloat("rho+delta (x espaciado)", rp.hoppe_support, 1.0, 10.0, "%.1f")
                psim.TreePop()
            if psim.TreeNode("RBF (Carr 2001)"):
                _, rp.rbf_centers = psim.SliderInt("centros", rp.rbf_centers, 100, 2000)
                _, rp.rbf_eps = psim.SliderFloat("eps (frac. diag)", rp.rbf_eps, 0.002, 0.05, "%.3f")
                psim.TreePop()
            if psim.TreeNode("Poisson (FFT)"):
                _, rp.poisson_sigma = psim.SliderFloat("suavizado de V (celdas)", rp.poisson_sigma, 0.0, 4.0, "%.2f")
                _, rp.poisson_adaptive = psim.Checkbox("peso por area local", rp.poisson_adaptive)
                psim.TreePop()
            if "Open3D" in self.avail and psim.TreeNode("Open3D (screened Poisson)"):
                _, rp.o3d_depth = psim.SliderInt("profundidad octree", rp.o3d_depth, 5, 10)
                _, rp.o3d_trim = psim.SliderFloat("recorte por densidad (cuantil)", rp.o3d_trim, 0.0, 0.2, "%.2f")
                psim.TreePop()
            if psim.Button("Reconstruir"):
                self.request("methods")
            psim.SameLine()
            if psim.Button("Todo de nuevo (nube + normales + mallas)"):
                self.request("scene")

        # ---------------------------------------------------------- resultados
        if _header("4. Resultados"):
            redraw = False
            for m in self.res:
                if psim.RadioButton(m + "##act", (m == self.active) and not self.side_by_side):
                    self.active, self.side_by_side = m, False
                    redraw = True
                psim.SameLine()
            if psim.RadioButton("lado a lado", self.side_by_side):
                self.side_by_side = True
                redraw = True
            ch1, self.color_by_error = psim.Checkbox("colorear por error", self.color_by_error)
            psim.SameLine()
            ch2, self.show_edges = psim.Checkbox("aristas", self.show_edges)
            ch3, self.err_max = psim.SliderFloat("error maximo del color", self.err_max, 0.001, 0.03, "%.3f")
            if redraw or ch1 or ch2 or ch3:
                self.draw_meshes()
                self.draw_slice()
            if redraw:
                self.frame_camera()
            psim.TextUnformatted("errores en milesimas de la diagonal")
            self.results_table()
            r = self.res.get(self.active)
            if r is not None:
                st = r.stats
                psim.TextUnformatted(f"{self.active}: V={st['V']} E={st['E']} F={st['F']}  chi = V-E+F = {st['chi']}")
                ex = r.field.extras
                if "undefined" in ex:
                    psim.TextUnformatted(f"  Hoppe: {100 * ex['undefined']:.1f}% de la grilla sin definir (rho+delta={ex['rho_delta']:.4f})")
                if "system" in ex:
                    psim.TextUnformatted(f"  RBF: sistema {ex['system']}x{ex['system']}, eps medio {ex['eps_mean']:.4f} diag,"
                                         f" solve {ex['t_solve']:.2f}s")
                if "iso" in ex:
                    psim.TextUnformatted(f"  Poisson: iso-valor = promedio en las muestras = {ex['iso']:.4g}")
            psim.TextUnformatted("err = vertices -> superficie real; compl = superficie real -> malla")

        # ---------------------------------------------------------- corte
        if _header("5. Corte del campo implicito"):
            ch0, self.slice_on = psim.Checkbox("mostrar corte", self.slice_on)
            ch1 = False
            for j, lab in enumerate("xyz"):
                psim.SameLine()
                if psim.RadioButton(lab + "##ax", self.slice_axis == j):
                    self.slice_axis = j
                    ch1 = True
            ch2, self.slice_pos = psim.SliderFloat("posicion", self.slice_pos, 0.0, 1.0, "%.3f")
            ch3, self.show_iso = psim.Checkbox("iso-linea (marching squares)", self.show_iso)
            ch4, self.show_V = psim.Checkbox("campo V (Poisson)", self.show_V)
            psim.SameLine()
            ch5, self.show_constraints = psim.Checkbox("restricciones RBF / centros Hoppe", self.show_constraints)
            if any((ch0, ch1, ch2, ch3, ch4, ch5)):
                self.draw_slice()
            if self.slice_on and self.active == "Open3D":
                psim.TextUnformatted("(Open3D no expone su campo: no hay corte)")

        if self.log:
            psim.Separator()
            for line in self.log[-3:]:
                psim.TextUnformatted(line)
        psim.PopItemWidth()


# --------------------------------------------------------------------------- #
def _init(headless: bool) -> None:
    if headless and hasattr(ps, "set_allow_headless_backends"):
        ps.set_allow_headless_backends(True)
    ps.set_program_name("SGP - reconstruccion de superficies")
    ps.set_verbosity(0)
    try:
        ps.set_print_prefix("[polyscope] ")
    except Exception:  # noqa: BLE001
        pass
    ps.init()
    if hasattr(ps, "set_right_gui_pane_width"):
        try:
            ps.set_right_gui_pane_width(620)
        except Exception:  # noqa: BLE001
            pass
    try:
        ps.set_ground_plane_mode("shadow_only")
    except Exception:  # noqa: BLE001
        pass


def _set_camera(app: App, lo=None, hi=None) -> None:
    """Vista oblicua desde arriba (blob, z hacia arriba) o de frente (modelos, y arriba)."""
    up = "z_up" if app.sp.model == "blob" else "y_up"
    ps.set_up_dir(up)
    if hasattr(ps, "set_front_dir"):
        try:
            ps.set_front_dir("neg_y_front" if up == "z_up" else "z_front")
        except Exception:  # noqa: BLE001
            pass
    if lo is None:
        lo, hi = app.nr.P.min(0), app.nr.P.max(0)
    c = 0.5 * (lo + hi)
    d = np.array([0.25, -0.80, 1.0]) if up == "z_up" else np.array([0.35, 0.30, 1.0])
    d = d / np.linalg.norm(d)
    dist = 1.1 * np.linalg.norm(hi - lo)
    try:
        ps.look_at(tuple(c + dist * d), tuple(c))
    except Exception:  # noqa: BLE001
        ps.reset_camera_to_home_view()


def show(app: App) -> None:
    _init(headless=False)
    _set_camera(app)
    app.draw_all()
    ps.set_user_callback(app.callback)
    ps.show()


def _tick(n: int) -> None:
    """Dibuja n cuadros llamando al panel (frame_tick en Polyscope >= 2)."""
    if hasattr(ps, "frame_tick"):
        for _ in range(n):
            ps.frame_tick()
    else:
        try:
            ps.show(forFrames=n)
        except TypeError:
            pass


def screenshots(app: App, prefix: str) -> list:
    """Sin pantalla: guarda un PNG por estado (nube, cada metodo, lado a lado, corte)."""
    _init(headless=True)
    _set_camera(app)
    app.draw_all()
    ps.set_user_callback(app.callback)
    _tick(3)                        # ejercita el panel (detecta errores de imgui)
    files = []

    def shot(tag):
        fn = f"{prefix}_{tag}.png"
        try:
            ps.screenshot(fn, transparent_bg=False)
        except TypeError:
            ps.screenshot(fn)
        files.append(fn)

    # 1) nube + orientacion de normales
    for m in ALL_METHODS:
        _remove(m)
    app.show_normals = True
    app.draw_cloud()
    shot("0_nube")
    app.show_normals = False
    app.draw_cloud()
    cloud = ps.get_point_cloud("nube de entrada")
    cloud.set_enabled(False)
    # 2) cada metodo
    for m in list(app.res):
        app.active, app.side_by_side = m, False
        app.draw_meshes()
        cloud.set_enabled(False)
        shot(f"1_{m}")
    # 3) lado a lado
    app.side_by_side = True
    app.draw_meshes()
    app.frame_camera()
    shot("2_lado_a_lado")
    # 4) corte del campo para cada metodo con campo
    app.side_by_side, app.slice_on = False, True
    app.slice_axis = 2 if app.sp.model == "blob" else 0
    app.frame_camera()
    for m in [m for m in app.res if app.res[m].field.vals is not None]:
        app.active = m
        app.show_V = m == "Poisson"
        app.draw_meshes()
        app.draw_slice()
        cloud.set_enabled(False)
        shot(f"3_corte_{m}")
    return files
