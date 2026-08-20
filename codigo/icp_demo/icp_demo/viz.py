"""Visualizacion interactiva del ICP en Polyscope.

Muestra la animacion iteracion por iteracion: la nube fuente moviendose sobre
la objetivo, las correspondencias usadas en cada paso, el residual por punto y
la curva de convergencia. Desde el panel se puede volver a ejecutar el
algoritmo con otra variante o backend sin salir del visor.
"""

from __future__ import annotations

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from scipy.spatial import cKDTree

from .geometry import apply_transform
from .icp import ICPParams, ICPResult, icp
from .scenario import Scenario

COL_TARGET = (0.35, 0.45, 0.60)
COL_SOURCE0 = (0.75, 0.35, 0.35)
COL_MOVING = (0.95, 0.65, 0.15)
COL_GT = (0.30, 0.70, 0.45)

_VARIANTS = ["point2point", "point2plane"]
_BACKENDS = ["kdtree", "brute", "brute-loop"]

def _separator_text(label: str) -> None:
    if hasattr(psim, "SeparatorText"):
        psim.SeparatorText(label)
    else:
        psim.Separator()
        psim.Text(label)

class ICPViewer:
    def __init__(self, scenario: Scenario, result: ICPResult,
                 max_corr_lines: int = 400):
        self.sc = scenario
        self.result = result
        self.max_corr_lines = max_corr_lines
        self.tree = cKDTree(scenario.target)

        self.frame = 0
        self.playing = False
        self.speed = 8              # frames de render por iteracion
        self._tick = 0
        self.show_corr = True
        self.show_ghost = True

        # controles del panel de re-ejecucion
        self.ui_variant = _VARIANTS.index(result.params.variant)
        self.ui_backend = _BACKENDS.index(result.params.nn_backend)
        self.ui_max_iter = result.params.max_iter
        self.ui_sample = result.params.sample_size or 0
        self.ui_reject = result.params.auto_reject_factor or 0.0
        self.status = ""

    # ------------------------------------------------------------------ #
    def register(self) -> None:
        sc = self.sc
        self.ps_target = ps.register_point_cloud("objetivo (fijo)", sc.target)
        self.ps_target.set_color(COL_TARGET)
        self.ps_target.set_radius(0.0035)
        self.ps_target.add_vector_quantity("normales", sc.target_normals,
                                           enabled=False, length=0.015,
                                           radius=0.0012, color=(0.2, 0.5, 0.8))

        self.ps_ghost = ps.register_point_cloud("fuente (pose inicial)", sc.source)
        self.ps_ghost.set_color(COL_SOURCE0)
        self.ps_ghost.set_radius(0.0025)
        self.ps_ghost.set_transparency(0.35)

        self.ps_gt = ps.register_point_cloud(
            "fuente (ground truth)", apply_transform(sc.T_gt, sc.source))
        self.ps_gt.set_color(COL_GT)
        self.ps_gt.set_radius(0.0025)
        self.ps_gt.set_enabled(False)

        self.ps_moving = ps.register_point_cloud("fuente (ICP)", sc.source.copy())
        self.ps_moving.set_color(COL_MOVING)
        self.ps_moving.set_radius(0.0035)

        self.ps_corr = None
        self.update_frame(0, force=True)

    # ------------------------------------------------------------------ #
    def update_frame(self, i: int, force: bool = False) -> None:
        i = int(np.clip(i, 0, len(self.result.history) - 1))
        if i == self.frame and not force:
            return
        self.frame = i
        rec = self.result.history[i]
        moved = apply_transform(rec.T, self.sc.source)
        self.ps_moving.update_point_positions(moved)

        dist, _ = self.tree.query(moved, k=1, workers=-1)
        self.ps_moving.add_scalar_quantity("distancia al vecino", dist,
                                           enabled=True, cmap="viridis")

        if self.show_corr and rec.corr_src is not None and len(rec.corr_src):
            k = min(self.max_corr_lines, len(rec.corr_src))
            step = max(1, len(rec.corr_src) // k)
            si = rec.corr_src[::step][:k]
            ti = rec.corr_tgt[::step][:k]
            inl = (rec.corr_inlier[::step][:k].astype(float)
                   if rec.corr_inlier is not None else np.ones(len(si)))
            nodes = np.vstack([moved[si], self.sc.target[ti]])
            edges = np.column_stack([np.arange(len(si)), np.arange(len(si)) + len(si)])
            self.ps_corr = ps.register_curve_network("correspondencias", nodes, edges,
                                                     radius=0.0008)
            self.ps_corr.add_scalar_quantity("aceptada (1) / rechazada (0)", inl,
                                             defined_on="edges", enabled=True,
                                             cmap="coolwarm", vminmax=(0.0, 1.0))
        elif self.ps_corr is not None and not self.show_corr:
            ps.remove_curve_network("correspondencias", False)
            self.ps_corr = None

    # ------------------------------------------------------------------ #
    def rerun(self) -> None:
        p = ICPParams(
            variant=_VARIANTS[self.ui_variant],
            nn_backend=_BACKENDS[self.ui_backend],
            max_iter=int(self.ui_max_iter),
            sample_size=int(self.ui_sample) if self.ui_sample > 0 else None,
            auto_reject_factor=self.ui_reject if self.ui_reject > 0 else None,
            store_corr=self.result.params.store_corr,
        )
        self.result = icp(self.sc.source, self.sc.target, p,
                          target_normals=self.sc.target_normals,
                          source_normals=self.sc.source_normals,
                          T_gt=self.sc.T_gt)
        self.status = self.result.summary()
        print(self.status)
        self.frame = -1
        self.update_frame(0, force=True)

    # ------------------------------------------------------------------ #
    def callback(self) -> None:
        hist = self.result.history
        n = len(hist) - 1
        rec = hist[self.frame]

        psim.TextWrapped(self.sc.description)
        _separator_text("Reproduccion")

        if self.playing:
            self._tick += 1
            if self._tick >= self.speed:
                self._tick = 0
                self.update_frame(self.frame + 1 if self.frame < n else 0)

        if psim.Button("Play" if not self.playing else "Pausa"):
            self.playing = not self.playing
        psim.SameLine()
        if psim.Button("<< paso"):
            self.playing = False
            self.update_frame(self.frame - 1)
        psim.SameLine()
        if psim.Button("paso >>"):
            self.playing = False
            self.update_frame(self.frame + 1)
        psim.SameLine()
        if psim.Button("Reiniciar"):
            self.playing = False
            self.update_frame(0)

        changed, val = psim.SliderInt("iteracion", self.frame, 0, max(n, 1))
        if changed:
            self.playing = False
            self.update_frame(val)
        _, self.speed = psim.SliderInt("frames por iteracion", self.speed, 1, 40)
        ch, self.show_corr = psim.Checkbox("mostrar correspondencias", self.show_corr)
        if ch:
            self.update_frame(self.frame, force=True)
        psim.SameLine()
        ch2, self.show_ghost = psim.Checkbox("pose inicial", self.show_ghost)
        if ch2:
            self.ps_ghost.set_enabled(self.show_ghost)

        # ------------------------------------------------------------- #
        _separator_text("Estado de la iteracion")
        psim.Text(f"iteracion {rec.it} / {n}      pares usados: {rec.n_pairs} "
                  f"({100*rec.inlier_ratio:.1f}% aceptados)")
        psim.Text(f"RMSE correspondencias : {rec.rmse:.6e}")
        if rec.gt_rmse is not None:
            psim.Text(f"RMSE vs ground truth  : {rec.gt_rmse:.6e}")
            psim.Text(f"error rotacion  : {rec.rot_err_deg:8.4f} grados")
            psim.Text(f"error traslacion: {rec.trans_err:.3e}")
        psim.Text(f"paso aplicado: d_rot {rec.delta_rot_deg:.4f} deg, "
                  f"d_trans {rec.delta_trans:.2e}")
        psim.Text(f"tiempos: vecinos {rec.t_query*1e3:.2f} ms | "
                  f"minimizacion {rec.t_solve*1e3:.2f} ms")

        curve = self.result.rmse_curve()
        logc = np.log10(np.maximum(curve, 1e-12)).astype(np.float32)
        psim.PlotLines("log10 RMSE", logc, graph_size=(0.0, 70.0))
        gt = self.result.gt_curve()
        if np.isfinite(gt).all():
            psim.PlotLines("log10 err. GT", np.log10(np.maximum(gt, 1e-12)).astype(np.float32),
                           graph_size=(0.0, 70.0))

        psim.Text(f"total: {self.result.n_iter} iteraciones, "
                  f"{self.result.total_time*1e3:.1f} ms "
                  f"(vecinos {self.result.query_time*1e3:.1f} ms, "
                  f"minimizacion {self.result.solve_time*1e3:.1f} ms)")

        # ------------------------------------------------------------- #
        _separator_text("Volver a ejecutar")
        _, self.ui_variant = psim.Combo("metrica", self.ui_variant, _VARIANTS)
        _, self.ui_backend = psim.Combo("busqueda NN", self.ui_backend, _BACKENDS)
        _, self.ui_max_iter = psim.SliderInt("max iteraciones", self.ui_max_iter, 1, 200)
        _, self.ui_sample = psim.SliderInt("submuestreo (0 = todos)", self.ui_sample,
                                           0, len(self.sc.source))
        _, self.ui_reject = psim.SliderFloat("rechazo (x mediana, 0 = off)",
                                             self.ui_reject, 0.0, 10.0)
        if psim.Button("Ejecutar ICP"):
            self.playing = False
            self.rerun()
        if self.status:
            psim.TextWrapped(self.status)


def show(scenario: Scenario, result: ICPResult, max_corr_lines: int = 400) -> None:
    """Abre el visor de Polyscope con la animacion del registro."""
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    viewer = ICPViewer(scenario, result, max_corr_lines)
    viewer.register()
    ps.set_user_callback(viewer.callback)
    ps.show()


def show_comparison(scenario: Scenario, results: dict[str, ICPResult]) -> None:
    """Muestra en paralelo el resultado final de varias configuraciones."""
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    tgt = ps.register_point_cloud("objetivo (fijo)", scenario.target)
    tgt.set_color(COL_TARGET)
    tgt.set_radius(0.0035)
    ghost = ps.register_point_cloud("fuente (pose inicial)", scenario.source)
    ghost.set_color(COL_SOURCE0)
    ghost.set_transparency(0.35)

    palette = [(0.95, 0.65, 0.15), (0.30, 0.70, 0.45), (0.65, 0.40, 0.85),
               (0.90, 0.35, 0.55), (0.20, 0.75, 0.85)]
    lines = []
    for i, (name, res) in enumerate(results.items()):
        pc = ps.register_point_cloud(f"resultado: {name}",
                                     apply_transform(res.T, scenario.source))
        pc.set_color(palette[i % len(palette)])
        pc.set_radius(0.003)
        pc.set_enabled(i == 0)
        lines.append(res.summary())

    def cb():
        psim.TextWrapped(scenario.description)
        _separator_text("Resultados")
        for ln in lines:
            psim.Text(ln)
        psim.TextWrapped("Activa/desactiva cada nube en el panel de estructuras "
                         "para comparar los resultados finales.")

    ps.set_user_callback(cb)
    ps.show()
