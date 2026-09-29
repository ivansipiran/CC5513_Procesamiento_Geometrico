"""
Visualizacion interactiva con Polyscope.

Se abre una sola ventana con la configuracion inicial y los resultados de cada
metodo, y se cambia entre ellos con los radio buttons del panel. No se anima el
progreso: solo antes / despues, mas las correspondencias que produjo el
descriptor y la pose de ground truth como referencia.

En un entorno sin pantalla (contenedor, servidor) Polyscope arranca con el
backend EGL headless y en vez de abrir la ventana guarda un PNG por estado.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

try:
    import polyscope as ps
    import polyscope.imgui as psim
    HAVE_POLYSCOPE = True
except ImportError:                                   # pragma: no cover
    HAVE_POLYSCOPE = False


SRC_COLOR = (0.878, 0.482, 0.224)     # naranja
TGT_COLOR = (0.227, 0.486, 0.647)     # azul
KP_COLOR = (0.10, 0.10, 0.12)
INLIER = (0.10, 0.65, 0.35)
OUTLIER = (0.84, 0.20, 0.18)


@dataclass
class Pose:
    """Un estado que se puede mostrar: un nombre, una pose y su descripcion."""
    name: str
    T: np.ndarray
    info: str = ""
    show_correspondences: bool = False
    extra: dict = field(default_factory=dict)


def _fmt_error(T, T_gt):
    from . import metrics
    rot = metrics.rotation_error_deg(T, T_gt)
    tr = metrics.translation_error(T, T_gt)
    ok = "OK" if (rot < 5 and tr < 0.05) else "FALLA"
    return f"error de rotacion {rot:6.2f} deg   traslacion {tr:.4f}   [{ok}]"


def show(pair,
         poses: list[Pose],
         keypoints=None,
         correspondences=None,
         inliers=None,
         screenshot_prefix: str | None = None,
         window_title: str = "Global Registration"):
    """Abre Polyscope con `pair` y una lista de poses intercambiables.

    pair            : RegistrationPair (source, target, T_gt)
    keypoints       : (kp_src, kp_tgt) en coordenadas de cada nube, o None
    correspondences : arreglo (M,2) de indices sobre kp_src / kp_tgt, o None
    inliers         : mascara booleana (M,) para colorear las correspondencias
    """
    if not HAVE_POLYSCOPE:
        raise RuntimeError(
            "polyscope no esta instalado.  pip install polyscope\n"
            "(o corre el demo con --no-view)")

    ps.set_allow_headless_backends(True)
    ps.set_program_name(window_title)
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")
    ps.set_SSAA_factor(2)

    S = np.asarray(pair.source.points)
    T = np.asarray(pair.target.points)

    ps_tgt = ps.register_point_cloud("destino", T, color=TGT_COLOR)
    ps_tgt.set_radius(0.0026, relative=True)
    ps_src = ps.register_point_cloud("origen", S, color=SRC_COLOR)
    ps_src.set_radius(0.0026, relative=True)

    ps_kp_s = ps_kp_t = ps_corr = None
    if keypoints is not None:
        kp_s, kp_t = keypoints
        ps_kp_s = ps.register_point_cloud("keypoints origen", kp_s, color=KP_COLOR)
        ps_kp_s.set_radius(0.008, relative=True)
        ps_kp_s.set_enabled(False)
        ps_kp_t = ps.register_point_cloud("keypoints destino", kp_t, color=KP_COLOR)
        ps_kp_t.set_radius(0.008, relative=True)
        ps_kp_t.set_enabled(False)

        if correspondences is not None and len(correspondences):
            nodes = np.vstack([kp_s, kp_t])
            edges = np.stack([correspondences[:, 0],
                              correspondences[:, 1] + len(kp_s)], axis=1)
            ps_corr = ps.register_curve_network("correspondencias", nodes, edges)
            ps_corr.set_radius(0.0016, relative=True)
            if inliers is not None:
                col = np.where(np.asarray(inliers)[:, None],
                               np.array(INLIER), np.array(OUTLIER))
                ps_corr.add_color_quantity("correcta / incorrecta", col,
                                           defined_on="edges", enabled=True)
            else:
                ps_corr.set_color(INLIER)

    # los keypoints del origen viajan con la nube del origen
    def apply(pose: Pose):
        for st in (ps_src, ps_kp_s):
            if st is not None:
                st.set_transform(pose.T)
        if ps_corr is not None:
            ps_corr.set_enabled(pose.show_correspondences)

    state = {"i": 0, "kp": False}
    apply(poses[0])

    def callback():
        psim.Begin("Global Registration", True)
        psim.TextUnformatted(f"{pair.name}   |origen|={len(S)}  "
                             f"|destino|={len(T)}   solapamiento {pair.overlap:.2f}")
        psim.Separator()
        for i, p in enumerate(poses):
            if psim.RadioButton(p.name, state["i"] == i) and state["i"] != i:
                state["i"] = i
                apply(poses[i])
        psim.Separator()
        for line in poses[state["i"]].info.split("\n"):
            psim.TextUnformatted(line)
        if ps_kp_s is not None:
            psim.Separator()
            ch, val = psim.Checkbox("mostrar keypoints", state["kp"])
            if ch:
                state["kp"] = val
                ps_kp_s.set_enabled(val)
                ps_kp_t.set_enabled(val)
        psim.End()

    ps.set_user_callback(callback)

    if ps.is_headless() or screenshot_prefix:
        prefix = screenshot_prefix or "polyscope"
        out = []
        for i, p in enumerate(poses):
            state["i"] = i
            apply(p)
            ps.reset_camera_to_home_view()   # reencuadrar: al alinearse, acerca
            fn = f"{prefix}_{i}_{_slug(p.name)}.png"
            ps.screenshot(fn, transparent_bg=False)
            out.append(fn)
        state["i"] = 0
        apply(poses[0])
        return out

    ps.show()
    return []


def _slug(s: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in s]
    return "".join(keep).strip("_").replace("__", "_")[:28]
