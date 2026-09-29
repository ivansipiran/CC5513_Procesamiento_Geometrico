"""
Compatibilidad con versiones viejas de Polyscope (mismos shims que los demos anteriores).

  - imgui.SeparatorText no existe antes de imgui 1.89     -> Separator() + Text()
  - BeginTable puede no estar expuesto                     -> texto monoespaciado
  - frame_tick() no existe en Polyscope 1.x                -> show(forFrames=1)
  - la fuente de Polyscope <= 2.3 es Latin-1: letras griegas, ‰, etc. salen como "?"
    -> todo el texto del panel pasa por T()
  - screenshot(): la firma cambió entre versiones
"""
import numpy as np

try:
    from importlib.metadata import version as _version
    PS_VERSION = tuple(int(x) for x in _version("polyscope").split(".")[:2])
except Exception:  # pragma: no cover
    PS_VERSION = (0, 0)

_UNICODE_OK = PS_VERSION >= (2, 4)

_REPL = {
    "χ": "chi", "λ": "lambda", "β": "beta", "α": "alpha", "δ": "delta", "π": "pi",
    "‰": "o/oo", "≈": "~", "≤": "<=", "≥": ">=", "→": "->", "·": "*", "Σ": "sum ",
    "²": "^2", "∞": "inf", "—": "-", "–": "-", "±": "+/-",
}


def T(s):
    """Texto seguro para la fuente de imgui."""
    if _UNICODE_OK:
        return s
    for a, b in _REPL.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


def separator_text(psim, s):
    if hasattr(psim, "SeparatorText"):
        psim.SeparatorText(T(s))
    else:
        psim.Separator()
        psim.TextUnformatted(T(s))


def has_tables(psim):
    return all(hasattr(psim, n) for n in ("BeginTable", "EndTable", "TableNextRow",
                                           "TableNextColumn", "TableSetupColumn",
                                           "TableHeadersRow"))


def table(psim, tid, header, rows):
    """Tabla simple; si no hay BeginTable, filas de texto alineadas."""
    if has_tables(psim):
        flags = 0
        for fl in ("ImGuiTableFlags_Borders", "ImGuiTableFlags_RowBg", "ImGuiTableFlags_SizingFixedFit"):
            flags |= getattr(psim, fl, 0)
        if psim.BeginTable(tid, len(header), flags):
            for h in header:
                psim.TableSetupColumn(T(h))
            psim.TableHeadersRow()
            for r in rows:
                psim.TableNextRow()
                for c in r:
                    psim.TableNextColumn()
                    psim.TextUnformatted(T(str(c)))
            psim.EndTable()
        return
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
         for i, h in enumerate(header)]
    psim.TextUnformatted(T("  ".join(str(h).rjust(w[i]) for i, h in enumerate(header))))
    for r in rows:
        psim.TextUnformatted(T("  ".join(str(c).rjust(w[i]) for i, c in enumerate(r))))


def text_colored(psim, rgb, s):
    try:
        psim.TextColored((float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0), T(s))
    except Exception:
        psim.TextUnformatted(T(s))


def frame_tick(ps):
    if hasattr(ps, "frame_tick"):
        ps.frame_tick()
    else:
        ps.show(forFrames=1)


def screenshot(ps, fn, include_ui=False):
    try:
        ps.screenshot(fn, transparent_bg=False, include_UI=include_ui)
    except TypeError:
        try:
            ps.screenshot(fn, transparent_bg=False)
        except TypeError:
            ps.screenshot(fn)


def init_polyscope(ps, headless=False):
    if headless and hasattr(ps, "set_allow_headless_backends"):
        ps.set_allow_headless_backends(True)
    ps.init()


def set_ground(ps, mode="shadow_only"):
    try:
        ps.set_ground_plane_mode(mode)
    except Exception:
        pass


def radio(psim, label, active):
    """RadioButton que devuelve True si se hizo clic."""
    return bool(psim.RadioButton(T(label), active))


def collapsing(psim, label, default_open=True):
    try:
        if default_open and hasattr(psim, "SetNextItemOpen"):
            psim.SetNextItemOpen(True, getattr(psim, "ImGuiCond_FirstUseEver", 4))
        return psim.CollapsingHeader(T(label))
    except TypeError:
        return psim.CollapsingHeader(T(label), 0)


def input_int(psim, label, v):
    try:
        return psim.InputInt(T(label), int(v))
    except TypeError:
        return psim.InputInt(T(label), int(v), 1)


def add_colors(struct, name, colors, defined_on=None, enabled=True):
    """defined_on solo para mallas ('vertices' / 'faces'); en nubes de puntos se omite."""
    colors = np.ascontiguousarray(colors, dtype=np.float64)
    kw = dict(enabled=enabled)
    if defined_on is not None:
        kw["defined_on"] = defined_on
    return struct.add_color_quantity(name, colors, **kw)


def add_scalar(struct, name, values, defined_on=None, enabled=True, cmap="viridis",
               vminmax=None):
    kw = dict(enabled=enabled, cmap=cmap)
    if defined_on is not None:
        kw["defined_on"] = defined_on
    if vminmax is not None:
        kw["vminmax"] = vminmax
    try:
        return struct.add_scalar_quantity(name, np.asarray(values, float), **kw)
    except TypeError:
        kw.pop("vminmax", None)
        return struct.add_scalar_quantity(name, np.asarray(values, float), **kw)
