"""Compatibilidad entre versiones de Polyscope (probado en 1.3.4, 2.0, 2.1 y 2.6).

· imgui < 1.89 no tiene SeparatorText ni BeginTable.
· Polyscope < 2 no tiene frame_tick (se usa show(forFrames=n)) ni
  set_allow_headless_backends.
· screenshot() cambió de firma; PlotLines pide list en 1.x y ndarray float32 en 2.x.
"""
import numpy as np
import polyscope as ps
import polyscope.imgui as _imgui

# La fuente de Polyscope < 2.4 (aprox.) solo trae Latin-1: tildes y ñ sí, pero
# χ, λ, v̄, ⁻¹, ‰ salen como "?". Todo texto del panel pasa por T().
_SUBS = [("\u0304", "bar"), ("⁻¹", "^-1"), ("⁺", "+"), ("⁻", "^-"), ("ᵀ", "^T"),
         ("₁", "1"), ("₂", "2"), ("Δ", "D"), ("λ", "lambda"), ("χ", "chi"), ("Σ", "Sum"),
         ("√", "sqrt"), ("‰", "o/oo"), ("—", "-"), ("–", "-"), ("−", "-"), ("⟂", "perp."),
         ("→", "->"), ("■", "#"), ("≈", "~"), ("≤", "<="), ("≥", ">="), ("…", "..."),
         ("∈", "en"), ("‖", "|")]


def T(s):
    """Texto seguro para la fuente Latin-1 de Polyscope."""
    if not isinstance(s, str):
        return s
    for a, b in _SUBS:
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


class _SafeImgui:
    """imgui con todos los argumentos de texto pasados por T()."""

    def __getattr__(self, name):
        obj = getattr(_imgui, name)
        if not callable(obj):
            return obj

        def call(*args, **kw):
            return obj(*[T(a) for a in args], **{k: T(v) for k, v in kw.items()})
        return call


psim = _SafeImgui()


def separator_text(label):
    if hasattr(_imgui, "SeparatorText"):
        psim.SeparatorText(label)
    else:
        psim.Separator()
        psim.TextColored((0.55, 0.75, 1.0, 1.0), label)


def plot_lines(label, values, overlay="", size=(0, 70), vmin=None, vmax=None):
    vals = np.asarray(values, dtype=np.float32)
    if len(vals) == 0:
        return
    lo = float(vals.min()) if vmin is None else vmin
    hi = float(vals.max()) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1.0
    try:
        _imgui.PlotLines(T(label), vals, 0, T(overlay), lo, hi, size)
    except TypeError:
        _imgui.PlotLines(T(label), vals.tolist(), 0, T(overlay), lo, hi, size)


def default_open_flag():
    return getattr(_imgui, "ImGuiTreeNodeFlags_DefaultOpen", 32)


def header(label, open_default=True):
    flags = default_open_flag() if open_default else 0
    r = psim.CollapsingHeader(label, flags)
    return r[0] if isinstance(r, tuple) else r


def table(rows, header_row=None, key="tbl"):
    """Tabla: BeginTable si existe, si no texto monoespaciado alineado."""
    ncol = len(rows[0]) if rows else (len(header_row) if header_row else 0)
    if ncol == 0:
        return
    if hasattr(_imgui, "BeginTable"):
        flags = getattr(_imgui, "ImGuiTableFlags_Borders", 0) | getattr(_imgui, "ImGuiTableFlags_RowBg", 0) | \
            getattr(_imgui, "ImGuiTableFlags_SizingFixedFit", 0)
        if psim.BeginTable(key, ncol, flags):
            if header_row:
                for h in header_row:
                    psim.TableSetupColumn(h)
                psim.TableHeadersRow()
            for r in rows:
                psim.TableNextRow()
                for c, val in enumerate(r):
                    psim.TableSetColumnIndex(c)
                    psim.TextUnformatted(str(val))
            psim.EndTable()
        return
    allrows = ([header_row] if header_row else []) + [[str(v) for v in r] for r in rows]
    w = [max(len(r[c]) for r in allrows) for c in range(ncol)]
    for i, r in enumerate(allrows):
        psim.TextUnformatted("  ".join(s.rjust(w[c]) if c else s.ljust(w[c]) for c, s in enumerate(r)))
        if i == 0 and header_row:
            psim.Separator()


def frame_tick(n=1):
    if hasattr(ps, "frame_tick"):
        for _ in range(n):
            ps.frame_tick()
    else:
        ps.show(forFrames=n)


def screenshot(fname, ui=False):
    try:
        ps.screenshot(fname, transparent_bg=False, include_UI=ui)
    except TypeError:
        try:
            ps.screenshot(fname, transparent_bg=False)
        except TypeError:
            ps.screenshot(fname)


def allow_headless():
    if hasattr(ps, "set_allow_headless_backends"):
        ps.set_allow_headless_backends(True)


def radio(label, active):
    """RadioButton(label, activo) → True si se hizo clic."""
    r = psim.RadioButton(label, active)
    return r[0] if isinstance(r, tuple) else r


def slider_int(label, v, lo, hi):
    return psim.SliderInt(label, int(v), int(lo), int(hi))


def slider_float(label, v, lo, hi, fmt="%.3f"):
    try:
        return psim.SliderFloat(label, float(v), float(lo), float(hi), fmt)
    except TypeError:
        return psim.SliderFloat(label, float(v), float(lo), float(hi))
