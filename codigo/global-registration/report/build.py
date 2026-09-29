"""Arma el reporte HTML: genera los graficos SVG y embebe las figuras."""
from __future__ import annotations

import base64
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
W, H = 760, 250
PADL, PADR, PADT, PADB = 46, 14, 14, 40


def data_uri(path: pathlib.Path) -> str:
    b = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{b}"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- #
def line_chart(series, xlabels, ymax=1.0, ylab_fmt="{:.0%}"):
    """series: lista de (nombre, var_css, [valores])"""
    x0, x1 = PADL, W - PADR
    y0, y1 = H - PADB, PADT
    n = len(xlabels)
    xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]

    def Y(v):
        return y1 + (y0 - y1) * (1 - v / ymax)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="Grafico de lineas de repetibilidad">']
    for g in (0, .25, .5, .75, 1.0):
        y = Y(g * ymax)
        out.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 8}" y="{y + 4:.1f}" '
                   f'text-anchor="end">{ylab_fmt.format(g * ymax)}</text>')
    out.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}"/>')
    for i, lb in enumerate(xlabels):
        out.append(f'<text class="tick" x="{xs[i]:.1f}" y="{y0 + 18}" '
                   f'text-anchor="middle">{esc(lb)}</text>')
    out.append(f'<text class="tick" x="{(x0 + x1) / 2:.0f}" y="{H - 6}" '
               f'text-anchor="middle">tolerancia ε (voxels)</text>')
    ends = sorted(range(len(series)), key=lambda k: Y(series[k][2][-1]))
    ypos, last = {}, -1e9
    for k in ends:
        y = max(Y(series[k][2][-1]) + 4, last + 13)
        ypos[k], last = y, y
    for si, (name, var, vals) in enumerate(series):
        pts = " ".join(f"{xs[i]:.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="var({var})" '
                   f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for i, v in enumerate(vals):
            out.append(f'<circle cx="{xs[i]:.1f}" cy="{Y(v):.1f}" r="4" '
                       f'fill="var({var})" stroke="var(--surface)" stroke-width="2"/>')
        out.append(f'<text class="slab" x="{xs[-1] + 8:.1f}" y="{ypos[si]:.1f}" '
                   f'fill="var({var})">{vals[-1]:.2f}</text>')
    out.append("</svg>")
    return "\n".join(out)


def grouped_bars(groups, series, ymax=1.0, height=250, ylabel_fmt="{:.2f}",
                 vfmt=None, xtitle=None, padb=None,
                 aria="Grafico de barras agrupadas"):
    """groups: [etiqueta,...]   series: [(nombre, var, [valores por grupo])]"""
    h = height
    y0, y1 = h - (padb or PADB), PADT + 6
    x0, x1 = PADL, W - PADR
    vfmt = vfmt or ylabel_fmt
    ng, ns = len(groups), len(series)
    gw = (x1 - x0) / ng
    bw = min(46, (gw - 18) / ns)

    def Y(v):
        return y1 + (y0 - y1) * (1 - v / ymax)

    out = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{esc(aria)}">']
    for g in (0, .25, .5, .75, 1.0):
        y = Y(g * ymax)
        out.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 8}" y="{y + 4:.1f}" '
                   f'text-anchor="end">{ylabel_fmt.format(g * ymax)}</text>')
    out.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}"/>')
    for gi, gl in enumerate(groups):
        cx = x0 + gw * (gi + .5)
        lines = str(gl).split("\n")
        spans = "".join(
            f'<tspan x="{cx:.1f}" dy="{0 if i == 0 else 14}">{esc(t)}</tspan>'
            for i, t in enumerate(lines))
        out.append(f'<text class="tick" x="{cx:.1f}" y="{y0 + 18}" '
                   f'text-anchor="middle">{spans}</text>')
        total = ns * bw + (ns - 1) * 2
        for si, (name, var, vals) in enumerate(series):
            v = vals[gi]
            bx = cx - total / 2 + si * (bw + 2)
            by = Y(v)
            bh = max(y0 - by, 0.8)
            out.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" '
                       f'height="{bh:.1f}" rx="3" fill="var({var})"/>')
            out.append(f'<text class="vlab" x="{bx + bw / 2:.1f}" y="{by - 6:.1f}" '
                       f'text-anchor="middle">{vfmt.format(v)}</text>')
    if xtitle:
        out.append(f'<text class="tick" x="{(x0 + x1) / 2:.0f}" y="{h - 6}" '
                   f'text-anchor="middle">{esc(xtitle)}</text>')
    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def main():
    tpl = (ROOT / "report" / "template.html").read_text()

    charts = {
        "CHART_REPEAT": line_chart(
            [("Harris 3D", "--s1", [0.26, 0.58, 0.82, 0.92, 0.97]),
             ("Harris 3D (Noble)", "--s2", [0.25, 0.59, 0.83, 0.94, 0.99]),
             ("ISS", "--s3", [0.10, 0.32, 0.41, 0.55, 0.73]),
             ("aleatorio", "--s4", [0.11, 0.36, 0.63, 0.80, 0.94])],
            ["1", "2", "3", "4", "6"]),
        "CHART_DESC": grouped_bars(
            ["solapamiento ≈ 0.86", "solapamiento ≈ 0.63", "solapamiento ≈ 0.29"],
            [("Spin images", "--s1", [0.62, 0.22, 0.05]),
             ("FPFH", "--s2", [0.30, 0.13, 0.03])],
            ymax=0.8, ylabel_fmt="{:.2f}",
            aria="Razon de inliers por descriptor y solapamiento"),
        "CHART_SOLVER": grouped_bars(
            ["solapamiento > 0.5", "solapamiento 0.35 – 0.5", "solapamiento < 0.35"],
            [("Agrupamiento (slides)", "--s1", [1.00, 0.00, 0.00]),
             ("Clique por distancias", "--s2", [1.00, 0.00, 0.00]),
             ("RANSAC", "--s3", [1.00, 0.50, 0.58])],
            ymax=1.0, ylabel_fmt="{:.2f}",
            aria="Recall por solver y solapamiento"),
        "CHART_SCALE": grouped_bars(
            ["n = 500\n0.05 s / 0.04 s", "n = 1 000\n0.09 s / 0.07 s",
             "n = 2 000\n1.24 s / 0.15 s", "n = 4 000\n5.27 s / 0.51 s"],
            [("Aceleracion", "--s1", [1.3, 1.4, 8.3, 10.4])],
            ymax=12.0, ylabel_fmt="{:.0f}x", vfmt="{:.1f}x", height=290, padb=76,
            xtitle="puntos en la nube de destino  ·  tiempo 4PCS / Super4PCS",
            aria="Aceleracion de Super4PCS sobre 4PCS al crecer la nube"),
    }
    for k, v in charts.items():
        tpl = tpl.replace("{{" + k + "}}", v)

    figs = {
        "FIG_STAGES": "fig_stages.png",
        "FIG_HARRIS": "fig_harris_mesh.png",
        "FIG_CORR": "fig_corr_overlap.png",
        "FIG_SPIN": "fig_spin.png",
    }
    for k, fn in figs.items():
        tpl = tpl.replace("{{" + k + "}}", data_uri(ROOT / "figs" / fn))

    assert "{{" not in tpl, "quedaron placeholders sin reemplazar"
    out = ROOT / "report" / "reporte.html"
    out.write_text(tpl)
    print(out, f"{out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    sys.exit(main())
