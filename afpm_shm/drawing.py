"""Live 2D drawing of the machine: cut-away top view and axial cross-section (all sizes in mm)."""
import math

import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

from . import model as M

PHASE_COLOR = {"A": "#e08a1e", "B": "#2e9e6b", "C": "#7a6fc4"}
IRON = "#c9ced6"
IRON_DARK = "#aab1bc"
MAG_N = "#d64545"
MAG_S = "#3a6fd8"
COPPER = "#d9a066"
SHAFT = "#6b7280"


def _tooth_poly(theta, ri, ro, wi, wo):
    u = np.array([math.cos(theta), math.sin(theta)])
    t = np.array([-math.sin(theta), math.cos(theta)])
    return np.array([ri * u - wi / 2 * t, ro * u - wo / 2 * t, ro * u + wo / 2 * t, ri * u + wi / 2 * t])


def draw_top(ax, P):
    mm = 1e3
    Ro, Ri, rsh = P.Ro * mm, P.Ri * mm, P.rsh * mm
    Q, p = int(P.Q), int(P.p)
    L = M.winding_layout(Q, p)
    lim = Ro * 1.12
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")
    clip_l = Rectangle((-lim, -lim), lim, 2 * lim, transform=ax.transData)
    clip_r = Rectangle((0, -lim), lim, 2 * lim, transform=ax.transData)

    def add(patch, clip):
        ax.add_patch(patch); patch.set_clip_path(clip)
        return patch

    # ---- left half: stator (looking at the winding side)
    add(Wedge((0, 0), Ro * 1.04, 0, 360, width=Ro * 1.04 - rsh * 1.4, fc=IRON, ec="#8a929e", lw=0.8), clip_l)
    pitch = 2 * math.pi / Q
    for k in range(Q):
        th = (k + 0.5) * pitch
        wi, wo = P.wti * mm, P.wto * mm
        # coil: tooth plus coil build on both sides, kept inside the slot pitch
        ci = min(wi + 2 * P.wc * mm, 2 * math.pi * Ri / Q * 0.97)
        co = min(wo + 2 * P.wc * mm, 2 * math.pi * Ro / Q * 0.97)
        col = PHASE_COLOR[L.phase[k]]
        add(Polygon(_tooth_poly(th, Ri, Ro, ci, co), closed=True, fc=col, ec="#444", lw=0.6,
                    hatch=None if L.sign[k] > 0 else "////", alpha=0.85), clip_l)
        add(Polygon(_tooth_poly(th, Ri, Ro, wi, wo), closed=True, fc=IRON_DARK, ec="#555", lw=0.6), clip_l)
        rm = (Ri + Ro) / 2
        x, y = rm * math.cos(th), rm * math.sin(th)
        if x < -2:
            ax.text(x, y, "%s%s" % (L.phase[k], "+" if L.sign[k] > 0 else "−"), ha="center", va="center",
                    fontsize=7, fontweight="bold", color="#222")
    ax.add_patch(Circle((0, 0), rsh, fc=SHAFT, ec="#333", lw=0.8, zorder=5))

    # ---- right half: rotor (magnets)
    add(Wedge((0, 0), Ro * 1.04, 0, 360, width=Ro * 1.04 - rsh * 1.4, fc="#e6e8ec", ec="#8a929e", lw=0.8), clip_r)
    npole = 2 * p
    pp = 360.0 / npole
    for j in range(npole):
        c = (j + 0.5) * pp
        a = P.alpha * pp
        add(Wedge((0, 0), Ro, c - a / 2, c + a / 2, width=Ro - Ri, fc=MAG_N if j % 2 == 0 else MAG_S,
                  ec="#222", lw=0.6), clip_r)
        x, y = (Ro + Ri) / 2 * math.cos(math.radians(c)), (Ro + Ri) / 2 * math.sin(math.radians(c))
        if x > 2:
            ax.text(x, y, "N" if j % 2 == 0 else "S", ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold")
    ax.plot([0, 0], [-lim * 0.97, lim * 0.97], color="#555", lw=0.8, ls=(0, (4, 3)))
    ax.text(-lim * 0.98, lim * 0.97, "Stator\n%d slots" % Q, fontsize=8, ha="left", va="top", color="#333")
    ax.text(lim * 0.98, lim * 0.97, "Rotor\n%d poles" % npole, fontsize=8, ha="right", va="top", color="#333")
    ax.text(0, -lim * 1.0, "Ro = %.1f mm   Ri = %.1f mm" % (Ro, Ri), fontsize=8, ha="center", va="top", color="#333")


DIM = "#1f2937"
EXT = "#6b7280"


def _hdim(ax, x0, x1, y, text, ext_to=None, fs=6.8, above=True):
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color=DIM, shrinkA=0, shrinkB=0, mutation_scale=6))
    if ext_to is not None:
        for x in (x0, x1):
            ax.plot([x, x], [y, ext_to], color=EXT, lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.text((x0 + x1) / 2, y + (0.8 if above else -0.8), text, ha="center", va="bottom" if above else "top",
            fontsize=fs, color=DIM, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85), zorder=9)


def _vdim(ax, x, y0, y1, fs=6.8):
    ax.annotate("", xy=(x, y0), xytext=(x, y1),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color=DIM, shrinkA=0, shrinkB=0, mutation_scale=5))


def _spread(ys, gap):
    """Move labels apart so that texts at neighbouring (thin) layers do not overlap."""
    order = sorted(range(len(ys)), key=lambda i: -ys[i])      # top to bottom
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[a] - out[b] < gap:
            out[b] = out[a] - gap
    return out


def _stack(P):
    """Layers from top to bottom for the developed view: (label, thickness mm, kind)."""
    mm = 1e3
    if P.topology == "SSSR":
        return [("stator yoke", P.hsy * mm, "yoke"), ("slot / coil height", P.hs * mm, "slot_dn"),
                ("air gap g", P.g * mm, "gap"), ("magnet hm", P.hm * mm, "mag"),
                ("rotor back-iron", P.hby * mm, "back")]
    return [("rotor back-iron", P.hby * mm, "back"), ("magnet hm", P.hm * mm, "mag"),
            ("air gap g", P.g * mm, "gap"), ("slot / coil height", P.hs * mm, "slot_up"),
            ("support plate", P.hplate * mm, "yoke"), ("slot / coil height", P.hs * mm, "slot_dn"),
            ("air gap g", P.g * mm, "gap"), ("magnet hm", P.hm * mm, "mag"),
            ("rotor back-iron", P.hby * mm, "back")]


def _crossing(ax, x, y, up, r=0.9):
    """Conductor symbol: cross = current into the page, dot = out of the page."""
    ax.add_patch(Circle((x, y), r, fc="white", ec="#222", lw=0.6, zorder=6))
    if up:
        ax.add_patch(Circle((x, y), r * 0.32, fc="#222", ec="none", zorder=7))
    else:
        d = r * 0.68
        ax.plot([x - d, x + d], [y - d, y + d], color="#222", lw=0.6, zorder=7)
        ax.plot([x - d, x + d], [y + d, y - d], color="#222", lw=0.6, zorder=7)


def draw_developed(ax, P):
    """Circumferential-axial section at the mean radius, drawn like a textbook machine sketch."""
    mm = 1e3
    Ro, Ri = P.Ro * mm, P.Ri * mm
    rm = (Ro + Ri) / 2
    Q, p = int(P.Q), int(P.p)
    ts = 2 * math.pi * rm / Q
    wt = (P.wti + P.wto) / 2 * mm
    b0 = P.b0 * mm
    sw = max(ts - wt, 0.5)
    tp = math.pi * rm / p
    wm = P.alpha * tp
    nshow = max(2, min(Q, 4))
    W = nshow * ts
    L = M.winding_layout(Q, p)
    layers = _stack(P)
    H = sum(t for _, t, _ in layers)
    hsh = min(1.5, 0.12 * P.hs * mm)
    # y runs from 0 (top) downwards (negative)
    ys = [0.0]
    for _, t, _ in layers:
        ys.append(ys[-1] - t)
    clip = Rectangle((0, -H - 1), W, H + 2, transform=ax.transData)

    def add(patch):
        ax.add_patch(patch); patch.set_clip_path(clip); return patch

    nmag = 0
    for (label, t, kind), ytop, ybot in zip(layers, ys, ys[1:]):
        if kind == "gap":
            continue
        if kind == "back":
            add(Rectangle((0, ybot), W, t, fc=IRON_DARK, ec="#555", lw=0.6))
        elif kind == "yoke":
            add(Rectangle((0, ybot), W, t, fc=IRON, ec="#555", lw=0.6))
        elif kind == "mag":
            nmag += 1                           # second rotor of an SSDR: opposite pole faces the first one
            for j in range(-3, int(W / tp) + 4):
                c = ts / 2 + j * tp
                isn = (j % 2 == 0) == (nmag == 1)
                add(Rectangle((c - wm / 2, ybot), wm, t, fc=MAG_N if isn else MAG_S, ec="#222", lw=0.6))
                if 0 < c < W and t > 1.5:
                    ax.text(c, (ytop + ybot) / 2, "N" if isn else "S", ha="center", va="center",
                            fontsize=6.5, color="white", fontweight="bold", zorder=8)
        else:                                   # slot layer: iron with slots, shoes and coil sides
            down = kind == "slot_dn"            # air gap below (True) or above (False)
            add(Rectangle((0, ybot), W, t, fc=IRON, ec="#555", lw=0.6))
            y_edge = ybot if down else ytop     # air-gap side
            y_far = ytop if down else ybot      # yoke side
            sgn = 1 if down else -1             # direction from the gap edge into the iron
            y_in = y_edge + sgn * hsh
            op = min(b0, sw)
            for i in range(-1, nshow + 1):
                xc = (i + 1) * ts
                xl, xr = xc - sw / 2, xc + sw / 2
                add(Polygon([(xl, y_far), (xr, y_far), (xr, y_in), (xc + op / 2, y_in), (xc + op / 2, y_edge),
                             (xc - op / 2, y_edge), (xc - op / 2, y_in), (xl, y_in)], closed=True,
                            fc="white", ec="#555", lw=0.6, zorder=2))
                # two coil sides: right side of tooth i, left side of tooth i+1
                for side, k in ((0, i % Q), (1, (i + 1) % Q)):
                    val = (-L.sign[k]) if side == 0 else L.sign[k]
                    cw = max((sw - 0.5) / 2, 0.3)
                    cx0 = xl + 0.2 if side == 0 else xc + 0.05
                    ya, yb = sorted((y_far - sgn * 0.0, y_in + sgn * 0.2))
                    add(Rectangle((cx0, ya), cw, yb - ya, fc=PHASE_COLOR[L.phase[k]], ec="#444", lw=0.5, zorder=3, alpha=0.95))
                    cxm = cx0 + cw / 2
                    if 0 < cxm < W and cw > 1.2:
                        _crossing(ax, cxm, (ya + yb) / 2, val > 0, r=min(0.9, cw * 0.32))
                        ax.text(cxm, (ya + yb) / 2 - min(2.6, (yb - ya) * 0.25), L.phase[k], ha="center", va="center",
                                fontsize=5.5, color="#222", zorder=8, fontweight="bold")

    # ------------------------------------------------ dimension arrows
    top_y, bot_y = ys[0], ys[-1]
    # vertical thickness arrows on the right, with spread-out labels
    xd = W + 2.5
    mids = [(a + b) / 2 for a, b in zip(ys, ys[1:])]
    lab_y = _spread(mids, 3.0)
    for (label, t, kind), a, b, ly in zip(layers, ys, ys[1:], lab_y):
        _vdim(ax, xd, a, b)
        ax.plot([W, xd + 0.8], [a, a], color=EXT, lw=0.4, ls=(0, (2, 2)))
        ax.plot([W, xd + 0.8], [b, b], color=EXT, lw=0.4, ls=(0, (2, 2)))
        ax.text(xd + 2.0, ly, "%s %.3g mm" % (label, t) if "= " not in label else label, ha="left", va="center",
                fontsize=6.6, color=DIM)
    # top rows: slot pitch, tooth width, slot opening (extension lines go down to the stator)
    slot_layers = [(ytop, ybot, k) for (_, _, k), ytop, ybot in zip(layers, ys, ys[1:]) if k.startswith("slot")]
    y_feat = slot_layers[0][0] if layers[0][2].startswith("slot") else slot_layers[0][0]
    r1, r2, r3 = top_y + 5.0, top_y + 10.5, top_y + 16.0
    x_t0 = 1.5 * ts
    _hdim(ax, 0.5 * ts, 1.5 * ts, r1, "slot pitch \u03c4s = %.1f mm" % ts, ext_to=y_feat)
    _hdim(ax, x_t0 - wt / 2, x_t0 + wt / 2, r2, "tooth width = %.1f mm" % wt, ext_to=y_feat)
    xs0 = 2 * ts
    _hdim(ax, xs0 - min(b0, sw) / 2, xs0 + min(b0, sw) / 2, r3, "slot opening b0 = %.1f mm" % b0, ext_to=y_feat)
    # bottom rows: pole pitch, magnet width (extension lines go up to the magnets)
    mag_y = [ybot for (_, _, k), ytop, ybot in zip(layers, ys, ys[1:]) if k == "mag"]
    y_m = mag_y[-1] if layers[0][2] == "yoke" else mag_y[-1]
    b1, b2 = bot_y - 5.0, bot_y - 10.5
    c0 = ts / 2 + 0 * tp
    c1 = c0 + tp
    _hdim(ax, c0, c1, b1, "pole pitch \u03c4p = %.1f mm" % tp, ext_to=y_m, above=False)
    _hdim(ax, c1 - wm / 2, c1 + wm / 2, b2, "magnet width = %.1f mm (\u03b1p = %.2g)" % (wm, P.alpha),
          ext_to=y_m, above=False)
    ax.set_xlim(-3, W + 3 + 0.55 * W)
    ax.set_ylim(bot_y - 16, top_y + 29)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0, top_y + 28.5, "Developed cross-section at mean radius %.1f mm   ( \u00d7 into page, \u2022 out of page )" % rm,
            fontsize=7.2, va="top", ha="left", color="#333")


def draw_section(ax, P):
    """Radial-axial section (shaft on the left) with the radius dimensions."""
    mm = 1e3
    Ro, Ri, rsh = P.Ro * mm, P.Ri * mm, P.rsh * mm
    layers = list(reversed(_stack(P)))          # bottom to top
    H = sum(t for _, t, _ in layers)
    ax.add_patch(Rectangle((0, 0), rsh, H, fc=SHAFT, ec="#333", lw=0.8))
    y = 0.0
    for label, t, kind in layers:
        if kind == "back":
            ax.add_patch(Rectangle((rsh, y), Ro - rsh, t, fc=IRON_DARK, ec="#555", lw=0.6))
        elif kind == "mag":
            ax.add_patch(Rectangle((Ri, y), Ro - Ri, t, fc=MAG_N, ec="#222", lw=0.6))
        elif kind == "yoke":
            ax.add_patch(Rectangle((Ri, y), Ro - Ri, t, fc=IRON, ec="#555", lw=0.6))
        elif kind.startswith("slot"):
            ax.add_patch(Rectangle((Ri, y), Ro - Ri, t, fc=COPPER, ec="#555", lw=0.6, hatch="..", alpha=0.9))
        y += t
    rows = [H + 5, H + 11]
    _hdim(ax, 0, Ro, -6, "Ro = %.1f mm" % Ro, ext_to=0, above=False)
    _hdim(ax, 0, Ri, -12, "Ri = %.1f mm" % Ri, ext_to=0, above=False)
    _hdim(ax, 0, rsh, -18, "shaft radius = %.1f mm" % rsh, ext_to=0, above=False)
    ax.set_xlim(-2, Ro * 1.15); ax.set_ylim(-24, H + 8)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0, H + 7.5, "Radial section (radius on x)", fontsize=7.2, va="bottom", ha="left", color="#333")


def draw_motor(fig, P, message=None):
    fig.clear()
    if message:
        fig.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, color="#a33", wrap=True)
        return
    gs = fig.add_gridspec(3, 1, height_ratios=[0.95, 1.0, 0.7], hspace=0.04, left=0.01, right=0.99, top=0.995,
                          bottom=0.005)
    draw_top(fig.add_subplot(gs[0]), P)
    draw_developed(fig.add_subplot(gs[1]), P)
    draw_section(fig.add_subplot(gs[2]), P)
