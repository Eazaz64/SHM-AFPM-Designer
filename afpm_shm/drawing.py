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


def _stack(P):
    """Layers from bottom to top: (label, thickness mm, kind)."""
    mm = 1e3
    if P.topology == "SSSR":
        return [("Stator yoke", P.hsy * mm, "yoke"), ("Teeth + coil", P.hs * mm, "coil"),
                ("Air gap", P.g * mm, "gap"), ("Magnet", P.hm * mm, "mag"),
                ("Rotor back-iron", P.hby * mm, "back")]
    return [("Rotor back-iron", P.hby * mm, "back"), ("Magnet", P.hm * mm, "mag"),
            ("Air gap", P.g * mm, "gap"), ("Teeth + coil", P.hs * mm, "coil"),
            ("Support plate", P.hplate * mm, "yoke"), ("Teeth + coil", P.hs * mm, "coil"),
            ("Air gap", P.g * mm, "gap"), ("Magnet", P.hm * mm, "mag"),
            ("Rotor back-iron", P.hby * mm, "back")]


def draw_section(ax, P):
    mm = 1e3
    Ro, Ri, rsh = P.Ro * mm, P.Ri * mm, P.rsh * mm
    layers = _stack(P)
    H = sum(t for _, t, _ in layers)
    ax.set_xlim(-2, Ro * 1.62); ax.set_ylim(-H * 0.08, H * 1.08); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), rsh, H, fc=SHAFT, ec="#333", lw=0.8))
    y = 0.0
    for label, t, kind in layers:
        x0 = {"back": rsh, "mag": Ri, "gap": None, "coil": Ri, "yoke": Ri}[kind]
        if kind == "gap":
            pass
        elif kind == "mag":
            n = 0
            ax.add_patch(Rectangle((x0, y), Ro - x0, t, fc=MAG_N, ec="#222", lw=0.6))
        elif kind == "back":
            ax.add_patch(Rectangle((x0, y), Ro - x0, t, fc=IRON_DARK, ec="#555", lw=0.6))
        elif kind == "yoke":
            ax.add_patch(Rectangle((x0, y), Ro - x0, t, fc=IRON, ec="#555", lw=0.6))
        else:
            ax.add_patch(Rectangle((x0, y), Ro - x0, t, fc=COPPER, ec="#555", lw=0.6, hatch="..", alpha=0.9))
        if kind != "gap" or t > 0:
            xt = Ro + 2
            ax.text(xt, y + t / 2, "%s  %.2g mm" % (label, t), fontsize=7, va="center", ha="left", color="#222")
        y += t
    ax.text(rsh / 2, -H * 0.04, "shaft", fontsize=6.5, ha="center", va="top", color="#333")
    ax.annotate("", xy=(Ri, -H * 0.03), xytext=(Ro, -H * 0.03),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="#333"))
    ax.text((Ri + Ro) / 2, -H * 0.05, "active length %.1f mm" % (Ro - Ri), fontsize=7, ha="center", va="top")
    ax.text(0, H * 1.04, "Axial cross-section (radius on x, one side of the shaft)", fontsize=7.5, va="bottom",
            ha="left", color="#333")


def draw_motor(fig, P, message=None):
    fig.clear()
    if message:
        fig.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, color="#a33", wrap=True)
        return
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 0.85], hspace=0.06, left=0.02, right=0.98, top=0.99, bottom=0.01)
    draw_top(fig.add_subplot(gs[0]), P)
    draw_section(fig.add_subplot(gs[1]), P)
