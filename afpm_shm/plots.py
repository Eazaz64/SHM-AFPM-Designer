"""Plot definitions. Every plot is a PlotSpec (data + labels), so the graph that is drawn
and the CSV that is exported always contain exactly the same numbers."""
import csv
import math
import os
from dataclasses import dataclass, field

import numpy as np

from . import model as M

COL = {"blue": "#1f5fa8", "orange": "#d4691f", "green": "#2e8b57", "red": "#c0392b",
       "grey": "#555555", "purple": "#7b4fa3"}


@dataclass
class Series:
    label: str
    x: np.ndarray
    y: np.ndarray
    kind: str = "line"              # line | points | bar
    style: str = "-"
    color: str = None
    marker: str = None
    axis: int = 1                   # 1 = left y axis, 2 = right y axis
    offset: float = 0.0             # for grouped bars (in x units)
    width: float = 0.8


@dataclass
class PlotSpec:
    key: str
    title: str
    xlabel: str = ""
    ylabel: str = ""
    series: list = field(default_factory=list)
    ylabel2: str = None
    kind: str = "xy"                # xy | contour | star
    xlim: tuple = None
    ylim: tuple = None
    ylim2: tuple = None
    xticks: list = None
    hlines: list = field(default_factory=list)      # (y, label)
    vlines: list = field(default_factory=list)      # (x, label)
    vspan: tuple = None                             # (x0, x1, label)
    note: str = ""
    extra: dict = field(default_factory=dict)


def _arr(v):
    return np.asarray(v, dtype=float)


# ------------------------------------------------------------------ plot builders
def plot_backemf(res):
    P, E = res.P, res.E
    th = np.rad2deg(E.th)
    pk = th[E.ipk]
    yl = 1.15 * np.max(np.abs(E.eab))
    return PlotSpec(
        "backemf", "Back-EMF at %.0f rpm" % P.N, "Electrical angle (deg)", "Back-EMF (V)",
        [Series("Phase e_a", th, E.ea, color=COL["blue"]),
         Series("Line e_ab", th, E.eab, color="black")],
        xlim=(0, 360), ylim=(-yl, yl), xticks=[0, 90, 180, 270, 360],
        vspan=(pk - 30, pk + 30, "60 deg conduction"),
        note="Phase fundamental peak %.2f V\nLine peak %.2f V, 60 deg average %.2f V\nKe = %.4f V s/rad"
             % (E.E1, E.Epk, E.Eavg, E.Ke))


def plot_airgap_B(res):
    P = res.P
    th = np.linspace(0, 2 * math.pi, 1441)
    radii = [P.Ri + 0.5e-3, P.rm, P.Ro - 0.5e-3]
    sty = [":", "-", "--"]
    s = [Series("r = %.1f mm" % (r * 1e3), np.rad2deg(th), M.airgap_B(P, r, th), style=sty[i], color=COL["blue"])
         for i, r in enumerate(radii)]
    return PlotSpec("airgap_B", "Air-gap flux density at the stator surface",
                    "Electrical angle (deg)", "B_z (T)", s, xlim=(0, 360), xticks=[0, 90, 180, 270, 360])


def plot_spectrum_B(res):
    P = res.P
    nn = np.arange(1, 16, 2)
    Bs = np.array([abs(float(M.shm_Bn(n, P.rm, P))) for n in nn])
    return PlotSpec("spectrum_B", "Harmonic spectrum of the air-gap flux density (mean radius)",
                    "Harmonic order n", "|B_n| (T)",
                    [Series("|B_n|", nn, Bs, kind="bar", color=COL["blue"], width=1.2)], xticks=list(nn))


def plot_spectrum_emf(res):
    E = res.E
    nn = E.nn[:13]
    ph = 100 * E.Eph[:13] / E.Eph[0]
    ln = 100 * E.Eln[:13] / E.Eln[0]
    return PlotSpec("spectrum_emf", "Harmonic spectrum of the back-EMF", "Harmonic order n",
                    "Amplitude (% of fundamental)",
                    [Series("Phase EMF (THD %.1f %%)" % (100 * E.THDph), nn, ph, kind="bar", color=COL["blue"], offset=-0.55, width=1.0),
                     Series("Line EMF (THD %.2f %%)" % (100 * E.THDline), nn, ln, kind="bar", color=COL["orange"], offset=0.55, width=1.0)],
                    xticks=list(nn))


def plot_winding_factor(res):
    P, L = res.P, res.L
    nn = np.arange(1, 26, 2)
    kw = np.array([M.winding_factor(n, P, L)[0] for n in nn])
    return PlotSpec("winding_factor", "Winding factor of each space harmonic", "Harmonic order n",
                    "k_w", [Series("k_w", nn, kw, kind="bar", color=COL["green"], width=1.2)],
                    xticks=list(nn), ylim=(0, 1.05))


def plot_star(res):
    P, L = res.P, res.L
    coils = []
    for k in range(P.Q):
        ang = L.angle[k] + (180.0 if L.sign[k] < 0 else 0.0)
        coils.append((k + 1, L.phase[k], int(L.sign[k]), ang % 360))
    return PlotSpec("star_of_slots", "Star of slots (fundamental), %d slots / %d poles" % (P.Q, 2 * P.p),
                    kind="star", extra={"coils": coils})


def plot_torque_current(res):
    P, R = res.P, res.R
    I = np.linspace(0, 1.15 * max(P.Ilim, R.Idc), 100)
    T = R.Kt * I
    return PlotSpec("torque_current", "Torque versus DC-link current (Kt)", "DC-link current (A)", "Torque (N m)",
                    [Series("T = Kt x I", I, T, color=COL["blue"]),
                     Series("Rated point", _arr([R.Idc]), _arr([R.T]), kind="points", marker="o", color=COL["red"])],
                    vlines=[(P.Ilim, "Current limit %.0f A" % P.Ilim)], xlim=(0, I[-1]), ylim=(0, None),
                    note="Kt = %.4f N m/A\nRated: %.1f A, %.2f N m" % (R.Kt, R.Idc, R.T))


def plot_ke(res, overlay=None):
    P, E = res.P, res.E
    n = np.linspace(0, 1.5 * P.N, 61)
    w = 2 * math.pi * n / 60
    s = [Series("Line EMF, 60 deg average (Ke = %.4f)" % E.Ke, n, E.Ke * w, color=COL["blue"]),
         Series("Line EMF, peak", n, E.Epk / P.wm * w, color=COL["orange"], style="--"),
         Series("Phase EMF, fundamental peak", n, E.E1 / P.wm * w, color=COL["green"], style=":")]
    if overlay is not None:
        s.append(Series("Measured (as entered)", overlay[0], overlay[1], kind="points", marker="o", color="black"))
    return PlotSpec("ke", "Back-EMF versus speed (Ke)", "Speed (rpm)", "Back-EMF (V)", s, xlim=(0, n[-1]), ylim=(0, None),
                    note="Ke = %.4f V s/rad (60 deg average basis)\nPeak basis: %.4f V s/rad"
                         % (E.Ke, E.Epk / P.wm))


def plot_speed_torque(res, overlay=None):
    P = res.P
    st = M.speed_torque(res)
    # constant-torque part (current limited) then the voltage-limited line
    x = np.concatenate([[0.0], st.n[::-1]])
    y = np.concatenate([[st.Tlim], st.T[::-1]])
    s = [Series("SHM prediction, %.0f V, %.0f A limit" % (P.Vdc, P.Ilim), x, y, color=COL["blue"]),
         Series("Rated point", _arr([P.N]), _arr([res.R.T]), kind="points", marker="o", color=COL["red"])]
    if overlay is not None:
        s.append(Series("Measured (as entered)", overlay[0], overlay[1], kind="points", marker="o", color="black"))
    return PlotSpec("speed_torque", "Speed-torque characteristic at %.0f V (SHM prediction)" % P.Vdc,
                    "Speed (rpm)", "Torque (N m)", s, xlim=(0, None), ylim=(0, None),
                    note="Ideal no-load speed %.0f rpm\nCurrent-limited torque %.2f N m up to %.0f rpm"
                         % (st.n0, st.Tlim, st.ncorner))


def plot_power_eff(res):
    P = res.P
    st = M.speed_torque(res)
    return PlotSpec("power_efficiency", "Output power and efficiency versus torque at %.0f V" % P.Vdc,
                    "Torque (N m)", "Output power (W)",
                    [Series("Output power", st.T, st.Po, color=COL["blue"]),
                     Series("Efficiency", st.T, 100 * st.eta, color=COL["orange"], style="--", axis=2)],
                    ylabel2="Efficiency (%)", xlim=(0, None), ylim=(0, None), ylim2=(50, 100))


def plot_eff_map(res):
    NN, TT, Z = M.efficiency_map(res)
    R, P = res.R, res.P
    return PlotSpec("efficiency_map", "Efficiency map at %.0f V (%%)" % P.Vdc, "Speed (rpm)", "Torque (N m)",
                    kind="contour", extra={"X": NN, "Y": TT, "Z": Z, "point": (P.N, R.T, "Rated point"),
                                           "levels": [50, 60, 70, 80, 85, 88, 90, 91, 92, 93, 94, 95]})


def plot_torque_ripple(res):
    E = res.E
    return PlotSpec("torque_ripple", "EMF-shape torque ripple, ideal 120 deg currents", "Electrical angle (deg)",
                    "T / T_avg", [Series("Ripple %.1f %%" % (100 * E.ripple), np.rad2deg(E.th), E.env / E.env.mean(), color=COL["blue"])],
                    xlim=(0, 360), xticks=[0, 60, 120, 180, 240, 300, 360])


def plot_tooth_B(res):
    P, F = res.P, res.F
    return PlotSpec("tooth_B", "Peak flux density in the tooth along the radius", "Radius (mm)", "B tooth (T)",
                    [Series("Tooth B", P.r * 1e3, F.Btooth, color=COL["blue"], marker="o")],
                    hlines=[(1.5, "1.5 T (typical onset of saturation)")], ylim=(0, 2.0),
                    note="Max %.2f T\nRotor back iron %.2f T%s" % (
                        F.BtoothMax, F.Bbackiron, "\nStator yoke %.2f T" % F.Byoke if P.yoke else ""))


def plot_losses(res):
    R = res.R
    names = ["Copper", "Core", "Rotational", "Stray"]
    vals = np.array([R.Pcu, R.Pfe, R.Prot, R.Pstr])
    return PlotSpec("losses", "Losses at the rated point (efficiency %.1f %%)" % (100 * R.eta), "Loss component", "Loss (W)",
                    [Series("Loss (W)", np.arange(1, 5), vals, kind="bar", color=COL["orange"], width=0.6)],
                    xticks=list(range(1, 5)), extra={"xticklabels": names})


PLOTS = [
    ("backemf", "Back-EMF waveform", plot_backemf),
    ("airgap_B", "Air-gap flux density", plot_airgap_B),
    ("spectrum_B", "Spectrum: air-gap B", plot_spectrum_B),
    ("spectrum_emf", "Spectrum: back-EMF", plot_spectrum_emf),
    ("winding_factor", "Winding factors", plot_winding_factor),
    ("star_of_slots", "Star of slots", plot_star),
    ("torque_current", "Torque vs current (Kt)", plot_torque_current),
    ("ke", "Back-EMF vs speed (Ke)", plot_ke),
    ("speed_torque", "Speed-torque", plot_speed_torque),
    ("power_efficiency", "Power and efficiency", plot_power_eff),
    ("efficiency_map", "Efficiency map", plot_eff_map),
    ("torque_ripple", "Torque ripple", plot_torque_ripple),
    ("tooth_B", "Tooth flux density", plot_tooth_B),
    ("losses", "Losses", plot_losses),
]
OVERLAY_KEYS = ("ke", "speed_torque")


def build_all(res, overlays=None):
    overlays = overlays or {}
    out = []
    for key, title, fn in PLOTS:
        out.append(fn(res, overlays.get(key)) if key in OVERLAY_KEYS else fn(res))
    return out


# ------------------------------------------------------------------ drawing
def draw(fig, spec, dpi_scale=1.0):
    """Draw spec onto the matplotlib Figure fig (created with the object API, no pyplot)."""
    fig.clear()
    ax = fig.add_subplot(111)
    if spec.kind == "star":
        _draw_star(ax, spec); fig.tight_layout(); return fig
    if spec.kind == "contour":
        e = spec.extra
        cf = ax.contourf(e["X"], e["Y"], e["Z"], levels=e["levels"], cmap="viridis", extend="both")
        cs = ax.contour(e["X"], e["Y"], e["Z"], levels=[85, 90, 92], colors="k", linewidths=0.8)
        ax.clabel(cs, fmt="%d", fontsize=8)
        fig.colorbar(cf, ax=ax, label="Efficiency (%)")
        px, py, pl = e["point"]
        ax.plot([px], [py], "o", color="red", label=pl)
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax2 = ax.twinx() if any(s.axis == 2 for s in spec.series) else None
        handles = []
        for s in spec.series:
            a = ax2 if s.axis == 2 else ax
            if s.kind == "bar":
                h = a.bar(s.x + s.offset * 0.5 * (1 if s.width >= 1 else 0.5), s.y, width=s.width * 0.5 if s.width >= 1 else s.width,
                          color=s.color, label=s.label)
            elif s.kind == "points":
                h, = a.plot(s.x, s.y, linestyle="none", marker=s.marker or "o", color=s.color, label=s.label, markersize=6)
            else:
                h, = a.plot(s.x, s.y, linestyle=s.style, color=s.color, marker=s.marker, markersize=4, linewidth=1.6, label=s.label)
            handles.append(h)
        for y, lab in spec.hlines:
            ax.axhline(y, color="grey", linestyle="--", linewidth=1, label=lab)
        for x, lab in spec.vlines:
            ax.axvline(x, color="grey", linestyle="--", linewidth=1, label=lab)
        if spec.vspan:
            ax.axvspan(spec.vspan[0], spec.vspan[1], color="0.85", alpha=0.7, label=spec.vspan[2])
        if spec.ylabel2 and ax2 is not None:
            ax2.set_ylabel(spec.ylabel2)
            if spec.ylim2: ax2.set_ylim(*spec.ylim2)
        h1, l1 = ax.get_legend_handles_labels()
        if ax2 is not None:
            h2, l2 = ax2.get_legend_handles_labels(); h1, l1 = h1 + h2, l1 + l2
        if h1:
            ax.legend(h1, l1, loc="best", fontsize=8)
        if spec.xlim: ax.set_xlim(*spec.xlim)
        if spec.ylim: ax.set_ylim(*spec.ylim)
        if spec.xticks is not None: ax.set_xticks(spec.xticks)
        if "xticklabels" in spec.extra: ax.set_xticklabels(spec.extra["xticklabels"])
        ax.grid(True, alpha=0.35)
        if spec.note:
            _place_note(fig, ax, spec)
    ax.set_xlabel(spec.xlabel); ax.set_ylabel(spec.ylabel); ax.set_title(spec.title, fontsize=11)
    fig.tight_layout()
    return fig


def _place_note(fig, ax, spec):
    """Put the note box in the corner of the axes that hides the fewest data points."""
    nlines = spec.note.count("\n") + 1
    bw, bh = 0.46, 0.07 * nlines + 0.05
    corners = {"lower right": (1 - bw, 0.0), "upper left": (0.0, 1 - bh), "lower left": (0.0, 0.0),
               "upper right": (1 - bw, 1 - bh)}
    pts = []
    for s in spec.series:
        if s.axis == 2:
            continue
        xy = np.column_stack([s.x, s.y]).astype(float)
        pts.append(xy[np.all(np.isfinite(xy), axis=1)])
    pts = np.vstack(pts) if pts else np.zeros((0, 2))
    best, best_n = "lower right", None
    if len(pts):
        fr = ax.transAxes.inverted().transform(ax.transData.transform(pts))
        for name, (x0, y0) in corners.items():
            n = int(np.sum((fr[:, 0] >= x0 - 0.02) & (fr[:, 0] <= x0 + bw + 0.02) &
                           (fr[:, 1] >= y0 - 0.02) & (fr[:, 1] <= y0 + bh + 0.02)))
            if best_n is None or n < best_n:
                best, best_n = name, n
    x0, y0 = corners[best]
    ha = "right" if "right" in best else "left"
    va = "top" if "upper" in best else "bottom"
    ax.text(0.98 if ha == "right" else 0.02, 0.97 if va == "top" else 0.03, spec.note,
            transform=ax.transAxes, ha=ha, va=va, fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.92))


def _draw_star(ax, spec):
    colors = {"A": "#c0392b", "B": "#2e8b57", "C": "#1f5fa8"}
    seen = {}
    for k, ph, sg, ang in spec.extra["coils"]:
        key = round(ang, 3)
        j = seen.get(key, 0); seen[key] = j + 1          # coils pointing the same way: stagger
        L = 1.0 - 0.18 * j
        a = math.radians(ang)
        ax.annotate("", xy=(L * math.cos(a), L * math.sin(a)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=colors[ph], lw=1.6))
        r = 1.2 + 0.2 * j
        ax.text(r * math.cos(a), r * math.sin(a), "%d(%s%s)" % (k, "-" if sg < 0 else "", ph),
                ha="center", va="center", fontsize=8, color=colors[ph])
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(spec.title, fontsize=11)


# ------------------------------------------------------------------ export
def spec_to_rows(spec):
    """Rows (list of lists, first row = header) with the numbers behind the graph."""
    if spec.kind == "contour":
        e = spec.extra
        rows = [["speed_rpm", "torque_Nm", "efficiency_percent"]]
        for xr, yr, zr in zip(e["X"], e["Y"], e["Z"]):
            for a, b, c in zip(xr, yr, zr):
                rows.append(["%.6g" % a, "%.6g" % b, "" if np.isnan(c) else "%.6g" % c])
        return rows
    if spec.kind == "star":
        rows = [["coil", "phase", "sign", "arrow_angle_deg"]]
        rows += [[k, ph, sg, "%.3f" % ang] for k, ph, sg, ang in spec.extra["coils"]]
        return rows
    ser = spec.series
    labels = spec.extra.get("xticklabels")
    same_x = all(len(s.x) == len(ser[0].x) and np.allclose(s.x, ser[0].x) for s in ser)
    if same_x:
        head = [spec.xlabel or "x"] + [s.label for s in ser]
        if labels: head = ["component_index", "component"] + head[1:]
        rows = [head]
        for i in range(len(ser[0].x)):
            r = ["%.8g" % ser[0].x[i]]
            if labels: r.append(labels[i])
            r += ["%.8g" % s.y[i] for s in ser]
            rows.append(r)
        return rows
    head = []
    for s in ser:
        yl = spec.ylabel2 if (s.axis == 2 and spec.ylabel2) else spec.ylabel
        head += ["%s [%s]" % (spec.xlabel or "x", s.label), "%s [%s]" % (yl or "y", s.label)]
    n = max(len(s.x) for s in ser)
    rows = [head]
    for i in range(n):
        r = []
        for s in ser:
            if i < len(s.x): r += ["%.8g" % s.x[i], "%.8g" % s.y[i]]
            else: r += ["", ""]
        rows.append(r)
    return rows


def write_csv(path, rows):
    # utf-8 with BOM so Excel opens it correctly; semicolons are not used
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows(rows)


def export_spec(spec, folder, formats=("png",), dpi=300):
    """Save graph (png/svg/pdf) and data (csv) of one plot into folder. Returns written paths."""
    from matplotlib.figure import Figure
    os.makedirs(folder, exist_ok=True)
    paths = []
    fig = Figure(figsize=(7.0, 4.6) if spec.kind != "star" else (5.2, 5.2))
    draw(fig, spec)
    for ext in formats:
        p = os.path.join(folder, "%s.%s" % (spec.key, ext))
        fig.savefig(p, dpi=dpi, facecolor="white")
        paths.append(p)
    p = os.path.join(folder, "%s.csv" % spec.key)
    write_csv(p, spec_to_rows(spec))
    paths.append(p)
    return paths


def export_all(res, folder, overlays=None, formats=("png",), dpi=300):
    """Results table, inputs and every graph with its data. Returns list of paths."""
    import json
    os.makedirs(folder, exist_ok=True)
    out = []
    rows = [["Section", "Quantity", "Value", "Unit"]] + [list(r) for r in M.results_table(res)]
    p = os.path.join(folder, "results.csv"); write_csv(p, rows); out.append(p)
    p = os.path.join(folder, "inputs.csv")
    write_csv(p, [["Input", "Value"]] + [[k, v] for k, v in M.params_to_dict(res.P).items()]); out.append(p)
    p = os.path.join(folder, "design.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(M.params_to_dict(res.P), fh, indent=2)
    out.append(p)
    for spec in build_all(res, overlays):
        out += export_spec(spec, folder, formats, dpi)
    return out


def read_xy_csv(path):
    """Read two numeric columns (x, y) from a CSV/TXT file; a header row is skipped."""
    xs, ys = [], []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        text = fh.read()
    delim = ";" if text.count(";") > text.count(",") else ("\t" if text.count("\t") > text.count(",") else ",")
    for row in csv.reader(text.splitlines(), delimiter=delim):
        try:
            xs.append(float(row[0])); ys.append(float(row[1]))
        except (ValueError, IndexError):
            continue
    if len(xs) < 2:
        raise ValueError("The file needs at least two rows with two numeric columns (x, y).")
    return np.array(xs), np.array(ys)
