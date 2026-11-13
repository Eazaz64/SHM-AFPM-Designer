"""Tkinter user interface of the AFPM-BLDC SHM Designer (offline, no network access)."""
import json
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import __version__
from . import model as M
from . import plots as PL
from . import drawing as DR

# (group, key, label, unit, factor [display = SI value / factor], integer?)
FIELDS = [
    ("Operating point", "N", "Rated speed", "rpm", 1, False),
    ("Operating point", "Vdc", "DC bus voltage", "V", 1, False),
    ("Operating point", "Vsw", "Switch voltage drop (2 switches)", "V", 1, False),
    ("Operating point", "Ilim", "Controller DC current limit", "A", 1, False),
    ("Operating point", "Tw", "Winding temperature", "degC", 1, False),
    ("Machine", "Ro", "Outer active radius Ro", "mm", 1e-3, False),
    ("Machine", "Ri", "Inner active radius Ri", "mm", 1e-3, False),
    ("Machine", "Q", "Number of slots Q", "", 1, True),
    ("Machine", "p", "Pole pairs p", "", 1, True),
    ("Machine", "g", "Mechanical air gap (per side)", "mm", 1e-3, False),
    ("Machine", "Ns", "Radial slices (quasi-3D)", "", 1, True),
    ("Machine", "rsh", "Shaft radius", "mm", 1e-3, False),
    ("Magnets", "Br", "Remanence Br", "T", 1, False),
    ("Magnets", "mur", "Recoil permeability", "", 1, False),
    ("Magnets", "hm", "Magnet thickness", "mm", 1e-3, False),
    ("Magnets", "alpha", "Pole-arc ratio", "", 1, False),
    ("Magnets", "nmax", "Highest space harmonic", "", 1, True),
    ("Stator", "b0", "Slot opening b0", "mm", 1e-3, False),
    ("Stator", "wti", "Tooth width at Ri", "mm", 1e-3, False),
    ("Stator", "wto", "Tooth width at Ro", "mm", 1e-3, False),
    ("Stator", "hs", "Slot (coil) height", "mm", 1e-3, False),
    ("Stator", "hsy", "Stator yoke thickness (SSSR)", "mm", 1e-3, False),
    ("Stator", "hplate", "Support plate thickness (SSDR)", "mm", 1e-3, False),
    ("Stator", "hby", "Rotor back-iron thickness", "mm", 1e-3, False),
    ("Winding", "Nc", "Turns per coil", "", 1, True),
    ("Winding", "nstrand", "Strands in hand", "", 1, True),
    ("Winding", "dwire", "Bare wire diameter", "mm", 1e-3, False),
    ("Winding", "fill", "Coil fill factor", "", 1, False),
    ("Winding", "wc", "Coil build (thickness)", "mm", 1e-3, False),
    ("Winding", "J", "Rated rms current density", "A/mm2", 1e6, False),
    ("Losses", "kh", "Hysteresis coefficient kh", "W/kg/Hz/T^b", 1, False),
    ("Losses", "beta", "Steinmetz exponent", "", 1, False),
    ("Losses", "ke", "Eddy coefficient ke", "W/kg/Hz2/T2", 1, False),
    ("Losses", "kharm", "Field-harmonic allowance", "", 1, False),
    ("Losses", "stray", "Stray loss (fraction of output)", "", 1, False),
    ("Losses", "Prot", "Rotational loss at rated speed", "W", 1, False),
    ("Losses", "mplate", "Support plate mass", "kg", 1, False),
    ("Materials", "rhoFe", "Density, laminated steel", "kg/m3", 1, False),
    ("Materials", "rhoBI", "Density, rotor back iron", "kg/m3", 1, False),
    ("Materials", "rhoCu", "Density, copper", "kg/m3", 1, False),
    ("Materials", "rhoPM", "Density, magnet", "kg/m3", 1, False),
    ("Materials", "rho20", "Copper resistivity at 20 degC", "1e-8 ohm m", 1e-8, False),
    ("Materials", "alphaCu", "Copper temperature coefficient", "1e-3 /K", 1e-3, False),
    ("Calibration (optional)", "kE", "Back-EMF (Ke, Kt) factor", "", 1, False),
    ("Calibration (optional)", "kL", "Inductance factor", "", 1, False),
]

HELP_TEXT = (
    "1. Choose the topology (SSSR or SSDR) and edit the inputs on the left.\n"
    "2. Press Calculate (F5). Results appear in the Results tab and all graphs are updated.\n"
    "3. In the Graphs tab, select a graph in the list. 'Save graph' writes PNG, SVG or PDF; "
    "'Save data (CSV)' writes the numbers behind the graph.\n"
    "4. File > Export everything writes results.csv, inputs.csv and every graph (PNG, 300 dpi) "
    "with its CSV into a folder you choose.\n"
    "5. Overlay > Load measured data adds your measurements to the Speed-torque and Back-EMF vs "
    "speed graphs. The file needs two numeric columns: x then y (a header row is allowed).\n\n"
    "Notes\n"
    "- Ke and Kt are on the 60-degree average line-to-line basis used in a six-step drive; the "
    "peak-basis Ke is shown separately in the Results tab and in the Ke graph.\n"
    "- The speed-torque curve is an average model: it ignores inductance and commutation delay.\n"
    "- The calibration factors multiply the predicted Ke/Kt and the inductance. Leave them at 1.0 "
    "for a pure SHM result.\n"
    "- The program works fully offline."
)

ABOUT_TEXT = ("AFPM-BLDC SHM Designer %s\n\nSpace Harmonic Method design tool for axial flux permanent "
              "magnet BLDC motors (SSSR and SSDR) with six-step drive.\n\nField model after Taghipour "
              "Boroujeni et al., IET Electric Power Applications, 2016.\n\nAll calculations run on this "
              "computer; no data is sent anywhere." % __version__)


class App:
    def __init__(self, root):
        self.root = root
        root.title("AFPM-BLDC SHM Designer %s" % __version__)
        root.geometry("1280x800")
        root.minsize(1000, 640)
        self.vars = {}
        self.topology = tk.StringVar(value="SSSR")
        self.status = tk.StringVar(value="Ready. Press Calculate (F5).")
        self.res = None
        self.specs = []
        self.overlays = {}
        self.fig = Figure(figsize=(6.5, 4.5), dpi=100)
        self.dfig = Figure(figsize=(4.2, 7.0), dpi=100)
        self._draw_job = None
        self._build_menu()
        self._build_toolbar()
        self._build_body()
        ttk.Label(root, textvariable=self.status, relief="sunken", anchor="w").pack(side="bottom", fill="x")
        root.bind("<F5>", lambda e: self.calculate())
        self.load_into_form(M.default_params("SSSR"))
        self.redraw_motor()
        self.calculate()

    # ------------------------------------------------------------------ layout
    def _build_menu(self):
        mb = tk.Menu(self.root)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Open design...", command=self.open_design)
        fm.add_command(label="Save design...", command=self.save_design)
        fm.add_separator()
        fm.add_command(label="Export everything...", command=self.export_everything)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.root.destroy)
        mb.add_cascade(label="File", menu=fm)
        om = tk.Menu(mb, tearoff=0)
        om.add_command(label="Load measured speed-torque data...", command=lambda: self.load_overlay("speed_torque"))
        om.add_command(label="Load measured back-EMF vs speed data...", command=lambda: self.load_overlay("ke"))
        om.add_command(label="Clear overlays", command=self.clear_overlays)
        mb.add_cascade(label="Overlay", menu=om)
        hm = tk.Menu(mb, tearoff=0)
        hm.add_command(label="How to use", command=lambda: messagebox.showinfo("How to use", HELP_TEXT))
        hm.add_command(label="About", command=lambda: messagebox.showinfo("About", ABOUT_TEXT))
        mb.add_cascade(label="Help", menu=hm)
        self.root.config(menu=mb)

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=(6, 4))
        bar.pack(side="top", fill="x")
        ttk.Label(bar, text="Topology:").pack(side="left")
        cb = ttk.Combobox(bar, textvariable=self.topology, values=["SSSR", "SSDR"], state="readonly", width=7)
        cb.pack(side="left", padx=(4, 10))
        cb.bind("<<ComboboxSelected>>", lambda e: self.on_topology())
        ttk.Button(bar, text="Defaults for this topology", command=self.load_defaults).pack(side="left", padx=2)
        ttk.Button(bar, text="Calculate (F5)", command=self.calculate).pack(side="left", padx=10)
        ttk.Button(bar, text="Export everything...", command=self.export_everything).pack(side="left", padx=2)

    def _build_body(self):
        pw = ttk.PanedWindow(self.root, orient="horizontal")
        pw.pack(side="top", fill="both", expand=True)
        left = ttk.Frame(pw)
        right = ttk.Frame(pw)
        drawf = ttk.LabelFrame(pw, text="Motor drawing (updates while you type)", padding=2)
        pw.add(left, weight=0)
        pw.add(right, weight=1)
        pw.add(drawf, weight=0)
        self.dcanvas = FigureCanvasTkAgg(self.dfig, master=drawf)
        self.dcanvas.get_tk_widget().configure(width=400)
        ttk.Button(drawf, text="Save drawing...", command=self.save_drawing).pack(side="bottom", pady=2)
        self.dcanvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        # scrollable input form
        canvas = tk.Canvas(left, width=340, highlightthickness=0)
        sb = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", lambda ev: canvas.yview_scroll(int(-ev.delta / 120), "units")))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        group, frame, row = None, None, 0
        for g, key, label, unit, factor, is_int in FIELDS:
            if g != group:
                group = g
                frame = ttk.LabelFrame(inner, text=g, padding=6)
                frame.pack(fill="x", padx=6, pady=4)
                row = 0
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=1)
            v = tk.StringVar()
            self.vars[key] = v
            try:
                v.trace_add("write", lambda *a: self.schedule_drawing())
            except Exception:
                pass
            e = ttk.Entry(frame, textvariable=v, width=11, justify="right")
            e.grid(row=row, column=1, padx=4)
            e.bind("<Return>", lambda ev: self.calculate())
            ttk.Label(frame, text=unit, width=11).grid(row=row, column=2, sticky="w")
            row += 1
        # right side: notebook
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)
        rt = ttk.Frame(nb)
        gt = ttk.Frame(nb)
        nb.add(rt, text="Results")
        nb.add(gt, text="Graphs")
        self.tree = ttk.Treeview(rt, columns=("q", "v", "u"), show="headings", height=30)
        self.tree.heading("q", text="Quantity")
        self.tree.heading("v", text="Value")
        self.tree.heading("u", text="Unit")
        self.tree.column("q", width=360)
        self.tree.column("v", width=170, anchor="e")
        self.tree.column("u", width=90)
        self.tree.tag_configure("sec", background="#dde6f2")
        tsb = ttk.Scrollbar(rt, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tsb.pack(side="right", fill="y")
        # graphs: list on the left, figure on the right
        self.listbox = tk.Listbox(gt, width=26, exportselection=False)
        for key, title, fn in PL.PLOTS:
            self.listbox.insert("end", title)
        self.listbox.pack(side="left", fill="y", padx=(6, 2), pady=6)
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.show_plot())
        gr = ttk.Frame(gt)
        gr.pack(side="left", fill="both", expand=True)
        btns = ttk.Frame(gr)
        btns.pack(side="top", fill="x", pady=4)
        ttk.Button(btns, text="Save graph...", command=self.save_graph).pack(side="left", padx=4)
        ttk.Button(btns, text="Save data (CSV)...", command=self.save_data).pack(side="left", padx=4)
        self.canvas = FigureCanvasTkAgg(self.fig, master=gr)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        tb = NavigationToolbar2Tk(self.canvas, gr)
        tb.update()
        self.listbox.selection_set(0)

    # ------------------------------------------------------------------ form <-> params
    def load_into_form(self, P):
        self.topology.set(P.topology)
        d = M.params_to_dict(P)
        for g, key, label, unit, factor, is_int in FIELDS:
            self.vars[key].set("%.6g" % (d[key] / factor))

    def read_form(self):
        d = M.params_to_dict(M.default_params(self.topology.get()))
        d["topology"] = self.topology.get()
        for g, key, label, unit, factor, is_int in FIELDS:
            txt = self.vars[key].get().strip().replace(",", ".")
            try:
                val = float(txt)
            except ValueError:
                raise ValueError("'%s' is not a number (field: %s)." % (txt, label))
            if is_int:
                if val != int(val):
                    raise ValueError("%s must be a whole number." % label)
                d[key] = int(val)
            else:
                d[key] = val * factor
        return M.params_from_dict(d)

    def schedule_drawing(self):
        """Redraw the motor shortly after the last key press (no full calculation needed)."""
        if self._draw_job is not None:
            try:
                self.root.after_cancel(self._draw_job)
            except Exception:
                pass
        try:
            self._draw_job = self.root.after(250, self.redraw_motor)
        except Exception:
            self.redraw_motor()

    def redraw_motor(self):
        self._draw_job = None
        try:
            P = self.read_form()
            err, warn = M.validate(P)
            if err:
                DR.draw_motor(self.dfig, P, message="Cannot draw yet:\n" + "\n".join(err))
            else:
                DR.draw_motor(self.dfig, P)
        except ValueError as exc:
            DR.draw_motor(self.dfig, None, message="Cannot draw yet:\n%s" % exc)
        except Exception as exc:
            DR.draw_motor(self.dfig, None, message="Drawing failed: %s" % exc)
        self.dcanvas.draw_idle()

    def save_drawing(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile="motor_drawing",
            filetypes=[("PNG image", "*.png"), ("SVG vector", "*.svg"), ("PDF", "*.pdf")])
        if path:
            self.dfig.savefig(path, dpi=300)
            self.status.set("Saved " + path)

    def on_topology(self):
        base = M.default_params(self.topology.get())
        for key in ("Ilim", "Prot", "mplate"):
            self.vars[key].set("%.6g" % getattr(base, key))
        self.calculate()

    def load_defaults(self):
        self.load_into_form(M.default_params(self.topology.get()))
        self.calculate()

    # ------------------------------------------------------------------ actions
    def calculate(self):
        try:
            P = self.read_form()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return
        err, warn = M.validate(P)
        if err:
            messagebox.showerror("Input error", "\n".join(err))
            return
        t0 = time.time()
        try:
            self.res = M.solve(P)
        except Exception as exc:  # show the problem instead of closing the program
            messagebox.showerror("Calculation failed", "%s: %s" % (type(exc).__name__, exc))
            return
        self.fill_results()
        self.refresh_plots()
        msg = "Calculated in %.1f s. %s: Ke = Kt = %.4f, rated %.2f N m, %.0f W." % (
            time.time() - t0, P.topology, self.res.E.Ke, self.res.R.T, self.res.R.P)
        if warn:
            msg += "  WARNING: " + " ".join(warn)
            messagebox.showwarning("Check the inputs", "\n".join(warn))
        self.status.set(msg)

    def fill_results(self):
        self.tree.delete(*self.tree.get_children())
        sec = None
        for s, q, v, u in M.results_table(self.res):
            if s != sec:
                sec = s
                self.tree.insert("", "end", values=(s, "", ""), tags=("sec",))
            self.tree.insert("", "end", values=("    " + q, v, u))

    def refresh_plots(self):
        self.specs = PL.build_all(self.res, self.overlays)
        self.show_plot()

    def current_spec(self):
        sel = self.listbox.curselection()
        idx = sel[0] if sel else 0
        return self.specs[idx] if self.specs else None

    def show_plot(self):
        spec = self.current_spec()
        if spec is None:
            return
        PL.draw(self.fig, spec)
        self.canvas.draw_idle()

    def save_graph(self):
        spec = self.current_spec()
        if spec is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=spec.key,
            filetypes=[("PNG image", "*.png"), ("SVG vector", "*.svg"), ("PDF", "*.pdf")])
        if not path:
            return
        fig = Figure(figsize=(7.0, 4.6) if spec.kind != "star" else (5.2, 5.2))
        PL.draw(fig, spec)
        try:
            fig.savefig(path, dpi=300, facecolor="white")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.status.set("Graph saved: %s" % path)

    def save_data(self):
        spec = self.current_spec()
        if spec is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=spec.key,
                                            filetypes=[("CSV file", "*.csv")])
        if not path:
            return
        try:
            PL.write_csv(path, PL.spec_to_rows(spec))
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.status.set("Graph data saved: %s" % path)

    def export_everything(self):
        if self.res is None:
            self.calculate()
            if self.res is None:
                return
        folder = filedialog.askdirectory(title="Choose the folder for the exported files")
        if not folder:
            return
        out = os.path.join(folder, "AFPM_%s_results" % self.res.P.topology)
        try:
            files = PL.export_all(self.res, out, self.overlays, formats=("png",), dpi=300)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.status.set("Exported %d files to %s" % (len(files), out))
        messagebox.showinfo("Export finished", "%d files written to:\n%s" % (len(files), out))

    def save_design(self):
        try:
            P = self.read_form()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile="design",
                                            filetypes=[("Design file", "*.json")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(M.params_to_dict(P), fh, indent=2)
        self.status.set("Design saved: %s" % path)

    def open_design(self):
        path = filedialog.askopenfilename(filetypes=[("Design file", "*.json")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                P = M.params_from_dict(json.load(fh))
        except Exception as exc:
            messagebox.showerror("Open failed", "Not a valid design file: %s" % exc)
            return
        self.load_into_form(P)
        self.calculate()

    def load_overlay(self, key):
        path = filedialog.askopenfilename(filetypes=[("CSV / text", "*.csv *.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.overlays[key] = PL.read_xy_csv(path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        if self.res is not None:
            self.refresh_plots()
        self.status.set("Measured data loaded from %s" % path)

    def clear_overlays(self):
        self.overlays.clear()
        if self.res is not None:
            self.refresh_plots()


def run():
    try:                                    # sharp text on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()
