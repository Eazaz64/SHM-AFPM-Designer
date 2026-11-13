"""Exercise every handler of the GUI with stand-in Tk widgets (no display needed).
This tests the program logic; the real window is checked by `--selftest` in the Windows build."""
import importlib
import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeVar:
    def __init__(self, value=""): self._v = value
    def get(self): return self._v
    def set(self, v): self._v = v
    def trace_add(self, *a, **k): pass


class W:
    """Stand-in widget: accepts any constructor arguments; every method is a mock."""
    def __init__(self, *a, **k): pass
    def __getattr__(self, name): return MagicMock()


class FakeListbox(W):
    _sel = 0
    def curselection(self): return (self._sel,)
    def selection_set(self, i): self._sel = i


@pytest.fixture()
def gui(monkeypatch, tmp_path):
    tk = types.ModuleType("tkinter")
    tk.Tk = W; tk.Canvas = W; tk.Menu = W
    tk.StringVar = FakeVar; tk.Listbox = FakeListbox
    ttk = types.ModuleType("tkinter.ttk")
    for n in ("Frame", "Label", "Entry", "Button", "Combobox", "LabelFrame", "Notebook", "Treeview",
              "Scrollbar", "PanedWindow"):
        setattr(ttk, n, W)
    mb = types.ModuleType("tkinter.messagebox")
    mb.showinfo = MagicMock(); mb.showerror = MagicMock(); mb.showwarning = MagicMock()
    fd = types.ModuleType("tkinter.filedialog")
    tk.ttk, tk.messagebox, tk.filedialog = ttk, mb, fd
    back = types.ModuleType("matplotlib.backends.backend_tkagg")
    back.FigureCanvasTkAgg = W; back.NavigationToolbar2Tk = W
    for name, mod in (("tkinter", tk), ("tkinter.ttk", ttk), ("tkinter.messagebox", mb),
                      ("tkinter.filedialog", fd), ("matplotlib.backends.backend_tkagg", back)):
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules.pop("afpm_shm.gui", None)
    g = importlib.import_module("afpm_shm.gui")
    yield g, mb, fd, tmp_path
    sys.modules.pop("afpm_shm.gui", None)


def test_gui_flow(gui):
    g, mb, fd, tmp = gui
    app = g.App(W())
    assert app.res is not None and len(app.specs) == 14
    assert abs(app.res.R.T - 3.55) < 0.01
    # change topology -> SSDR defaults
    app.topology.set("SSDR"); app.on_topology()
    assert abs(app.res.R.T - 7.10) < 0.01 and app.vars["Ilim"].get() == "40"
    # edit an input with a unit factor (mm) and recalculate
    app.vars["Ro"].set("60"); app.calculate()
    assert app.res.P.Ro == pytest.approx(0.060)
    # bad input -> error dialog, no crash, previous result kept
    app.vars["Br"].set("abc"); before = app.res; app.calculate()
    assert mb.showerror.called and app.res is before
    app.vars["Br"].set("1.2")
    app.vars["Q"].set("12.5"); app.calculate()
    assert app.res is before
    app.vars["Q"].set("12"); app.vars["Ro"].set("55"); app.calculate()
    # live drawing, valid and invalid input
    app.redraw_motor(); app.vars['Q'].set('abc'); app.redraw_motor(); app.vars['Q'].set('12')
    app.vars['Ri'].set('70'); app.redraw_motor(); app.vars['Ri'].set('27.5')
    fd.asksaveasfilename = MagicMock(return_value=str(tmp / 'd.png')); app.save_drawing()
    assert (tmp / 'd.png').exists()
    # every graph can be shown
    for i in range(14):
        app.listbox.selection_set(i); app.show_plot()
    # save graph / data
    app.listbox.selection_set(8)
    fd.asksaveasfilename = MagicMock(return_value=str(tmp / "g.svg")); app.save_graph()
    fd.asksaveasfilename = MagicMock(return_value=str(tmp / "d.csv")); app.save_data()
    assert (tmp / "g.svg").stat().st_size > 1000
    assert "Speed (rpm) [SHM prediction" in (tmp / "d.csv").read_text(encoding="utf-8-sig")
    # export everything
    fd.askdirectory = MagicMock(return_value=str(tmp)); app.export_everything()
    out = tmp / "AFPM_SSDR_results"
    assert (out / "results.csv").exists() and (out / "ke.png").exists() and (out / "ke.csv").exists()
    # design save / open
    fd.asksaveasfilename = MagicMock(return_value=str(tmp / "des.json")); app.save_design()
    app.vars["N"].set("900")
    fd.askopenfilename = MagicMock(return_value=str(tmp / "des.json")); app.open_design()
    assert app.vars["N"].get() == "1200"
    # overlay
    m = tmp / "meas.csv"; m.write_text("speed,torque\n800,7.2\n1200,7.0\n")
    fd.askopenfilename = MagicMock(return_value=str(m)); app.load_overlay("speed_torque")
    assert "speed_torque" in app.overlays
    app.clear_overlays(); assert not app.overlays
    bad = tmp / "bad.csv"; bad.write_text("nothing here\n")
    fd.askopenfilename = MagicMock(return_value=str(bad)); mb.showerror.reset_mock(); app.load_overlay("ke")
    assert mb.showerror.called


def test_decimal_comma_accepted(gui):
    g, mb, fd, tmp = gui
    app = g.App(W())
    app.vars["Br"].set("1,25"); app.calculate()
    assert app.res.P.Br == 1.25
