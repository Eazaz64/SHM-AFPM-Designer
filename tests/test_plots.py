import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from afpm_shm import model as M, plots as PL


def test_all_plots_draw_and_export(tmp_path):
    from matplotlib.figure import Figure
    for topo in ("SSSR", "SSDR"):
        res = M.solve(M.default_params(topo))
        ov = {"speed_torque": (np.array([800., 1000, 1200]), np.array([7.2, 7.1, 7.0])),
              "ke": (np.array([400., 800, 1200]), np.array([11.3, 22.3, 33.7]))}
        specs = PL.build_all(res, ov)
        assert len(specs) == 14
        for s in specs:
            fig = Figure(); PL.draw(fig, s)
            rows = PL.spec_to_rows(s)
            assert len(rows) > 1 and all(len(r) == len(rows[0]) for r in rows)
        out = PL.export_all(res, str(tmp_path / topo), ov, formats=("png", "svg", "pdf"), dpi=100)
        names = {os.path.basename(p) for p in out}
        for key, _, _ in PL.PLOTS:
            assert key + ".png" in names and key + ".csv" in names
        assert "results.csv" in names and "inputs.csv" in names and "design.json" in names


def test_roundtrip_params(tmp_path):
    P = M.default_params("SSDR"); P.Br = 1.25
    d = M.params_to_dict(P)
    assert M.params_from_dict(d) == P


def test_read_xy_csv(tmp_path):
    f = tmp_path / "m.csv"; f.write_text("speed,torque\n800,7.2\n1000,7.1\n1200,7.0\n")
    x, y = PL.read_xy_csv(str(f))
    assert list(x) == [800, 1000, 1200] and list(y) == [7.2, 7.1, 7.0]
    f2 = tmp_path / "m2.csv"; f2.write_text("800;7,2\n")
    try:
        PL.read_xy_csv(str(f2)); assert False
    except ValueError:
        pass


def test_motor_drawing_variants(tmp_path):
    from matplotlib.figure import Figure
    from afpm_shm import drawing as DR
    for topo in ("SSSR", "SSDR"):
        f = Figure(); DR.draw_motor(f, M.default_params(topo))
        assert len(f.axes) == 3
        f.savefig(str(tmp_path / (topo + ".png")))
    P = M.default_params("SSSR"); P.Q = 9; P.p = 4; P.wto = 14e-3; P.wti = 5e-3; P.b0 = 5e-3
    f = Figure(); DR.draw_motor(f, P); assert len(f.axes) == 3
    f = Figure(); DR.draw_motor(f, None, message="x"); assert len(f.axes) == 0
