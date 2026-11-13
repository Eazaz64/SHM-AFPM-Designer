import sys


def selftest(outdir=None):
    """Headless check used by the build: solve, export and open the window once."""
    import os
    import tempfile
    from . import model as M, plots as PL
    outdir = outdir or tempfile.mkdtemp(prefix="afpm_selftest_")
    for topo, T in (("SSSR", 3.55), ("SSDR", 7.10)):
        res = M.solve(M.default_params(topo))
        assert abs(res.R.T - T) < 0.01, "torque check failed for " + topo
        assert abs(res.E.Ke - 0.228) < 0.001, "Ke check failed for " + topo
        files = PL.export_all(res, os.path.join(outdir, topo), dpi=80)
        assert len(files) >= 30, "export incomplete"
    import tkinter as tk
    from .gui import App
    root = tk.Tk()
    app = App(root)
    root.update_idletasks()
    root.update()
    assert app.res is not None and len(app.specs) == 14
    app.redraw_motor(); root.update()
    assert len(app.dfig.axes) == 3, "motor drawing failed"
    root.destroy()
    with open(os.path.join(outdir, "selftest_ok.txt"), "w") as fh:
        fh.write("ok\n")
    return outdir


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--version":
        from . import __version__
        print(__version__)
        return 0
    if argv and argv[0] == "--selftest":
        try:
            out = selftest(argv[1] if len(argv) > 1 else None)
        except Exception:
            import os
            import traceback
            folder = argv[1] if len(argv) > 1 else "."
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "selftest_failed.txt"), "w") as fh:
                fh.write(traceback.format_exc())
            return 1
        return 0
    from .gui import run
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
