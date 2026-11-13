"""Check the Python SHM against the expected output listed in the MATLAB README."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from afpm_shm import model as M


@pytest.fixture(scope="module")
def sssr(): return M.solve(M.default_params("SSSR"))


@pytest.fixture(scope="module")
def ssdr(): return M.solve(M.default_params("SSDR"))


def test_winding_layout():
    L = M.winding_layout(12, 7)
    assert L.phase == "ACCBBAACCBBA"
    assert list(L.sign) == [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]


def test_carter_and_field(sssr):
    P = sssr.P
    kc = [float(M.carter_coeff(r, P)) for r in (P.Ri, P.rm, P.Ro)]
    assert kc == pytest.approx([1.42, 1.25, 1.17], abs=0.006)
    assert sssr.F.B1m == pytest.approx(1.04, abs=0.006)


def test_winding_factors(sssr):
    kw = [M.winding_factor(n, sssr.P, sssr.L)[0] for n in (1, 3, 5, 7, 9, 11, 13)]
    assert kw == pytest.approx([0.933, 0.5, 0.067, 0.067, 0.5, 0.933, 0.933], abs=0.0006)


@pytest.mark.parametrize("fx", ["sssr", "ssdr"])
def test_backemf(fx, request):
    r = request.getfixturevalue(fx)
    assert r.E.E1 == pytest.approx(17.6, abs=0.06)
    assert 100 * r.E.THDph == pytest.approx(8.2, abs=0.06)
    assert r.E.Epk == pytest.approx(30.3, abs=0.06)
    assert r.E.Eavg == pytest.approx(28.6, abs=0.06)
    assert r.E.Ke == pytest.approx(0.228, abs=0.0006)
    assert 100 * r.E.ripple == pytest.approx(15.2, abs=0.1)   # 15.146 -> README rounds to 15.2


def test_circuit(sssr, ssdr):
    assert sssr.C.R20 * 1e3 == pytest.approx(53.9, abs=0.06)
    assert sssr.C.R * 1e3 == pytest.approx(65.5, abs=0.06)
    assert ssdr.C.R20 * 1e3 == pytest.approx(26.9, abs=0.06)
    assert [sssr.C.Lg * 1e6, sssr.C.Lslot * 1e6, sssr.C.Lend * 1e6] == pytest.approx([151, 61, 8.8], abs=0.6)
    assert sssr.C.L * 1e6 == pytest.approx(220, abs=0.6)
    assert ssdr.C.L * 1e6 == pytest.approx(110, abs=0.6)
    assert sssr.C.tau * 1e3 == pytest.approx(3.4, abs=0.06)
    assert abs(sssr.C.Mg) < 1e-9


def test_iron(sssr, ssdr):
    assert sssr.F.BtoothMax == pytest.approx(0.99, abs=0.006)
    assert sssr.F.Bbackiron == pytest.approx(1.22, abs=0.006)
    assert sssr.F.Byoke == pytest.approx(1.06, abs=0.006)


def test_performance_sssr(sssr):
    R = sssr.R
    assert (R.Idc, R.T, R.P) == pytest.approx((15.6, 3.55, 446), abs=0.6)
    assert (R.Pcu, R.Pfe, R.Prot, R.Pstr) == pytest.approx((31.8, 7.5, 3.0, 2.2), abs=0.06)
    assert 100 * R.eta == pytest.approx(90.9, abs=0.06)
    assert R.Vreq == pytest.approx(31.7, abs=0.06)
    assert R.Nmax == pytest.approx(1382, abs=1)
    assert R.mass == pytest.approx(1.92, abs=0.006)
    assert R.Tdens == pytest.approx(1.85, abs=0.006)
    assert R.shear / 1e3 == pytest.approx(11.6, abs=0.06)


def test_performance_ssdr(ssdr):
    R = ssdr.R
    assert (R.Idc, R.T, R.P) == pytest.approx((31.1, 7.10, 892), abs=0.6)
    assert (R.Pcu, R.Pfe, R.Prot, R.Pstr) == pytest.approx((63.6, 10.0, 5.0, 4.5), abs=0.06)
    assert 100 * R.eta == pytest.approx(91.5, abs=0.06)
    assert R.mass == pytest.approx(3.43, abs=0.006)
    assert R.Tdens == pytest.approx(2.07, abs=0.006)
    assert R.shear / 1e3 == pytest.approx(23.3, abs=0.06)


def test_kt_equals_ke(sssr):
    assert sssr.R.Kt == sssr.E.Ke


def test_speed_torque_physics(sssr):
    st = M.speed_torque(sssr)
    P = sssr.P
    assert st.n0 == pytest.approx(60 / (2 * math.pi) * (P.Vdc - P.Vsw) / sssr.E.Ke, rel=1e-9)
    assert st.T[0] == 0 and st.T[-1] == pytest.approx(sssr.E.Ke * P.Ilim)
    assert np.all(np.diff(st.n) < 0)          # speed falls as current rises


def test_calibration_scales_ke_and_inductance():
    base = M.solve(M.default_params("SSDR"))
    P = M.default_params("SSDR"); P.kE = 1.12; P.kL = 1.16
    cal = M.solve(P)
    assert cal.E.Ke == pytest.approx(1.12 * base.E.Ke)
    assert cal.R.T == pytest.approx(1.12 * base.R.T)
    assert cal.C.L == pytest.approx(1.16 * base.C.L)


def test_validate():
    P = M.default_params("SSSR")
    assert M.validate(P)[0] == []
    P.Ri = 0.06
    assert M.validate(P)[0]
    P = M.default_params("SSSR"); P.Q = 10
    assert M.validate(P)[0]


def test_other_machine_runs():
    P = M.default_params("SSSR"); P.Q = 9; P.p = 4; P.wto = 14e-3; P.wti = 5e-3; P.b0 = 5e-3
    L = M.winding_layout(9, 4)
    assert len(L.phase) == 9 and set(L.phase) == set("ABC")
    r = M.solve(P)
    assert r.E.Ke > 0
