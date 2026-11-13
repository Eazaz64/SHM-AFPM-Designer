"""Space Harmonic Method (SHM) model of an axial flux PM BLDC motor.

Python port of the MATLAB files AFPM_params, carter_coeff, shm_Bn, winding_factor,
shm_backemf, flux_check, circuit_params and performance (Chapter 4 of the thesis).
All quantities are SI (metres, tesla, volts, amperes) unless a name says otherwise.
"""
import math
from dataclasses import dataclass, asdict, replace, fields
from types import SimpleNamespace

import numpy as np

MU0 = 4 * math.pi * 1e-7
_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------- inputs
@dataclass
class Params:
    topology: str = "SSSR"      # 'SSSR' or 'SSDR'
    # operating point
    N: float = 1200.0           # rated speed (rpm)
    Vdc: float = 36.0           # DC bus voltage (V)
    Vsw: float = 1.0            # voltage drop of the two conducting switches (V)
    Ilim: float = 25.0          # controller DC current limit (A)
    # main dimensions
    Ro: float = 55e-3
    Ri: float = 27.5e-3
    Q: int = 12                 # slots (= teeth = coils)
    p: int = 7                  # pole pairs
    m: int = 3
    g: float = 1.0e-3           # mechanical air gap per side (m)
    Ns: int = 16                # radial slices (quasi-3D)
    # magnets
    Br: float = 1.2
    mur: float = 1.05
    hm: float = 4e-3
    alpha: float = 0.8          # pole-arc ratio
    nmax: int = 59              # highest odd space harmonic
    # stator
    b0: float = 7.2e-3
    wti: float = 7.2e-3
    wto: float = 21.6e-3
    hs: float = 20e-3
    hsy: float = 6e-3
    hplate: float = 5e-3
    hby: float = 5e-3
    rsh: float = 7.5e-3
    # winding
    Nc: int = 16
    nstrand: int = 3
    dwire: float = 0.914e-3
    fill: float = 0.5
    wc: float = 3.4e-3
    J: float = 6.46e6
    Tw: float = 75.0
    rho20: float = 1.72e-8
    alphaCu: float = 3.93e-3
    # losses
    kh: float = 0.025
    beta: float = 2.0
    ke: float = 1.07e-4
    kharm: float = 1.5
    stray: float = 0.005
    Prot: float = 3.0
    mplate: float = 0.0
    # densities
    rhoFe: float = 7650.0
    rhoBI: float = 7850.0
    rhoCu: float = 8960.0
    rhoPM: float = 7500.0
    # optional calibration against measurements (1.0 = pure SHM)
    kE: float = 1.0             # multiplies the back-EMF (Ke, Kt)
    kL: float = 1.0             # multiplies the inductance

    # ---- derived quantities ----
    @property
    def nrot(self): return 2 if self.topology == "SSDR" else 1
    @property
    def paths(self): return 2 if self.topology == "SSDR" else 1
    @property
    def yoke(self): return self.topology == "SSSR"
    @property
    def Ht(self): return self.hs if self.topology == "SSSR" else 2 * self.hs + self.hplate
    @property
    def wm(self): return 2 * math.pi * self.N / 60
    @property
    def f(self): return self.p * self.N / 60
    @property
    def dr(self): return (self.Ro - self.Ri) / self.Ns
    @property
    def r(self): return self.Ri + (np.arange(1, self.Ns + 1) - 0.5) * self.dr
    @property
    def rm(self): return (self.Ro + self.Ri) / 2
    @property
    def Nph(self): return (self.Q // self.m) * self.Nc
    @property
    def harm(self): return np.arange(1, int(self.nmax) + 1, 2)


def default_params(topology="SSSR"):
    t = topology.upper()
    if t == "SSSR":
        return Params(topology="SSSR", Ilim=25.0, Prot=3.0, mplate=0.0)
    if t == "SSDR":
        return Params(topology="SSDR", Ilim=40.0, Prot=5.0, mplate=0.070)
    raise ValueError("topology must be 'SSSR' or 'SSDR'")


def params_to_dict(P):
    return asdict(P)


def params_from_dict(d):
    names = {f.name for f in fields(Params)}
    clean = {k: v for k, v in d.items() if k in names}
    base = default_params(clean.get("topology", "SSSR"))
    return replace(base, **clean)


def validate(P):
    """Return (errors, warnings) as lists of strings."""
    err, warn = [], []
    pos = ["N", "Vdc", "Ilim", "Ro", "Ri", "g", "Br", "mur", "hm", "alpha", "b0", "wti", "wto",
           "hs", "hby", "rsh", "Nc", "nstrand", "dwire", "fill", "wc", "J", "rhoFe", "rhoBI",
           "rhoCu", "rhoPM", "kE", "kL"]
    for k in pos:
        if not getattr(P, k) > 0:
            err.append("%s must be greater than zero." % k)
    if err:
        return err, warn
    if P.Ro <= P.Ri: err.append("Outer radius Ro must be larger than inner radius Ri.")
    if P.Ri <= P.rsh: err.append("Inner radius Ri must be larger than the shaft radius.")
    if P.Q < 3 or P.Q % P.m: err.append("Number of slots Q must be a multiple of 3.")
    if P.p < 1 or int(P.p) != P.p: err.append("Pole pairs p must be a positive integer.")
    if not 0 < P.alpha <= 1: err.append("Pole-arc ratio must be between 0 and 1.")
    if P.nmax < 13: err.append("Highest harmonic nmax must be at least 13.")
    if P.Ns < 2: err.append("Number of slices Ns must be at least 2.")
    if err:
        return err, warn
    t = math.gcd(int(P.Q), int(P.p))
    if (P.Q / (P.m * t)) != int(P.Q / (P.m * t)):
        warn.append("The Q/p combination does not give a balanced three-phase winding.")
    for r, w, lab in ((P.Ri, P.wti, "Ri"), (P.Ro, P.wto, "Ro")):
        ts = 2 * math.pi * r / P.Q
        if w >= ts:
            err.append("Tooth width at %s (%.1f mm) is not smaller than the slot pitch (%.1f mm)."
                       % (lab, w * 1e3, ts * 1e3))
        elif abs(ts - w - P.b0) > 0.5e-3:
            warn.append("Slot opening b0 differs from (slot pitch - tooth width) at %s by more than 0.5 mm." % lab)
    if P.Nph * 1.0 <= 0: err.append("Turns per phase must be positive.")
    return err, warn


# --------------------------------------------------------------------------- winding
def winding_layout(Q, p):
    """Coil phases and signs of a double-layer tooth-coil winding from the star of slots.
    Gives 'ACCBBAACCBBA' / [1 1 -1 -1 1 1 -1 -1 1 1 -1 -1] for 12 slots / 7 pole pairs."""
    Q, p = int(Q), int(p)
    phases, signs, angles = [], [], []
    table = [("A", 1), ("C", -1), ("B", 1), ("A", -1), ("C", 1), ("B", -1)]
    for k in range(Q):
        thQ = (k * 360 * p) % (360 * Q)                  # angle x Q (exact integers)
        sector = ((thQ + 30 * Q) % (360 * Q)) // (60 * Q)
        ph, sg = table[int(sector)]
        phases.append(ph); signs.append(sg); angles.append(thQ / Q)
    return SimpleNamespace(phase="".join(phases), sign=np.array(signs), angle=np.array(angles))


def winding_factor(n, P, L):
    alpha = 2 * math.pi * P.p / P.Q
    kp = abs(math.sin(n * alpha / 2))
    idx = [i for i, c in enumerate(L.phase) if c == "A"]
    kd = abs(sum(L.sign[i] * np.exp(1j * n * i * alpha) for i in idx)) / len(idx)
    return kp * kd, kp, kd


# --------------------------------------------------------------------------- field
def carter_coeff(r, P):
    r = np.asarray(r, dtype=float)
    ts = 2 * math.pi * r / P.Q
    x = P.b0 / (2 * P.g)
    gamma = 4 / math.pi * (x * math.atan(x) - math.log(math.sqrt(1 + x * x)))
    return ts / (ts - gamma * P.g)


def shm_Bn(n, r, P):
    """Amplitude of the n-th harmonic of the axial air-gap flux density at radius r (Eq. 4.14)."""
    r = np.asarray(r, dtype=float)
    gp = carter_coeff(r, P) * P.g
    k = n * P.p / r
    Mn = 4 * P.Br / (n * math.pi) * math.sin(n * math.pi * P.alpha / 2)
    return Mn * np.sinh(k * P.hm) / (P.mur * np.cosh(k * P.hm) * np.sinh(k * gp)
                                     + np.sinh(k * P.hm) * np.cosh(k * gp))


def airgap_B(P, r, th, nmax=79):
    B = np.zeros_like(th)
    for n in range(1, nmax + 1, 2):
        B += float(shm_Bn(n, r, P)) * np.cos(n * th)
    return B


# --------------------------------------------------------------------------- back-EMF
def backemf(P, L):
    th = np.linspace(0, 2 * math.pi, 3601)
    ea = np.zeros_like(th); eb = np.zeros_like(th)
    for n in P.harm:
        s = np.sum(2 * shm_Bn(n, P.r, P) * P.r * P.dr)
        En = P.wm * P.Nph * winding_factor(n, P, L)[0] * s
        ea += En * np.sin(n * th)
        eb += En * np.sin(n * (th - 2 * math.pi / 3))
    eab = ea - eb
    i = int(np.argmax(eab)); Epk = eab[i]
    w = len(th) // 12
    idx = np.arange(i - w, i + w) % len(th)
    Eavg = eab[idx].mean()
    nn = np.arange(1, 60, 2)
    Eph = np.array([abs(_trapz(ea * np.sin(k * th), th) / math.pi) for k in nn])
    Eln = np.array([abs(complex(_trapz(eab * np.sin(k * th), th) / math.pi,
                                _trapz(eab * np.cos(k * th), th) / math.pi)) for k in nn])
    ec = np.interp(np.mod(th + 2 * math.pi / 3, 2 * math.pi), th, ea)
    eb2 = np.interp(np.mod(th - 2 * math.pi / 3, 2 * math.pi), th, ea)
    env = np.max([ea - eb2, ea - ec, eb2 - ec, eb2 - ea, ec - ea, ec - eb2], axis=0)
    return SimpleNamespace(
        th=th, ea=ea, eab=eab, env=env, ipk=i, Epk=Epk, Eavg=Eavg, Ke=Eavg / P.wm,
        E1=Eph[0], THDph=math.sqrt(np.sum(Eph[1:] ** 2)) / Eph[0],
        THDline=math.sqrt(np.sum(Eln[1:] ** 2)) / Eln[0],
        ripple=(env.max() - env.min()) / env.mean(), nn=nn, Eph=Eph, Eln=Eln)


# --------------------------------------------------------------------------- iron flux
def flux_check(P):
    harm = P.harm
    Bn = np.array([shm_Bn(n, P.r, P) for n in harm])            # (nh, Ns)
    npp = (harm * P.p).astype(float)[:, None]
    span = 2 * np.sin(npp * math.pi / P.Q) / npp
    wt = P.wti + (P.r - P.Ri) * (P.wto - P.wti) / (P.Ro - P.Ri)
    shifts = np.linspace(0, 2 * math.pi / P.p, 60)
    best = 0.0
    Btooth = np.zeros(P.Ns)
    yokeFlux = 0.0
    c = np.arange(P.Q) * 2 * math.pi / P.Q

    def slice_flux(ang, sh):
        return P.r * np.sum(Bn * span * np.cos(npp * (ang - sh)), axis=0)

    for sh in shifts:
        f = slice_flux(0.0, sh)
        phi = np.sum(f * P.dr)
        if abs(phi) > best:
            best = abs(phi); Btooth = np.abs(f) / wt
        phit = np.array([np.sum(slice_flux(cq, sh) * P.dr) for cq in c])
        cum = np.cumsum(phit); cum = cum - cum.mean()
        yokeFlux = max(yokeFlux, np.max(np.abs(cum)))
    B1m = float(shm_Bn(1, P.rm, P))
    phiPole = 2 / math.pi * B1m * math.pi * (P.Ro ** 2 - P.Ri ** 2) / (2 * P.p)
    return SimpleNamespace(
        wt=wt, Btooth=Btooth, BtoothMax=Btooth.max(), phiTooth=best,
        Byoke=yokeFlux / (P.hsy * (P.Ro - P.Ri)),
        Bbackiron=phiPole / 2 / (P.hby * (P.Ro - P.Ri)), B1m=B1m)


# --------------------------------------------------------------------------- circuit
def _winding_function(phase, P, L, thm):
    Nw = np.zeros_like(thm)
    for k in range(P.Q):
        if L.phase[k] != phase:
            continue
        a0 = k * 2 * math.pi / P.Q
        inside = np.mod(thm - a0, 2 * math.pi) < 2 * math.pi / P.Q
        Nw = Nw + L.sign[k] * P.Nc * inside
    return Nw - Nw.mean()


def circuit_params(P, L):
    C = SimpleNamespace()
    C.Aw = P.nstrand * math.pi / 4 * P.dwire ** 2
    C.hc = P.Nc * C.Aw / P.fill / P.wc
    side = math.hypot(P.Ro - P.Ri, (P.wto - P.wti) / 2)
    C.MLT = 2 * side + P.wti + P.wto + math.pi * P.wc
    R1 = P.rho20 * P.Nph * C.MLT / C.Aw
    C.R20 = R1 / P.paths
    C.R = C.R20 * (1 + P.alphaCu * (P.Tw - 20))
    thm = np.linspace(0, 2 * math.pi, 7201)[:-1]
    NA = _winding_function("A", P, L, thm)
    NB = _winding_function("B", P, L, thm)
    ge = carter_coeff(P.r, P) * P.g + P.hm / P.mur
    perm = np.sum(MU0 * P.r * P.dr / ge)
    C.Lg = perm * _trapz(NA * NA, thm) / P.paths
    C.Mg = perm * _trapz(NA * NB, thm) / P.paths
    lam = C.hc / (3 * P.b0)
    C.Lslot = 4 * 2 * MU0 * P.Nc ** 2 * (P.Ro - P.Ri) * lam / P.paths
    As = P.wc * C.hc
    tau = np.array([P.wti, P.wto]) + P.wc
    Le1 = np.sum(MU0 * P.Nc ** 2 * tau / 8 * np.log(tau ** 2 * math.pi / (4 * As)))
    C.Lend = 4 * Le1 / P.paths
    for nm in ("Lg", "Mg", "Lslot", "Lend"):
        setattr(C, nm, getattr(C, nm) * P.kL)
    C.L = C.Lg + C.Lslot + C.Lend
    C.tau = C.L / C.R
    return C


# --------------------------------------------------------------------------- performance
def core_loss(P, F, R, fr):
    """Core loss (W) at speed ratio fr = N / N_rated (Steinmetz model, Eq. 4.25)."""
    f = P.f * np.asarray(fr)
    return P.kharm * (P.kh * f * R.Sb + P.ke * f ** 2 * R.S2)


def performance(P, E, F, C):
    R = SimpleNamespace()
    R.Irms = P.J * C.Aw * P.paths
    R.Idc = R.Irms / math.sqrt(2 / 3)
    R.Kt = E.Ke
    R.T = R.Kt * R.Idc
    R.P = R.T * P.wm
    mT = P.Q * F.wt * P.dr * P.Ht * P.rhoFe
    R.mTeeth = float(np.sum(mT))
    R.mYoke = (1.0 if P.yoke else 0.0) * math.pi * (P.Ro ** 2 - P.Ri ** 2) * P.hsy * P.rhoFe
    R.mCu = P.Q * P.Nc * C.MLT * C.Aw * P.rhoCu * P.paths
    R.mPM = P.nrot * P.alpha * math.pi * (P.Ro ** 2 - P.Ri ** 2) * P.hm * P.rhoPM
    R.mBI = P.nrot * math.pi * (P.Ro ** 2 - P.rsh ** 2) * P.hby * P.rhoBI
    R.mass = R.mTeeth + R.mYoke + R.mCu + R.mPM + R.mBI + P.mplate
    R.Sb = float(np.sum(mT * F.Btooth ** P.beta) + R.mYoke * F.Byoke ** P.beta)
    R.S2 = float(np.sum(mT * F.Btooth ** 2) + R.mYoke * F.Byoke ** 2)
    R.Pcu = 2 * C.R * R.Idc ** 2
    R.Pfe = float(core_loss(P, F, R, 1.0))
    R.Prot = P.Prot
    R.Pstr = P.stray * R.P
    R.eta = R.P / (R.P + R.Pcu + R.Pfe + R.Prot + R.Pstr)
    R.Vreq = E.Eavg + 2 * C.R * R.Idc + P.Vsw
    R.Nmax = P.N * (P.Vdc - 2 * C.R * R.Idc - P.Vsw) / E.Eavg
    R.shear = R.T / (2 * math.pi / 3 * (P.Ro ** 3 - P.Ri ** 3))
    R.Ai = 2 * P.m * P.Nph * R.Irms / P.paths / (2 * math.pi * P.Ri)
    R.Tdens = R.T / R.mass
    R.Pdens = R.P / R.mass
    return R


def solve(P):
    """Run the whole SHM design for the parameters P. Returns a namespace of results."""
    L = winding_layout(P.Q, P.p)
    E = backemf(P, L)
    k = P.kE
    if k != 1.0:                                  # optional calibration of the EMF
        for a in ("ea", "eab", "Epk", "Eavg", "Ke", "E1", "Eph", "Eln"):
            setattr(E, a, getattr(E, a) * k)
    F = flux_check(P)
    C = circuit_params(P, L)
    R = performance(P, E, F, C)
    return SimpleNamespace(P=P, L=L, E=E, F=F, C=C, R=R)


# --------------------------------------------------------------------------- speed-torque
def speed_torque(res, npts=400):
    """Six-step steady-state characteristic at Vdc (average model, Section 4.x).
    DC-link current I: Vdc - Vsw = Ke*w + 2*R*I, T = Kt*I. Current limited to P.Ilim."""
    P, E, C, F, R = res.P, res.E, res.C, res.F, res.R
    Vm = P.Vdc - P.Vsw
    I = np.linspace(0, P.Ilim, npts)
    w = (Vm - 2 * C.R * I) / E.Ke
    n = 60 * w / (2 * math.pi)
    T = E.Ke * I
    Po = T * w
    fr = n / P.N
    Pfe = core_loss(P, F, R, fr)
    Pin = Po + 2 * C.R * I ** 2 + Pfe + P.Prot * fr + P.stray * Po
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(Pin > 0, Po / Pin, np.nan)
    n0 = 60 * Vm / E.Ke / (2 * math.pi)
    return SimpleNamespace(I=I, n=n, T=T, Po=Po, eta=eta, n0=n0, Tlim=T[-1], ncorner=n[-1])


def efficiency_map(res, nx=120, ny=100, nmax=None):
    P, E, C, F, R = res.P, res.E, res.C, res.F, res.R
    nmax = nmax or 1.4 * P.N
    NN, TT = np.meshgrid(np.linspace(100, nmax, nx), np.linspace(0.05, 1.5 * R.T, ny))
    I = TT / R.Kt
    w = 2 * math.pi * NN / 60
    Po = TT * w
    fr = NN / P.N
    Pfe = core_loss(P, F, R, fr)
    eta = Po / (Po + 2 * C.R * I ** 2 + Pfe + P.Prot * fr + P.stray * Po)
    V = E.Eavg * fr + 2 * C.R * I + P.Vsw
    eta = np.where((V > P.Vdc) | (I > P.Ilim), np.nan, eta)
    return NN, TT, 100 * eta


# --------------------------------------------------------------------------- tables
def results_table(res):
    """List of (section, quantity, value, unit) rows for display and CSV export."""
    P, L, E, F, C, R = res.P, res.L, res.E, res.F, res.C, res.R
    st = speed_torque(res)
    rows = []

    def add(sec, q, v, u=""):
        rows.append((sec, q, v, u))

    lam = P.Ri / P.Ro
    lamOpt = 1 / math.sqrt(3)
    add("Sizing", "Ri / Ro", "%.3f" % lam)
    add("Sizing", "Torque capability vs optimum Ri/Ro = 1/sqrt(3)",
        "%.1f" % (100 * lam * (1 - lam ** 2) / (lamOpt * (1 - lamOpt ** 2))), "%")
    add("Sizing", "Shear stress", "%.1f" % (R.shear / 1e3), "kPa")
    add("Sizing", "Electric loading at Ri", "%.1f" % (R.Ai / 1e3), "kA/m")
    add("Field", "Carter coefficient at Ri / mean / Ro", "%.2f / %.2f / %.2f" % tuple(
        float(carter_coeff(x, P)) for x in (P.Ri, P.rm, P.Ro)))
    add("Field", "Fundamental air-gap B1 at mean radius", "%.2f" % F.B1m, "T")
    for n in range(1, 14, 2):
        kw, kp, kd = winding_factor(n, P, L)
        add("Winding factors", "n = %d  (kp / kd / kw)" % n, "%.3f / %.3f / %.3f" % (kp, kd, kw))
    add("Back-EMF at rated speed", "Rated speed", "%.0f" % P.N, "rpm")
    add("Back-EMF at rated speed", "Series turns per phase (per path)", "%d" % P.Nph)
    add("Back-EMF at rated speed", "Phase EMF, fundamental peak", "%.2f" % E.E1, "V")
    add("Back-EMF at rated speed", "Phase EMF THD", "%.1f" % (100 * E.THDph), "%")
    add("Back-EMF at rated speed", "Line EMF peak", "%.2f" % E.Epk, "V")
    add("Back-EMF at rated speed", "Line EMF average over 60 deg", "%.2f" % E.Eavg, "V")
    add("Back-EMF at rated speed", "Line EMF THD", "%.2f" % (100 * E.THDline), "%")
    add("Back-EMF at rated speed", "Ke (line-to-line, 60 deg average)", "%.4f" % E.Ke, "V s/rad")
    add("Back-EMF at rated speed", "Ke (line-to-line, peak basis)", "%.4f" % (E.Epk / P.wm), "V s/rad")
    add("Back-EMF at rated speed", "Kt = Ke", "%.4f" % R.Kt, "N m/A")
    add("Circuit", "Coil height", "%.1f" % (C.hc * 1e3), "mm")
    add("Circuit", "Mean turn length", "%.1f" % (C.MLT * 1e3), "mm")
    add("Circuit", "Phase resistance at 20 degC", "%.1f" % (C.R20 * 1e3), "mOhm")
    add("Circuit", "Phase resistance at winding temperature", "%.1f" % (C.R * 1e3), "mOhm")
    add("Circuit", "Lg (air gap)", "%.0f" % (C.Lg * 1e6), "uH")
    add("Circuit", "Lslot", "%.0f" % (C.Lslot * 1e6), "uH")
    add("Circuit", "Lend", "%.1f" % (C.Lend * 1e6), "uH")
    add("Circuit", "L total", "%.0f" % (C.L * 1e6), "uH")
    add("Circuit", "Mutual inductance Mg", "%.2f" % (C.Mg * 1e6), "uH")
    add("Circuit", "Time constant L/R", "%.2f" % (C.tau * 1e3), "ms")
    add("Torque ripple", "EMF-shape torque ripple (six-step)", "%.1f" % (100 * E.ripple), "%")
    add("Torque ripple", "Cogging periods per revolution", "%d" % (P.Q * 2 * P.p // math.gcd(int(P.Q), 2 * int(P.p))))
    add("Iron flux density", "Tooth B, narrow end", "%.2f" % F.BtoothMax, "T")
    add("Iron flux density", "Stator yoke B", "%.2f" % F.Byoke if P.yoke else "-", "T" if P.yoke else "")
    add("Iron flux density", "Rotor back-iron B", "%.2f" % F.Bbackiron, "T")
    add("Rated-point performance", "DC-link current", "%.1f" % R.Idc, "A")
    add("Rated-point performance", "Torque", "%.2f" % R.T, "N m")
    add("Rated-point performance", "Output power", "%.0f" % R.P, "W")
    add("Rated-point performance", "Copper loss", "%.1f" % R.Pcu, "W")
    add("Rated-point performance", "Core loss", "%.1f" % R.Pfe, "W")
    add("Rated-point performance", "Rotational loss", "%.1f" % R.Prot, "W")
    add("Rated-point performance", "Stray loss", "%.1f" % R.Pstr, "W")
    add("Rated-point performance", "Efficiency", "%.1f" % (100 * R.eta), "%")
    add("Rated-point performance", "Voltage required", "%.1f" % R.Vreq, "V")
    add("Rated-point performance", "Max speed at rated current", "%.0f" % R.Nmax, "rpm")
    add("Rated-point performance", "Active mass", "%.2f" % R.mass, "kg")
    add("Rated-point performance", "Torque density", "%.2f" % R.Tdens, "N m/kg")
    add("Rated-point performance", "Power density", "%.0f" % R.Pdens, "W/kg")
    add("Speed-torque at DC bus", "DC bus voltage", "%.1f" % P.Vdc, "V")
    add("Speed-torque at DC bus", "Ideal no-load speed (Vdc - Vsw) / Ke", "%.0f" % st.n0, "rpm")
    add("Speed-torque at DC bus", "Torque at current limit", "%.2f" % st.Tlim, "N m")
    add("Speed-torque at DC bus", "Speed where the current limit ends", "%.0f" % st.ncorner, "rpm")
    return rows


def inputs_table(P):
    return [("Input", k, "%s" % v, "") for k, v in params_to_dict(P).items()]
