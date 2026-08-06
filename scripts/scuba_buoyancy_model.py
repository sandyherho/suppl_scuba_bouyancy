#!/usr/bin/env python
"""
Shared model, numerics, and plotting substrate for the diver buoyancy study.

This module is the single source of truth for every diagnostic script in the
supplement. It defines the vertical buoyancy plant, the delayed
proportional-derivative controller exercised by the diver, the canonical diver
configurations, and the numerical routines used throughout. Each figure script
imports from here and remains independently runnable.

Model
    A diver at depth z (positive downward) stabilises an inherently unstable
    equilibrium. The carried gas (wetsuit or drysuit plus buoyancy compensator)
    compresses with depth by Boyle's law, so buoyancy falls as the diver sinks,
    producing a negative stiffness. The linearised closed loop is the delay
    differential equation

        x'' + 2 zeta x' - x + kappa_p x(s - theta) + kappa_d x'(s - theta) = 0,

    written in nondimensional time s = omega0 t, with omega0 = sqrt(beta / m_eff).

Gold-standard numerics
    Stability spectrum : pseudospectral discretisation of the infinitesimal
                         generator of the solution semigroup on Chebyshev
                         Gauss-Lobatto nodes, after Breda, Maset, and Vermiglio.
                         The rightmost characteristic roots are recovered as
                         eigenvalues of a finite matrix and converge spectrally.
    Time integration   : fourth-order Runge-Kutta method of steps on a grid
                         whose spacing divides the delay, with the interior-stage
                         delayed values supplied by the cubic Hermite continuous
                         extension so the delay term retains fourth-order
                         accuracy rather than dropping to first order.
    Ordinal analysis   : Bandt-Pompe permutation entropy and the Rosso
                         Jensen-Shannon statistical complexity, computed from
                         their definitions.

Author: Sandy H. S. Herho <sandy.herho@email.ucr.edu>
Date: 2026-08-01
"""

from dataclasses import dataclass, field, replace

import numpy as np


# --------------------------------------------------------------------------- #
# Physical constants (SI)
# --------------------------------------------------------------------------- #
RHO = 1025.0        # seawater density, kg m^-3
G = 9.81            # gravitational acceleration, m s^-2
P0 = 101325.0       # surface pressure, Pa


# --------------------------------------------------------------------------- #
# Diver configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DiverConfig:
    """A single canonical diver and its nominal control law."""
    key: str
    label: str
    m_eff: float        # effective mass incl. added mass, kg
    V_inc: float        # incompressible displaced volume, m^3
    V_s0: float         # suit gas volume at surface pressure, m^3
    Delta: float        # net weight excess m g - rho g V_inc, N
    z_star: float       # target (neutral) depth, m
    C_d: float          # drag coefficient
    S: float            # reference frontal area, m^2
    c1: float           # linear drag coefficient, N s m^-1
    tau: float          # sensorimotor delay, s
    kp: float           # proportional gain, m^3 m^-1  (gas per unit depth error)
    kd: float           # derivative gain,  m^3 s m^-1
    q_max: float        # compensator gas capacity at surface pressure, m^3


# Four canonical configurations. D1 is the baseline. D2 is a deeper technical
# drysuit dive with a larger compressible volume. D3 adds a long reaction delay
# (cold or task loaded). D4 is a poorly tuned novice with a long delay and an
# overdriven proportional gain.
DIVERS = {
    "D1": DiverConfig(
        key="D1", label="Recreational wetsuit",
        m_eff=100.0, V_inc=0.070, V_s0=0.006, Delta=40.0, z_star=15.0,
        C_d=1.1, S=0.28, c1=12.0, tau=2.0, kp=1.6e-3, kd=2.6e-3, q_max=0.020,
    ),
    "D2": DiverConfig(
        key="D2", label="Technical drysuit",
        m_eff=120.0, V_inc=0.078, V_s0=0.012, Delta=60.0, z_star=30.0,
        C_d=1.1, S=0.32, c1=14.0, tau=2.0, kp=2.2e-3, kd=3.4e-3, q_max=0.030,
    ),
    "D3": DiverConfig(
        key="D3", label="Cold, task loaded",
        m_eff=100.0, V_inc=0.070, V_s0=0.006, Delta=40.0, z_star=15.0,
        C_d=1.1, S=0.28, c1=12.0, tau=3.5, kp=1.6e-3, kd=1.2e-3, q_max=0.020,
    ),
    "D4": DiverConfig(
        key="D4", label="Novice, poorly tuned",
        m_eff=100.0, V_inc=0.070, V_s0=0.006, Delta=40.0, z_star=15.0,
        C_d=1.1, S=0.28, c1=12.0, tau=3.0, kp=5.2e-3, kd=0.4e-3, q_max=0.020,
    ),
}

DIVER_ORDER = ["D1", "D2", "D3", "D4"]


# --------------------------------------------------------------------------- #
# Derived quantities
# --------------------------------------------------------------------------- #
def pressure(z):
    """Absolute pressure at depth z (m), Pa."""
    return P0 + RHO * G * z


def derived(cfg):
    """Return the equilibrium gas load and the linearisation coefficients.

    q_star  : compensator gas at surface pressure giving neutral buoyancy at z*
    G_star  : total compressible surface-referenced gas, V_s0 + q_star
    beta    : magnitude of the destabilising negative stiffness, N m^-1
    gamma   : buoyancy authority per unit surface-referenced gas, N m^-3
    omega0  : instability rate sqrt(beta / m_eff), s^-1
    """
    Pz = pressure(cfg.z_star)
    G_star = cfg.Delta * Pz / (RHO * G * P0)
    q_star = G_star - cfg.V_s0
    beta = (RHO * G) ** 2 * G_star * P0 / Pz ** 2
    gamma = RHO * G * P0 / Pz
    omega0 = np.sqrt(beta / cfg.m_eff)
    return dict(Pz=Pz, G_star=G_star, q_star=q_star, beta=beta,
                gamma=gamma, omega0=omega0)


def nondim(cfg):
    """Return the four dimensionless groups (zeta, kappa_p, kappa_d, theta)."""
    d = derived(cfg)
    zeta = cfg.c1 / (2.0 * cfg.m_eff * d["omega0"])
    kappa_p = d["gamma"] * cfg.kp / d["beta"]
    kappa_d = d["gamma"] * cfg.kd * d["omega0"] / d["beta"]
    theta = d["omega0"] * cfg.tau
    return dict(zeta=zeta, kappa_p=kappa_p, kappa_d=kappa_d, theta=theta)


# --------------------------------------------------------------------------- #
# Nonlinear plant with delayed PD control and gas saturation
# --------------------------------------------------------------------------- #
def buoyancy_force(z, G_surf):
    """Upward buoyancy from the compressible gas, N (surface-referenced G)."""
    return RHO * G * G_surf * P0 / pressure(z)


def control_gas(cfg, z_del, v_del):
    """Delayed PD demand for surface-referenced compensator gas, saturated.

    A positive depth error (too deep) calls for more gas. The demand is clamped
    to the physically available range [0, q_max].
    """
    q = derived(cfg)["q_star"] + cfg.kp * (z_del - cfg.z_star) + cfg.kd * v_del
    return np.clip(q, 0.0, cfg.q_max)


def plant_acceleration(cfg, z, v, q_bcd, breath=0.0):
    """Vertical acceleration of the diver, m s^-2 (downward positive).

    breath is an optional surface-referenced tidal volume perturbation added to
    the compressible gas (used only by the forcing diagnostics).
    """
    G_surf = cfg.V_s0 + q_bcd + breath
    F_b = buoyancy_force(z, G_surf)
    drag = cfg.c1 * v + 0.5 * RHO * cfg.C_d * cfg.S * np.abs(v) * v
    return (cfg.Delta - F_b - drag) / cfg.m_eff


def integrate_dde(cfg, t_end, dt=None, z0=None, v0=None,
                  saturate=True, breath_amp=0.0, breath_freq=0.0,
                  noise_std=0.0, seed=0, n_per_delay=200, escape=500.0):
    """Fourth-order continuous method-of-steps integration of the plant.

    The step is chosen as tau / n_per_delay so the delayed state at whole
    Runge-Kutta stages lands exactly on a stored grid node. The half-step
    delayed values needed by the interior stages are supplied by the cubic
    Hermite continuous extension, using the stored velocity and acceleration as
    the interpolation slopes, so the delay contribution is fourth-order accurate
    rather than reduced to first order by a piecewise-constant lookup. The
    history on [-tau, 0] is the constant initial state. Returns (t, z, v).
    """
    d = derived(cfg)
    if dt is None:
        dt = cfg.tau / n_per_delay
    m = int(round(cfg.tau / dt))
    dt = cfg.tau / m
    n = int(np.ceil(t_end / dt))
    rng = np.random.default_rng(seed)

    zz0 = cfg.z_star if z0 is None else z0
    vv0 = 0.0 if v0 is None else v0
    z = np.empty(n + 1)
    v = np.empty(n + 1)
    a = np.empty(n + 1)      # acceleration at each node, for Hermite slopes
    z[0] = zz0
    v[0] = vv0

    def breath(t):
        if breath_amp == 0.0:
            return 0.0
        return breath_amp * np.sin(2.0 * np.pi * breath_freq * t)

    def accel(t, zz, vv, z_del, v_del):
        if saturate:
            q = control_gas(cfg, z_del, v_del)
        else:
            q = d["q_star"] + cfg.kp * (z_del - cfg.z_star) + cfg.kd * v_del
        return plant_acceleration(cfg, zz, vv, q, breath(t))

    def node(j):
        """State and acceleration at grid index j; constant history for j<=0."""
        if j <= 0:
            aj = a[0] if j == 0 else 0.0
            zj = z[0] if j == 0 else zz0
            vj = v[0] if j == 0 else vv0
            return zj, vj, aj
        return z[j], v[j], a[j]

    # acceleration at node 0 (delayed argument m steps back is in the history)
    zd0, vd0, _ = node(-m)
    a[0] = accel(0.0, z[0], v[0], zd0, vd0)

    for i in range(n):
        t = i * dt
        zj, vj, aj = node(i - m)           # delayed at stage time t
        zjp, vjp, ajp = node(i - m + 1)     # delayed at stage time t + dt
        # cubic Hermite midpoint for the delayed state at t + dt/2
        z_dh = 0.5 * (zj + zjp) + 0.125 * dt * (vj - vjp)
        v_dh = 0.5 * (vj + vjp) + 0.125 * dt * (aj - ajp)

        k1z, k1v = v[i], a[i]
        k2z = v[i] + 0.5 * dt * k1v
        k2v = accel(t + 0.5 * dt, z[i] + 0.5 * dt * k1z, k2z, z_dh, v_dh)
        k3z = v[i] + 0.5 * dt * k2v
        k3v = accel(t + 0.5 * dt, z[i] + 0.5 * dt * k2z, k3z, z_dh, v_dh)
        k4z = v[i] + dt * k3v
        k4v = accel(t + dt, z[i] + dt * k3z, k4z, zjp, vjp)
        z[i + 1] = z[i] + (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
        v[i + 1] = v[i] + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        if noise_std > 0.0:
            z[i + 1] += noise_std * rng.standard_normal()
        zjn, vjn, _ = node(i + 1 - m)
        a[i + 1] = accel((i + 1) * dt, z[i + 1], v[i + 1], zjn, vjn)
        if not np.isfinite(z[i + 1]) or abs(z[i + 1] - cfg.z_star) > escape:
            z = z[:i + 2]; v = v[:i + 2]
            break
    t_arr = dt * np.arange(len(z))
    return t_arr, z, v


# --------------------------------------------------------------------------- #
# Linear characteristic equation and Hopf conditions
# --------------------------------------------------------------------------- #
def char_eq(mu, zeta, kappa_p, kappa_d, theta):
    """Nondimensional characteristic function; roots are the growth rates."""
    return (mu ** 2 + 2.0 * zeta * mu - 1.0
            + (kappa_p + kappa_d * mu) * np.exp(-mu * theta))


def hopf_residuals(Omega, theta, zeta, kappa_p, kappa_d):
    """Real and imaginary parts of the characteristic function at mu = i Omega."""
    c = np.cos(Omega * theta)
    s = np.sin(Omega * theta)
    real = -Omega ** 2 - 1.0 + kappa_p * c + kappa_d * Omega * s
    imag = 2.0 * zeta * Omega - kappa_p * s + kappa_d * Omega * c
    return real, imag


def hopf_kappa_p_curve(zeta, kappa_d, Omega):
    """Closed-form Hopf boundary parametrised by the crossing frequency Omega.

    Solving the two residual conditions for the pair (kappa_p, theta) at a given
    Omega. Returns (kappa_p, theta) with theta taken in its principal branch.
    """
    # From the two conditions, treat (c, s) = (cos, sin)(Omega theta) as unknown
    # unit vector and solve the linear system
    #   kappa_p c + kappa_d Omega s = Omega^2 + 1
    #  -kappa_p s + kappa_d Omega c = -2 zeta Omega
    a = Omega ** 2 + 1.0
    b = -2.0 * zeta * Omega
    kdO = kappa_d * Omega
    # [ c  s ] [kappa_p c ; ...] is nonlinear; instead solve for c, s directly
    # kappa_p c + kdO s = a
    # -kappa_p s + kdO c = b
    # unknowns kappa_p, and angle phi = Omega theta with c=cos, s=sin.
    # Rearranged: kappa_p c = a - kdO s ; kappa_p s = kdO c - b
    # => (a - kdO s) s = (kdO c - b) c  ->  a s - kdO s^2 = kdO c^2 - b c
    # => a s + b c = kdO (c^2 + s^2) = kdO
    # A single linear relation: a sin phi + b cos phi = kdO
    # Solve for phi, then kappa_p from kappa_p = (a - kdO s)/c.
    R = np.hypot(a, b)
    if kdO / R > 1.0 or kdO / R < -1.0:
        return np.nan, np.nan
    delta = np.arctan2(b, a)
    phi = np.arcsin(np.clip(kdO / R, -1.0, 1.0)) - delta
    if phi <= 0:
        phi += 2.0 * np.pi
    c = np.cos(phi)
    s = np.sin(phi)
    if abs(c) < 1e-12:
        kappa_p = (kdO * c - b) / s
    else:
        kappa_p = (a - kdO * s) / c
    theta = phi / Omega
    return kappa_p, theta


# --------------------------------------------------------------------------- #
# Gold-standard stability spectrum: pseudospectral generator discretisation
# --------------------------------------------------------------------------- #
def _cheb(N):
    """Chebyshev Gauss-Lobatto nodes on [-1, 1] and differentiation matrix."""
    if N == 0:
        return np.array([1.0]), np.array([[0.0]])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = (np.outer(c, 1.0 / c)) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return x, D


def rightmost_roots(zeta, kappa_p, kappa_d, theta, N=30, n_return=8):
    """Rightmost characteristic roots via pseudospectral generator collocation.

    The linear DDE is written as y' = L0 y(t) + L1 y(t - theta) with y = (x, x').
    The infinitesimal generator is discretised on N+1 Chebyshev nodes mapped to
    [-theta, 0]; the node at s = 0 carries the boundary condition. Eigenvalues
    of the resulting matrix approximate the characteristic roots.
    """
    d = 2
    L0 = np.array([[0.0, 1.0], [1.0, -2.0 * zeta]])
    L1 = np.array([[0.0, 0.0], [-kappa_p, -kappa_d]])
    x, Dc = _cheb(N)
    # map x in [-1,1] to s in [-theta,0]: s = (theta/2)(x-1); d/ds = (2/theta) d/dx
    Ds = (2.0 / theta) * Dc
    M = np.kron(Ds, np.eye(d))
    # node 0 (x=1 -> s=0) boundary row: y'(0) = L0 y(0) + L1 y(-theta)
    M[0:d, :] = 0.0
    M[0:d, 0:d] = L0
    M[0:d, N * d:(N + 1) * d] = L1
    ev = np.linalg.eigvals(M)
    ev = ev[np.argsort(-ev.real)]
    return ev[:n_return]


def spectral_abscissa(zeta, kappa_p, kappa_d, theta, N=30):
    """Real part of the rightmost characteristic root (negative means stable)."""
    return float(rightmost_roots(zeta, kappa_p, kappa_d, theta, N=N,
                                 n_return=1)[0].real)


# --------------------------------------------------------------------------- #
# Ordinal analysis: permutation entropy and statistical complexity
# --------------------------------------------------------------------------- #
def _ordinal_distribution(x, dim, delay):
    """Bandt-Pompe ordinal pattern probabilities for embedding (dim, delay)."""
    x = np.asarray(x, dtype=float)
    n = len(x) - (dim - 1) * delay
    if n <= 0:
        raise ValueError("series too short for the requested embedding")
    idx = np.arange(dim) * delay
    from math import factorial
    counts = np.zeros(factorial(dim), dtype=float)
    # rank each window to a unique permutation index (Lehmer code)
    fact = [factorial(k) for k in range(dim)]
    for i in range(n):
        w = x[i + idx]
        order = np.argsort(w, kind="mergesort")
        # Lehmer code of the permutation "order"
        code = 0
        seen = []
        for j, o in enumerate(order):
            smaller = sum(1 for p in seen if p < o)
            code += (o - smaller) * fact[dim - 1 - j]
            seen.append(o)
        counts[code] += 1.0
    p = counts / counts.sum()
    return p


def permutation_entropy_complexity(x, dim=6, delay=1):
    """Normalised Bandt-Pompe entropy H and Rosso Jensen-Shannon complexity C.

    H is the Shannon entropy of the ordinal distribution divided by log of the
    number of admissible patterns. C = Q_J(P, U) * H, where Q_J is the
    Jensen-Shannon divergence to the uniform distribution normalised by its
    maximum. Definitions follow Bandt and Pompe and Rosso and colleagues.
    """
    from math import factorial
    p = _ordinal_distribution(x, dim, delay)
    Nfac = factorial(dim)
    nz = p[p > 0]
    S = -np.sum(nz * np.log(nz))
    H = S / np.log(Nfac)
    u = np.full(Nfac, 1.0 / Nfac)
    pm = 0.5 * (p + u)
    nzpm = pm[pm > 0]
    S_pm = -np.sum(nzpm * np.log(nzpm))
    S_p = -np.sum(nz * np.log(nz))
    S_u = np.log(Nfac)
    JS = S_pm - 0.5 * S_p - 0.5 * S_u
    Q0 = -0.5 * (((Nfac + 1.0) / Nfac) * np.log(Nfac + 1.0)
                 - 2.0 * np.log(2.0 * Nfac) + np.log(Nfac))
    Q = JS / Q0
    C = Q * H
    return float(H), float(C)


# --------------------------------------------------------------------------- #
# Plotting substrate: shared style, panel labels, single bottom legend
# --------------------------------------------------------------------------- #
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
SAVE_DPI = 600
SAVE_EXTS = ("pdf", "png", "eps")


def configure_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.titlesize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.labelsize": 11,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
    })


def panel_label(ax, letter, **kw):
    """Place a bold (letter) tag just outside the axes, above the top-left.

    Using the left title slot keeps the tag out of the data area and lets the
    layout engine reserve space for it, so it never overlaps neighbouring panels.
    """
    ax.set_title(f"({letter})", loc="left", fontweight="bold", fontsize=11,
                 pad=6, **kw)


def bottom_legend(fig, handles, labels, ncol=None, **kw):
    """Attach one shared legend in a reserved band beneath all panels.

    Requires the figure to use constrained layout so the band does not overlap
    the axis labels of the bottom row.
    """
    if ncol is None:
        ncol = min(len(labels), 4)
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol,
               frameon=False, handlelength=1.8, columnspacing=1.6)


def save_figure(fig, fig_dir, stem, **kwargs):
    """Save a figure trio (pdf, png, eps) at 600 dpi.

    Layout is handled by the figure's constrained layout, so no manual margin
    adjustment is applied here (that would fight the layout engine).
    """
    from pathlib import Path
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    for ext in SAVE_EXTS:
        fig.savefig(f"{fig_dir}/{stem}.{ext}", dpi=SAVE_DPI, bbox_inches="tight")


# --------------------------------------------------------------------------- #
# Text report helpers
# --------------------------------------------------------------------------- #
BAR = "=" * 76
SUB = "-" * 76


def report_header(title_lines, notes=None):
    from datetime import datetime
    L = [BAR]
    for t in title_lines:
        L.append(" " + t)
    L.append(SUB)
    L.append(" Author    : Sandy H. S. Herho <sandy.herho@email.ucr.edu>")
    L.append(f" Generated : {datetime.now().isoformat(timespec='seconds')}")
    if notes:
        L.append(SUB)
        for nt in notes:
            L.append(" NOTE: " + nt)
    L.append(BAR)
    L.append("")
    return L


def write_report(lines, out_path):
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
