#!/usr/bin/env python
"""
Open-loop instability of the uncontrolled buoyancy plant.

With the carried gas held fixed at its neutral value the equilibrium at the
target depth is a saddle: a downward excursion compresses the gas, reduces
buoyancy, and accelerates the descent, while an upward excursion runs away in
the opposite sense. This script draws the uncontrolled phase portrait for the
baseline diver and the exponential divergence of small perturbations for all
four configurations, and reports the saddle eigenvalues and e-folding times.

Output
    ../figures/open_loop_instability.{pdf,png,eps}   1x2 panel
    ../calculations/open_loop_instability.txt         numerical report

Author: Sandy H. S. Herho <sandy.herho@email.ucr.edu>
Date: 2026-08-01
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scuba_buoyancy_model as M

FIG_DIR = Path("../figures")
CALC_DIR = Path("../calculations")
STEM = "open_loop_instability"


def open_loop_accel(cfg, z, v):
    """Acceleration with gas fixed at the neutral load (no control)."""
    q = M.derived(cfg)["q_star"]
    return M.plant_acceleration(cfg, z, v, q)


def open_loop_eigs(cfg):
    """Saddle eigenvalues of the uncontrolled linearisation, s^-1."""
    d = M.derived(cfg)
    w0 = d["omega0"]
    zeta = cfg.c1 / (2.0 * cfg.m_eff * w0)
    # x'' + 2 zeta w0 x' - w0^2 x = 0  ->  lam = w0(-zeta +/- sqrt(zeta^2+1))
    r = np.sqrt(zeta ** 2 + 1.0)
    return w0 * (-zeta + r), w0 * (-zeta - r)


def integrate_open_loop(cfg, dz0, t_end, dt):
    n = int(t_end / dt)
    z = np.empty(n + 1)
    v = np.empty(n + 1)
    z[0] = cfg.z_star + dz0
    v[0] = 0.0
    for i in range(n):
        k1z, k1v = v[i], open_loop_accel(cfg, z[i], v[i])
        k2z, k2v = v[i] + 0.5 * dt * k1v, open_loop_accel(cfg, z[i] + 0.5 * dt * k1z, v[i] + 0.5 * dt * k1v)
        k3z, k3v = v[i] + 0.5 * dt * k2v, open_loop_accel(cfg, z[i] + 0.5 * dt * k2z, v[i] + 0.5 * dt * k2v)
        k4z, k4v = v[i] + dt * k3v, open_loop_accel(cfg, z[i] + dt * k3z, v[i] + dt * k3v)
        z[i + 1] = z[i] + (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
        v[i + 1] = v[i] + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        if abs(z[i + 1] - cfg.z_star) > 60.0:
            z = z[:i + 2]; v = v[:i + 2]; break
    return dt * np.arange(len(z)), z, v


def main():
    M.configure_style()
    cfg = M.DIVERS["D1"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), layout="constrained")

    # Panel (a): uncontrolled phase portrait for the baseline diver
    ax = axes[0]
    zr = np.linspace(-8, 8, 41)
    vr = np.linspace(-1.2, 1.2, 41)
    ZZ, VV = np.meshgrid(zr, vr)
    AA = open_loop_accel(cfg, cfg.z_star + ZZ, VV)
    speed = np.hypot(VV, AA)
    ax.streamplot(zr, vr, VV, AA, density=1.1, color="0.7", linewidth=0.6,
                  arrowsize=0.7)
    for dz0 in (-4.0, -2.0, 2.0, 4.0):
        _, z, v = integrate_open_loop(cfg, dz0, t_end=40.0, dt=0.01)
        ax.plot(z - cfg.z_star, v, color=M.PALETTE[0], lw=1.4)
    ax.plot(0, 0, "o", color=M.PALETTE[3], ms=6, zorder=5)
    ax.set_xlabel(r"depth error $z-z^\ast$ (m)")
    ax.set_ylabel(r"vertical velocity $\dot z$ (m s$^{-1}$)")
    ax.set_xlim(-8, 8); ax.set_ylim(-1.2, 1.2)
    M.panel_label(ax, "a")

    # Panel (b): exponential divergence of small perturbations.
    # The open-loop plant ignores the controller, so divers that share hull,
    # gas, and weighting have identical open-loop dynamics; one curve per plant.
    ax = axes[1]
    groups = {}
    order = []
    for k in M.DIVER_ORDER:
        c = M.DIVERS[k]
        sig = (c.m_eff, c.V_inc, c.V_s0, c.Delta, c.z_star, c.c1)
        if sig not in groups:
            groups[sig] = []
            order.append(sig)
        groups[sig].append(k)
    handles, labels = [], []
    for sig, color in zip(order, M.PALETTE):
        ks = groups[sig]
        cfg_k = M.DIVERS[ks[0]]
        t, z, v = integrate_open_loop(cfg_k, 0.05, t_end=90.0, dt=0.01)
        (h,) = ax.plot(t, np.abs(z - cfg_k.z_star), color=color, lw=1.4)
        lam_p, _ = open_loop_eigs(cfg_k)
        ax.plot(t, 0.05 * np.exp(lam_p * t), color=color, lw=0.8, ls="--")
        handles.append(h)
        labels.append(f"{', '.join(ks)}: {cfg_k.label}")
    ax.set_yscale("log")
    ax.set_xlabel(r"time $t$ (s)")
    ax.set_ylabel(r"$|z-z^\ast|$ (m)")
    ax.set_xlim(0, 90); ax.set_ylim(1e-2, 1e2)
    M.panel_label(ax, "b")

    M.bottom_legend(fig, handles, labels, ncol=len(labels))
    M.save_figure(fig, FIG_DIR, STEM)
    plt.close(fig)

    # ----------------------------------------------------------------- report
    L = M.report_header(
        ["OPEN-LOOP INSTABILITY OF THE UNCONTROLLED BUOYANCY PLANT",
         "Saddle structure with the carried gas fixed at the neutral load"],
        notes=[
            "The dashed lines in panel (b) are the analytic growth envelopes",
            "  0.05 * exp(lambda_+ t) with lambda_+ the positive saddle eigenvalue.",
            "Eigenvalues are of x'' + 2 zeta omega0 x' - omega0^2 x = 0, so the",
            "  equilibrium is a saddle for every configuration (one positive root).",
            "Open-loop dynamics ignore the controller, so D1, D3, and D4 share one",
            "  plant and coincide in panel (b); D2 is the only distinct plant.",
        ])
    for k in M.DIVER_ORDER:
        cfg_k = M.DIVERS[k]
        d = M.derived(cfg_k)
        lam_p, lam_m = open_loop_eigs(cfg_k)
        L.append(M.BAR)
        L.append(f" {k}: {cfg_k.label}   (z* = {cfg_k.z_star:.1f} m)")
        L.append(M.SUB)
        L.append(f"   Neutral gas load q*        [L, surface]   = {d['q_star']*1e3:10.4f}")
        L.append(f"   Total compressible G*      [L, surface]   = {d['G_star']*1e3:10.4f}")
        L.append(f"   Negative stiffness beta    [N m^-1]       = {d['beta']:10.6f}")
        L.append(f"   Buoyancy authority gamma   [N m^-3]       = {d['gamma']:10.2f}")
        L.append(f"   Instability rate omega0    [s^-1]         = {d['omega0']:10.6f}")
        L.append(f"   Saddle eigenvalue lambda_+ [s^-1]         = {lam_p:+10.6f}")
        L.append(f"   Saddle eigenvalue lambda_- [s^-1]         = {lam_m:+10.6f}")
        L.append(f"   e-folding time 1/lambda_+  [s]            = {1.0/lam_p:10.4f}")
        L.append("")
    L.append(M.BAR)
    M.write_report(L, CALC_DIR / f"{STEM}.txt")
    print(f"[{STEM}] figure and report written")


if __name__ == "__main__":
    main()
