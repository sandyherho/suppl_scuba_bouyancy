#!/usr/bin/env python
"""
Hopf bifurcation of the controlled plant as the reaction delay grows.

For the baseline diver the neutral hover is stable at short delay and gives way
to a bounded porpoising limit cycle as the delay crosses a critical value. Panel
(a) is the bifurcation diagram, the steady oscillation amplitude against delay,
with the pseudospectral onset marked. Panel (b) tests the criticality: near
onset the squared amplitude grows linearly in the delay excess, the signature of
a supercritical Hopf whose amplitude is limited by quadratic drag.

Output
    ../figures/bifurcation.{pdf,png,eps}   1x2 panel
    ../calculations/bifurcation.txt         numerical report

Author: Sandy H. S. Herho <sandy.herho@email.ucr.edu>
Date: 2026-08-01
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import scuba_buoyancy_model as M

FIG_DIR = Path("../figures")
CALC_DIR = Path("../calculations")
STEM = "bifurcation"

BASE = M.DIVERS["D1"]


def onset_delay():
    """Critical delay tau_c where the spectral abscissa vanishes (baseline)."""
    nd = M.nondim(BASE)
    w0 = M.derived(BASE)["omega0"]
    f = lambda th: M.spectral_abscissa(nd["zeta"], nd["kappa_p"],
                                       nd["kappa_d"], th)
    th_c = brentq(f, 0.05, 1.5, xtol=1e-8)
    return th_c / w0, th_c, w0


def limit_amplitude(tau, t_end=1400.0, tail=0.30):
    """Steady peak-to-peak amplitude of z at a given delay."""
    cfg = replace(BASE, tau=tau)
    t, z, v = M.integrate_dde(cfg, t_end=t_end, z0=BASE.z_star + 0.8,
                              n_per_delay=80, saturate=True, escape=40.0)
    if len(z) < 10:
        return np.nan
    seg = z[int((1 - tail) * len(z)):]
    return float(np.ptp(seg))


def main():
    M.configure_style()
    tau_c, th_c, w0 = onset_delay()

    taus = np.linspace(0.6 * tau_c, 1.7 * tau_c, 26)
    amps = np.array([limit_amplitude(t) for t in taus])
    half_amp = 0.5 * amps

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), layout="constrained")

    # Panel (a): bifurcation diagram
    ax = axes[0]
    stable = taus < tau_c
    (h_st,) = ax.plot(taus[stable], half_amp[stable], "o", ms=4,
                      color=M.PALETTE[0], label="stable hover")
    (h_lc,) = ax.plot(taus[~stable], half_amp[~stable], "o", ms=4,
                      color=M.PALETTE[3], label="porpoising amplitude")
    (h_on,) = ax.plot([tau_c, tau_c], [0, np.nanmax(half_amp) * 1.05],
                      color="k", ls="--", lw=1.0, label="Hopf onset")
    ax.set_xlabel(r"reaction delay $\tau$ (s)")
    ax.set_ylabel(r"oscillation amplitude (m)")
    ax.set_ylim(bottom=-0.02)
    M.panel_label(ax, "a")

    # Panel (b): supercriticality, amplitude squared vs delay excess
    ax = axes[1]
    mask = (~stable) & np.isfinite(half_amp) & (half_amp > 1e-3)
    dex = taus[mask] - tau_c
    a2 = half_amp[mask] ** 2
    near = dex < 0.5 * (taus.max() - tau_c)
    coef = np.polyfit(dex[near], a2[near], 1)
    fit = np.poly1d(coef)
    ss_res = np.sum((a2[near] - fit(dex[near])) ** 2)
    ss_tot = np.sum((a2[near] - a2[near].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    (h_pt,) = ax.plot(dex, a2, "o", ms=4, color=M.PALETTE[3],
                      label=r"$A^2$ (numerical)")
    xx = np.linspace(0, dex.max(), 50)
    (h_ft,) = ax.plot(xx, fit(xx), color="k", lw=1.2,
                      label="linear fit near onset")
    ax.set_xlabel(r"delay excess $\tau-\tau_c$ (s)")
    ax.set_ylabel(r"squared amplitude $A^2$ (m$^2$)")
    ax.set_xlim(left=0)
    M.panel_label(ax, "b")

    handles = [h_st, h_lc, h_on, h_ft]
    labels = ["stable hover", "porpoising amplitude", "Hopf onset",
              "linear fit near onset"]
    M.bottom_legend(fig, handles, labels, ncol=4)
    M.save_figure(fig, FIG_DIR, STEM)
    plt.close(fig)

    # ----------------------------------------------------------------- report
    nd = M.nondim(BASE)
    r = M.rightmost_roots(nd["zeta"], nd["kappa_p"], nd["kappa_d"], th_c,
                          n_return=1)
    Omega_H = abs(r[0].imag)
    L = M.report_header(
        ["HOPF BIFURCATION OF THE CONTROLLED PLANT (baseline diver)",
         "Amplitude branch versus reaction delay and criticality test"],
        notes=[
            "Amplitude is the peak-to-peak of z over the final 30 percent of a",
            f"  {1400:.0f} s integration; half of it is plotted as the amplitude.",
            "Onset tau_c is the delay where the pseudospectral spectral abscissa",
            "  crosses zero. A positive linear slope of A^2 versus (tau - tau_c)",
            "  near onset indicates a supercritical Hopf; the drag nonlinearity",
            "  sets the saturated amplitude.",
        ])
    L.append(M.BAR)
    L.append(f"   Onset delay tau_c            [s]   = {tau_c:.6f}")
    L.append(f"   Onset dimensionless theta_c       = {th_c:.6f}")
    L.append(f"   Onset frequency Omega_H           = {Omega_H:.6f}")
    L.append(f"   Onset porpoising period 2pi/(w0 Omega_H) [s] = {2*np.pi/(w0*Omega_H):.4f}")
    L.append(f"   Supercriticality slope dA^2/d(tau) [m^2 s^-1] = {coef[0]:.6f}")
    L.append(f"   Linear fit R^2 near onset         = {r2:.5f}")
    L.append(f"   Criticality verdict               = "
             f"{'supercritical' if coef[0] > 0 else 'not supercritical'}")
    L.append("")
    L.append(M.BAR)
    L.append("   tau [s]     amplitude (m)   regime")
    for t, a in zip(taus, half_amp):
        reg = "hover" if t < tau_c else "porpoising"
        L.append(f"   {t:8.4f}    {a:10.5f}    {reg}")
    L.append("")
    L.append(M.BAR)
    M.write_report(L, CALC_DIR / f"{STEM}.txt")
    print(f"[{STEM}] figure and report written  (tau_c = {tau_c:.4f} s)")


if __name__ == "__main__":
    main()
