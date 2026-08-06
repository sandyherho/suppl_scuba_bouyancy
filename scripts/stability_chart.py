#!/usr/bin/env python
"""
Linear stability chart and the porpoising onset.

The closed loop loses stability through a Hopf bifurcation as the reaction delay
or the proportional gain grows. Panel (a) maps the spectral abscissa, the real
part of the rightmost characteristic root obtained by pseudospectral generator
collocation, over the dimensionless delay and proportional gain, with the Hopf
boundary and the four diver operating points overlaid. Panel (b) gives the
onset frequency and the corresponding porpoising period along the boundary.

Output
    ../figures/stability_chart.{pdf,png,eps}   1x2 panel
    ../calculations/stability_chart.txt         numerical report

Author: Sandy H. S. Herho <sandy.herho@email.ucr.edu>
Date: 2026-08-01
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import scuba_buoyancy_model as M

FIG_DIR = Path("../figures")
CALC_DIR = Path("../calculations")
STEM = "stability_chart"

# representative slice for the background map (baseline diver)
ND = M.nondim(M.DIVERS["D1"])
ZETA_SLICE = ND["zeta"]
KD_SLICE = ND["kappa_d"]


def abscissa_map(theta_grid, kp_grid, N=24):
    A = np.empty((len(kp_grid), len(theta_grid)))
    for i, kp in enumerate(kp_grid):
        for j, th in enumerate(theta_grid):
            A[i, j] = M.spectral_abscissa(ZETA_SLICE, kp, KD_SLICE, th, N=N)
    return A


def analytic_boundary(zeta, kappa_d, Omega_grid):
    kp, th, Om = [], [], []
    for Omega in Omega_grid:
        kpc, thc = M.hopf_kappa_p_curve(zeta, kappa_d, Omega)
        if np.isfinite(kpc) and np.isfinite(thc) and thc > 0:
            kp.append(kpc); th.append(thc); Om.append(Omega)
    return np.array(th), np.array(kp), np.array(Om)


def main():
    M.configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), layout="constrained")

    # Panel (a): spectral-abscissa map with the Hopf boundary
    ax = axes[0]
    theta_grid = np.linspace(0.05, 1.6, 100)
    kp_grid = np.linspace(0.2, 14.0, 100)
    A = abscissa_map(theta_grid, kp_grid)
    vmax = np.nanmax(np.abs(A))
    pcm = ax.pcolormesh(theta_grid, kp_grid, A, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    cs = ax.contour(theta_grid, kp_grid, A, levels=[0.0], colors="k",
                    linewidths=1.2)
    th_b, kp_b, Om_b = analytic_boundary(ZETA_SLICE, KD_SLICE,
                                         np.linspace(0.05, 3.0, 400))
    ax.plot(th_b, kp_b, color="k", ls="--", lw=1.0)
    for k, color in zip(M.DIVER_ORDER, M.PALETTE):
        nd = M.nondim(M.DIVERS[k])
        sa = M.spectral_abscissa(nd["zeta"], nd["kappa_p"], nd["kappa_d"],
                                 nd["theta"])
        filled = sa < 0
        ax.plot(nd["theta"], nd["kappa_p"], marker="o", ms=8, mfc=(color if filled else "white"),
                mec=color, mew=1.6, ls="none", zorder=6)
        ax.annotate(k, (nd["theta"], nd["kappa_p"]), textcoords="offset points",
                    xytext=(6, 5), fontsize=8)
    ax.set_xlabel(r"dimensionless delay $\theta$")
    ax.set_ylabel(r"proportional gain $\kappa_p$")
    ax.set_xlim(theta_grid[0], theta_grid[-1])
    ax.set_ylim(kp_grid[0], kp_grid[-1])
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"spectral abscissa $\max\Re\,\mu$", fontsize=9)
    cb.ax.tick_params(labelsize=7)
    M.panel_label(ax, "a")

    # Panel (b): onset frequency and porpoising period along the boundary
    ax = axes[1]
    order = np.argsort(th_b)
    thb, Omb = th_b[order], Om_b[order]
    w0 = M.derived(M.DIVERS["D1"])["omega0"]
    (h1,) = ax.plot(thb, Omb, color=M.PALETTE[0], lw=1.6,
                    label=r"onset frequency $\Omega_H$")
    ax.set_xlabel(r"dimensionless delay $\theta$")
    ax.set_ylabel(r"onset frequency $\Omega_H$", color=M.PALETTE[0])
    ax.tick_params(axis="y", labelcolor=M.PALETTE[0])
    ax.set_xlim(thb.min(), thb.max())
    axr = ax.twinx()
    period = 2.0 * np.pi / (w0 * Omb)
    (h2,) = axr.plot(thb, period, color=M.PALETTE[3], lw=1.6, ls="--",
                     label=r"porpoising period $T$ (s)")
    axr.set_ylabel(r"porpoising period $T$ (s)", color=M.PALETTE[3])
    axr.tick_params(axis="y", labelcolor=M.PALETTE[3])
    axr.spines["top"].set_visible(True)
    M.panel_label(ax, "b")

    M.bottom_legend(fig, [h1, h2],
                    [r"onset frequency $\Omega_H$", r"porpoising period $T$ (s)"],
                    ncol=2)
    M.save_figure(fig, FIG_DIR, STEM)
    plt.close(fig)

    # ----------------------------------------------------------------- report
    L = M.report_header(
        ["LINEAR STABILITY CHART AND PORPOISING ONSET",
         "Hopf boundary from pseudospectral generator collocation"],
        notes=[
            f"Background map fixes zeta = {ZETA_SLICE:.4f} and kappa_d = {KD_SLICE:.4f}",
            "  (the baseline diver values); the solid black curve is the zero",
            "  level of the pseudospectral spectral abscissa and the dashed curve",
            "  is the closed-form Hopf boundary. Diver markers use each diver's",
            "  own kappa_d, so a marker may sit slightly off the fixed-slice curve.",
            "Porpoising period uses the baseline omega0 = "
            f"{w0:.6f} s^-1 for the T axis.",
        ])
    L.append(M.BAR)
    L.append(" HOPF BOUNDARY SAMPLE (fixed-slice kappa_d, zeta)")
    L.append(M.SUB)
    L.append("   theta      kappa_p,crit   Omega_H      T_onset [s]")
    for th, kp, Om in zip(thb[::40], kp_b[order][::40], Omb[::40]):
        L.append(f"   {th:7.4f}    {kp:9.4f}    {Om:8.4f}    {2*np.pi/(w0*Om):9.4f}")
    L.append("")
    L.append(M.BAR)
    L.append(" PER-DIVER CLASSIFICATION (own kappa_d)")
    L.append(M.SUB)
    for k in M.DIVER_ORDER:
        cfg = M.DIVERS[k]
        nd = M.nondim(cfg)
        r = M.rightmost_roots(nd["zeta"], nd["kappa_p"], nd["kappa_d"],
                              nd["theta"], n_return=2)
        sa = r[0].real
        cls = "STABLE (hover)" if sa < 0 else "UNSTABLE (porpoising)"
        L.append(f" {k}: {cfg.label}")
        L.append(f"    theta={nd['theta']:.4f}  kappa_p={nd['kappa_p']:.4f}  "
                 f"kappa_d={nd['kappa_d']:.4f}  zeta={nd['zeta']:.4f}")
        L.append(f"    rightmost root = {sa:+.5f} {r[0].imag:+.5f}i   -> {cls}")
        if abs(r[0].imag) > 1e-6:
            w0k = M.derived(cfg)["omega0"]
            L.append(f"    associated period 2pi/(omega0 |Im|) = "
                     f"{2*np.pi/(w0k*abs(r[0].imag)):.3f} s")
        L.append("")
    L.append(M.BAR)
    M.write_report(L, CALC_DIR / f"{STEM}.txt")
    print(f"[{STEM}] figure and report written")


if __name__ == "__main__":
    main()
