#!/usr/bin/env python
"""
Ordinal analysis of the buoyancy regimes on the complexity-entropy plane.

Depth series are mapped to Bandt-Pompe permutation entropy and Rosso
Jensen-Shannon statistical complexity. Panel (a) places the four divers, a white
noise reference, and a clean oscillation on the plane bounded by the theoretical
minimum and maximum complexity curves. Panel (b) tracks the two measures as the
reaction delay is ramped through the Hopf onset, so the coordinate leaves the
stochastic corner and moves toward the periodic edge as porpoising sets in.

Output
    ../figures/complexity_entropy.{pdf,png,eps}   1x2 panel
    ../calculations/complexity_entropy.txt         numerical report

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
import ordpy

import scuba_buoyancy_model as M

FIG_DIR = Path("../figures")
CALC_DIR = Path("../calculations")
STEM = "complexity_entropy"

DIM, DELAY = 6, 1
DT_SAMPLE = 0.25
NOISE = 0.02          # observational depth noise, m


def depth_series(cfg, t_end, noise=NOISE, seed=0, dz0=1.0):
    t, z, v = M.integrate_dde(cfg, t_end=t_end, z0=cfg.z_star + dz0,
                              n_per_delay=60, saturate=True, noise_std=0.0,
                              seed=seed, escape=cfg.z_star + 6.0)
    dt = t[1] - t[0]
    step = max(1, int(round(DT_SAMPLE / dt)))
    zs = z[::step]
    rng = np.random.default_rng(seed)
    zs = zs + noise * rng.standard_normal(len(zs))
    return zs[len(zs) // 5:]     # drop the first fifth as transient


def onset_delay():
    nd = M.nondim(M.DIVERS["D1"])
    w0 = M.derived(M.DIVERS["D1"])["omega0"]
    th_c = brentq(lambda th: M.spectral_abscissa(nd["zeta"], nd["kappa_p"],
                  nd["kappa_d"], th), 0.05, 1.5, xtol=1e-8)
    return th_c / w0


def main():
    M.configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), layout="constrained")

    # Panel (a): the plane with theoretical bounds
    ax = axes[0]
    cmax = np.array(ordpy.maximum_complexity_entropy(dx=DIM))
    cmin = np.array(ordpy.minimum_complexity_entropy(dx=DIM))
    ax.plot(cmax[:, 0], cmax[:, 1], color="0.4", lw=0.9)
    ax.plot(cmin[:, 0], cmin[:, 1], color="0.4", lw=0.9)

    pts = []
    for i, (k, color) in enumerate(zip(M.DIVER_ORDER, M.PALETTE)):
        cfg = M.DIVERS[k]
        x = depth_series(cfg, t_end=2600.0, seed=10 + i)
        H, C = M.permutation_entropy_complexity(x, DIM, DELAY)
        ax.plot(H, C, "o", ms=7, color=color, zorder=5)
        off = {"D1": (6, 6), "D2": (6, -13), "D3": (7, 3), "D4": (7, 3)}
        ax.annotate(k, (H, C), textcoords="offset points",
                    xytext=off.get(k, (6, 4)), fontsize=8)
        pts.append((k, cfg.label, H, C))

    rng = np.random.default_rng(7)
    xw = rng.standard_normal(9000)
    Hw, Cw = M.permutation_entropy_complexity(xw, DIM, DELAY)
    (h_w,) = ax.plot(Hw, Cw, "s", ms=6, color="0.3", zorder=5, label="white noise")
    ts = np.arange(9000) * DT_SAMPLE
    xs = np.sin(2 * np.pi * ts / 28.0) + NOISE * rng.standard_normal(9000)
    Hs, Cs = M.permutation_entropy_complexity(xs, DIM, DELAY)
    (h_s,) = ax.plot(Hs, Cs, "^", ms=7, color="0.3", zorder=5, label="clean oscillation")

    ax.set_xlabel(r"permutation entropy $H$")
    ax.set_ylabel(r"statistical complexity $C$")
    ax.set_xlim(0, 1.02)
    M.panel_label(ax, "a")

    # Panel (b): H and C along a delay ramp through onset
    ax = axes[1]
    tau_c = onset_delay()
    taus = np.linspace(0.5 * tau_c, 1.5 * tau_c, 22)
    Hs_r, Cs_r = [], []
    for tau in taus:
        cfg = replace(M.DIVERS["D1"], tau=tau)
        x = depth_series(cfg, t_end=1800.0, seed=2, dz0=0.8)
        H, C = M.permutation_entropy_complexity(x, DIM, DELAY)
        Hs_r.append(H); Cs_r.append(C)
    Hs_r = np.array(Hs_r); Cs_r = np.array(Cs_r)
    (h_H,) = ax.plot(taus, Hs_r, "-o", ms=3, color=M.PALETTE[0], label=r"entropy $H$")
    ax.set_xlabel(r"reaction delay $\tau$ (s)")
    ax.set_ylabel(r"permutation entropy $H$", color=M.PALETTE[0])
    ax.tick_params(axis="y", labelcolor=M.PALETTE[0])
    axr = ax.twinx()
    (h_C,) = axr.plot(taus, Cs_r, "-s", ms=3, color=M.PALETTE[3], label=r"complexity $C$")
    axr.set_ylabel(r"statistical complexity $C$", color=M.PALETTE[3])
    axr.tick_params(axis="y", labelcolor=M.PALETTE[3])
    ax.axvline(tau_c, color="k", ls="--", lw=1.0)
    M.panel_label(ax, "b")

    handles = [h_w, h_s, h_H, h_C]
    labels = ["white noise", "clean oscillation", r"entropy $H$", r"complexity $C$"]
    M.bottom_legend(fig, handles, labels, ncol=4)
    M.save_figure(fig, FIG_DIR, STEM)
    plt.close(fig)

    # ----------------------------------------------------------------- report
    L = M.report_header(
        ["COMPLEXITY-ENTROPY ANALYSIS OF THE BUOYANCY REGIMES",
         "Bandt-Pompe entropy and Rosso Jensen-Shannon complexity"],
        notes=[
            f"Embedding dimension {DIM}, delay {DELAY}; series sampled at "
            f"{DT_SAMPLE:.2f} s.",
            f"Observational depth noise {NOISE:.3f} m added to every series.",
            "Bounds are the maximum and minimum complexity curves for this",
            "  embedding dimension. The native estimator is verified against the",
            "  ordpy reference implementation.",
        ])
    L.append(M.BAR)
    L.append(" PLANE COORDINATES")
    L.append(M.SUB)
    L.append("   key   H          C         label")
    for k, lab, H, C in pts:
        L.append(f"   {k}    {H:.5f}   {C:.5f}   {lab}")
    L.append(f"   WN    {Hw:.5f}   {Cw:.5f}   white noise")
    L.append(f"   OSC   {Hs:.5f}   {Cs:.5f}   clean oscillation")
    L.append("")
    L.append(M.BAR)
    L.append(f" DELAY RAMP THROUGH ONSET (tau_c = {tau_c:.4f} s)")
    L.append(M.SUB)
    L.append("   tau [s]     H          C")
    for tau, H, C in zip(taus, Hs_r, Cs_r):
        L.append(f"   {tau:8.4f}   {H:.5f}   {C:.5f}")
    L.append("")
    L.append(M.BAR)
    M.write_report(L, CALC_DIR / f"{STEM}.txt")
    print(f"[{STEM}] figure and report written")


if __name__ == "__main__":
    main()
