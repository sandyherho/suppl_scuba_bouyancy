#!/usr/bin/env python
"""
Critical slowing down as an early warning of porpoising onset.

As the reaction delay approaches the Hopf boundary from the stable side the
rightmost characteristic root drifts toward the imaginary axis, so perturbations
decay ever more slowly. Under weak observational noise this raises the lag-one
autocorrelation and the variance of the depth signal and lowers the measured
recovery rate. Each indicator is computed from stationary runs at a sequence of
delays and plotted against the spectral abscissa, the exact distance to onset.

Output
    ../figures/early_warning.{pdf,png,eps}   2x2 panel
    ../calculations/early_warning.txt         numerical report

Author: Sandy H. S. Herho <sandy.herho@email.ucr.edu>
Date: 2026-08-01
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import kendalltau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import scuba_buoyancy_model as M

FIG_DIR = Path("../figures")
CALC_DIR = Path("../calculations")
STEM = "early_warning"

BASE = M.DIVERS["D1"]
DT_SAMPLE = 0.5
NOISE = 0.03


def onset_delay():
    nd = M.nondim(BASE)
    w0 = M.derived(BASE)["omega0"]
    th_c = brentq(lambda th: M.spectral_abscissa(nd["zeta"], nd["kappa_p"],
                  nd["kappa_d"], th), 0.05, 1.5, xtol=1e-8)
    return th_c / w0, w0


def stationary_indicators(cfg, seed=0):
    t, z, v = M.integrate_dde(cfg, t_end=3000.0, z0=cfg.z_star, v0=0.0,
                              n_per_delay=60, saturate=False,
                              noise_std=NOISE * 0.02, seed=seed,
                              escape=cfg.z_star + 6.0)
    dt = t[1] - t[0]
    step = max(1, int(round(DT_SAMPLE / dt)))
    x = z[::step]
    x = x[len(x) // 4:] - np.mean(x[len(x) // 4:])
    var = float(np.var(x))
    ar1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    return ar1, var, x


def measured_recovery(cfg):
    """Asymptotic decay rate (s^-1) from the tail of a perturbation envelope.

    The early transient carries faster-decaying modes, so the envelope is fitted
    over its later portion to isolate the rightmost characteristic mode, which is
    what the spectral abscissa predicts.
    """
    t, z, v = M.integrate_dde(cfg, t_end=900.0, z0=cfg.z_star + 1.0, v0=0.0,
                              n_per_delay=80, saturate=False,
                              escape=cfg.z_star + 6.0)
    x = np.abs(z - cfg.z_star)
    idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    idx = idx[x[idx] > 1e-7]
    if len(idx) < 5:
        return np.nan
    tail = idx[len(idx) // 3:]          # drop the early multi-mode transient
    coef = np.polyfit(t[tail], np.log(x[tail]), 1)
    return -coef[0]


def main():
    M.configure_style()
    tau_c, w0 = onset_delay()
    taus = np.linspace(0.45 * tau_c, 0.97 * tau_c, 14)

    absc, ar1s, vars_, recs = [], [], [], []
    examples = {}
    example_taus = [taus[0], taus[len(taus) // 2], taus[-1]]
    for tau in taus:
        cfg = replace(BASE, tau=tau)
        nd = M.nondim(cfg)
        a = M.spectral_abscissa(nd["zeta"], nd["kappa_p"], nd["kappa_d"],
                                nd["theta"])
        ar1, var, x = stationary_indicators(cfg, seed=3)
        rec = measured_recovery(cfg)
        absc.append(a); ar1s.append(ar1); vars_.append(var); recs.append(rec)
        if any(abs(tau - te) < 1e-9 for te in example_taus):
            examples[tau] = x
    absc = np.array(absc); ar1s = np.array(ar1s)
    vars_ = np.array(vars_); recs = np.array(recs)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), layout="constrained")

    # (a) example stationary series at three distances to onset
    ax = axes[0, 0]
    colors = [M.PALETTE[0], M.PALETTE[2], M.PALETTE[3]]
    for (tau, x), color in zip(sorted(examples.items()), colors):
        tt = np.arange(len(x)) * DT_SAMPLE
        ax.plot(tt[:800], x[:800] + 0, color=color, lw=0.8,
                label=fr"$\tau={tau:.2f}$ s")
    ax.set_xlabel(r"time $t$ (s)")
    ax.set_ylabel(r"depth fluctuation (m)")
    M.panel_label(ax, "a")

    # (b) lag-1 autocorrelation vs distance to onset
    ax = axes[0, 1]
    ax.plot(-absc, ar1s, "-o", ms=4, color=M.PALETTE[0])
    ax.set_xlabel(r"distance to onset $-\max\Re\,\mu$")
    ax.set_ylabel(r"lag-1 autocorrelation")
    ax.invert_xaxis()
    M.panel_label(ax, "b")

    # (c) variance vs distance to onset
    ax = axes[1, 0]
    ax.plot(-absc, vars_, "-o", ms=4, color=M.PALETTE[1])
    ax.set_xlabel(r"distance to onset $-\max\Re\,\mu$")
    ax.set_ylabel(r"depth variance (m$^2$)")
    ax.invert_xaxis()
    M.panel_label(ax, "c")

    # (d) measured recovery rate vs spectral prediction
    ax = axes[1, 1]
    pred = -absc * w0
    ax.plot(pred, recs, "o", ms=5, color=M.PALETTE[3])
    lim = [0, np.nanmax(pred) * 1.05]
    ax.plot(lim, lim, color="k", lw=0.9, ls="--")
    ax.set_xlabel(r"predicted rate $-\omega_0\,\max\Re\,\mu$ (s$^{-1}$)")
    ax.set_ylabel(r"measured recovery rate (s$^{-1}$)")
    M.panel_label(ax, "d")

    handles = [Line2D([0], [0], color=colors[i], lw=1.2) for i in range(3)]
    labels = [fr"$\tau={tau:.2f}$ s" for tau in sorted(examples)]
    handles.append(Line2D([0], [0], color="k", ls="--", lw=0.9))
    labels.append("1:1 line")
    M.bottom_legend(fig, handles, labels, ncol=4)
    M.save_figure(fig, FIG_DIR, STEM)
    plt.close(fig)

    # trend statistics against distance to onset (increasing as onset nears)
    d = -absc
    tau_ar, p_ar = kendalltau(-d, ar1s)
    tau_var, p_var = kendalltau(-d, vars_)
    tau_rec, p_rec = kendalltau(-d, recs)

    # ----------------------------------------------------------------- report
    L = M.report_header(
        ["CRITICAL SLOWING DOWN AS EARLY WARNING OF ONSET",
         "Autocorrelation, variance, and recovery rate versus distance to onset"],
        notes=[
            "Delays span the stable side up to 0.97 of the critical delay.",
            "Indicators are computed from stationary runs under weak noise; the",
            "  recovery rate is fitted to the envelope of a deterministic decay",
            "  and compared with the pseudospectral prediction.",
            "Kendall tau is taken against decreasing distance to onset, so a",
            "  positive value means the indicator rises as onset is approached.",
        ])
    L.append(M.BAR)
    L.append(f"   Critical delay tau_c            [s]   = {tau_c:.6f}")
    L.append(f"   Kendall tau, autocorrelation trend    = {tau_ar:+.4f}  (p={p_ar:.3g})")
    L.append(f"   Kendall tau, variance trend           = {tau_var:+.4f}  (p={p_var:.3g})")
    L.append(f"   Kendall tau, recovery-rate trend      = {tau_rec:+.4f}  (p={p_rec:.3g})")
    L.append("")
    L.append(M.BAR)
    L.append("   tau [s]   spectral_abscissa   AR(1)      variance    recovery[s^-1]")
    for tau, a, ar, vr, rc in zip(taus, absc, ar1s, vars_, recs):
        L.append(f"   {tau:7.4f}   {a:+.5f}          {ar:.4f}    {vr:.3e}   {rc:.5f}")
    L.append("")
    L.append(M.BAR)
    M.write_report(L, CALC_DIR / f"{STEM}.txt")
    print(f"[{STEM}] figure and report written")


if __name__ == "__main__":
    main()
