#!/usr/bin/env python
"""
Sensitivity of stability and onset to the physical parameters.

Around the baseline diver each physical parameter is perturbed by a small
relative amount and the resulting change in two targets is measured by central
finite differences: the spectral abscissa at the operating point, and the
critical delay at which porpoising begins. Results are reported as
semi-elasticities, the change per unit fractional change in the parameter, and
ranked. The drag shape parameters enter only the amplitude-limiting nonlinearity
and so leave the linear onset unchanged, which the ranking recovers.

Output
    ../figures/sensitivity.{pdf,png,eps}   1x2 tornado panel
    ../calculations/sensitivity.txt         numerical report

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
from matplotlib.patches import Patch

import scuba_buoyancy_model as M

FIG_DIR = Path("../figures")
CALC_DIR = Path("../calculations")
STEM = "sensitivity"

BASE = M.DIVERS["D1"]
PARAMS = ["m_eff", "V_inc", "V_s0", "Delta", "z_star", "C_d", "S", "c1",
          "tau", "kp", "kd"]
PLABEL = {"m_eff": r"$m_{\rm eff}$", "V_inc": r"$V_{\rm inc}$",
          "V_s0": r"$V_{s0}$", "Delta": r"$\Delta$", "z_star": r"$z^\ast$",
          "C_d": r"$C_d$", "S": r"$S$", "c1": r"$c_1$", "tau": r"$\tau$",
          "kp": r"$k_p$", "kd": r"$k_d$"}
REL = 1e-3


def abscissa_of(cfg):
    nd = M.nondim(cfg)
    return M.spectral_abscissa(nd["zeta"], nd["kappa_p"], nd["kappa_d"],
                               nd["theta"])


def tau_c_of(cfg):
    nd = M.nondim(cfg)
    w0 = M.derived(cfg)["omega0"]
    f = lambda th: M.spectral_abscissa(nd["zeta"], nd["kappa_p"], nd["kappa_d"], th)
    try:
        th_c = brentq(f, 0.02, 3.0, xtol=1e-8)
    except ValueError:
        return np.nan
    return th_c / w0


def semi_elasticity(target, pname):
    p0 = getattr(BASE, pname)
    hp = replace(BASE, **{pname: p0 * (1.0 + REL)})
    hm = replace(BASE, **{pname: p0 * (1.0 - REL)})
    return (target(hp) - target(hm)) / (2.0 * REL)


def main():
    M.configure_style()

    sa = {p: semi_elasticity(abscissa_of, p) for p in PARAMS}
    tc = {p: semi_elasticity(tau_c_of, p) for p in PARAMS if p != "tau"}

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), layout="constrained")
    pos_c = M.PALETTE[3]
    neg_c = M.PALETTE[0]

    # Panel (a): spectral abscissa semi-elasticity
    ax = axes[0]
    order = sorted(sa, key=lambda k: abs(sa[k]))
    y = np.arange(len(order))
    ax.barh(y, [sa[k] for k in order],
            color=[pos_c if sa[k] >= 0 else neg_c for k in order])
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([PLABEL[k] for k in order])
    ax.set_xlabel(r"$\partial(\max\Re\,\mu)/\partial\ln p$")
    M.panel_label(ax, "a")

    # Panel (b): critical delay semi-elasticity
    ax = axes[1]
    order2 = sorted(tc, key=lambda k: abs(tc[k]))
    y2 = np.arange(len(order2))
    ax.barh(y2, [tc[k] for k in order2],
            color=[pos_c if tc[k] >= 0 else neg_c for k in order2])
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(y2); ax.set_yticklabels([PLABEL[k] for k in order2])
    ax.set_xlabel(r"$\partial\tau_c/\partial\ln p$ (s)")
    M.panel_label(ax, "b")

    handles = [Patch(color=neg_c), Patch(color=pos_c)]
    M.bottom_legend(fig, handles, ["stabilising (negative)",
                                   "destabilising (positive)"], ncol=2)
    M.save_figure(fig, FIG_DIR, STEM)
    plt.close(fig)

    # ----------------------------------------------------------------- report
    a0 = abscissa_of(BASE)
    t0 = tau_c_of(BASE)
    L = M.report_header(
        ["PARAMETER SENSITIVITY OF STABILITY AND ONSET (baseline diver)",
         "Central-difference semi-elasticities, change per fractional change"],
        notes=[
            f"Baseline spectral abscissa = {a0:+.6f}; baseline tau_c = {t0:.6f} s.",
            f"Relative perturbation {REL:.0e}. Positive abscissa sensitivity is",
            "  destabilising; positive tau_c sensitivity widens the stable range.",
            "C_d and S act only through quadratic drag, absent from the linear",
            "  operator, so their linear sensitivities are numerically zero.",
        ])
    L.append(M.BAR)
    L.append(" SPECTRAL ABSCISSA SEMI-ELASTICITY (ranked)")
    L.append(M.SUB)
    for k in sorted(sa, key=lambda k: -abs(sa[k])):
        L.append(f"   {k:8s} = {sa[k]:+.6e}")
    L.append("")
    L.append(M.BAR)
    L.append(" CRITICAL DELAY SEMI-ELASTICITY (ranked)")
    L.append(M.SUB)
    for k in sorted(tc, key=lambda k: -abs(tc[k])):
        L.append(f"   {k:8s} = {tc[k]:+.6e} s")
    L.append("")
    L.append(M.BAR)
    M.write_report(L, CALC_DIR / f"{STEM}.txt")
    print(f"[{STEM}] figure and report written")


if __name__ == "__main__":
    main()
