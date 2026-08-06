#!/usr/bin/env python
"""
The three closed-loop regimes: hover, porpoising, and runaway.

A well-tuned diver settles to neutral (stable hover). A poorly tuned diver past
the Hopf boundary holds a bounded porpoising limit cycle. A diver started
outside the recoverable set saturates the compensator and escapes: here a
shallow, rising diver runs away to the surface, the uncontrolled ascent. Each
row shows the depth history and the phase portrait for one regime.

Output
    ../figures/regime_gallery.{pdf,png,eps}   3x2 panel
    ../calculations/regime_gallery.txt         numerical report

Author: Sandy H. S. Herho <sandy.herho@email.ucr.edu>
Date: 2026-08-01
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import scuba_buoyancy_model as M

FIG_DIR = Path("../figures")
CALC_DIR = Path("../calculations")
STEM = "regime_gallery"

# (label, diver key, z0 offset, v0, colour, t_end)
REGIMES = [
    ("Hover",      "D1", +1.5, 0.0, M.PALETTE[0], 220.0),
    ("Porpoising", "D4", +0.5, 0.0, M.PALETTE[3], 260.0),
    ("Runaway",    "D1", -12.0, -0.7, M.PALETTE[1], 90.0),
]


def measure_period(t, z):
    """Dominant oscillation period from upward zero-crossings of z - mean."""
    x = z - np.mean(z)
    s = np.sign(x)
    up = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    if len(up) < 3:
        return np.nan
    return float(np.mean(np.diff(t[up])))


def main():
    M.configure_style()
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.4), layout="constrained")
    results = []

    for row, (name, key, dz0, v0, color, t_end) in enumerate(REGIMES):
        cfg = M.DIVERS[key]
        esc = cfg.z_star + 6.0
        t, z, v = M.integrate_dde(cfg, t_end=t_end, z0=cfg.z_star + dz0, v0=v0,
                                  n_per_delay=100, saturate=True, escape=esc)
        # depth history (depth axis inverted; surface at top)
        ax = axes[row, 0]
        ax.plot(t, z, color=color, lw=1.4)
        ax.axhline(cfg.z_star, color="0.6", lw=0.8, ls=":")
        if name == "Runaway":
            ax.axhline(0.0, color="0.3", lw=0.8)
        ax.invert_yaxis()
        ax.set_ylabel("depth $z$ (m)")
        if row == 2:
            ax.set_xlabel(r"time $t$ (s)")
        M.panel_label(ax, "abc"[row * 2 + 0] if False else chr(ord("a") + row * 2))

        # phase portrait
        ax = axes[row, 1]
        ax.plot(z - cfg.z_star, v, color=color, lw=1.2)
        ax.plot(z[0] - cfg.z_star, v[0], "o", color=color, ms=5, mfc="white")
        ax.plot(0, 0, "+", color="0.3", ms=8)
        ax.set_ylabel(r"$\dot z$ (m s$^{-1}$)")
        if row == 2:
            ax.set_xlabel(r"depth error $z-z^\ast$ (m)")
        M.panel_label(ax, chr(ord("a") + row * 2 + 1))

        per = measure_period(t, z) if name == "Porpoising" else np.nan
        amp = float(np.ptp(z[len(z) // 2:])) if name != "Runaway" else np.nan
        reached_surface = bool(np.min(z) <= 0.05)
        results.append(dict(name=name, key=key, dz0=dz0, v0=v0, t_end=t_end,
                            per=per, amp=amp, end=z[-1], zmin=float(np.min(z)),
                            zmax=float(np.max(z)),
                            escaped=bool(abs(z[-1] - cfg.z_star) > 5.0),
                            surfaced=reached_surface, n=len(z)))

    handles = [Line2D([0], [0], color=c, lw=1.6) for _, _, _, _, c, _ in REGIMES]
    labels = [r[0] for r in REGIMES]
    M.bottom_legend(fig, handles, labels, ncol=3)
    M.save_figure(fig, FIG_DIR, STEM)
    plt.close(fig)

    # ----------------------------------------------------------------- report
    L = M.report_header(
        ["CLOSED-LOOP REGIMES: HOVER, PORPOISING, RUNAWAY",
         "Depth history and phase portrait for one representative case each"],
        notes=[
            "Runaway uses the baseline diver started shallow and ascending, so",
            "  the compensator vents to empty and the diver escapes to the surface.",
            "Porpoising period is the mean interval between upward mean-crossings.",
        ])
    for r in results:
        cfg = M.DIVERS[r["key"]]
        L.append(M.BAR)
        L.append(f" {r['name'].upper()}   diver {r['key']} ({cfg.label})")
        L.append(M.SUB)
        L.append(f"   Initial depth error        [m]   = {r['dz0']:+.2f}")
        L.append(f"   Initial velocity           [m/s] = {r['v0']:+.2f}")
        L.append(f"   Integration horizon        [s]   = {r['t_end']:.1f}")
        L.append(f"   Final depth                [m]   = {r['end']:.3f}")
        L.append(f"   Depth range visited        [m]   = [{r['zmin']:.2f}, {r['zmax']:.2f}]")
        if np.isfinite(r["per"]):
            L.append(f"   Porpoising period          [s]   = {r['per']:.3f}")
        if np.isfinite(r["amp"]):
            L.append(f"   Late peak-to-peak amplitude [m]  = {r['amp']:.3f}")
        L.append(f"   Reached surface                  = {r['surfaced']}")
        L.append(f"   Escaped neutral (|dev|>5 m)      = {r['escaped']}")
        L.append("")
    L.append(M.BAR)
    M.write_report(L, CALC_DIR / f"{STEM}.txt")
    print(f"[{STEM}] figure and report written")


if __name__ == "__main__":
    main()
