#!/usr/bin/env python
"""
Safe operating envelope of the buoyancy loop.

For each diver the plane of initial depth error and initial vertical velocity is
swept and every start is classified as bounded, meaning the diver keeps depth
control, or runaway, meaning the compensator saturates and the diver escapes to
the surface or the deep. The bounded set is the safe operating envelope; its
boundary is fixed by control saturation acting against the negative buoyancy
stiffness. The whole initial-condition ensemble is advanced together with a
vectorised method-of-steps Runge-Kutta scheme on a grid aligned to the delay.

Output
    ../figures/safe_envelope.{pdf,png,eps}   2x2 panel
    ../calculations/safe_envelope.txt         numerical report

Author: Sandy H. S. Herho <sandy.herho@email.ucr.edu>
Date: 2026-08-01
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import scuba_buoyancy_model as M

FIG_DIR = Path("../figures")
CALC_DIR = Path("../calculations")
STEM = "safe_envelope"

NZ, NV = 41, 41
DZ_MAX = 16.0
T_END = 150.0
N_PER_DELAY = 40
DEEP = 20.0        # runaway if excursion below neutral exceeds this (m)


def classify_grid(cfg, dz, vv):
    """Vectorised basin classification; returns a bounded/runaway grid."""
    d = M.derived(cfg)
    qstar = d["q_star"]
    dt = cfg.tau / N_PER_DELAY
    m = N_PER_DELAY
    n = int(T_END / dt)
    DZ, VV = np.meshgrid(dz, vv)
    z = (cfg.z_star + DZ).ravel().astype(float)
    v = VV.ravel().astype(float)
    K = z.size
    hz = np.empty((n + 1, K)); hv = np.empty((n + 1, K)); ha = np.empty((n + 1, K))
    hz[0] = z; hv[0] = v
    z0 = z.copy(); v0 = v.copy()
    maxdev = np.abs(z - cfg.z_star)
    minz = z.copy()

    def acc(zz, vv_, zd, vd):
        q = np.clip(qstar + cfg.kp * (zd - cfg.z_star) + cfg.kd * vd,
                    0.0, cfg.q_max)
        Gs = cfg.V_s0 + q
        Fb = M.RHO * M.G * Gs * M.P0 / (M.P0 + M.RHO * M.G * zz)
        drag = cfg.c1 * vv_ + 0.5 * M.RHO * cfg.C_d * cfg.S * np.abs(vv_) * vv_
        return (cfg.Delta - Fb - drag) / cfg.m_eff

    def node(j):
        if j <= 0:
            return (z0, v0, ha[0] if j == 0 else np.zeros(K))
        return hz[j], hv[j], ha[j]

    ha[0] = acc(z0, v0, *node(-m)[:2])

    for i in range(n):
        zj, vj, aj = node(i - m)
        zjp, vjp, ajp = node(i - m + 1)
        z_dh = 0.5 * (zj + zjp) + 0.125 * dt * (vj - vjp)
        v_dh = 0.5 * (vj + vjp) + 0.125 * dt * (aj - ajp)
        k1z, k1v = v, ha[i]
        k2z = v + 0.5 * dt * k1v
        k2v = acc(z + 0.5 * dt * k1z, k2z, z_dh, v_dh)
        k3z = v + 0.5 * dt * k2v
        k3v = acc(z + 0.5 * dt * k2z, k3z, z_dh, v_dh)
        k4z = v + dt * k3v
        k4v = acc(z + dt * k3z, k4z, zjp, vjp)
        z = z + (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
        v = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        z = np.clip(z, -60.0, cfg.z_star + 300.0)
        v = np.clip(v, -20.0, 20.0)
        hz[i + 1] = z; hv[i + 1] = v
        ha[i + 1] = acc(z, v, *node(i + 1 - m)[:2])
        maxdev = np.maximum(maxdev, np.abs(z - cfg.z_star))
        minz = np.minimum(minz, z)
    runaway = (minz <= 0.05) | ((z - cfg.z_star) > DEEP) | (np.abs(z - cfg.z_star) > 5.0)
    bounded = (~runaway).reshape(DZ.shape).astype(float)
    return bounded


def main():
    M.configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2), layout="constrained")
    dz = np.linspace(-DZ_MAX, DZ_MAX, NZ)
    summary = []

    for idx, key in enumerate(M.DIVER_ORDER):
        cfg = M.DIVERS[key]
        vmax = min(1.2, cfg.z_star / 30.0 + 0.8)
        vv = np.linspace(-vmax, vmax, NV)
        grid = classify_grid(cfg, dz, vv)

        ax = axes[idx // 2, idx % 2]
        ax.pcolormesh(dz, vv, grid, cmap="Blues", vmin=0, vmax=1.6,
                      shading="auto", rasterized=True)
        ax.contour(dz, vv, grid, levels=[0.5], colors="k", linewidths=1.0)
        ax.plot(0, 0, "+", color=M.PALETTE[3], ms=9)
        ax.set_xlim(-DZ_MAX, DZ_MAX); ax.set_ylim(-vmax, vmax)
        if idx // 2 == 1:
            ax.set_xlabel(r"initial depth error $z_0-z^\ast$ (m)")
        if idx % 2 == 0:
            ax.set_ylabel(r"initial velocity $\dot z_0$ (m s$^{-1}$)")
        ax.annotate(f"{key}: {cfg.label}", (0.03, 0.05),
                    xycoords="axes fraction", fontsize=8)
        M.panel_label(ax, "abcd"[idx])

        row0 = grid[np.argmin(np.abs(vv))]
        safe_dz = dz[row0 > 0.5]
        summary.append(dict(key=key, frac=float(grid.mean()),
                            dz_down=float(safe_dz.max()) if safe_dz.size else np.nan,
                            dz_up=float(safe_dz.min()) if safe_dz.size else np.nan))
        print(f"[{STEM}] {key} bounded fraction {grid.mean():.3f}")

    handles = [Patch(facecolor=plt.get_cmap("Blues")(0.7 / 1.6), edgecolor="k"),
               Patch(facecolor="white", edgecolor="k")]
    M.bottom_legend(fig, handles, ["bounded (safe)", "runaway"], ncol=2)
    M.save_figure(fig, FIG_DIR, STEM)
    plt.close(fig)

    L = M.report_header(
        ["SAFE OPERATING ENVELOPE OF THE BUOYANCY LOOP",
         "Bounded versus runaway initial conditions per diver"],
        notes=[
            f"Grid {NZ} x {NV} over depth error in [-{DZ_MAX:.0f}, {DZ_MAX:.0f}] m",
            f"  and velocity in [-vmax, vmax]; horizon {T_END:.0f} s, delay-aligned",
            f"  step tau/{N_PER_DELAY}. The ensemble is advanced together.",
            "Runaway means the diver reaches the surface, sinks past the deep",
            f"  bound of {DEEP:.0f} m below neutral, or fails to return within 5 m.",
            "dz_down and dz_up are the largest recoverable depth errors at zero",
            "  initial velocity (positive is deeper than neutral).",
        ])
    for s in summary:
        cfg = M.DIVERS[s["key"]]
        L.append(M.BAR)
        L.append(f" {s['key']}: {cfg.label}")
        L.append(M.SUB)
        L.append(f"   Bounded fraction of the sampled plane = {s['frac']:.4f}")
        L.append(f"   Max recoverable depth error (deeper)  = {s['dz_down']:+.3f} m")
        L.append(f"   Max recoverable depth error (shallow) = {s['dz_up']:+.3f} m")
        L.append("")
    L.append(M.BAR)
    M.write_report(L, CALC_DIR / f"{STEM}.txt")
    print(f"[{STEM}] figure and report written")


if __name__ == "__main__":
    main()
