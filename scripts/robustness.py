#!/usr/bin/env python
"""
Robustness of the diagnostics: observational noise and breathing forcing.

Two checks that the reported signatures survive realistic conditions. Panels (a)
and (b) track the permutation entropy and statistical complexity of a hover and a
porpoising record as observational depth noise grows, showing the two regimes
stay separated until the noise floods the signal. Panels (c) and (d) add a
periodic breathing perturbation to the carried gas and sweep its frequency: the
depth response peaks near the slow buoyancy resonance and rolls off toward normal
breathing rates, so the loop behaves as a low-pass filter on tidal volume.

Output
    ../figures/robustness.{pdf,png,eps}   2x2 panel
    ../calculations/robustness.txt         numerical report

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
STEM = "robustness"

DIM, DELAY = 6, 1
DT_SAMPLE = 0.25
BREATH_AMP = 0.0008     # surface-referenced tidal gas modulation, m^3


def clean_series(cfg, t_end, dz0):
    t, z, v = M.integrate_dde(cfg, t_end=t_end, z0=cfg.z_star + dz0,
                              n_per_delay=60, saturate=True,
                              escape=cfg.z_star + 6.0)
    dt = t[1] - t[0]
    step = max(1, int(round(DT_SAMPLE / dt)))
    zs = z[::step]
    return zs[len(zs) // 5:]


def forced_amplitude(cfg, freq, t_end=500.0):
    t, z, v = M.integrate_dde(cfg, t_end=t_end, z0=cfg.z_star, v0=0.0,
                              n_per_delay=80, saturate=True,
                              breath_amp=BREATH_AMP, breath_freq=freq,
                              escape=cfg.z_star + 6.0)
    seg = z[len(z) // 2:]
    return float(np.ptp(seg)), t, z


def main():
    M.configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), layout="constrained")

    # clean records for the two regimes
    hover = clean_series(M.DIVERS["D1"], 2600.0, dz0=1.0)
    porp = clean_series(M.DIVERS["D4"], 2600.0, dz0=0.5)
    n = min(len(hover), len(porp))
    hover, porp = hover[:n], porp[:n]

    noise_levels = np.logspace(np.log10(0.005), np.log10(0.4), 16)
    rng = np.random.default_rng(11)
    Hh, Ch, Hp, Cp = [], [], [], []
    for s in noise_levels:
        xh = hover + s * rng.standard_normal(n)
        xp = porp + s * rng.standard_normal(n)
        h1, c1 = M.permutation_entropy_complexity(xh, DIM, DELAY)
        h2, c2 = M.permutation_entropy_complexity(xp, DIM, DELAY)
        Hh.append(h1); Ch.append(c1); Hp.append(h2); Cp.append(c2)

    # (a) entropy vs noise
    ax = axes[0, 0]
    (h_hi,) = ax.semilogx(noise_levels, Hh, "-o", ms=3, color=M.PALETTE[0],
                          label="hover")
    (h_pi,) = ax.semilogx(noise_levels, Hp, "-s", ms=3, color=M.PALETTE[3],
                          label="porpoising")
    ax.set_xlabel(r"observational noise $\sigma$ (m)")
    ax.set_ylabel(r"permutation entropy $H$")
    M.panel_label(ax, "a")

    # (b) complexity vs noise
    ax = axes[0, 1]
    ax.semilogx(noise_levels, Ch, "-o", ms=3, color=M.PALETTE[0])
    ax.semilogx(noise_levels, Cp, "-s", ms=3, color=M.PALETTE[3])
    ax.set_xlabel(r"observational noise $\sigma$ (m)")
    ax.set_ylabel(r"statistical complexity $C$")
    M.panel_label(ax, "b")

    # (c) breathing entrainment: response amplitude vs frequency (log sweep)
    ax = axes[1, 0]
    freqs = np.logspace(np.log10(0.004), np.log10(0.4), 26)
    w0 = M.derived(M.DIVERS["D1"])["omega0"]
    nd = M.nondim(M.DIVERS["D1"])
    f_n = w0 / (2.0 * np.pi)
    f_r = f_n * np.sqrt(max(0.0, 1.0 - 2.0 * nd["zeta"] ** 2))
    amps = np.array([forced_amplitude(M.DIVERS["D1"], f)[0] for f in freqs])
    f_peak = freqs[np.argmax(amps)]
    (h_amp,) = ax.semilogx(freqs, amps * 100.0, "-o", ms=3, color=M.PALETTE[2],
                           label="breathing response")
    ax.axvline(f_n, color="k", ls="--", lw=0.9)
    ax.set_xlabel(r"breathing frequency (Hz)")
    ax.set_ylabel(r"depth response (cm)")
    M.panel_label(ax, "c")

    # (d) example forced series near the corner
    ax = axes[1, 1]
    f_ex = freqs[np.argmax(amps)]
    _, t, z = forced_amplitude(M.DIVERS["D1"], f_ex, t_end=400.0)
    ax.plot(t, z - M.DIVERS["D1"].z_star, color=M.PALETTE[2], lw=1.0)
    ax.set_xlabel(r"time $t$ (s)")
    ax.set_ylabel(r"depth error (m)")
    M.panel_label(ax, "d")

    handles = [h_hi, h_pi, h_amp]
    labels = ["hover", "porpoising", "breathing response"]
    M.bottom_legend(fig, handles, labels, ncol=3)
    M.save_figure(fig, FIG_DIR, STEM)
    plt.close(fig)

    # ----------------------------------------------------------------- report
    sep_H = np.abs(np.array(Hh) - np.array(Hp))
    sep_C = np.abs(np.array(Ch) - np.array(Cp))
    L = M.report_header(
        ["ROBUSTNESS: OBSERVATIONAL NOISE AND BREATHING FORCING",
         "Ordinal separation under noise and the breathing resonance"],
        notes=[
            f"Embedding dimension {DIM}, delay {DELAY}, sample {DT_SAMPLE:.2f} s.",
            f"Breathing modulates the carried gas by {BREATH_AMP*1e3:.2f} L at",
            "  surface reference; amplitude is steady peak-to-peak depth.",
            f"Baseline buoyancy rate omega0 = {w0:.5f} s^-1 "
            f"({w0/(2*np.pi):.5f} Hz).",
        ])
    L.append(M.BAR)
    L.append(" ORDINAL SEPARATION VERSUS NOISE")
    L.append(M.SUB)
    L.append("   sigma[m]    H_hover   H_porp    C_hover   C_porp    |dH|    |dC|")
    for s, h1, h2, c1, c2, dH, dC in zip(noise_levels, Hh, Hp, Ch, Cp,
                                         sep_H, sep_C):
        L.append(f"   {s:7.4f}    {h1:.4f}   {h2:.4f}    {c1:.4f}   {c2:.4f}"
                 f"   {dH:.4f}  {dC:.4f}")
    L.append("")
    L.append(M.BAR)
    L.append(" BREATHING FORCING (low-pass response)")
    L.append(M.SUB)
    L.append(f"   Undamped corner f_n         [Hz]  = {f_n:.4f}")
    L.append(f"   Damped resonance f_r        [Hz]  = {f_r:.4f}")
    L.append(f"   Peak of the swept response  [Hz]  = {f_peak:.4f}")
    L.append(f"   Peak depth response         [cm]  = {amps.max()*100:.4f}")
    L.append(f"   Response at 0.25 Hz         [cm]  = "
             f"{np.interp(0.25, freqs, amps)*100:.4f}")
    L.append("")
    L.append(M.BAR)
    M.write_report(L, CALC_DIR / f"{STEM}.txt")
    print(f"[{STEM}] figure and report written")


if __name__ == "__main__":
    main()
