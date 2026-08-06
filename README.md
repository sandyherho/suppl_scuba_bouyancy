# Supplementary Analysis Scripts: *Delayed buoyancy control in underwater diving as a nonlinear dynamical system: porpoising, runaway, and information-theoretic diagnostics*

This repository holds the analysis scripts that produce every computed figure and
numerical report of the accompanying manuscript. The study is analysis-only: a
diver stabilising an inherently unstable buoyancy equilibrium under sensorimotor
delay is modelled as a delay differential equation, and the scripts derive its
stability, bifurcation, safe operating envelope, and ordinal-pattern diagnostics
directly from that model. There is no external solver and no archived simulation
data. A single shared module, `scuba_buoyancy_model.py`, holds the model, the
numerical routines, and the plotting substrate; each figure script imports from
it and is independently runnable.

## Scripts

All live in `scripts/` and are run from that directory. Each writes a figure to
`../figures/` in three formats (PDF, PNG, EPS) at 600 dpi and a plain-text report
to `../calculations/`.

| Script | Figure | Produces |
|---|---|---|
| `scuba_buoyancy_model.py` | (shared) | Model, delay-aligned RK4 integrator, pseudospectral stability spectrum, ordinal estimators, and plotting substrate. Imported by every script; not run on its own |
| `open_loop_instability.py` | 2 | Uncontrolled phase portrait and saddle structure; exponential divergence of small perturbations with analytic e-folding envelopes; 1×2 panel |
| `stability_chart.py` | 3 | Spectral-abscissa map over delay and gain by pseudospectral generator collocation, Hopf boundary, and onset frequency with porpoising period; 1×2 panel |
| `bifurcation.py` | 4 | Limit-cycle amplitude against reaction delay with the pseudospectral onset, and the supercriticality test on the squared amplitude; 1×2 panel |
| `regime_gallery.py` | 5 | Depth history and phase portrait for the three regimes: stable hover, bounded porpoising, and runaway ascent; 3×2 panel |
| `safe_envelope.py` | 6 | Bounded versus runaway initial conditions in the depth-error and velocity plane for each diver, from a vectorised ensemble integration; 2×2 panel |
| `complexity_entropy.py` | 7 | Bandt-Pompe permutation entropy and Rosso statistical complexity of the regimes on the plane with theoretical bounds, and their drift along a delay ramp; 1×2 panel |
| `early_warning.py` | 8 | Critical slowing down approaching onset: lag-one autocorrelation, variance, and measured recovery rate against the spectral abscissa; 2×2 panel |
| `sensitivity.py` | S1 | Central-difference semi-elasticities of the spectral abscissa and the critical delay to the physical parameters, ranked; 1×2 tornado panel |
| `robustness.py` | S2 | Ordinal separation of hover and porpoising under observational noise, and the low-pass depth response to breathing forcing; 2×2 panel |

## Four canonical divers

Several scripts process the same four configurations, which span the stable and
unstable regimes:

1. `D1` recreational wetsuit, warm water, moderately trained (baseline, stable)
2. `D2` technical drysuit, deeper, larger compressible gas volume (stable)
3. `D3` cold or task loaded, long reaction delay (porpoising)
4. `D4` novice, poorly tuned high gain and long delay (porpoising)

## Requirements

Python 3.9 or newer, with:

```bash
pip install numpy scipy matplotlib ordpy
```

`ordpy` supplies the theoretical complexity-entropy boundary curves; the
permutation entropy and complexity estimators used for the data are implemented
in the shared module and are verified against `ordpy`.

## Usage

Run any script from inside `scripts/`. The `figures/` and `calculations/`
directories are created automatically if absent.

```bash
cd scripts
python open_loop_instability.py
python stability_chart.py
python bifurcation.py
python regime_gallery.py
python safe_envelope.py
python complexity_entropy.py
python early_warning.py
python sensitivity.py
python robustness.py
```

Each script is self-contained through the shared model module and reads no
external files, so the scripts may be run in any order.

### Directory layout

```
.
├── scripts/          # shared model + nine analysis scripts (tracked)
├── figures/          # generated figures      (not tracked)
└── calculations/     # generated text reports (not tracked)
```

Only the scripts are version-controlled; all generated outputs are git-ignored
and distributed through the OSF archive.

## Related resources

- **Archived outputs (figures, text reports):** https://doi.org/10.17605/OSF.IO/XXXXX

## Authors

Sandy H. S. Herho, Iwan P. Anwar, Faruq Khadami, Alfita P. Handayani,
Karina A. Sujatmiko, Rusmawan Suwarman, and Dasapta E. Irawan

Correspondence: Sandy H. S. Herho — <sh001@ucr.edu>

## License

MIT License - See [LICENSE](LICENSE) for details.
