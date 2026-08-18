# Supplementary Scripts

### *An idealized delay-differential model of scuba diver porpoising and runaway ascent*

[![arXiv](https://img.shields.io/badge/arXiv-2603.13715-b31b1b.svg?style=flat-square)](https://doi.org/10.48550/arXiv.2608.14978)
[![DOI](https://zenodo.org/badge/1325460305.svg)](https://doi.org/10.5281/zenodo.21940225)


This repository holds the analysis scripts that produce every computed figure and
numerical report of the accompanying manuscript. The study is analysis-only: a
diver stabilizing an inherently unstable buoyancy equilibrium under sensorimotor
delay is modeled as a delay differential equation, and the scripts derive its
stability, bifurcation, safe operating envelope, and ordinal-pattern diagnostics
directly from that model. There is no external solver and no archived simulation
data. A single shared module, `scuba_buoyancy_model.py`, holds the model, the
numerical routines, and the plotting substrate; each figure script imports from
it and is independently runnable.

## Model and governing equations

A diver at depth $z$ (positive downward) holds a neutral hover that is inherently
unstable: the carried gas (wetsuit or drysuit plus buoyancy compensator)
compresses with depth by Boyle's law, so the buoyant force weakens as the diver
sinks and strengthens as the diver rises. The vertical force balance, with
effective mass $m_\mathrm{eff}$, net weight excess $\Delta = m g - \rho g V_\mathrm{inc}$,
compressible buoyancy referenced to its surface volume $G_\mathrm{surf}$, and
linear-plus-quadratic drag, is

$$m_\mathrm{eff}\,\ddot z = \Delta - \rho g\,G_\mathrm{surf}\,\frac{P_0}{P(z)} - c_1 \dot z - \tfrac{1}{2}\rho\,C_d\,S\,|\dot z|\,\dot z, \qquad P(z) = P_0 + \rho g z.$$

The diver regulates the compensator gas after a finite reaction delay $\tau$
through a proportional-derivative law,

$$q(t) = q^\ast + k_p\,\bigl(z(t-\tau) - z^\ast\bigr) + k_d\,\dot z(t-\tau), \qquad \text{clamped to } [0, q_\mathrm{max}].$$

Linearizing about the neutral state $z^\ast$ gives a negative (destabilizing)
stiffness $\beta = (\rho g)^2 G^\ast P_0 / P_z^2$, a control authority
$\gamma = \rho g P_0 / P_z$, and an instability rate
$\omega_0 = \sqrt{\beta / m_\mathrm{eff}}$. In nondimensional time $s = \omega_0 t$
the closed loop reduces to the delay differential equation

$$x'' + 2\zeta\,x' - x + \kappa_p\,x(s-\theta) + \kappa_d\,x'(s-\theta) = 0,$$

with the four dimensionless groups

$$\zeta = \frac{c_1}{2 m_\mathrm{eff}\,\omega_0}, \qquad \kappa_p = \frac{\gamma k_p}{\beta}, \qquad \kappa_d = \frac{\gamma k_d\,\omega_0}{\beta}, \qquad \theta = \omega_0\,\tau.$$

Its characteristic function is the quasi-polynomial

$$\chi(\mu) = \mu^2 + 2\zeta\mu - 1 + \bigl(\kappa_p + \kappa_d\mu\bigr)e^{-\mu\theta} = 0,$$

whose rightmost root governs stability. With the gas frozen the plant is a saddle
($x'' + 2\zeta x' - x = 0$), so any oscillation is generated entirely by the
delayed control, and porpoising sets in through a supercritical Hopf bifurcation
as $\theta$ (or the gain) grows.

## Gold-standard numerics

- **Stability spectrum:** pseudospectral discretization of the infinitesimal
  generator of the solution semigroup on Chebyshev-Gauss-Lobatto nodes, after
  Breda, Maset, and Vermiglio. The rightmost roots of $\chi(\mu)$ are recovered
  as eigenvalues of a finite matrix and converge spectrally; they agree with a
  direct Newton solution to a residual of order $10^{-13}$.
- **Time integration:** fourth-order Runge-Kutta method of steps on a grid whose
  spacing divides the delay, with interior-stage delayed values supplied by the
  cubic Hermite continuous extension. The saturation on the compensator gas is
  retained, so the integrated dynamics include the amplitude-limiting
  nonlinearity.
- **Ordinal analysis:** Bandt-Pompe permutation entropy and the Rosso
  Jensen-Shannon statistical complexity, computed from their definitions at
  embedding dimension six and unit lag, and verified against `ordpy`.

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
| `safe_envelope.py` | 6 | Bounded versus runaway initial conditions in the depth-error and velocity plane for each diver, from a vectorized ensemble integration; 2×2 panel |
| `complexity_entropy.py` | 7 | Bandt-Pompe permutation entropy and Rosso statistical complexity of the regimes on the plane with theoretical bounds, and their drift along a delay ramp; 1×2 panel |
| `early_warning.py` | 8 | Critical slowing down approaching onset: lag-one autocorrelation, variance, and measured recovery rate against the spectral abscissa; 2×2 panel |
| `sensitivity.py` | 9 | Central-difference semi-elasticities of the spectral abscissa and the critical delay to the physical parameters, ranked; 1×2 tornado panel |
| `robustness.py` | 10 | Ordinal separation of hover and porpoising under observational noise, and the low-pass depth response to breathing forcing; 2×2 panel |

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

- **Archived outputs (figures, computational notes):** https://doi.org/10.17605/OSF.IO/VNMWS

## Authors

Sandy H. S. Herho, Faizal A. R. Abdullah, Iwan P. Anwar, Faruq Khadami, Alfita P. Handayani,
Karina A. Sujatmiko, Rusmawan Suwarman, and Dasapta E. Irawan

Correspondence: Sandy H. S. Herho — <sandyherho@itb.ac.id>

## License

MIT License - See [LICENSE](LICENSE) for details.