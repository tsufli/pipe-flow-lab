# Barkley puff-front tracking: r = 0.6

Dedicated GitHub Pages subsite:
**https://tsufli.github.io/pipe-flow-lab/barkley/front-tracking/**

This is a separate website **within the existing pipe-flow-lab repository**, because
the connected GitHub integration allows writing this repository but does not expose
new-repository creation. It reuses the spatial Barkley equations, parameters and
stochastic time-stepper from `web/barkley/spatial/index.html` and
`scripts/barkley_spatial.py`. Simulations run offline, not in the browser.

## First ensemble

- Control parameter `r=0.6`; parameters `delta=.1, eps1=.1, eps2=.2,
  zeta=.8, D=.5, U0=2, Ub=1`.
- 100 *independent* pseudorandom streams; deterministic puff at t=350 warm-up,
  translated once before each stochastic run. All trajectories remain in the
  lab frame; positions are unwrapped across the periodic seam.
- Periodic L=100, N=300, dx=1/3; explicit dt=.04, saved every .4, duration T=120.
- Noise `sigma=.5` (the earlier **nonzero** stochastic setting): multiplicative
  independent space-time white noise with Itô increments
  `sigma * q_old * sqrt(dt/dx) * Normal(0,1)`.
  The original browser's launch-state slider is `sigma=0`, which cannot
  provide stochastic drift or diffusion estimates.
- Heun (second order deterministic) with first-order noise kick, upwind for
  q advection, Rusanov conservative flux for u, centered q Laplacian.
- A Gaussian smoothing of width 1.6 **grid cells** is applied only to
  *front detection*, not to time evolution. The front is an interpolated
  `q_smooth=0.5` crossing of the largest contiguous superthreshold region.
  We separately measure ambiguous extra regions. Intervals with ambiguity
  at either endpoint are excluded from inference.

## Conditional inference

Length is `ell = x_down - x_up`.
For each lag `h in {.4,.8,1.6,3.2}`, for each length bin `[k,k+1)`,
we compute:

```
A_up(ell)   = E[delta x_up | ell] / h
A_down(ell) = E[delta x_down | ell] / h
D_ij(ell)   = Cov(delta x_i,delta x_j | ell) / (2h)
A_length    = A_down - A_up
D_length    = D_dd + D_uu - 2 D_ud
```

Bins with fewer than 60 usable paired snapshots or fewer than ten independent
realizations are not estimated. A stable length-drift zero is a crossing
from positive to negative drift; roots are located by linear interpolation
between consecutive supported bins. Uncertainties use a **whole-realization
bootstrap**, 150 replicates, which avoids treating correlated snapshots as
independent runs. Lag dependence is presented explicitly.

These are **finite-time conditional** Kramers–Moyal estimates. We do not assume
either front is an Ornstein–Uhlenbeck process, or that length alone is Markovian.
A censoring bias is possible near decay, and the estimates can be sensitive
to smoothing, lag, resolution, and noise strength.

## Asymptotic comparison

We use the epsilon→0 slow–fast construction from Frishman & Grafke,
*Phys. Rev. E* **105**, 045108 (2022), section III.C:

```
q_plus(u) = 1 + sqrt((r+u-U0)/(r+delta))
S(u) = sqrt(D) * [3 sqrt((r+u-U0)/2) - sqrt((r+delta)/2)]
c_up^0 = U0-zeta-S(U0)
u_down-zeta+S(u_down) = c_up^0
ell_bulk^0 = integral_{u_down}^{U0}
             (u-c_up^0) / [-eps1*(U0-u+kappa*(Ub-u)*q_plus(u))] du
kappa=eps2/eps1.
```

For r=.6: `c_up^0≈.4564`, `u_down≈1.4272`,
`q_plus(u_down)≈1.1972`, `ell_bulk^0≈4.2507`.
This is **not** the full finite-epsilon distance between q=0.5 crossings:
the fast transition layers add O(1) corrections. The site reports
the deterministic threshold width separately.

## Reproduce

```sh
python -m pip install numpy scipy numba
python scripts/barkley_front_tracking.py --runs 100 --T 120 --sigma .5 --seed 20261123
python -m unittest discover -s tests -p 'test_barkley_front_tracking.py'
```

GitHub Actions workflow `.github/workflows/barkley-front-tracking.yml`
runs the simulation, commits the resulting static JSON to
`web/barkley/front-tracking/data/r_0.6.json`, then dispatches the established Pages workflow.
The JSON includes trajectories, conditional estimates at all four lags,
bootstrap intervals, tracked quality statistics, sampled profiles,
fixed seeds, and wall time. No fabricated/example data are plotted.

## Interpretation caveats

- The original web spatial solver has sigma=0 on first load, but the
  stochastic experiments use sigma=.5 as an explicit nonzero choice.
- Threshold crossings can fragment when sigma=.5; tracking smoothing,
  secondary-component flags and missing-data rates are essential diagnostics.
- Estimates near very short lengths are conditional on puff survival.
- A zero of conditional drift at finite lag need not equal the mode or mean
  of the stationary length distribution, nor deterministic puff length.
- The asymptotic bulk length omits the O(1) front shifts and only becomes
  quantitatively controlled as epsilon decreases.
- This run has not established numerical convergence or reduction validity.

## Source and references

- Existing code: `web/barkley/spatial/index.html` and
  `scripts/barkley_spatial.py` in this repository.
- Frishman & Grafke (2022), "Dynamical landscape of transitional pipe flow",
  https://doi.org/10.1103/PhysRevE.105.045108.
- Barkley (2016), "Theoretical perspective on the route to turbulence in a pipe",
  https://doi.org/10.1017/jfm.2016.465.
