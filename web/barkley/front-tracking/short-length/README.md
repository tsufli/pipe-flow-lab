# Short-length Barkley front tracking at r=0.6

Dedicated site: https://tsufli.github.io/pipe-flow-lab/barkley/front-tracking/short-length/

This experiment extends the first 100-run study with 500 natural realizations, T=300,
and up to four independent continuations per naturally encountered short parent,
with up to 40 short-high-q parents and 40 weak edge-like parents.
The original 100-run data are preserved as a reference. All simulations run off-browser
on GitHub Actions.

## Physical model and discretization

Identical to the original stochastic Barkley model in scripts/barkley_front_tracking.py:
r=0.6, sigma=.5, delta=.1, eps1=.1, eps2=.2, zeta=.8, D=.5,
U0=2, Ub=1. Periodic L=100, N=300, dt=.04; snapshots every .2.
The noise is sigma*q_old*sqrt(dt/dx)*Normal(0,1), interpreted as Ito.

## Two distinctly defined measured lengths

The old crossing criterion uses q=0.5. A second q=0.15 threshold lets
us follow weaker puffs approaching the numerical decay edge, which has
qmax about 0.299 and is therefore invisible at q=0.5. Each q profile is
Gaussian-smoothed over 1.6 grid cells for measurement only. We identify
the widest contiguous superthreshold component, interpolate the threshold
positions, unwrap lab-frame positions periodically, flag competing components,
and discard any finite-lag increment containing a missing or ambiguous frame.

The deterministic edge reference uses the nearest preexisting numerical
traveling-wave result at r=0.595714; the threshold width is resampled and
computed separately. The leading-order epsilon-to-zero bulk puff length
about 4.25 is not a threshold width and not the decay-edge length.

## Explicit decay confirmation

An observed disappearance of threshold crossings is NOT itself physical decay.
To confirm decay we require qmax<.05 and integral(max(q,0))dx<.25 continuously
for 12 model-time units, with max(|u-U0|)<.15 at confirmation.
After confirmation the simulated path terminates and the inference is
retrospectively censored from the beginning of the persistent quiet episode.
Pre-confirmation loss of measurable crossings remains observational censoring;
shortest-length drift and diffusion may consequently still suffer survival bias.
Decay-event totals are just diagnostics, not a decay-probability estimate.

## Natural and short-restarted populations

All 500 natural paths begin from the same relaxed deterministic puff with
independent noise seeds. Restart parents are saved at the first time t>12
when the q=.5 measured width lies in [1.4,3.5] and the raw peak q>.53.
Save the FULL q(x) and u(x), not just a scalar length.
We additionally select independent edge-like parent runs with 0.18<=qmax<=0.48
and a valid q=.15 width in [.55,5.5]. At most one parent per original
natural run is selected, with up to four fresh-noise branches of T=60 each. Restarts form a selected population and are NEVER pooled as though
they were independent stationary samples. Confidence intervals use whole
independent natural-run or restart-ancestor bootstrap clusters.

## Conditional inference

At lags h=.2,.4,.8,1.6 and initial-length bins of width .5 below l=6,
1 above l=6, estimate

    A_i(l) = E[delta x_i | l]/h
    D_ij(l) = Cov(delta x_i,delta x_j | l)/(2h)
    A_l = A_down-A_up
    D_l = D_dd+D_uu-2D_ud.

Bins must contain at least 75 valid increment pairs and 12 independent
ancestors. Drift zeros are interpolated; 95% intervals use 160
whole-ancestor bootstrap samples. Finite-lag dependence is shown. No
Ornstein-Uhlenbeck drift, closed Markovian dynamics, or continuous-time
diffusion limit is assumed.

## Reproduce

    python -m pip install numpy scipy numba
    python -m unittest discover -s tests -p test_barkley_shortlength.py -v
    python scripts/barkley_shortlength.py --runs 500 --T 300 --parents 40 --edge-parents 40 --branches 4 --branch-T 60 --sigma .5 --seed 20261129

The GitHub Actions workflow .github/workflows/barkley-short-puffs.yml
performs this experiment, commits web/barkley/front-tracking/data/shortlength_r_0.6.json,
and triggers the established Pages publisher. Output retains algorithm metadata,
parent identities, seed schedule, decay confirmation criteria, coefficients,
and selected trajectories.

## Limitations

The hidden velocity field u can cause length-history dependence near the edge.
True physical decay and the loss of front measurements are not equivalent.
Survival selection affects the shortest bins. The two threshold definitions
are intentionally distinct and are not pooled. Numerical convergence in
resolution, lag, noise strength and threshold is not established.
