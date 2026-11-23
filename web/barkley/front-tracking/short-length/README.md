# Short-length Barkley front tracking at r=0.6

Dedicated site: https://tsufli.github.io/pipe-flow-lab/barkley/front-tracking/short-length/

This experiment extends the first 100-run study with 2000 natural realizations,
T=300, and up to six independent continuations from each of 100 genuine
turbulent puff snapshots taken from natural trajectories. There are NO
low-amplitude edge-like restart parents and NO synthetic/scaled/interpolated
initial states. The full measured q(x) and u(x) profiles are restarted.
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
independent noise seeds. Restart parents are saved *directly* from natural simulations (no fabricated
profiles). A parent must have, at t>20, a q=0.5 width between 3 and 5.5,
qmax>=1, mass>=4, and at least one unit of preceding uninterrupted,
unambiguous q=0.5 puff tracking with qmax>=0.9 and width>=2.5. Only the
first qualifying state of each independent run is accepted. Full q(x) and
u(x) fields are preserved. Up to 100 independent parents receive six fresh
noise continuations of T=60. The deliberately selected restarts remain
separate from the unselected 2000 natural trajectories, with confidence
intervals calculated by resampling independent parent clusters.

## Conditional inference

At lags h=.2,.4,.8,1.6 and initial-length bins of width .5 from l=2.5 to l=6,
1 above l=6, estimate

    A_i(l) = E[delta x_i | l]/h
    D_ij(l) = Cov(delta x_i,delta x_j | l)/(2h)
    A_l = A_down-A_up
    D_l = D_dd+D_uu-2D_ud.

Both endpoints AND all intermediate observations must have measured length >=2.5.
Measurements below 2.5 are excluded rather than reported. This additional
censoring can bias conditional increments in the first trustworthy bin.
Bins must contain at least 75 valid increment pairs and 12 independent
ancestors. When more than one stable zero exists, a single bootstrap root
interval would mix different branches, so it is suppressed. Drift zeros are interpolated; 95% intervals use 160
whole-ancestor bootstrap samples. Finite-lag dependence is shown. No
Ornstein-Uhlenbeck drift, closed Markovian dynamics, or continuous-time
diffusion limit is assumed.

## Reproduce

    python -m pip install numpy scipy numba
    python -m unittest discover -s tests -p test_barkley_shortlength.py -v
    python scripts/barkley_shortlength.py --runs 2000 --T 300 --parents 100 --branches 6 --branch-T 60 --sigma .5 --seed 20261129

The GitHub Actions workflow .github/workflows/barkley-short-puffs.yml
performs this experiment, commits web/barkley/front-tracking/data/shortlength_r_0.6.json,
and saves the result. A fresh GitHub Pages publication can be triggered by
running the existing pages.yml workflow after an Actions-generated dataset commit;
ordinary user/app pushes to main trigger pages.yml automatically. Output retains algorithm metadata,
parent identities, seed schedule, decay confirmation criteria, coefficients,
and selected trajectories.

## Limitations

The hidden velocity field u can cause length-history dependence near the edge.
True physical decay and the loss of front measurements are not equivalent.
Survival selection affects the shortest bins. The two threshold definitions
are intentionally distinct and are not pooled. Numerical convergence in
resolution, lag, noise strength and threshold is not established.

## Smoothness and interpretation

Fourfold more natural trajectories reduce independent-sampling uncertainty by
approximately a factor two where sampling dominates. They cannot cure
threshold fragmentation, length-history dependence, or censoring bias.
The website may draw a labeled, optional local-polynomial display trend
alongside the raw half-unit bins and bootstrap intervals. No smoothing is
applied to the physics, stored coefficients or zero-finding algorithm.

## Completed larger ensemble (2026-11-23)

The completed updated run includes 2000 independent natural realizations
(T=300), **751** confirmed decays by the stated diagnostic and
**100 genuine turbulent puff snapshots** with six stochastic continuations
each (600 restarts). These are all complete snapshots directly from natural
PDE trajectories. The smallest recorded restart-parent q peak is 1.047 and
all parent q=0.5 widths are 3.074 to 5.499.

For the q=.15 channel at lag .4, the observed drift A_l in
[4.5,5.0) is **+0.0632**, parent-bootstrap 95% interval
[+0.0187,+0.1201]. The q=.15 length-drift zero now resolved in
the trusted domain is stable at l≈10.359 (and the q=.5 stable zero is
l≈6.678). The previously apparent unstable root near length 5
in the uncut 500-run analysis is NOT reproduced under the strict
l>=2.5 interval filtering. This difference cannot be attributed to
increased sample size alone, because the selection rule also changed.
There is genuine right-censoring/survival selection at the cutoff;
the new positive short-length drift should not be interpreted as proof
that all short puffs recover. The full-field numerical edge reference
still has qpeak≈0.299 and q=.15 width≈5.007.

Completed local simulation wall-time: about 442 seconds, excluding workflow
environment setup and Pages publishing.
