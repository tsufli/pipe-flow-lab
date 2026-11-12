# Barkley decay-edge eigenmode and manifold scan

This subsite displays data generated from the **deterministic** 1D Barkley model for eight values of `r` from 0.43 to 0.72. It is not a direct Navier–Stokes simulation and `r` is a model control parameter, not the physical Reynolds number.

## Reproduction

From the repository root (Python 3.11+):

```bash
python -m pip install numpy scipy numba
python scripts/build_barkley_edge_scan.py
```

The script regenerates `web/barkley/edge-scan/data/summary.json` and the eight `r_*.json` files. It also produces `puff_*.npz` and `edge_*.npz` in that directory; these are **not** required for the site and can be ignored by Git.

## Method

1. Relax a localized turbulent seed to a puff at `r=0.6`, solve the traveling-wave equations in a comoving frame by Newton iteration with a phase condition.
2. Use finite-time basin bisection to construct a candidate decay-edge profile and Newton-refine it at the same `r`.
3. Continue both solutions to eight `r` values in steps ≤0.009.
4. Construct the discrete Jacobian at each edge state, compute eigenpairs by sparse shift–invert, and compute adjoint modes using the transpose Jacobian. Normalize each pair by `∑ Δx conj(left)·right = 1`.
5. Evolve the two branches obtained by perturbing the edge along ± the unstable right eigenfunction, using the same spatial discretization and the edge's comoving frame. Verify that the branches approach the puff and laminar basins.

Parameters: `U0=2`, `Ub=1`, `zeta=0.8`, `D=0.5`, `delta=0.1`, `epsilon=0.1`, `kappa=2`, `L=160` periodic, `N=640`, `dx=0.25`. Second-order backward differences for `q_x` and `u_x`, centered differences for `q_xx`, 4th-order Runge–Kutta for time evolution. Profiles and eigenfunctions are subsampled by factor four for site visualization. The site plots only the **near-axis portion** of the spectrum, not the full 1280 eigenvalues.

## Limits

This is a single-grid exploratory numerical calculation; grid convergence and spectral-branch tracking should be completed before drawing continuum conclusions. At `r ≥ 0.554`, the selected leading stable eigenpair is complex, so displaying its real part is not equivalent to finding a real stable eigenfunction. Different wave speeds mean the puff and edge are separately stationary only in their own frames. Unstable-manifold plots remove translation to compare profiles. Full convergence of finite-time forward paths to invariant attractors is not established beyond the reported endpoint metrics.