# Barkley instanton explorer (restricted-action experiment)

The interactive website is published at the Barkley project's `/barkley/instanton/` tab.

The two datasets, `data/r_0.5.json` and `data/r_0.6.json`, are **numerically computed restricted-family action candidates**, **not globally minimized instantons**. They contain the full-resolution solver's diagnostics plus display-downsampled profiles of the puff, edge, lower-action pulse, baseline pulse, and reversed deterministic edge-to-puff reference.

## Equations and action

The model is

```
q_t = D q_xx − (u−ζ) q_x + q[r + u − U0 − (r+δ)(q−1)^2] + q a(x,t)
u_t = −u u_x + ε[U0−u + κ(Ubar−u)q]
```

with stochastic-control cost (excluding the factor `σ^{-2}`)

```
I = 1/2 ∫dt ∫dx a(x,t)^2.
```

The restriction is a Gaussian pulse:

```
a(x,t) = −A exp(−dist_periodic(x,x0)^2/(2w^2)) sin²(πt/τ)  [0 ≤ t ≤ τ],
         = 0                                                 [t > τ].
```

The pulse is **multiplicative in q**, not added directly to the u equation. A coarse grid and Nelder–Mead search identified the stored `(x0,w,τ)` pulse parameters. For each shape, a bisection in `A` brackets the two deterministic outcomes. The resulting trajectory shadows the computed decay edge for finite time and later departs along its unstable direction. The source data reports both the edge proximity and amplitude bracket.

## Reproduction

From the repository root:

```bash
python -m pip install numpy scipy numba
python scripts/build_barkley_instanton.py
```

This starts from the existing eight-point edge-scan JSON, uses periodic cubic interpolation to obtain a Newton seed at exactly `r=0.5` and `r=0.6`, and recomputes the converged states and controlled PDE trajectories with `N=320`, `Δx=0.5`, `L=160`, RK4 time step ~0.09. The numerical action and edge-distance diagnostics are computed before display downsampling to 160 points. Running `python scripts/build_barkley_instanton.py --optimize` repeats the **restricted-family Gaussian optimization** (slower); it does not solve the unrestricted minimum-action problem.

The reversed deterministic edge-to-puff curve and the linear puff–edge interpolation are strictly **geometric references**. Reversing the deterministic solution or interpolating both fields generally violates the `u_t = F_u` constraint and therefore cannot be assigned the displayed finite noise-control action.

## Limitations

- Restricted Gaussian family only, with two values of r; this is not a converged gMAM or full optimal-control instanton.
- Single finite-resolution periodic discretization and time step; no continuum convergence proof.
- The displayed paths are selected by basin-threshold bracketing, not fixed-time endpoint-constrained action minimization. They approach an edge **neighborhood**, not the edge exactly at finite time.
- Weak-noise predictions for lifetimes require independent rare-event simulations.
- The viewer's “follow the peak” coordinate shift is presentation only; the action is computed on the unshifted PDE fields.