# Barkley two-start full-field minimum-action experiment

View at **`/barkley/instanton/`** in the existing GitHub Pages site.

This replaces the earlier Gaussian-pulse experiment. The primary goal is to test
whether **two geometrically different starting guesses** for a puff→decay-edge
transition converge toward the same numerical minimum-action trajectory.

## Numerical problem

We use the deterministic Barkley two-field PDE in the co-moving frame of the
stable puff. The control `a(x,t)` enters the q equation multiplicatively:

```
q_t = D q_xx -(u-zeta-c) q_x + q*[r + u-U0 -(r+delta)(q-1)^2] + q*a
u_t = -(u-c) u_x + eps*[U0-u + kappa*(Ubar-u)*q]
```

With unit white-noise covariance, the control cost is

```
I[a] = (1/2) sum_{n,j} (a[n,j])^2 * dt * dx .
```

Unlike the old experiment, **every entry of the `320 × 160` space–time
control array is an independent optimization variable** (subject only to broad
amplitude bounds). The optimizer is differentiable through the PDE time-stepper.
The unforced `u` equation is enforced exactly by the forward time integration.

To accommodate the fact that the decay edge is a traveling wave, the terminal
misfit is computed against a translated edge profile; the single translation is
estimated smoothly using the first spatial Fourier moment of `q(T)` and
applied to **both** edge fields. No spatial translations enter the PDE or the
stochastic action. The objective is

```
J_rho = I[a] + rho/2 * dx * sum_j [ (qT-qE_translated)^2 + 3*(uT-uE_translated)^2 ] .
```

This is a **soft finite-time endpoint penalty**, not an exact endpoint
constraint or an infinite-time quasipotential.

## Two initial conditions for the minimizer

1. **Linear interpolation:** interpolate the puff q profile to the edge q
   profile with a smooth ramp in time.
2. **Reversed edge unstable branch:** compute the deterministic unstable branch
   from the edge toward the puff and reverse its sequence of q profiles,
   registering the reference shapes by spatial translation before reversal.

Neither reference q/u history is an admissible stochastic realization, because
u cannot be forced directly. Instead, construct a control seed by inverting
the q equation along the reference q path while integrating u deterministically.
For numerical stability the resulting control is damped by a factor of 0.05
*before* the actual optimization. Thus the stored `reference` lines are
**geometric initialization templates**, not the initial PDE rollout.

Starting from each of these control seeds, the same L-BFGS-B numerical
optimization is performed at endpoint-penalty strengths `rho = 10, 30, 100`,
warm-starting each stage from the previous one. The website displays all of
the resulting PDE histories, actions, endpoint errors, and convergence records.

## Cases and reproducibility

The two cases are `r = 0.5` and `r = 0.6`. Spatial domain `L=160`,
`N=160`, `dx=1`, temporal horizon `T=80` with `320` controlled steps of
`dt=.25`; the q and u states use second-order backward advection and q uses
a centered diffusion discretization. The PDE is integrated with Heun's RK2.

The exact traveling puff and edge are recomputed using Newton iteration from
the previous edge-scan JSONs (`r=0.512857` and `r=0.595714`) and continued to
the two target parameters.

From repository root:

```
python -m pip install numpy scipy numba 'jax[cpu]'
python scripts/build_barkley_full_action.py --r 0.5 --maxiter 350
python scripts/build_barkley_full_action.py --r 0.6 --maxiter 350
```

The default run recomputes and replaces the two data files. These are
computationally intensive, and different iteration budgets change the last
unconverged checkpoint. A separate cheap data-consistency test validates the
committed files:

```
python -m unittest discover -s tests -p 'test_barkley_full_action.py'
```

## Important limitations

- The optimizer searches all **discrete** control components but can settle at
  **local** minima; **global optimality is not certified**.
- The last high-penalty optimization checkpoints at the two r values are not
  all gradient-converged. The endpoint distances are nonzero.
- This is a fixed time horizon and fixed mesh; neither convergence in `T` nor
  the continuum limit has been established.
- The chosen temporal boundary conditions differ from a true infinite-time
  instanton, whose approach to a saddle is asymptotic.
- Different final translations of the edge represent the same physical
  traveling-wave family, so raw final-state differences should not be treated
  as physical separation without quotienting spatial translation.
- The actions from this experiment must **not** be directly compared with the
  previously published Gaussian-pulse actions, which used a different grid and
  endpoint selection procedure.

All published profiles are actual controlled-PDE rollouts, not manually
interpolated states. The no-noise u constraint is satisfied at each numerical
time step.