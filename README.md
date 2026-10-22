# Pipe Flow Laboratory — 3D DNS transition explorer

A reproducible, finite-budget 3D Navier–Stokes pipe-flow experiment, executed by GitHub Actions and archived on GitHub Pages. Based on the **[dnsjax](https://github.com/gokhanyalniz/dnsjax)** pipe DNS solver, pinned to version `0.1.0.1` (Apache-2.0 license).

This repository runs actual simulations, not a 2D surrogate or synthetic time series. It records solver provenance, flow diagnostics, velocity/vorticity/Q images, mean profiles and finite-time comparisons.

## Get started

1. Create a new **public** GitHub repository named `pipe-flow-lab` with the default branch **main**. Upload the **contents** of this starter folder into its root, preserving `.github/workflows` (including hidden files). Alternatively, use `git init`, `git add .`, `git commit`, and `git push`.
2. Go to **Settings → Pages → Build and deployment → Source: GitHub Actions**. Visit **Actions** and enable workflows if GitHub prompts for approval.
3. Open **Actions → Run 3D pipe DNS → Run workflow** and accept the default case: `Re=2500`, amplitude `0.3`, short box `L=8`, `fast` grid, physical duration `t=200`, wall-clock budget `42 min`.
4. The first run installs dependencies, executes the 3D solver, extracts plots and statistics, commits small processed output to `data/runs/`, and uploads the raw snapshot files as a **temporary 30-day artifact** on its Actions run page.
5. The commit triggers **Publish results dashboard**, updating `https://YOUR_USERNAME.github.io/pipe-flow-lab/`. The first site may deploy before any runs and show an empty archive. Reload after the DNS action finishes and the publish action completes.
6. For the laminar control run, choose amplitude `0.0` (not a small random disturbance). The unperturbed state should remain Poiseuille, with `D/(2/Re)=1` and Darcy friction `64/Re`.

**Note:** The GitHub Pages dashboard is read-only. Starting a solver job requires clicking **Launch simulation** and using the GitHub Actions workflow form; it cannot securely dispatch an authenticated GitHub workflow from an unauthenticated static site.

## Parameters

| Input | Units and meaning | Default |
|---|---|---:|
| `re` | Bulk-velocity × pipe-diameter Reynolds number | 2500 |
| `amplitude` | Localized 3D roll's peak disturbance velocity (0 = perfectly laminar) | 0.30 |
| `width` | Axial perturbation half-width / radius | 1.5 |
| `length` | Pipe length in radii, `L/D = length/2` | 8 |
| `t_end` | Solver advective time units (not minutes) | 200 |
| `preset` | `fast`: 24×32×32; `refined`: 48×48×48 (z, r, theta) | fast |
| `wall_minutes` | Solver wall-time cap, 1..44 | 42 |

`dt=0.01`, `iterative-cn` stepping, `fd_order=8` and full azimuthal extent are used in both presets. Constant-bulk-velocity forcing is used so drag changes are visible as pressure corrections. GitHub Actions has its own 58-minute job timeout. **Total runtime under an hour is a budget target, not a speed guarantee**: package setup, JAX JIT compilation, plotting, runner contention or numerical stiffness can shorten the simulated duration or cause a job failure. The wall-time stop is gracefully handled by dnsjax but does not guarantee the requested simulated horizon is reached. Inspect `observed_end_time` and completion markers in each run.

### Physics and diagnostics

For pipe radius `R=1`, laminar centerline speed `U_c=1` and bulk speed `U_b=1/2`:

- Laminar total velocity: `u_z(r) = 1-r²`.
- Constant bulk flow is driven by pressure `−dP/dz = 4/Re + (−dP/dz′)`.
- Laminar dissipation in dnsjax's normalization: `D_lam = 2/Re`.
- Darcy friction: `f_D = 64/Re + 16 (−dP/dz′)`.
- `E′` is *perturbation* energy relative to the laminar state, **not** automatically turbulence kinetic energy; departure of the evolving mean profile also contributes.
- The snapshot turbulence intensity is the RMS of deviations from the azimuthally and axially averaged instantaneous radial velocity profiles, divided by `U_b`. It is measured only at published snapshot times, not on every `stats.dat` time sample.
- Vorticity magnitude and 3D positive-Q structures are computed from spatial derivatives of the reconstructed **total** velocity field, using cylindrical-to-Cartesian derivatives.

The outcome heatmap uses the tail of `D/D_lam` as an **exploratory finite-time indicator**: >1.5 = active at end, <1.15 = near laminar at end, otherwise ambiguous. Different seeds, transients, initial disturbances and box lengths can change this result. This is *not* a mathematically established turbulence threshold.

### Scientific caveats

At four pipe diameters the domain is a **minimal periodic cell**. It cannot capture the dynamics of long localized turbulent puffs and cannot quantify sustained turbulence at onset. Production studies should include at least a longer 3D pipe (usually tens of diameters), multiple initial conditions, sensitivity to timestep/grid, physical-time extensions and independent validation. The two provided grids are convenience presets, *not* a convergence certification. `dnsjax` itself is still under active development and must be validated for a particular scientific use case.

### Results retained

- `data/runs/<id>/result.json`: inputs, solver version, exit status, physical-time horizon, time-series diagnostics, snapshot-profile data, and analysis errors.
- `data/runs/<id>/images/`: compact PNGs of velocity, vorticity and Q, extracted from a maximum of five simulation snapshots.
- GitHub Actions **artifacts**, retained for 30 days: original solver `state*.tar` checkpoints, `stats.dat`, solver log. They are not persisted forever in the Git repository (which would become very large).

The dashboard is rebuilt on pushes to `main`, without external hosting services. Its interactive plots use Plotly loaded from a CDN; if an offline browser blocks that CDN, images and the archive still exist but plots will not load.

## Run locally (optional)

Install Python 3.13 and [uv](https://docs.astral.sh/uv/), then:

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python 'dnsjax==0.1.0.1' 'matplotlib>=3.9,<4'
.venv/bin/python scripts/run_case.py --run-id local-test --re 2500 --amplitude 0.3 --t-end 40 --max-wall-minutes 15
.venv/bin/python scripts/analyze.py --source work/local-test --dest data/runs/local-test
python scripts/build_site.py
python -m http.server --directory _site 8000
```

Open http://localhost:8000. For a convergence check, compare `fast` and `refined` at identical physics parameters with a sufficiently long run time.

## Source and credit

Solver: [Gökhan Yalnız, dnsjax](https://github.com/gokhanyalniz/dnsjax) (Apache-2.0; external dependency, not vendored into this repository). The front-end, workflow and extraction code in this repository are separate project glue. Solver behavior and API are version-pinned and should be revisited before upgrading.
