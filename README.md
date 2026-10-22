# Pipe Flow Laboratory — fluctuation-energy movies from 3D DNS

A reproducible 3D pipe-flow experiment run by GitHub Actions and published to GitHub Pages. The solver is **[dnsjax](https://github.com/gokhanyalniz/dnsjax)** (version `0.1.0.1`, Apache-2.0).

## Main output

This repository is centered on **one primary visualization**:

- subtract the laminar Hagen–Poiseuille profile,
- compute the fluctuation kinetic energy density

  `k′ = 1/2 |u - u_lam|² = 1/2[(u_z-u_{z,lam})² + u_r² + u_θ²]`,

- render a **2D pipe cross-section** at the axial mid-plane,
- and assemble those frames into a **movie through time**.

**White means `k′ = 0`**, i.e. perfectly laminar flow. Colored regions indicate where fluctuation energy is present.

## What gets saved

Each processed run in `data/runs/<run-id>/` contains:

- `result.json`: metadata and per-frame information
- `frames/*.png`: individual cross-sectional fluctuation-energy frames
- `fluctuation_energy.gif`: animation across saved times

Raw solver outputs (`state*.tar`, `stats.dat`, `solver.log`) are uploaded as temporary GitHub Actions artifacts and are **not** committed permanently to the repository.

## Workflow

1. Open **Actions → Run 3D pipe DNS → Run workflow**.
2. Choose physics parameters such as `Re`, perturbation amplitude, and simulation length.
3. The workflow runs the DNS, saves snapshots, converts them into fluctuation-energy frames, assembles a GIF, and commits the small processed results.
4. A Pages deployment publishes the archive and viewer.

## Parameters

| Input | Meaning | Default |
|---|---|---:|
| `re` | Reynolds number based on bulk velocity and pipe diameter | 2500 |
| `amplitude` | Peak amplitude of the localized perturbation (`0` = exact laminar initial condition) | 0.30 |
| `width` | Axial half-width of the initial localized rolls | 1.5 |
| `length` | Pipe length in radii (`L/D = length/2`) | 8 |
| `t_end` | Simulated time | 200 |
| `preset` | `fast` = 24×32×32, `refined` = 48×48×48 | fast |
| `wall_minutes` | Wall-clock budget in minutes | 42 |

## Scientific note

The movie uses the **fluctuation field relative to laminar flow**, not the total axial velocity. In dnsjax pipe snapshots, the stored state is already the perturbation field, so the analysis naturally computes the desired quantity.

The default domain is a short periodic pipe and is mainly a proof of concept. It is useful for seeing how finite-amplitude disturbances appear, spread, and decay, but it is not yet a full research-grade study of puff dynamics in a long pipe.
