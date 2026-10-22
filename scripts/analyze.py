#!/usr/bin/env python3
"""Turn dnsjax pipe snapshots into fluctuation-energy movies and compact JSON."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


ENERGY_CMAP = LinearSegmentedColormap.from_list(
    'white_viridis',
    ['#ffffff', '#d9f0d3', '#78c679', '#238443', '#225ea8', '#54278f']
)


def finite_json(x):
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.ndarray):
        return [finite_json(y) for y in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [finite_json(y) for y in x]
    if isinstance(x, dict):
        return {str(k): finite_json(v) for k, v in x.items()}
    return x


def fluctuation_energy_density(uprime_z, u_r, u_theta):
    """Pointwise fluctuation kinetic energy density 1/2 |u-u_lam|^2.

    dnsjax stores the perturbation field for pipe flow, so u'_z is already
    the axial fluctuation relative to Hagen–Poiseuille, while u_r and u_theta
    are pure fluctuations.
    """
    return 0.5 * (uprime_z**2 + u_r**2 + u_theta**2)


def radial_weights(r):
    edges = np.empty(len(r) + 1)
    edges[0], edges[-1] = 0.0, 1.0
    edges[1:-1] = (r[:-1] + r[1:]) / 2
    weights = np.maximum(0, np.diff(edges**2))
    return weights / weights.sum()


def section_mean_energy(k, r):
    # Mean over theta and z, weighted by area in radius.
    by_r = np.mean(k, axis=(1, 2))
    return float(np.dot(radial_weights(r), by_r))


def render_energy_frame(k_section, r, theta, output_path, title, vmax=None):
    x = r[:, None] * np.cos(theta[None, :])
    y = r[:, None] * np.sin(theta[None, :])
    fig, ax = plt.subplots(figsize=(5.4, 4.9), layout='constrained')
    if vmax is None or not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    pc = ax.scatter(
        x.ravel(), y.ravel(), c=k_section.ravel(), s=22,
        cmap=ENERGY_CMAP, vmin=0.0, vmax=vmax
    )
    circle = plt.Circle((0, 0), 1.0, color='#222222', fill=False, lw=1.0)
    ax.add_patch(circle)
    ax.set(
        xlabel='x / R', ylabel='y / R', aspect='equal',
        xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), title=title
    )
    fig.colorbar(pc, ax=ax, label="Fluctuation kinetic energy density ½|u-u_lam|²")
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def build_gif(frame_paths, gif_path, fps=3):
    images = [Image.open(p).convert('P', palette=Image.ADAPTIVE) for p in frame_paths]
    if not images:
        return False
    duration = int(round(1000 / max(fps, 1)))
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return True


def analyze_case(source, dest):
    meta_file = source / 'run_metadata.json'
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    dest.mkdir(parents=True, exist_ok=True)
    frames_dir = dest / 'frames'
    frames_dir.mkdir(exist_ok=True)
    output = {
        'metadata': meta,
        'frames': [],
        'movie': None,
        'errors': [],
        'status': 'failed',
        'field_definition': "k'(x,t)=1/2 |u-u_lam|^2 on a pipe cross-section, using the dnsjax perturbation field.",
    }
    try:
        from dnsjax.analysis import read_state, read_meta
    except Exception as exc:
        output['errors'].append('Unable to load dnsjax analysis: ' + repr(exc))
        (dest / 'result.json').write_text(json.dumps(finite_json(output), indent=2))
        return output

    snaps = sorted(source.glob('state*.tar'))
    if not snaps:
        output['errors'].append('No readable snapshots were found')
        (dest / 'result.json').write_text(json.dumps(finite_json(output), indent=2))
        return output

    snapshot_payloads = []
    section_maxima = []
    for sn in snaps:
        try:
            st = read_state(str(sn))
            uzp, ur, uth = [np.asarray(u, dtype=float) for u in st.physical]
            r, th, z = [np.asarray(v, dtype=float).reshape(-1) for v in st.physical_coords]
            k = fluctuation_energy_density(uzp, ur, uth)
            z_index = k.shape[2] // 2
            k_section = k[:, :, z_index]
            t = None
            try:
                sm = read_meta(str(sn))
                if 't' in sm:
                    t = float(sm['t'])
            except Exception:
                pass
            snapshot_payloads.append({
                'snapshot': sn.name,
                't': t,
                'r': r,
                'theta': th,
                'k_section': k_section,
                'section_mean_energy': section_mean_energy(k, r),
                'section_max_energy': float(np.nanmax(k_section)),
                'z_index': int(z_index),
                'z_value': float(z[z_index]),
            })
            section_maxima.append(float(np.nanmax(k_section)))
        except Exception as exc:
            output['errors'].append(f'{sn.name}: {type(exc).__name__}: {exc}')

    if not snapshot_payloads:
        output['errors'].append('All snapshot reads failed')
        (dest / 'result.json').write_text(json.dumps(finite_json(output), indent=2))
        return output

    vmax = max(section_maxima) if section_maxima else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    frame_paths = []
    for i, item in enumerate(snapshot_payloads):
        title = f"Fluctuation kinetic energy cross-section"
        if item['t'] is not None:
            title += f" · t = {item['t']:.2f}"
        png_name = f"energy_{i:03d}.png"
        render_energy_frame(item['k_section'], item['r'], item['theta'], frames_dir / png_name, title, vmax=vmax)
        frame_paths.append(frames_dir / png_name)
        output['frames'].append({
            'snapshot': item['snapshot'],
            't': item['t'],
            'image': f'frames/{png_name}',
            'section_mean_energy': item['section_mean_energy'],
            'section_max_energy': item['section_max_energy'],
            'z_index': item['z_index'],
            'z_value': item['z_value'],
        })

    gif_path = dest / 'fluctuation_energy.gif'
    if build_gif(frame_paths, gif_path, fps=3):
        output['movie'] = {'gif': 'fluctuation_energy.gif', 'fps': 3, 'n_frames': len(frame_paths), 'vmax': vmax}

    output['status'] = 'processed' if not output['errors'] else 'partial'
    (dest / 'result.json').write_text(json.dumps(finite_json(output), indent=2))
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--dest', required=True)
    args = p.parse_args()
    out = analyze_case(Path(args.source), Path(args.dest))
    if out['errors']:
        print('Analysis warnings:', out['errors'], file=sys.stderr)
    print('Analysis status:', out['status'], 'frames:', len(out['frames']))
    if out['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
