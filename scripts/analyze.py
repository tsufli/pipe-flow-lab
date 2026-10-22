#!/usr/bin/env python3
"""Turn DNS diagnostics and snapshots into small, publishable JSON and images."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def finite_json(x):
    """JSON-friendly numbers, with no misleading NaN encodings."""
    if isinstance(x, (np.integer, int)): return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.ndarray): return [finite_json(y) for y in x.tolist()]
    if isinstance(x, (list, tuple)): return [finite_json(y) for y in x]
    if isinstance(x, dict): return {str(k): finite_json(v) for k,v in x.items()}
    return x


def derive_stats(stats, re):
    """D laminar = 2/Re; -dPdz' is the *excess* pressure force."""
    time = np.asarray(stats['t'], dtype=float)
    D = np.asarray(stats['D'], dtype=float)
    ep = np.asarray(stats["E'"], dtype=float) if "E'" in stats else np.full(len(time), np.nan)
    if "-dPdz'" in stats:
        delta_p = np.asarray(stats["-dPdz'"], dtype=float)
        pressure = (4/re) + delta_p
        friction = 16 * pressure  # Dpipe=2, Ub=0.5; f_D=16*(-dP/dz)
    else:
        delta_p = np.full(len(time), np.nan)
        friction = np.full(len(time), np.nan)
    return {'t': time, 'D': D, 'D_over_laminar': D/(2/re),
            'perturbation_energy': ep, 'pressure_correction': delta_p,
            'friction': friction, 'friction_laminar': 64/re}


def cylindrical_operators(u_z, u_r, u_th, r, theta, z):
    """Cartesian velocity gradient evaluated on native cylindrical nodes.

    r is strictly positive for dnsjax's half-Chebyshev grid.
    Q = -tr(grad u squared)/2 for solenoidal flow.
    """
    RR = r[:, None, None]
    co = np.cos(theta)[None, :, None]
    si = np.sin(theta)[None, :, None]
    ux = u_r*co - u_th*si
    uy = u_r*si + u_th*co
    uz = u_z
    dth = float(np.median(np.diff(theta)))
    dz = float(np.median(np.diff(z)))
    components = (ux, uy, uz)
    gradients = []
    for u in components:
        dr = np.gradient(u, r, axis=0, edge_order=2)
        dt = (np.roll(u, -1, axis=1)-np.roll(u, 1, axis=1))/(2*dth)
        ddz = (np.roll(u, -1, axis=2)-np.roll(u, 1, axis=2))/(2*dz)
        gradients.append((co*dr-si*dt/RR, si*dr+co*dt/RR, ddz))
    G = np.array(gradients)
    q = -0.5*sum(G[i,j]*G[j,i] for i in range(3) for j in range(3))
    vort_sq = sum((G[k,j]-G[j,k])**2 for k,j in [(0,1),(0,2),(1,2)])
    return q, np.sqrt(np.maximum(0, vort_sq))


def radial_weights(r):
    edges = np.empty(len(r)+1)
    edges[0], edges[-1] = 0.0, 1.0
    edges[1:-1] = (r[:-1] + r[1:])/2
    weights = np.maximum(0, np.diff(edges**2))
    return weights/weights.sum()


def turbulence_intensity(u_z, u_r, u_th, r):
    """RMS fluctuation about the radial, theta-z averaged *total* flow / Ub."""
    fields = [u_z, u_r, u_th]
    fluctuations = [(u-u.mean(axis=(1,2), keepdims=True)) for u in fields]
    energy_by_r = sum(np.mean(f*f, axis=(1,2)) for f in fluctuations)
    return float(np.sqrt(np.dot(radial_weights(r), energy_by_r)) / 0.5)


def render_frame(data, path, idx):
    uz_prime, ur, uth = [np.asarray(u, dtype=float) for u in data.physical]
    r, th, z = [np.asarray(v, dtype=float).reshape(-1) for v in data.physical_coords]
    uz = uz_prime + (1-r[:,None,None]**2)
    # Save scientifically useful small figures; raw velocity arrays are never published.
    images = {}
    cross = uz[:,:,uz.shape[2]//2]
    X = r[:,None]*np.cos(th[None,:]); Y = r[:,None]*np.sin(th[None,:])
    fig,ax=plt.subplots(figsize=(5.4,4.6),layout='constrained')
    pc=ax.scatter(X.ravel(),Y.ravel(),c=cross.ravel(),s=17,cmap='coolwarm',vmin=0,vmax=1.4)
    ax.set(xlabel='x / R',ylabel='y / R',title='Axial velocity, central cross-section',aspect='equal',xlim=(-1.05,1.05),ylim=(-1.05,1.05))
    fig.colorbar(pc, ax=ax, label='u_z / U_center,lam')
    name=f'frame_{idx:02d}_velocity.png';fig.savefig(path/name,dpi=125);plt.close(fig)
    images['velocity']='images/'+name
    q,vorticity=cylindrical_operators(uz,ur,uth,r,th,z)
    fig,ax=plt.subplots(figsize=(5.4,4.6),layout='constrained')
    vc=vorticity[:,:,vorticity.shape[2]//2]
    sc=ax.scatter(X.ravel(),Y.ravel(),c=vc.ravel(),s=17,cmap='magma')
    ax.set(xlabel='x / R',ylabel='y / R',title='Vorticity magnitude, cross-section',aspect='equal',xlim=(-1.05,1.05),ylim=(-1.05,1.05))
    fig.colorbar(sc,ax=ax,label='|ω|')
    name=f'frame_{idx:02d}_vorticity.png';fig.savefig(path/name,dpi=125);plt.close(fig)
    images['vorticity']='images/'+name
    fig=plt.figure(figsize=(7,4.7),layout='constrained'); ax=fig.add_subplot(111, projection='3d')
    # Filter numerical round-off to avoid false vortex markers in exact laminar flow.
    q_floor=max(1e-6, 0.01*float(np.nanmax(np.abs(q))))
    positives=q[(q>q_floor)&np.isfinite(q)]
    if positives.size:
        cutoff=np.quantile(positives, 0.97)
        select=np.argwhere(q>max(cutoff,q_floor))
        if len(select)>2500: select=select[np.linspace(0,len(select)-1,2500,dtype=int)]
        ri,ti,zi=select.T
        xx=r[ri]*np.cos(th[ti]); yy=r[ri]*np.sin(th[ti]); zz=z[zi]
        artist=ax.scatter(zz,xx,yy,c=q[ri,ti,zi],s=2.3,alpha=0.48,cmap='plasma')
        fig.colorbar(artist,ax=ax,shrink=0.7,label='Q')
    else:
        ax.text2D(0.12,0.54,'No positive-Q structures in snapshot',transform=ax.transAxes)
    ax.set(xlabel='z / R', ylabel='x / R',zlabel='y / R',title='3D positive-Q vortex structures')
    ax.set_box_aspect((max(float(z[-1]-z[0]),1),2,2), zoom=.85)
    name=f'frame_{idx:02d}_q.png';fig.savefig(path/name,dpi=125);plt.close(fig)
    images['q3d']='images/'+name
    return {'r':r.tolist(),'uz_profile':uz.mean(axis=(1,2)).tolist(),
            'rms_intensity': turbulence_intensity(uz,ur,uth,r),
            'images':images}


def analyze_case(source, dest):
    meta_file = source/'run_metadata.json'
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'images').mkdir(exist_ok=True)
    output={'metadata':meta,'series':{},'frames':[],'errors':[]}
    try:
        from dnsjax.analysis import read_dat,read_state,read_meta
    except Exception as e:
        output['errors'].append('Unable to load dnsjax analysis: '+repr(e))
        (dest/'result.json').write_text(json.dumps(finite_json(output),indent=2))
        return output
    if (source/'stats.dat').exists():
        try:
            s = read_dat(source/'stats.dat')
            if len(s.get('t',[])):
                output['series']=finite_json(derive_stats(s, meta['re']))
                output['observed_end_time']=float(np.asarray(s['t'])[-1])
        except Exception as e:
            output['errors'].append('Stats: '+repr(e))
    snaps = sorted(source.glob('state*.tar'))
    if snaps:
        positions=np.linspace(0,len(snaps)-1,min(len(snaps),5),dtype=int)
        for i,j in enumerate(sorted(set(positions))):
            sn=snaps[j]
            try:
                st=read_state(str(sn))
                frame=render_frame(st,dest/'images',i)
                # Snapshot metadata may contain actual time; do not infer from name.
                try:
                    sm=read_meta(str(sn))
                    if 't' in sm: frame['t']=float(sm['t'])
                except Exception: pass
                frame['snapshot']=sn.name
                output['frames'].append(frame)
            except Exception as e:
                output['errors'].append(f'{sn.name}: {type(e).__name__}: {e}')
    if not output['series'] and not output['frames']:
        output['errors'].append('No readable solver results')
    # No binary snapshots or giant logs committed to Git. Mark derived-data status accurately.
    output['status'] = ('processed' if output['series'] and not output['errors'] else
                        'partial' if output['series'] or output['frames'] else 'failed')
    (dest/'result.json').write_text(json.dumps(finite_json(output),indent=2))
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--dest',required=True)
    args=p.parse_args()
    out=analyze_case(Path(args.source),Path(args.dest))
    if out['errors']: print('Analysis warnings:',out['errors'],file=sys.stderr)
    print('Analysis status:',out['status'],'frames:',len(out['frames']))
    if out['status']=='failed':
        raise SystemExit(1)

if __name__=='__main__': main()
