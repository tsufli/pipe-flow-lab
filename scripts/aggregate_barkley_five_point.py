#!/usr/bin/env python3
"""Aggregate independent Barkley-2016, sigma=0.5 AMS campaign batches.

Source files are per-job, naturally sampled puff pools; do not aggregate
individual clones as if independent. Report hierarchical bootstrap intervals
sampling parent-pool batches and AMS repeats within batches.
"""
import argparse
from datetime import datetime, timezone
import json, math
from pathlib import Path
import numpy as np

RS=(.62,.64,.66,.68,.70)
BATCHES=(0,1,2)
H=100.
def tau(p):
    if p<=0:return None
    if p>=1:return 0.
    return -H/math.log1p(-p)
def wilson(k,n):
    if not n:return None
    ph=k/n;z=1.95996398454
    d=1+z*z/n
    c=(ph+z*z/(2*n))/d
    w=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/d
    return [max(0,c-w),min(1,c+w)]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    rows=[]
    for r in RS:
        chunks=[]
        for b in BATCHES:
            f=a.input_dir/f'r_{r:.2f}_batch_{b}.json'
            if not f.exists():raise FileNotFoundError(f)
            d=json.loads(f.read_text())
            par=d['parameters']
            assert abs(par['r']-r)<1e-10 and par['sigma']==.5
            assert par['dx']==.1 and par['dt']==.004 and par['L']==80
            assert len(d['ams'])==1 and d['ams'][0]['H']==H
            assert d['ams'][0]['replicas']==64 and d['ams'][0]['repetitions']==8
            chunks.append(d)
        p_batch=[[float(z['p']) for z in d['ams'][0]['samples']] for d in chunks]
        probs=np.asarray(p_batch,dtype=float)
        p=float(probs.mean())
        # Three independent parent cohorts (each supplies eight AMS runs).
        # Resample cohorts, then repeated AMS realizations inside cohorts.
        rng=np.random.default_rng(20261123+int(1e4*r))
        boot=np.empty(10000,dtype=float)
        for i in range(len(boot)):
            bs=rng.integers(0,len(BATCHES),len(BATCHES))
            draws=[probs[z,rng.integers(0,8,8)].mean() for z in bs]
            boot[i]=np.mean(draws)
        ci=[float(x) for x in np.quantile(boot,[.025,.975])]
        # All direct trajectories in the same horizon and nominal parameters.
        directs=[d0 for d in chunks for d0 in d['direct']]
        k=sum(int(d['decays']) for d in directs)
        n=sum(int(d['n']) for d in directs)
        pci=wilson(k,n)
        direct=dict(n=n,events=k,p=k/n if n else None,p_ci95=pci,
                    tau=tau(k/n) if n else None,
                    tau_ci95=[tau(pci[1]),tau(pci[0])] if pci else None)
        rows.append(dict(r=r,sigma=.5,H=H,dx=.1,dt=.004,L=80,
            n_parent_pools=3,n_ams_repeats=24,n_ams_clones=64,
            n_parent_states=sum(int(d['accepted_parents']) for d in chunks),
            p_ams=p,p_ams_ci95=ci,tau_ams=tau(p),
            tau_ams_ci95=[tau(ci[1]),tau(ci[0])],
            p_ams_by_parent_pool=[float(q.mean()) for q in probs],
            ams_K_by_pool=[[int(x['K']) for x in d['ams'][0]['samples']] for d in chunks],
            direct=direct,
            barkley_fit_tau=math.exp(math.exp(4.901*r-1.102)),
            status="exploratory: 3 independently prepared parent pools; hierarchical bootstrap; no grid convergence"))
    out=dict(schema=1,created_utc=datetime.now(timezone.utc).isoformat(),
        title="Barkley sigma=0.5 five-point AMS decay benchmark",
        complete=True,points=rows,
        numerical_method="Barkley 2016 stochastic Itô PDE; Euler-Maruyama, q upwind, u Rusanov",
        sampling="3 independent parent ensembles at each r; 8 independent AMS runs per parent ensemble, 64 clones",
        event="total positive q mass <=0.15 within 100 time units",
        limitations=["Fixed H=100 assumes approximately exponential survival and quasi-stationary parent ensemble",
        "Bootstrap resamples only 3 distinct parent pools and cannot establish full physical error budget",
        "AMS and direct estimates use the same underlying numerical discretization",
        "Split detection was used for accepting parent states, not as a competing-risk censor during evolution",
        "Grid and observation-horizon convergence have not yet been demonstrated",
        "The published Barkley curve is reference only, not fitted to numerical measurements"])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2))
    print("Wrote",a.output,flush=True)
    for z in rows:
        print(f"r={z['r']:.2f} AMS={z['tau_ams']:.5g} CI={z['tau_ams_ci95']} "
              f"Barkley={z['barkley_fit_tau']:.5g} direct={z['direct']['events']}/{z['direct']['n']}",flush=True)
if __name__=="__main__":main()
