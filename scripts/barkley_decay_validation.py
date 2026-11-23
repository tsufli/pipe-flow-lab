#!/usr/bin/env python3
"""Independent-puff ensemble, direct survival and offline AMS validation.

Uses Barkley (2016) stochastic equations, importing the main deterministic
and Ito solver. All parents are *naturally evolved* noisy puff states.
No browser simulation, artificially interpolated puff, or artificial mass
target in parent selection. Reporting a lifetime requires a Poisson check.
"""
from __future__ import annotations
import argparse, json, math, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from numba import njit, prange
from barkley_decay_ams import step, seed_puff, evolve

@njit(cache=True, parallel=True)
def age_ensemble(q0,u0,r,sigma,dx,dt,seeds,ages,death_mass):
    n=len(seeds); nx=len(q0)
    qs=np.zeros((n,nx)); us=np.zeros((n,nx)); alive=np.zeros(n,np.int8)
    for j in prange(n):
        np.random.seed(seeds[j]); q=q0.copy();u=u0.copy();ok=True
        for k in range(ages[j]):
            q,u=step(q,u,r,sigma,dx,dt)
            if k%50==49:
                mass=0.
                for i in range(nx):mass+=max(q[i],0.)*dx
                if mass<=death_mass:
                    ok=False;break
        if ok:
            qs[j]=q;us[j]=u;alive[j]=1
    return qs,us,alive

@njit(cache=True,parallel=True)
def direct_trajectories(qparents,uparents,r,sigma,dx,dt,seeds,steps,check_stride,death_mass):
    n=len(seeds);nx=qparents.shape[1]
    death=np.full(n,-1.,np.float64)
    for j in prange(n):
        np.random.seed(seeds[j]);q=qparents[j].copy();u=uparents[j].copy()
        for k in range(steps):
            q,u=step(q,u,r,sigma,dx,dt)
            if (k+1)%check_stride==0:
                m=0.
                for i in range(nx):m+=max(q[i],0.)*dx
                if m<=death_mass:
                    death[j]=(k+1)*dt;break
    return death

def components(q,dx,threshold=.5,min_width=1.2):
    smooth=(np.roll(q,2)+np.roll(q,1)+q+np.roll(q,-1)+np.roll(q,-2))/5
    mask=smooth>threshold
    if np.all(mask):return 1
    beg=np.where(mask & ~np.roll(mask,1))[0];count=0
    for b in beg:
        k=int(b);nn=0
        while mask[k]:nn+=1;k=(k+1)%len(q)
        if nn*dx>=min_width:count+=1
    return count

def make_parents(r,sigma,dx,dt,L,n,age_lo,age_hi,seed,death_mass=.15):
    clock=time.perf_counter()
    q0,u0=seed_puff(r,dx,dt,L,seed)
    rng=np.random.default_rng(seed+142)
    seeds=rng.integers(1,2**31-1,size=n,dtype=np.int64)
    ages=np.round(rng.uniform(age_lo,age_hi,size=n)/dt).astype(np.int64)
    qs,us,alive=age_ensemble(q0,u0,r,sigma,dx,dt,seeds,ages,death_mass)
    rows=[];accepted=[]
    for i in range(n):
        if not alive[i]:
            rows.append(dict(candidate=i,age=float(ages[i]*dt),status='decayed'))
            continue
        q=qs[i];u=us[i]
        mass=float(np.maximum(q,0).sum()*dx)
        peak=float(np.max(q));parts=components(q,dx)
        valid=parts==1 and peak>.85 and mass>death_mass
        rows.append(dict(candidate=i,age=float(ages[i]*dt),mass=mass,peak=peak,
                         components=parts,status='accepted' if valid else 'not_isolated'))
        if valid:accepted.append(i)
    print('parents',dict(r=r,sigma=sigma,n=n,accepted=len(accepted),
        mass_mean=float(np.mean([z['mass'] for z in rows if z['status']=='accepted'])) if accepted else None,
        elapsed=round(time.perf_counter()-clock,1)),flush=True)
    if len(accepted)<5:raise RuntimeError('Insufficient isolated surviving puff parents')
    return qs[accepted],us[accepted],rows

def trajectory(q,u,r,sigma,dt,dx,H,save_dt,seed,death_mass):
    stride=round(save_dt/dt);steps=round(H/dt)
    if steps%stride or abs(stride*dt-save_dt)>1e-7:
        raise ValueError('Choose integer H/save_dt and save_dt/dt')
    return evolve(q,u,r,sigma,dt,dx,steps,stride,seed,death_mass)

def ams_pool(poolq,poolu,r,sigma,dx,dt,H,save_dt,N,seed,death_mass=.15,maxit=4000):
    """AMS estimates P(absorption by H) from the empirical parent distribution.

    Rank by absolute minimum q mass across full field, not fractional decrease
    relative to each parent. Clone the entire Markov state at *first* threshold
    crossing; resume with independent Ito noise.
    """
    clock=time.perf_counter();rng=np.random.default_rng(seed)
    picks=rng.choice(len(poolq),size=N,replace=True)
    paths=[];scores=[];deaths=0
    for par in picks:
        qs,us,m,td=trajectory(poolq[par],poolu[par],r,sigma,dt,dx,H,save_dt,
                            int(rng.integers(1,2147483647)),death_mass)
        paths.append((qs,us,m,td))
        scores.append(-float(np.min(m)));deaths+=int(td>=0)
    k=0;level=-death_mass
    while min(scores)<level and k<maxit:
        i=int(np.argmin(scores));s=scores[i]
        donor_set=np.flatnonzero(np.asarray(scores)>s+1e-9)
        if not len(donor_set):raise RuntimeError('Tied AMS selection levels')
        donor=int(rng.choice(donor_set))
        qs,us,m,td=paths[donor]
        crossed=np.flatnonzero(-m>s+1e-9)
        if not len(crossed):raise RuntimeError('No donor crossing')
        j=int(crossed[0]);remaining=H-j*save_dt
        if remaining>0:
            qb,ub,mb,tb=trajectory(qs[j].astype(float),us[j].astype(float),
                              r,sigma,dt,dx,remaining,save_dt,
                              int(rng.integers(1,2147483647)),death_mass)
            qnew=np.concatenate((qs[:j],qb));unew=np.concatenate((us[:j],ub))
            mnew=np.concatenate((m[:j],mb));tnew=j+tb if tb>=0 else -1
        else:
            qnew,unew,mnew,tnew=qs.copy(),us.copy(),m.copy(),td
        paths[i]=(qnew,unew,mnew,tnew)
        scores[i]=-float(np.min(mnew));k+=1
        if k%100==0:print('AMS progress',r,sigma,H,'K',k,'score',-min(scores),
                         's',round(time.perf_counter()-clock,1),flush=True)
    if k>=maxit:raise RuntimeError('Reached AMS max iterations')
    p=(1-1/N)**k
    return dict(H=H,N=N,K=k,p=p,tau=-H/math.log1p(-p),
         initial_direct_deaths=deaths,seed=seed,wall_s=time.perf_counter()-clock)

def bootstrap_prob(values,reps=2000):
    a=np.asarray(values)
    if len(a)<2:return None
    rng=np.random.default_rng(5281)
    samples=np.mean(rng.choice(a,size=(reps,len(a)),replace=True),axis=1)
    return [float(z) for z in np.quantile(samples,[.025,.975])]

def run_case(r,sigma,dx,dt,L,parents,clones,repeats,H_values,direct_n,seed,age_lo,age_hi):
    poolq,poolu,poolrows=make_parents(r,sigma,dx,dt,L,parents,age_lo,age_hi,seed)
    out=dict(schema=2,generated_utc=datetime.now(timezone.utc).isoformat(),
        parameters=dict(r=r,sigma=sigma,dx=dx,dt=dt,L=L,age_range=[age_lo,age_hi]),
        parent_samples=poolrows,accepted_parents=len(poolq),direct=[],ams=[],
        barkley_fit_tau=math.exp(math.exp(4.901*r-1.102)) if sigma==.5 else None,
        status='exploratory, not converged')
    rng=np.random.default_rng(seed+12367)
    for H in H_values:
        if direct_n:
            selected=rng.choice(len(poolq),size=direct_n,replace=True)
            seeds=rng.integers(1,2147483647,size=direct_n,dtype=np.int64)
            clock=time.perf_counter()
            times=direct_trajectories(poolq[selected],poolu[selected],r,sigma,
                                     dx,dt,seeds,round(H/dt),round(.2/dt),.15)
            k=int(np.sum(times>=0));p=k/direct_n;z=1.96
            denominator=1+z*z/direct_n
            center=(p+z*z/(2*direct_n))/denominator
            half=z*math.sqrt(p*(1-p)/direct_n+z*z/(4*direct_n**2))/denominator
            direct=dict(H=H,n=direct_n,decays=k,p=p,
                p_ci95=[max(0,center-half),min(1,center+half)],
                tau=(-H/math.log1p(-p) if 0<p<1 else None),
                decay_times=[float(t) for t in times if t>=0],
                wall_s=time.perf_counter()-clock)
            print('direct',dict(H=H,n=direct_n,decays=k,tau=direct['tau']),flush=True)
            out['direct'].append(direct)
        samples=[]
        for rep in range(repeats):
            run=ams_pool(poolq,poolu,r,sigma,dx,dt,H,.2,clones,
                         seed+3457+10477*rep+int(H*99))
            samples.append(run);print('ams',run,flush=True)
        if samples:
            ps=[x['p'] for x in samples];pmean=float(np.mean(ps))
            ci=bootstrap_prob(ps)
            tci=[-H/math.log1p(-ci[1]),-H/math.log1p(-ci[0])] if ci and ci[0]>0 and ci[1]<1 else None
            out['ams'].append(dict(H=H,replicas=clones,repetitions=len(samples),
                samples=samples,p_mean=pmean,p_ci95=ci,
                tau=-H/math.log1p(-pmean),tau_ci95=tci))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--r',type=float,default=.62)
    ap.add_argument('--sigma',type=float,default=.5)
    ap.add_argument('--dx',type=float,default=.1);ap.add_argument('--dt',type=float,default=.004)
    ap.add_argument('--L',type=float,default=80.)
    ap.add_argument('--parents',type=int,default=36)
    ap.add_argument('--clones',type=int,default=64)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--H',type=float,nargs='+',default=[100.])
    ap.add_argument('--direct',type=int,default=256)
    ap.add_argument('--seed',type=int,default=20261127)
    ap.add_argument('--age-lo',type=float,default=100.)
    ap.add_argument('--age-hi',type=float,default=220.)
    ap.add_argument('--output',type=Path,default=Path('web/barkley/decay-lifetimes/data/validation.json'))
    a=ap.parse_args();clock=time.perf_counter()
    result=run_case(a.r,a.sigma,a.dx,a.dt,a.L,a.parents,a.clones,a.repeats,
         a.H,a.direct,a.seed,a.age_lo,a.age_hi)
    result['wall_s']=time.perf_counter()-clock
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2))
    print('DONE',a.output,round(result['wall_s'],1),flush=True)
if __name__=='__main__':main()
