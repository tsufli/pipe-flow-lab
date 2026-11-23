#!/usr/bin/env python3
"""Offline fixed-horizon adaptive multilevel splitting for Barkley puff decay.

Barkley (2016) (8.14), Itô noise sigma*q*dW with Var(dW_i)=dt/dx.
No browser-side simulation. The inferred lifetime tau=-H/log(1-p_H) requires
exponential survival and is NOT considered converged until validation.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
from numba import njit

@njit(cache=True)
def step(q,u,r,sigma,dx,dt):
    n=len(q); qn=np.empty(n); un=np.empty(n)
    for i in range(n):
        im=i-1 if i>0 else n-1
        ip=i+1 if i<n-1 else 0
        a=u[i]-.8
        grad=(q[i]-q[im])/dx if a>=0 else (q[ip]-q[i])/dx
        reac=q[i]*(r+u[i]-2.-(r+.1)*(q[i]-1.)**2)
        qn[i]=q[i]+dt*(-a*grad+.5*(q[im]-2*q[i]+q[ip])/dx**2+reac)
        fl=.25*(u[im]**2+u[i]**2)-.5*max(abs(u[im]),abs(u[i]))*(u[i]-u[im])
        fr=.25*(u[i]**2+u[ip]**2)-.5*max(abs(u[i]),abs(u[ip]))*(u[ip]-u[i])
        un[i]=u[i]+dt*(-(fr-fl)/dx+.1*(2.-u[i])+.2*(1.-u[i])*q[i])
    if sigma>0:
        s=sigma*math.sqrt(dt/dx)
        for i in range(n):qn[i]+=s*q[i]*np.random.normal()
    return qn,un

@njit(cache=True)
def evolve(q0,u0,r,sigma,dt,dx,steps,stride,seed,death_mass):
    np.random.seed(seed)
    n=steps//stride+1
    qs=np.empty((n,len(q0)),dtype=np.float32)
    us=np.empty((n,len(q0)),dtype=np.float32)
    mass=np.empty(n)
    q=q0.copy();u=u0.copy();tdeath=-1
    for j in range(n):
        qs[j]=q;us[j]=u
        m=0.
        for k in range(len(q)):m+=max(0.,q[k])*dx
        mass[j]=m
        if m<=death_mass:
            tdeath=j
            for h in range(j+1,n):
                qs[h]=q;us[h]=u;mass[h]=m
            break
        if j<n-1:
            for _ in range(stride):q,u=step(q,u,r,sigma,dx,dt)
    return qs,us,mass,tdeath

def seed_puff(r,dx,dt,L,seed):
    x=np.arange(int(round(L/dx)))*dx
    q=1.68/(1+np.exp(np.clip((np.abs(x-L/2)-7)/1.,-60,60)))
    u=np.full(q.shape,2.)
    np.random.seed(seed)
    for _ in range(int(round(350/dt))):q,u=step(q,u,r,0.,dx,dt)
    return q,u

def parent_puff(q,u,r,sigma,dt,dx,seed,age=80.,death_mass=.15):
    np.random.seed(seed)
    for j in range(int(round(age/dt))):
        q,u=step(q,u,r,sigma,dx,dt)
        if (j+1)%100==0 and np.maximum(q,0).sum()*dx<=death_mass:return None
    return q,u

def trajectory(q,u,r,sigma,dt,dx,H,sample_dt,seed,death_mass):
    stride=int(round(sample_dt/dt));steps=int(round(H/dt))
    if steps%stride:raise ValueError("H must be a multiple of sample_dt")
    return evolve(q,u,r,sigma,dt,dx,steps,stride,seed,death_mass)

def ams(r,sigma,dt,dx,L,H,sample_dt,n,seed,death_mass=.15,max_iterations=5000):
    clock=time.perf_counter()
    qdet,udet=seed_puff(r,dx,dt,L,seed)
    parent=None
    for retry in range(20):
        candidate=parent_puff(qdet.copy(),udet.copy(),r,sigma,dt,dx,seed+91+retry*517,death_mass=death_mass)
        if candidate is None:continue
        pq,pu=candidate
        pm=float(np.maximum(pq,0).sum()*dx)
        # Reject decaying remnants and overgrown/multi-puff parents.
        if 3.0<=pm<=14.0 and pq.max()>=.85:
            parent=candidate
            break
    if parent is None:raise RuntimeError("Could not prepare an active, isolated puff parent")
    qref,uref=parent
    ref_mass=float(np.maximum(qref,0).sum()*dx)
    rng=np.random.default_rng(seed+318)
    replicas=[];peaks=[];first_deaths=0
    for j in range(n):
        qh,uh,m,td=trajectory(qref,uref,r,sigma,dt,dx,H,sample_dt,seed+2001+7919*j,death_mass)
        replicas.append((qh,uh,m,td))
        peaks.append(1.-np.min(m)/ref_mass)
        first_deaths+=(td>=0)
    threshold=1.-death_mass/ref_mass
    k=0;trace=[]
    while min(peaks)<threshold and k<max_iterations:
        idx=int(np.argmin(peaks));score=peaks[idx]
        eligible=np.flatnonzero(np.array(peaks)>score+1e-12)
        if not len(eligible):raise RuntimeError("All trajectory scores tied")
        donor=int(rng.choice(eligible))
        qh,uh,m,td=replicas[donor]
        crosses=np.flatnonzero(1.-m/ref_mass>score+1e-12)
        if not len(crosses):raise RuntimeError("Donor never crossed threshold")
        j=int(crosses[0]);remaining=H-j*sample_dt
        # First sampled crossing, with the complete q/u state; new independent
        # noise after that crossing, and exact donor prefix before it.
        if remaining>1e-12:
            qb,ub,mb,tb=trajectory(qh[j].astype(float),uh[j].astype(float),
                 r,sigma,dt,dx,remaining,sample_dt,int(rng.integers(1,2147483647)),death_mass)
            qnew=np.concatenate((qh[:j],qb),axis=0)
            unew=np.concatenate((uh[:j],ub),axis=0)
            mnew=np.concatenate((m[:j],mb))
            death=int(j+tb) if tb>=0 else -1
        else:
            qnew,unew,mnew,death=qh.copy(),uh.copy(),m.copy(),td
        replicas[idx]=(qnew,unew,mnew,death)
        peaks[idx]=1.-np.min(mnew)/ref_mass
        k+=1
        if k%50==0:trace.append([k,round(float(score),5),int(sum(np.array(peaks)>=threshold))])
    if k>=max_iterations:raise RuntimeError("AMS exceeded iteration limit")
    p=(1.-1./n)**k
    tau=-H/math.log1p(-p)
    return dict(type="ams_fixed_horizon",r=r,sigma=sigma,probability=p,lifetime=tau,
                N=n,K=k,H=H,first_generation_decays=first_deaths,
                death_mass=death_mass,reference_mass=ref_mass,
                reference_peak=float(qref.max()),level_trace=trace,
                dt=dt,dx=dx,L=L,sample_dt=sample_dt,wall_s=time.perf_counter()-clock,
                estimator="p=(1-1/N)^K; tau=-H/log(1-p)",
                status="pilot_unvalidated",split_as_survival=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r",type=float,nargs="+",default=[.62,.64,.66,.68,.70,.72])
    p.add_argument("--sigma",type=float,nargs="+",default=[.5])
    p.add_argument("--dx",type=float,default=.1)
    p.add_argument("--dt",type=float,default=.004)
    p.add_argument("--L",type=float,default=80.)
    p.add_argument("--H",type=float,default=120.)
    p.add_argument("--sample-dt",type=float,default=.2)
    p.add_argument("--replicas",type=int,default=48)
    p.add_argument("--seed",type=int,default=23498)
    p.add_argument("--output",type=Path,default=Path("web/barkley/decay-lifetimes/data/results.json"))
    a=p.parse_args();rows=[]
    for sigma in a.sigma:
        for r in a.r:
            sd=a.seed+int(round(sigma*1000))*67+int(round(r*10000))*139
            row=ams(r,sigma,a.dt,a.dx,a.L,a.H,a.sample_dt,a.replicas,sd)
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!="level_trace"}),flush=True)
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(dict(schema=1,generated_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),rows=rows,reference=dict(source="Barkley 2016 Figure 32, NOT generated data",formula="tau=exp(exp(4.901*r-1.102))",sigma=.5)),indent=2))
if __name__=="__main__":main()
