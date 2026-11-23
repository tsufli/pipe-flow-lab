#!/usr/bin/env python3
"""Off-browser 100-run stochastic Barkley puff-front ensemble.

Matches scripts/barkley_spatial.py and web/barkley/spatial/index.html:
cellwise upwind q advection, Rusanov u flux, central q diffusion,
deterministic Heun and multiplicative Itô increments sigma*q_old*sqrt(dt/dx)*Z.
Results are exploratory finite-resolution, finite-lag conditional moments.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq
from scipy.integrate import quad

PARAMS = dict(r=.6, delta=.1, eps1=.1, eps2=.2, zeta=.8, D=.5, U0=2., Ub=1.,
              L=100., N=300, dt=.04, save_dt=.4, T=120., sigma=.5,
              threshold=.5, smooth_cells=1.6, warmup_T=350.)
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "web" / "barkley" / "front-tracking" / "data" / "r_0.6.json"

@njit(cache=True)
def rhs(q, u, dx, r, delta, eps1, eps2, zeta, D, U0, Ub):
    n = len(q)
    dq = np.empty(n); du = np.empty(n)
    for i in range(n):
        im = (i - 1) % n; ip = (i + 1) % n
        a = u[i] - zeta
        adv = a*(q[i]-q[im])/dx if a >= 0 else a*(q[ip]-q[i])/dx
        dq[i] = -adv + D*(q[im]-2*q[i]+q[ip])/dx**2 + q[i]*(r+u[i]-U0-(r+delta)*(q[i]-1)**2)
        fl = .25*(u[im]**2+u[i]**2)-.5*max(abs(u[im]),abs(u[i]))*(u[i]-u[im])
        fr = .25*(u[i]**2+u[ip]**2)-.5*max(abs(u[i]),abs(u[ip]))*(u[ip]-u[i])
        du[i] = -(fr-fl)/dx + eps1*(U0-u[i]) + eps2*(Ub-u[i])*q[i]
    return dq, du

@njit(cache=True)
def advance(q, u, dx, dt, sigma, r, delta, eps1, eps2, zeta, D, U0, Ub):
    a,b = rhs(q,u,dx,r,delta,eps1,eps2,zeta,D,U0,Ub)
    aa,bb = rhs(q+dt*a,u+dt*b,dx,r,delta,eps1,eps2,zeta,D,U0,Ub)
    qnext=q+.5*dt*(a+aa)
    unext=u+.5*dt*(b+bb)
    if sigma:
        qnext += sigma*q*np.sqrt(dt/dx)*np.random.normal(0,1,len(q))
    return qnext,unext

@njit(cache=True)
def run_steps(q0,u0,dx,dt,sigma,nsteps,stride,seed,r,delta,eps1,eps2,zeta,D,U0,Ub):
    np.random.seed(seed)
    nsave=nsteps//stride
    saved=np.empty((nsave+1,len(q0)))
    q=q0.copy();u=u0.copy()
    saved[0,:]=q
    for k in range(1,nsteps+1):
        q,u=advance(q,u,dx,dt,sigma,r,delta,eps1,eps2,zeta,D,U0,Ub)
        if k%stride==0: saved[k//stride,:]=q
    return saved,q,u

def regions(q, dx, threshold, smooth_cells):
    s=gaussian_filter1d(q,smooth_cells,mode="wrap")
    b=s>threshold
    begins=np.where(b & ~np.roll(b,1))[0]
    out=[]
    n=len(q); L=n*dx
    for j in begins:
        e=int(j)
        while b[e]: e=(e+1)%n
        last=(e-1)%n; before=(j-1)%n
        left=(before+.5+(threshold-s[before])/(s[j]-s[before]))*dx
        right=(last+.5+(threshold-s[last])/(s[e]-s[last]))*dx
        width=(right-left)%L
        out.append((float(width),float(left%L),float(right%L)))
    out.sort(reverse=True)
    return out

def tracking(saved, p):
    dx=p["L"]/p["N"]; L=p["L"]
    track=np.full((len(saved),5),np.nan)
    prev=None; prev_index=None
    for t,q in enumerate(saved):
        groups=regions(q,dx,p["threshold"],p["smooth_cells"])
        multiple=sum(z[0]>1.5 for z in groups)>1
        track[t,3]=float(multiple)
        track[t,4]=float(q.max())
        if not groups or groups[0][0]<2 or groups[0][0]>.42*L:
            continue
        width,left,_=groups[0]
        up=left if prev is None else left + L*round((prev-left)/L)
        if prev is not None and abs(up-prev)>max(6.,(t-prev_index)*1.8):
            continue
        track[t,:3]=[up,up+width,width]
        prev=up;prev_index=t
    return track

def analytical_length(p):
    r=p["r"]; d=p["delta"]; D=p["D"]; U0=p["U0"]; Ub=p["Ub"]; z=p["zeta"]
    k=p["eps2"]/p["eps1"]
    def S(u):
        return math.sqrt(D)*(3*math.sqrt(max(0.,r+u-U0)/2)-math.sqrt((r+d)/2))
    c=U0-z-S(U0)
    ud=brentq(lambda u:u-z+S(u)-c,U0-r+1e-10,U0)
    def qp(u): return 1+math.sqrt(max(0,r+u-U0)/(r+d))
    def g(u): return U0-u+k*(Ub-u)*qp(u)
    length=quad(lambda u:(u-c)/(-p["eps1"]*g(u)),ud,U0,epsabs=1e-10)[0]
    return {"bulk_length":length,"speed":c,"u_down":ud,
            "q_down":qp(ud),"definition":"leading-order slow-interior length, excluding O(1) front widths"}

def estimate(tracks, lag, p):
    # One conditional Kramers-Moyal bin at each integer-length midpoint.
    # Adjacent events are not independent: raw counts must not be treated as independent errors.
    L0=tracks[:,:-lag,:]; L1=tracks[:,lag:,:]
    good=np.isfinite(L0[:,:,2]) & np.isfinite(L1[:,:,2])
    good &= (L0[:,:,3]==0) & (L1[:,:,3]==0)
    good &= np.abs(L1[:,:,2]-L0[:,:,2]) < 5
    edges=np.arange(2,15,1.)
    rows=[]
    h=p["save_dt"]*lag
    for low,high in zip(edges[:-1],edges[1:]):
        chosen=good & (L0[:,:,2]>=low) & (L0[:,:,2]<high)
        n=int(chosen.sum()); run_count=int(chosen.any(axis=1).sum())
        row={"ell":float((low+high)/2),"n":n,"runs":run_count}
        if n>=60 and run_count>=10:
            du=(L1[:,:,0]-L0[:,:,0])[chosen]
            dd=(L1[:,:,1]-L0[:,:,1])[chosen]
            dl=dd-du
            mean_u=du.mean();mean_d=dd.mean()
            zdu=du-mean_u;zdd=dd-mean_d
            row.update(A_u=float(mean_u/h),A_d=float(mean_d/h),
                A_l=float(dl.mean()/h),
                D_uu=float(np.mean(zdu*zdu)/(2*h)),
                D_dd=float(np.mean(zdd*zdd)/(2*h)),
                D_ud=float(np.mean(zdu*zdd)/(2*h)),
                D_l=float(np.mean((zdd-zdu)**2)/(2*h)))
        rows.append(row)
    roots=zeros(rows)
    return {"lag":h,"rows":rows,"roots":roots,
            "valid_pairs":int(good.sum()),"total_pairs":int(good.size)}

def zeros(rows):
    out=[]
    valid=[x for x in rows if "A_l" in x]
    for a,b in zip(valid[:-1],valid[1:]):
        if b["ell"]-a["ell"]>1.01:continue
        ya=a["A_l"];yb=b["A_l"]
        if ya*yb<=0 and ya!=yb:
            x=a["ell"]-ya*(b["ell"]-a["ell"])/(yb-ya)
            out.append({"ell":float(x),"stability":"stable" if ya>yb else "unstable"})
    return out

def bootstrap(tracks, lag, p, base, repetitions=150):
    rng=np.random.default_rng(61926+lag)
    runs=len(tracks); pool=[]
    drift=np.full((repetitions,len(base["rows"])),np.nan)
    for b in range(repetitions):
        ids=rng.integers(0,runs,size=runs)
        est=estimate(tracks[ids],lag,p)
        candidates=[z["ell"] for z in est["roots"] if z["stability"]=="stable" and 4<z["ell"]<11]
        if candidates:pool.append(min(candidates,key=lambda x:abs(x-6.5)))
        for j,row in enumerate(est["rows"]):
            if "A_l" in row:drift[b,j]=row["A_l"]
    for j,row in enumerate(base["rows"]):
        vals=drift[:,j];vals=vals[np.isfinite(vals)]
        if len(vals)>=50 and "A_l" in row:
            row["A_l_CI95"]=[float(x) for x in np.quantile(vals,[.025,.975])]
    if len(pool)>=50:
        base["stable_root_bootstrap_CI95"]=[float(x) for x in np.quantile(pool,[.025,.975])]
        base["bootstrap_roots_found"]=len(pool)
    else:
        base["bootstrap_roots_found"]=len(pool)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--runs",type=int,default=100)
    parser.add_argument("--T",type=float,default=120)
    parser.add_argument("--sigma",type=float,default=.5)
    parser.add_argument("--seed",type=int,default=20261123)
    args=parser.parse_args()
    p=dict(PARAMS);p["T"]=float(args.T);p["sigma"]=float(args.sigma)
    dx=p["L"]/p["N"];steps=round(p["T"]/p["dt"]);stride=round(p["save_dt"]/p["dt"])
    assert steps%stride==0 and p["sigma"]>0
    start=time.perf_counter()
    x=(np.arange(p["N"])+.5)*dx
    q=1.65/(1+np.exp(np.minimum(100.,(abs(x-p["L"]/2)-6)/max(dx,1.))))
    u=np.full(p["N"],p["U0"])
    common=(p["r"],p["delta"],p["eps1"],p["eps2"],p["zeta"],p["D"],p["U0"],p["Ub"])
    warmup, q, u = run_steps(q,u,dx,p["dt"],0.,round(p["warmup_T"]/p["dt"]),
                              10,123,*common)
    # Translate entire relaxed puff + recovery tail once; subsequent trajectories
    # remain in lab coordinates, with only modulo-unwrapping for position continuity.
    t0=tracking(q[None,:],p)[0]
    old_center=(t0[0]+t0[1])/2
    shift=int(round((30-old_center)/dx))
    q=np.roll(q,shift);u=np.roll(u,shift)
    deterministic=tracking(q[None,:],p)[0]
    raw_p=dict(p,smooth_cells=0.0001)
    deterministic_raw=tracking(q[None,:],raw_p)[0]
    tracks=[]; profiles=[];failed=0
    for i in range(args.runs):
        saved,q_end,u_end=run_steps(q,u,dx,p["dt"],p["sigma"],steps,stride,
                                    args.seed+1009*i,*common)
        tr=tracking(saved,p)
        tracks.append(tr)
        if i in (0,1,2,3,4,9):
            profiles.append({"run":i+1,
               "frames":[{"t":round(j*p["save_dt"],4),
                          "q":[round(float(v),4) for v in saved[j,::3]]}
                         for j in [0,round(len(saved)*.1),round(len(saved)*.25),
                                   round(len(saved)*.5),round(len(saved)*.75),len(saved)-1]]})
        if not np.any(np.isfinite(tr[:,2])):failed+=1
        if (i+1)%20==0: print("completed",i+1,"of",args.runs,flush=True)
    tracks=np.stack(tracks)
    lags={}
    for k in (1,2,4,8):
        res=estimate(tracks,k,p)
        bootstrap(tracks,k,p,res,repetitions=150)
        lags[str(k)]=res
    finite=np.isfinite(tracks[:,:,2])
    ambiguous=tracks[:,:,3]==1
    thresholds={"q_threshold":p["threshold"],"Gaussian_smoothing_cells":p["smooth_cells"],
      "significant_secondary_region_length":1.5,"minimum_width":2.,"max_width_fraction":.42,
      "discard_ambiguous_pairs":True}
    result={"schema":1,"generated_utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
      "params":p,"seed_base":args.seed,"n_runs":args.runs,"wall_seconds":time.perf_counter()-start,
      "method":"spatial upwind/Rusanov + Heun + Itô sigma q_old sqrt(dt/dx) independent cell noise",
      "tracking":thresholds,
      "quality":{"valid_frames":int(finite.sum()),"all_frames":int(finite.size),
         "ambiguous_frames":int(ambiguous.sum()),"valid_at_final":int(finite[:,-1].sum()),
         "all_invalid_realizations":failed,
         "survival_conditioned":True},
      "asymptotic":analytical_length(p),
      "deterministic":{"threshold_width_smoothed":float(deterministic[2]),
                       "threshold_width_raw":float(deterministic_raw[2]),
                       "peak":float(q.max()),
                       "initial_profile_q":[round(float(v),4) for v in q[::3]]},
      "lag_analysis":lags,
      "trajectories":[{"run":i+1,
           "seed":args.seed+1009*i,
           "x_up":[None if not np.isfinite(y) else round(float(y),3) for y in tr[:,0]],
           "x_down":[None if not np.isfinite(y) else round(float(y),3) for y in tr[:,1]],
           "length":[None if not np.isfinite(y) else round(float(y),3) for y in tr[:,2]],
           "ambiguous":[int(z) for z in tr[:,3]]}
           for i,tr in enumerate(tracks)],
      "profiles":profiles}
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(result,separators=(",",":"),allow_nan=False))
    print("RESULT", json.dumps({"wall_s":result["wall_seconds"],
        "quality":result["quality"],"roots_at_dt04":lags["1"]["roots"],
        "roots_at_dt08":lags["2"]["roots"],"root_ci":lags["2"].get("stable_root_bootstrap_CI95"),
        "asymptotic":result["asymptotic"],"deterministic":result["deterministic"]}))
    print("Wrote",OUTPUT)

if __name__=="__main__":
    main()
