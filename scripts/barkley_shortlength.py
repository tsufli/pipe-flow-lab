#!/usr/bin/env python3
"""Long/branched stochastic Barkley puff study of short lengths at r=0.6.

Same physical PDE and stochastic step as barkley_front_tracking.py; richer
front observables, separate low/high thresholds and confirmed decay censoring.
Off-browser runs; whole-realization / ancestor-block bootstrap. No imposed OU.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from numba import njit
from barkley_front_tracking import PARAMS, advance, run_steps, regions, analytical_length

BASE = Path(__file__).resolve().parents[1]
OUTPUT = BASE / "web/barkley/front-tracking/data/shortlength_r_0.6.json"
EDGE_FILE = BASE / "web/barkley/edge-scan/data/r_0.595714.json"
BINS = np.r_[np.arange(0,6.5,.5),np.arange(7.,16.,1.)]
LAGS = (1,2,4,8)  # saved frames; h=.2,.4,.8,1.6
COEFF = ("A_u","A_d","A_l","D_uu","D_dd","D_ud","D_l")

@njit(cache=True)
def evolve_until(q0,u0,dx,dt,sigma,nsteps,stride,seed,common,
                 quiet_peak=.05,quiet_mass=.25,quiet_u=.15,confirm_time=12.):
    """Save q,u for later tracking and branch point selection.
    Early-stop ONLY after persistent low-q activity AND recovered u.
    Confirmation is retrospective: censor from beginning of quiet episode.
    """
    np.random.seed(seed)
    nmax=nsteps//stride+1
    qhist=np.empty((nmax,len(q0)),dtype=np.float64)
    uhist=np.empty((nmax,len(q0)),dtype=np.float64)
    diag=np.empty((nmax,3),dtype=np.float64)
    q=q0.copy();u=u0.copy()
    quiet_start=-1
    onset=-1; confirm=-1
    saved=0
    for k in range(nsteps+1):
        if k>0:
            q,u=advance(q,u,dx,dt,sigma,common[0],common[1],common[2],common[3],
                        common[4],common[5],common[6],common[7])
        if k%stride!=0: continue
        qhist[saved]=q;uhist[saved]=u
        peak=np.max(q);mass=np.sum(np.maximum(q,0))*dx
        udiff=np.max(np.abs(u-common[6]))
        diag[saved,0]=peak;diag[saved,1]=mass;diag[saved,2]=udiff
        if peak<quiet_peak and mass<quiet_mass:
            if quiet_start<0: quiet_start=saved
            if (saved-quiet_start)*stride*dt>=confirm_time and udiff<quiet_u:
                onset=quiet_start;confirm=saved
                saved+=1
                break
        else:
            quiet_start=-1
        saved+=1
    return qhist[:saved],uhist[:saved],diag[:saved],onset,confirm

def tracked(qhist,p,threshold,minwidth=.55):
    """Follow the widest threshold-crossing component; retain ambiguity and
    unwrapping flags. qpeak/mass measured on unsmoothed physical field.
    """
    dx=p["L"]/p["N"];L=p["L"]; dt=p["save_dt"]
    rows=np.full((len(qhist),5),np.nan)
    prev=None;prev_index=None
    for k,q in enumerate(qhist):
        groups=regions(q,dx,threshold,p["smooth_cells"])
        rows[k,3]=float(sum(z[0]>1.0 for z in groups)>1)
        rows[k,4]=float(q.max())
        if not groups or not (minwidth<=groups[0][0]<.42*L):continue
        width,left,_=groups[0]
        up=left if prev is None else left+L*round((prev-left)/L)
        # Don't allow a tracker jump from a component switch; this does NOT
        # truncate genuine shrinkage Δlength.
        if prev is not None and abs(up-prev)>max(4.,1.8*(k-prev_index)*dt):
            continue
        rows[k,:3]=[up,up+width,width]
        prev=up;prev_index=k
    return rows

def data_until_confirmed(track,onset):
    t=track.copy()
    if onset>=0: t[onset:,:3]=np.nan
    return t

def bin_sums(tracks,lag,p,edges=BINS):
    """Per-trajectory sufficient statistics; gives fast block bootstrap."""
    nbin=len(edges)-1
    out=np.zeros((len(tracks),nbin,6),dtype=np.float64)
    invalid_splits=0
    for i,tr in enumerate(tracks):
        if len(tr)<=lag:continue
        # Require *all intermediate saved states* to be continuously measurable.
        valid=np.isfinite(tr[:,2])&(tr[:,3]==0)
        bad=np.r_[0,np.cumsum(~valid)]
        good=(bad[lag+1:]-bad[:-lag-1])==0
        # Raw left/right crossing positions, not artificially centered.
        du=tr[lag:,0]-tr[:-lag,0]
        dd=tr[lag:,1]-tr[:-lag,1]
        dl=dd-du
        finite=np.isfinite(du)&np.isfinite(dd)&np.isfinite(tr[:-lag,2])
        good &= finite
        # Guard only discontinuous component switching, not genuine shrinkage.
        good &= (np.abs(dl)<=12.)
        invalid_splits += int(np.count_nonzero(finite & ~good))
        if not np.any(good):continue
        bins=np.searchsorted(edges,tr[:-lag,2][good],side="right")-1
        inside=(bins>=0)&(bins<nbin)
        bins=bins[inside];u=du[good][inside];d=dd[good][inside]
        for slot,w in enumerate((None,u,d,u*u,d*d,u*d)):
            out[i,:,slot]=np.bincount(bins,weights=w,minlength=nbin)
    return out,invalid_splits

def compute_coeff(tot,h):
    count=tot[:,0]
    with np.errstate(divide="ignore",invalid="ignore"):
        mu=tot[:,1]/count;md=tot[:,2]/count
        vu=np.maximum(0.,tot[:,3]/count-mu*mu)
        vd=np.maximum(0.,tot[:,4]/count-md*md)
        cud=tot[:,5]/count-mu*md
    return np.array([mu/h,md/h,(md-mu)/h,vu/(2*h),vd/(2*h),
                     cud/(2*h),np.maximum(0.,vu+vd-2*cud)/(2*h)]).T

def find_roots(rows):
    out=[]
    good=[r for r in rows if "A_l" in r]
    for a,b in zip(good[:-1],good[1:]):
        if (b["lo"]-a["hi"])>.0001:continue
        va=a["A_l"];vb=b["A_l"]
        if va*vb<0:
            ell=a["ell"]+(b["ell"]-a["ell"])*(-va)/(vb-va)
            out.append({"ell":round(float(ell),5),
                        "stability":"stable" if va>vb else "unstable"})
    return out

def fit(tracks,lag,p,group_ids=None,bootstrap=150,seed=33820):
    """Conditioned finite-lag moments and parent-block bootstrap.
    group_ids groups descendant branches descended from one natural snapshot.
    """
    arrays,rejected=bin_sums(tracks,lag,p)
    if group_ids is None:group_ids=list(range(len(tracks)))
    group_ids=np.asarray(group_ids)
    _,index=np.unique(group_ids,return_inverse=True)
    ng=int(index.max()+1)
    blocks=np.zeros((ng,len(BINS)-1,6))
    for i,b in enumerate(index):blocks[b]+=arrays[i]
    total=blocks.sum(axis=0)
    n_ancestors=(blocks[:,:,0]>0).sum(axis=0)
    h=lag*p["save_dt"]
    values=compute_coeff(total,h)
    rows=[]
    def available(j,bs=total):
        return bs[j,0]>=75 and n_ancestors[j]>=12
    for j,(lo,hi) in enumerate(zip(BINS[:-1],BINS[1:])):
        row={"lo":float(lo),"hi":float(hi),"ell":float((lo+hi)/2),
             "n":int(total[j,0]),"ancestors":int(n_ancestors[j])}
        if available(j):
            row.update({k:float(v) for k,v in zip(COEFF,values[j])})
        rows.append(row)
    roots=find_roots(rows)
    rng=np.random.default_rng(seed+lag)
    quantiles=np.full((bootstrap,len(rows),len(COEFF)),np.nan)
    roots_boot=[]
    for b in range(bootstrap):
        bs=blocks[rng.integers(ng,size=ng)].sum(axis=0)
        vals=compute_coeff(bs,h)
        valid=bs[:,0]>=75  # bootstrap already drawn from contributing ancestors
        quantiles[b,valid,:]=vals[valid,:]
        rb=[]
        for j in range(len(rows)):
            if valid[j]:
                rb.append(dict(rows[j], A_l=float(vals[j,2])))
            else:rb.append(dict(rows[j], A_l=None))
        cand=find_roots([r for r in rb if r["A_l"] is not None])
        stable=[z["ell"] for z in cand if z["stability"]=="stable" and 3<z["ell"]<11]
        if stable:roots_boot.append(min(stable,key=lambda x:abs(x-6.7)))
    for j,row in enumerate(rows):
        if "A_l" not in row:continue
        for c,k in enumerate(COEFF):
            arr=quantiles[:,j,c];arr=arr[np.isfinite(arr)]
            if len(arr)>=bootstrap*.6:
                row[k+"_CI95"]=[float(z) for z in np.quantile(arr,[.025,.975])]
    ans={"lag":h,"rows":rows,"roots":roots,"valid_pairs":int(total[:,0].sum()),
         "tracked_censoring_or_component_breaks":rejected,
         "independent_ancestors":ng,"bootstrap_replicates":bootstrap}
    if len(roots_boot)>=bootstrap*.5:
        ans["stable_root_bootstrap_CI95"]=[float(z) for z in np.quantile(roots_boot,[.025,.975])]
    return ans

def edge_reference(p):
    j=json.loads(EDGE_FILE.read_text())
    q=np.array(j["edge"]["q"],dtype=float)
    subdx=j["L"]/len(q)
    x=np.arange(len(q))*subdx
    grid=np.arange(round(j["L"]/(.3333333333)))*(.3333333333)
    qi=np.interp(grid,x,q,period=j["L"])
    features={}
    for threshold in (.15,.2,.5):
        parts=regions(qi,grid[1]-grid[0],threshold,p["smooth_cells"])
        features[str(threshold)]=float(parts[0][0]) if parts else None
    return {"reference_r":j["r"],"source":"edge-scan/r_0.595714.json (numerical decay edge, nearest scan point)",
            "peak":float(q.max()),"mass":float(j["edge_mass"]),
            "width_at_threshold":features,
            "profile_q":[round(float(z),5) for z in q],
            "profile_dx":subdx,"profile_L":j["L"]}

def slim_track(track,decimation=5):
    return {"up":[None if not np.isfinite(z) else round(float(z),3) for z in track[::decimation,0]],
            "down":[None if not np.isfinite(z) else round(float(z),3) for z in track[::decimation,1]],
            "ell":[None if not np.isfinite(z) else round(float(z),3) for z in track[::decimation,2]],
            "ambiguous":[int(z==1) for z in track[::decimation,3]]}

def json_ready(value):
    """JSON must distinguish an unmeasurable front (null) from a real number."""
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,dict):
        return {k:json_ready(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_ready(v) for v in value]
    return value

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=int,default=500)
    parser.add_argument("--T",type=float,default=300.)
    parser.add_argument("--branches",type=int,default=4)
    parser.add_argument("--parents",type=int,default=40)
    parser.add_argument("--edge-parents",type=int,default=40)
    parser.add_argument("--branch-T",type=float,default=60.)
    parser.add_argument("--sigma",type=float,default=.5)
    parser.add_argument("--seed",type=int,default=20261129)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    start=time.perf_counter()
    p=dict(PARAMS,T=float(args.T),save_dt=.2,sigma=args.sigma)
    dx=p["L"]/p["N"];stride=round(p["save_dt"]/p["dt"]);nsteps=round(p["T"]/p["dt"])
    assert abs(stride*p["dt"]-p["save_dt"])<1e-9
    assert nsteps%stride==0
    common=np.array([p[k] for k in ("r","delta","eps1","eps2","zeta","D","U0","Ub")])
    x=(np.arange(p["N"])+.5)*dx
    q=1.65/(1+np.exp(np.minimum(100.,(np.abs(x-p["L"]/2)-6)/max(dx,1.))))
    u=np.full(p["N"],p["U0"])
    _,q,u=run_steps(q,u,dx,p["dt"],0.,round(p["warmup_T"]/p["dt"]),10,123,*common)
    init_track=tracked(q[None,:],p,.5)
    origin=(init_track[0,0]+init_track[0,1])/2
    shift=round((30-origin)/dx)
    q=np.roll(q,shift);u=np.roll(u,shift)
    det={}
    for thr in (.15,.5):
        a=tracked(q[None,:],p,thr)[0]
        det[str(thr)]=float(a[2])
    parent_snaps=[]
    edge_parent_count=0;short_parent_count=0
    long_high=[];long_low=[];viz=[];samples=[];classification=[]
    peak_short=[]
    for i in range(args.runs):
        seed=args.seed+1009*i
        qhist,uhist,diag,onset,confirm=evolve_until(q,u,dx,p["dt"],p["sigma"],nsteps,stride,
                                                    seed,common)
        high=data_until_confirmed(tracked(qhist,p,.5),onset)
        low=data_until_confirmed(tracked(qhist,p,.15),onset)
        long_high.append(high);long_low.append(low)
        qc=np.copy(diag)
        if onset>=0:qc[onset:,:]=np.nan
        classification.append({"run":i+1,"seed":seed,"decayed":bool(confirm>=0),
           "onset_t":None if onset<0 else round(onset*p["save_dt"],3),
           "confirmed_t":None if confirm<0 else round(confirm*p["save_dt"],3),
           "computed_until_t":round((len(diag)-1)*p["save_dt"],3),
           "valid_high":int(np.isfinite(high[:,2]).sum()),
           "valid_low":int(np.isfinite(low[:,2]).sum())})
        # Two *distinct* selected populations. Keep at most one saved parent
        # per independent natural run to preserve ancestor bootstrap validity.
        eligible=(np.arange(len(high))*p["save_dt"]>12.)
        if onset>=0:eligible &= (np.arange(len(high))<onset)
        selected=None
        if edge_parent_count<args.edge_parents:
            # Low q-peak, approximately edge-height profiles: needed because
            # q=.5 crossing ceases before the decay-edge amplitude (~.30).
            hits=np.where(eligible&np.isfinite(low[:,2])&
                  (low[:,2]>=.55)&(low[:,2]<=5.5)&
                  (low[:,3]==0)&(diag[:,0]>=.18)&(diag[:,0]<=.48))[0]
            if len(hits):
                k=int(hits[0]); selected=("edge-like",k);edge_parent_count+=1
        if selected is None and short_parent_count<args.parents:
            hits=np.where(eligible&np.isfinite(high[:,2])&
                  (high[:,2]>=1.4)&(high[:,2]<=3.5)&
                  (high[:,3]==0)&(diag[:,0]>.53))[0]
            if len(hits):
                k=int(hits[0]);selected=("short-high",k);short_parent_count+=1
        if selected is not None:
            kind,k=selected
            parent_snaps.append({"parent":i+1,"kind":kind,"time":k*p["save_dt"],
                "q":qhist[k].copy(),"u":uhist[k].copy(),
                "start_ell_05":float(high[k,2]),
                "start_ell_015":float(low[k,2]),
                "qmax":float(diag[k,0])})
        if i<30:
            viz.append({"run":i+1,"seed":seed,"high":slim_track(high),
                        "low":slim_track(low)})
        if i in (0,1,2,3,4,9):
            picks=np.linspace(0,len(qhist)-1,7,dtype=int)
            samples.append({"run":i+1,"frames":[{"t":round(k*p["save_dt"],3),
              "q":[round(float(z),4) for z in qhist[k,::3]],
              "qmax":round(float(diag[k,0]),4),
              "mass":round(float(diag[k,1]),4)} for k in picks]})
        if (i+1)%50==0:print("natural",i+1,"/",args.runs,
                             "confirmed decays",sum(z["decayed"] for z in classification),
                             "parents",len(parent_snaps),"edge-like",edge_parent_count,flush=True)
    # All natural trajectories have been recorded. Their lengths are not
    # assumed to be Markovian and branches are reported separately.
    natural={}
    for label,tracks in (("0.5",long_high),("0.15",long_low)):
        natural[label]={str(k):fit(tracks,k,p,bootstrap=160,seed=args.seed) for k in LAGS}
    del long_high,long_low
    short_p=dict(p,T=float(args.branch_T))
    steps=round(short_p["T"]/p["dt"])
    all_restarts={"0.5":[],"0.15":[]}
    ancestors=[];branch_info=[];branch_viz=[]
    for i,snap in enumerate(parent_snaps):
        for b in range(args.branches):
            seed=args.seed+500000+1009*i+71*b
            qh,uh,diag,onset,confirmed=evolve_until(snap["q"],snap["u"],dx,p["dt"],
                          p["sigma"],steps,stride,seed,common)
            ht=data_until_confirmed(tracked(qh,p,.5),onset)
            lt=data_until_confirmed(tracked(qh,p,.15),onset)
            all_restarts["0.5"].append(ht)
            all_restarts["0.15"].append(lt)
            ancestors.append(i)
            branch_info.append({"parent_run":snap["parent"],"parent_time":round(snap["time"],2),
                "start_length_05":round(snap["start_ell_05"],3),
                "start_length_015":round(snap["start_ell_015"],3),
                "start_qmax":round(snap["qmax"],3),"kind":snap["kind"],
                "seed":seed,"decayed":bool(confirmed>=0),
                "confirmed_t":None if confirmed<0 else round(confirmed*p["save_dt"],2)})
            if i<5 and b<2:
                branch_viz.append({"parent":snap["parent"],"kind":snap["kind"],"branch":b+1,"high":slim_track(ht,2),
                                   "low":slim_track(lt,2)})
        if (i+1)%10==0: print("restart ancestors",i+1,"/",len(parent_snaps),flush=True)
    restarts={}
    if ancestors:
        for label,tracks in all_restarts.items():
            restarts[label]={str(k):fit(tracks,k,p,group_ids=ancestors,
                bootstrap=160,seed=args.seed+99) for k in LAGS}
    ref=edge_reference(p)
    quality={}
    for label in ("0.5","0.15"):
        results=natural[label]
        quality[label]={"valid_pairs":{key:val["valid_pairs"] for key,val in results.items()}}
    result={"schema":2,"generated_utc":datetime.now(timezone.utc).isoformat(),
      "method":"Barkley Heun/Rusanov/upwind + multiplicative Itô noise as original lab",
      "params":p,"seed_base":args.seed,"n_runs":args.runs,"T_restart":args.branch_T,
      "n_restart_parents":len(parent_snaps),"n_edge_like_parents":edge_parent_count,
      "n_short_high_parents":short_parent_count,"branches_per_parent":args.branches,
      "n_restarts":len(ancestors),"wall_seconds":time.perf_counter()-start,
      "decay_criteria":{"qmax_below":.05,"integral_positive_q_below":.25,
         "for_time":12.,"max_u_minus_U0_below_at_confirmation":.15,
         "event_time":"start of persistent quiet episode, retrospectively censored",
         "important":"Missing threshold crossing alone is not a confirmed decay"},
      "tracker":{"thresholds":[.5,.15],"gaussian_smooth_cells":p["smooth_cells"],
        "minimum_crossing_width":.55,"significant_secondary_width":1.,
        "maximum_crossing_width":.42*p["L"],
        "missing_or_ambiguous_intermediate_frames_excluded":True,
        "source":"largest contiguous threshold-exceeding component"},
      "edge":ref,
      "asymptotic":analytical_length(p),
      "deterministic":{"peak":float(q.max()),"threshold_widths":det},
      "decay_events":classification,"n_confirmed_decays":sum(z["decayed"] for z in classification),
      "natural":natural,"restarts":restarts,"restart_parents":[{"parent_run":z["parent"],
         "t":round(z["time"],2),"kind":z["kind"],"length_05":round(z["start_ell_05"],3),
         "length_015":round(z["start_ell_015"],3),"qmax":round(z["qmax"],3)} for z in parent_snaps],
      "restart_info":branch_info,
      "natural_examples":viz,"restart_examples":branch_viz,
      "profiles":samples,"quality":quality}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(json_ready(result),separators=(",",":"),allow_nan=False))
    def short(result,label,lag):
        v=result.get(label,{}).get(str(lag),{})
        return {"roots":v.get("roots"),"short":[{k:r.get(k) for k in
            ("ell","n","ancestors","A_l","D_l")} for r in v.get("rows",[]) if r["ell"]<4]}
    print("RESULT",json.dumps({"runs":args.runs,"confirmed_decays":result["n_confirmed_decays"],
         "restart_parents":len(parent_snaps),"edge_like":edge_parent_count,
         "short_high":short_parent_count,"branches":len(ancestors),
         "wall_seconds":result["wall_seconds"],
         "edge":ref["peak"],"edge_width_low":ref["width_at_threshold"]["0.15"],
         "natural_low":short(natural,"0.15",2),
         "natural_high":short(natural,"0.5",2),
         "restarted_low":short(restarts,"0.15",2)}),flush=True)
    print("Wrote",args.output,flush=True)

if __name__=="__main__":
    main()
