#!/usr/bin/env python3
"""Rebuild the restricted-action Barkley instanton-explorer data.

Requires numpy, scipy, numba and the existing Barkley edge-scan JSON inputs.
It is NOT a global stochastic instanton solver.  It optimizes a finite-parameter
negative Gaussian q-noise pulse within the family
    a(x,t)=-A exp(-(x-x0)^2/(2w^2)) sin^2(pi*t/tau), 0<t<tau
where q_t=F_q+q*a and u_t=F_u (no u noise).
The spatial covariance is ideal white noise on the grid, so
    I[a]=0.5*int dx dt a^2.
The saved optimized pulse parameters (x0,w,tau) originated in a coarse grid
+ local Nelder-Mead search. Pass --optimize to rerun that search. A is re-bracketed to locate the laminar/puff basin threshold.
All trajectories are integrated with the actual two-field PDE; no interpolated
q/u paths are called finite-action trajectories.

The instanton *candidate* is a pulse-optimized escape to the edge vicinity.
The unrestricted minimum action, instanton convergence, and continuum limits
are not claimed by this calculation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys,argparse
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import eigs
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parent))
import barkley_edge_scan_solver as sw
from barkley_instanton_core import N,DX,L,gaussian,run,run_end,pulse_action

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'web'/'barkley'/'edge-scan'/'data'
OUTPUT=ROOT/'web'/'barkley'/'instanton'/'data'
OUTPUT.mkdir(parents=True,exist_ok=True)
RS=(.5,.6)
INITIAL={
 .5:{'best':(.16941885,1.80430834,25.19210142),'baseline':(0.,4.,16.)},
 .6:{'best':(.33566619359,2.07600688335,17.39499921104),'baseline':(0.,4.,16.)}}
SOURCES={.5:'r_0.512857.json',.6:'r_0.595714.json'}
TIMES=np.array([0,2,4,6,8,10,12,14,16,18,20,23,26,30,34,38,42,46,50,55,60,65,70,75,80,90,100,110,120,130,140,150,160,170,180,200,220,240],dtype=float)
REV_TIMES=np.array([0,2,4,6,8,10,12,14,16,18,20,23,26,30,34,38,42,46,50,55,60,65,70,75,80,90,100,120,140,160,180],dtype=float)
STRIDE=2

def resample(y):
    y=np.asarray(y,float)
    xx=np.arange(len(y))*L/len(y)
    return CubicSpline(np.r_[xx,L],np.r_[y,y[0]],bc_type='periodic')(np.arange(N)*DX)

def exact_state(src,what,r):
    q=resample(src[what]['q']);u=resample(src[what]['u'])
    sol=sw.newton(q,u,r,c0=src['c_'+what],maxit=30)
    if not sol['success']:raise RuntimeError(f'Unable to recover {what} at r={r}: {sol["err"]}')
    shift=int(.38*N)-int(np.argmax(sol['q']))
    sol['q']=np.roll(sol['q'],shift);sol['u']=np.roll(sol['u'],shift)
    return sol

def eigenmode(edge,r):
    J=sw.jac(edge['q'],edge['u'],edge['c'],r)
    lam,vec=eigs(J,k=5,sigma=.13,tol=1e-11,which='LM')
    valid=np.where(np.abs(lam.imag)<1e-7,lam.real,-999.)
    i=int(np.argmax(valid));v=vec[:,i].real
    if valid[i]<0:raise RuntimeError('No real positive edge eigenmode')
    v*=np.sign(v[:N].sum()) or 1.
    v/=max(1e-12,np.max(abs(v[:N])))
    lv,lvec=eigs(J.T,k=5,sigma=lam[i].real,tol=1e-11,which='LM')
    w=lvec[:,int(np.argmin(abs(lv-lam[i].real)))].real
    norm=float(np.sum(w*v)*DX)
    w/=norm
    return float(lam[i].real),v,w

def threshold(puff,r,g,tau,amplitude_guess,T=240.,dt=.09):
    lo=.94*amplitude_guess;hi=1.06*amplitude_guess
    def mass(amp):
        m,peak=run_end(puff['q'],puff['u'],r,puff['c'],g,amp,tau,dt,T)
        return m
    # Verify a valid bracket, with opposite destinations.
    for tries in range(12):
        if mass(lo)>1.25 and mass(hi)<1.25:break
        lo*=.85;hi*=1.15
    else:raise RuntimeError(f'Unable to bracket edge at r={r}')
    for i in range(37):
        mid=(lo+hi)/2
        if mass(mid)>1.25:lo=mid
        else:hi=mid
    a=(lo+hi)/2
    return float(a),float(hi-lo),[float(mass(lo)),float(mass(hi))]

def distance_to_edge(q,u,qe,ue):
    # Geometric comparison only.  Translation removed in the metric, never in
    # the forcing/action calculation. w_u=3 compensates for field amplitude.
    shift=int(np.argmax(qe)-np.argmax(q))
    qa=np.roll(q,shift);ua=np.roll(u,shift)
    return float(np.sqrt(DX*np.sum((qa-qe)**2+3*(ua-ue)**2)))

def cumul_action(t,amp,shape,tau):
    s=max(0.,min(t,tau))
    integral=3*s/8 -tau/(4*np.pi)*np.sin(2*np.pi*s/tau)+tau/(32*np.pi)*np.sin(4*np.pi*s/tau)
    return float(.5*amp**2*np.sum(shape**2)*DX*integral)

def rnd(a,decimals=7):return [round(float(y),decimals) for y in a[::STRIDE]]

def candidate(r,p,e,which,x0,w,tau):
    g=gaussian(x0,w,np.argmax(p['q']))
    guess={.5:{'best':.1586324,'baseline':.1657811},.6:{'best':.36200405,'baseline':.31299637}}[r][which]
    amp,bracket,bracket_masses=threshold(p,r,g,tau,guess)
    q,u,ts,qs,us=run(p['q'],p['u'],r,p['c'],g,amp,tau,dt=.09,T=240.,save_times=TIMES)
    entries=[]
    for i,t in enumerate(ts):
        qi=qs[i];ui=us[i]
        entries.append({'t':round(float(t),4),'q':rnd(qi),'u':rnd(ui),
           'mass':round(float(qi.sum()*DX),6),'peak':round(float(max(qi)),6),
           'center':round(float(np.argmax(qi)*DX),3),
           'd_edge':round(distance_to_edge(qi,ui,e['q'],e['u']),6),
           'cumI':round(cumul_action(t,amp,g,tau),7)})
    candidates=[(i,s['d_edge']) for i,s in enumerate(entries) if s['t']>=tau+8 and s['t']<=160]
    besti=min(candidates,key=lambda a:a[1])[0]
    return {
      'id':which,'name':'Lower-action localized pulse' if which=='best' else 'Baseline Gaussian pulse',
      'amplitude':round(amp,11),'x0':x0,'width':w,'tau':tau,
      'action':round(pulse_action(g,amp,tau),8),
      'amplitude_bracket_width':bracket,'outcome_bracket_masses_at_t240':bracket_masses,
      'edge_frame_index':int(besti),'edge_encounter_time':entries[besti]['t'],
      'min_edge_distance':entries[besti]['d_edge'],
      'spatial_shape':rnd(g), 'frames':entries
    }

def reversed_edge_branch(r,e,vu):
    q0=e['q']+.025*np.max(e['q'])*vu[:N]
    u0=e['u']+.025*np.max(e['q'])*vu[N:]
    q,u,ts,qs,us=run(q0,u0,r,e['c'],np.zeros(N),0.,1.,dt=.09,T=180.,save_times=REV_TIMES)
    return [{'t_edge_forward':round(float(t),4),'q':rnd(qi),'u':rnd(ui),
         'mass':round(float(qi.sum()*DX),6),'peak':round(float(max(qi)),6)}
         for t,qi,ui in zip(ts,qs,us)]

def optimize_family(puff,r):
    """Optional fresh coarse-grid + local Nelder–Mead search, not global action optimization.

    The default build uses the published parameter minima for reproducibility,
    but --optimize runs the actual restricted-family search again.
    """
    def cost(z):
        x0,w,tau=map(float,z)
        if not (-12<=x0<=12 and 1.4<=w<=15 and 5<=tau<=60):return 1e4
        g=gaussian(x0,w,np.argmax(puff['q']))
        lo,hi=0.,.12
        threshold=1.25
        for tries in range(14):
            mass,_=run_end(puff['q'],puff['u'],r,puff['c'],g,hi,tau,dt=.18,T=180.)
            if mass<threshold:break
            hi*=1.6
        else:return 1e4
        for i in range(13):
            mid=(lo+hi)/2
            mass,_=run_end(puff['q'],puff['u'],r,puff['c'],g,mid,tau,dt=.18,T=180.)
            if mass>threshold:lo=mid
            else:hi=mid
        return pulse_action(g,(lo+hi)/2,tau)
    seeds=[]
    for x0 in [-7,-3,0,3,7]:
        for w in [2.,4.,7.,12.]:
            for tau in [8.,16.,30.,55.]:
                seeds.append(((x0,w,tau),cost((x0,w,tau))))
    seeds.sort(key=lambda v:v[1]);best=None
    for initial,_ in seeds[:2]:
        result=minimize(cost,np.array(initial,float),method='Nelder-Mead',
            options={'maxiter':85,'xatol':.15,'fatol':.0005})
        if best is None or result.fun<best.fun:best=result
    print('Fresh restricted-family search at r=',r,'cost=',best.fun,'x0,w,tau=',best.x,flush=True)
    return tuple(map(float,best.x))


def build(optimize=False):
    sw.N=N;sw.DX=DX;sw.x=np.arange(N)*DX
    allsum=[]
    for r in RS:
        source=json.loads((SOURCE/SOURCES[r]).read_text())
        puff=exact_state(source,'puff',r);edge=exact_state(source,'edge',r)
        lam,vu,wl=eigenmode(edge,r)
        shapes=INITIAL[r].copy()
        if optimize:shapes['best']=optimize_family(puff,r)
        controls={key:candidate(r,puff,edge,key,*pars) for key,pars in shapes.items()}
        rev=reversed_edge_branch(r,edge,vu)
        dist_ratio=controls['best']['action']/controls['baseline']['action']
        obj={'version':1,'r':r,'type':'restricted_Gaussian_pulse_action_candidate',
          'formula':'I=0.5 int dt dx a^2, q_t=Fq+q*a, u_t=Fu',
          'comparison_metric':'sqrt(dx*sum[(qaligned-qedge)^2+3*(ualigned-uedge)^2])',
          'raster':{'N_full':N,'N_display':N//STRIDE,'dx_full':DX,'dx_display':DX*STRIDE,'L':L,'frame':'puff co-moving','no_frame_recenter':True},
          'constants':{'U0':2,'Ub':1,'zeta':.8,'D':.5,'delta':.1,'epsilon':.1,'kappa':2},
          'puff':{'q':rnd(puff['q']),'u':rnd(puff['u']),'c':puff['c'],'residual':puff['err'],'mass':float(puff['q'].sum()*DX)},
          'edge':{'q':rnd(edge['q']),'u':rnd(edge['u']),'c':edge['c'],'residual':edge['err'],'mass':float(edge['q'].sum()*DX)},
          'edgeMode':{'lambda':lam,'right':{'q':rnd(vu[:N]),'u':rnd(vu[N:])},
              'adjointWeightedDrive':rnd(edge['q']*wl[:N]),
              'left':{'q':rnd(wl[:N]),'u':rnd(wl[N:])}},
          'candidates':controls, 'reverseDeterministic':rev,
          'warnings':[
             'Gaussian-family action optimization only; NOT a globally converged instanton.',
             'Reversing deterministic edge-to-puff dynamics is not an admissible stochastic path (u equation).',
             'State profile interpolation likewise violates the deterministic u equation.',
             'Threshold trajectories linger near the edge, then leave along its unstable direction.',
             'The displayed profiles are downsampled; all actions and endpoint diagnostics use N=320 full resolution.',
             'Weak-noise asymptotic probability is proportional to exp(-I/sigma^2) for white forcing.',
             'The frame is fixed to the unforced puff speed; no time-dependent recentering occurs.'
          ]}
        (OUTPUT/f'r_{r:.1f}.json').write_text(json.dumps(obj,separators=(',',':')))
        print(f'r={r} I_best={controls["best"]["action"]} I_baseline={controls["baseline"]["action"]} gain={(1-dist_ratio)*100:.1f}% encounter={controls["best"]["edge_encounter_time"]} distance={controls["best"]["min_edge_distance"]}',flush=True)
        allsum.append({'r':r,'bestAction':controls['best']['action'],'baselineAction':controls['baseline']['action'],
           'gainPct':round((1-dist_ratio)*100,3),'edgeEncounterTime':controls['best']['edge_encounter_time'],
           'edgeDistance':controls['best']['min_edge_distance']})
    (OUTPUT/'summary.json').write_text(json.dumps({'version':1,'source':'Barkley grid white noise restricted-family control','params':INITIAL,'scan':allsum},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--optimize',action='store_true',help='Repeat the restricted-family Gaussian pulse search; expensive')
    build(optimize=parser.parse_args().optimize)