#!/usr/bin/env python3
"""Reproduce the eight-point Barkley deterministic puff/decay-edge scan.

Usage: pip install numpy scipy numba
       python scripts/build_barkley_edge_scan.py

The script computes two traveling branches, a partial spectrum, left/right
modes, and ±unstable-manifold trajectories. The outputs are static JSON in
web/barkley/edge-scan/data for GitHub Pages. No hand-entered eigenvalues.
The finite-difference continuum limit is NOT established by this one scan.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import numpy as np
from pathlib import Path
from numba import njit
from scipy.sparse.linalg import eigs
import barkley_edge_scan_solver as m
B=m.BASE
R=np.linspace(.43,.72,8)


def solve_branches():
    # A wide turbulent seed relaxes to the puff traveling wave at r=.6.
    rel=(m.x-.38*m.L+m.L/2)%m.L-m.L/2
    q=1.7*.5*(np.tanh((rel+9)/1.5)-np.tanh((rel-9)/1.5))
    u=m.U0-.3*q
    q,u=m.evolve(q,u,.6,m.DX,m.DT,round(350/m.DT))
    p=m.newton(q,u,.6,maxit=35)
    if not p['success']:raise RuntimeError('Unable to converge initial puff')
    qP,uP=p['q'],p['u']
    # Initial threshold at r=.6 by finite-time deterministic basin bisection.
    # We use an earlier time near the edge as initial guess for stationary Newton.
    lo,hi=0.,1.; puff_mass=qP.sum()*m.DX
    for k in range(31):
        alpha=(lo+hi)/2
        qq,uu=m.evolve(alpha*qP,m.U0+alpha*(uP-m.U0),.6,m.DX,m.DT,round(200/m.DT))
        if qq.sum()*m.DX>.45*puff_mass: hi=alpha
        else: lo=alpha
    alpha=(lo+hi)/2
    qq,uu=m.evolve(alpha*qP,m.U0+alpha*(uP-m.U0),.6,m.DX,m.DT,round(50/m.DT))
    e=m.newton(qq,uu,.6,maxit=35)
    if not e['success'] or not (0.1<e['q'].max()<qP.max()/2):
        raise RuntimeError('Unable to converge initial decay edge')
    print(f"Reference r=.6: puff c={p['c']:.6f}, edge c={e['c']:.6f}",flush=True)
    for name,first in (('puff',p),('edge',e)):
        initial=(first['q'],first['u'],first['c'])
        for direction in (-1,1):
            q,u,c=initial; rprev=.6
            targets=sorted([float(r) for r in R if (r<.6 if direction<0 else r>.6)],reverse=direction<0)
            for r in targets:
                steps=max(1,int(np.ceil(abs(r-rprev)/.009)))
                for rr in np.linspace(rprev,r,steps+1)[1:]:
                    out=m.newton(q,u,float(rr),c0=c,maxit=25)
                    if not out['success']:
                        raise RuntimeError(f'Continuation failed: {name} at r={rr:.6f}: residual={out["err"]}')
                    q,u,c=out['q'],out['u'],out['c']
                np.savez_compressed(B/f'{name}_{r:.6f}.npz',q=q,u=u,c=c,r=r,dx=m.DX,L=m.L,res=out['err'])
                rprev=r
                print(f'{name} r={r:.6f}, speed={c:.6f}, residual={out["err"]:.2g}',flush=True)


@njit(cache=True)
def rhs_frame(q,u,r,dx,c):
    n=len(q);dq=np.empty(n);du=np.empty(n)
    for i in range(n):
        im=(i-1)%n;imm=(i-2)%n;ip=(i+1)%n;qi=q[i];ui=u[i]
        qx=(3*qi-4*q[im]+q[imm])/(2*dx)
        ux=(3*ui-4*u[im]+u[imm])/(2*dx)
        dq[i]=-(ui-m.ZETA-c)*qx+m.D*(q[ip]+q[im]-2*qi)/(dx*dx)+qi*(r+ui-m.U0-(r+m.DELTA)*(qi-1)**2)
        du[i]=-(ui-c)*ux+m.EPS*(m.U0-ui+m.KAPPA*(m.UB-ui)*qi)
    return dq,du


@njit(cache=True)
def step_frame(q,u,r,dx,c,dt):
    a,b=rhs_frame(q,u,r,dx,c)
    a2,b2=rhs_frame(q+dt*a/2,u+dt*b/2,r,dx,c)
    a3,b3=rhs_frame(q+dt*a2/2,u+dt*b2/2,r,dx,c)
    a4,b4=rhs_frame(q+dt*a3,u+dt*b3,r,dx,c)
    return q+dt*(a+2*a2+2*a3+a4)/6,u+dt*(b+2*b2+2*b3+b4)/6


@njit(cache=True)
def evolve_frame(q,u,r,dx,c,dt,n):
    for i in range(n):q,u=step_frame(q,u,r,dx,c,dt)
    return q,u


def eigmode(J,lam,complexmode=False):
    rv,rvec=eigs(J,k=4,sigma=lam,which='LM',tol=1e-10,maxiter=5000)
    j=np.argmin(abs(rv-lam));ev=rv[j];v=rvec[:,j]
    lv,lvec=eigs(J.T,k=4,sigma=np.conj(ev) if complexmode else ev,which='LM',tol=1e-10,maxiter=5000)
    k=np.argmin(abs(lv-(np.conj(ev) if complexmode else ev)));w=lvec[:,k]
    if abs(ev.imag)<1e-7:
        v=v.real;w=w.real
        v*=np.sign(np.sum(v[:m.N])) or 1
    else:
        v=v*np.exp(-1j*np.angle(v[np.argmax(abs(v[:m.N]))]))
    v=v/max(np.max(abs(v[:m.N])),1e-15)
    w=w/np.conj(np.vdot(w,v)*m.DX)
    return ev,v,w,{
        'right_res':float(np.linalg.norm(J@v-ev*v)/max(1e-30,np.linalg.norm(v))),
        'left_res':float(np.linalg.norm(J.T@w-np.conj(ev)*w)/max(1e-30,np.linalg.norm(w))),
        'pair':float(abs(np.vdot(w,v)*m.DX))}


def aligned_sample(q,u,stride=4):
    q,u=m.align(q,u)
    return {'q':[round(float(y),7) for y in q[::stride]], 'u':[round(float(y),7) for y in u[::stride]]}


def samplevec(v,stride=4):
    return {'q':[round(float(y),7) for y in v[:m.N:stride].real],
            'u':[round(float(y),7) for y in v[m.N::stride].real],
            'qi':[round(float(y),7) for y in v[:m.N:stride].imag],
            'ui':[round(float(y),7) for y in v[m.N::stride].imag]}


def trajectory(q0,u0,c,r,v,sign):
    strength=.012*max(q0)
    q=q0+sign*strength*v[:m.N].real
    u=u0+sign*strength*v[m.N:].real
    ts=[0,5,10,15,20,30,40,55,75,100,130,170,230,310,430,600]
    prev=0;out=[]
    for t in ts:
        if t>prev:q,u=evolve_frame(q,u,r,m.DX,c,m.DT,round((t-prev)/m.DT))
        snap=aligned_sample(q,u)
        snap.update({'t':t,'mass':round(float(np.sum(np.maximum(q,0))*m.DX),7),'peak':round(float(max(q)),7)})
        out.append(snap);prev=t
    return out


def process(r):
    edge=np.load(B/f'edge_{r:.6f}.npz');puff=np.load(B/f'puff_{r:.6f}.npz')
    q,u,c=edge['q'],edge['u'],float(edge['c']);J=m.jac(q,u,c,float(r))
    ev,_=eigs(J,k=52,sigma=-.105,which='LM',tol=1e-10)
    ev2,_=eigs(J,k=10,sigma=.12,which='LM',tol=1e-10)
    ev=np.concatenate([ev,ev2]);positives=[z.real for z in ev if abs(z.imag)<1e-6 and z.real>1e-4]
    if not positives:raise RuntimeError(f'No unstable eigenvalue at r={r}')
    realstables=[z.real for z in ev if abs(z.imag)<1e-7 and z.real<-.015]
    if realstables:
        lam_s=max(realstables);stable_type='real'
    else:
        sv=[z for z in ev if z.imag>1e-6 and z.real<-.015]
        lam_s=min(sv,key=lambda z:abs(z.imag)+.02*abs(z.real+.11));stable_type='complex'
    lam_u=max(positives)
    lu,vu,wu,val_u=eigmode(J,lam_u)
    ls,vs,ws,val_s=eigmode(J,lam_s,complexmode=stable_type=='complex')
    spec=[]
    for z in sorted(ev,key=lambda v:(v.real,v.imag)):
        if not any(abs(z-w)<1e-5 for w in spec):spec.append(z)
    paths=[]
    for sign in (-1,1):
        frames=trajectory(q,u,c,r,vu,sign)
        final_mass=frames[-1]['mass'];puff_mass=float(np.sum(puff['q'])*m.DX)
        destination='puff' if final_mass>.35*puff_mass else 'laminar'
        paths.append({'sign':sign,'destination':destination,'frames':frames})
    obj={'r':round(float(r),8),'dx':m.DX,'N':m.N,'L':m.L,'eps':m.EPS,
         'c_edge':c,'c_puff':float(puff['c']),
         'res_edge':float(edge['res']),'res_puff':float(puff['res']),
         'puff_mass':float(np.sum(puff['q'])*m.DX),'edge_mass':float(np.sum(q)*m.DX),
         'edge':aligned_sample(q,u),'puff':aligned_sample(puff['q'],puff['u']),
         'eigs':{'unstable':{'real':float(lu.real),'imag':float(lu.imag),'right':samplevec(vu),'left':samplevec(wu),'quality':val_u},
                 'stable':{'real':float(ls.real),'imag':float(ls.imag),'type':stable_type,'right':samplevec(vs),'left':samplevec(ws),'quality':val_s},
                 'neutral':{'real':float(min(spec,key=lambda x:abs(x)).real)},
                 'spectrum':[[float(z.real),float(z.imag)] for z in spec]},
         'manifold':paths}
    (B/f'r_{r:.6f}.json').write_text(json.dumps(obj,separators=(',',':')))
    print('eigen r=',r,'unstable=',float(lu.real),'stable=',ls,'branches=',[p['destination'] for p in paths],flush=True)
    return {'r':float(r),'edge_c':c,'puff_c':float(puff['c']),
            'edge_peak':float(max(q)),'puff_peak':float(max(puff['q'])),
            'edge_mass':obj['edge_mass'],'puff_mass':obj['puff_mass'],
            'unstable_real':float(lu.real),'stable_real':float(ls.real),
            'stable_imag':float(ls.imag),'stable_type':stable_type,
            'manifold_minus':paths[0]['destination'],'manifold_plus':paths[1]['destination']}


if __name__=='__main__':
    solve_branches()
    rows=[process(float(r)) for r in R]
    (B/'summary.json').write_text(json.dumps({'version':1,'params':{
        'U0':m.U0,'Ub':m.UB,'zeta':m.ZETA,'delta':m.DELTA,
        'D':m.D,'epsilon':m.EPS,'kappa':m.KAPPA,
        'L':m.L,'N':m.N,'dx':m.DX},'scan':rows},indent=2))
    print('Wrote scan JSON to',B)