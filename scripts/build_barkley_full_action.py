#!/usr/bin/env python3
"""Barkley: finite-horizon FREELY-GRIDDED minimum-action optimization from two seeds.

Run from repo root:
  pip install numpy scipy numba 'jax[cpu]'
  python scripts/build_barkley_full_action.py --r 0.5 --maxiter 350
  python scripts/build_barkley_full_action.py --r 0.6 --maxiter 350

The action is 0.5*sum_{t,x} a(x,t)^2 dt dx, and a acts ONLY in q_t:
   q_t = F_q(q,u)+q a;   u_t = F_u(q,u).

This is a full 320 x 160 grid of independently optimizable a values, not an
arbitrary Gaussian pulse. We use a SOFT terminal penalty, so the results are
local finite-time control candidates, *not* certified global instantons.
There is no mesh-size or time-horizon extrapolation here.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from scipy.optimize import minimize
from scipy.sparse.linalg import eigs
from numba import njit
jax.config.update('jax_enable_x64',True)
sys.path.insert(0,str(Path(__file__).resolve().parent))
import barkley_edge_scan_solver as wave
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'web'/'barkley'/'edge-scan'/'data'
OUT=ROOT/'web'/'barkley'/'instanton'/'data'
OUT.mkdir(parents=True,exist_ok=True)
N=160;L=160.;DX=1.;DT=.25;NT=320;T=80.;STEPS=np.arange(0,NT+1,8)
STAGES=(10.,30.,100.)
SOURCES={.5:'r_0.512857.json',.6:'r_0.595714.json'}
PAUSE={.5:72.,.6:62.}

@njit(cache=True)
def deterministic_rhs(q,u,r,c):
    dq=np.empty(len(q));du=np.empty(len(q));n=len(q)
    for i in range(n):
        im=(i-1)%n;im2=(i-2)%n;ip=(i+1)%n
        qx=(3*q[i]-4*q[im]+q[im2])/(2*DX)
        ux=(3*u[i]-4*u[im]+u[im2])/(2*DX)
        dq[i]=-(u[i]-.8-c)*qx+.5*(q[ip]+q[im]-2*q[i])/(DX*DX)+q[i]*(r+u[i]-2-(r+.1)*(q[i]-1)**2)
        du[i]=-(u[i]-c)*ux+.1*(2-u[i]+2*(1-u[i])*q[i])
    return dq,du

@njit(cache=True)
def evolve_edge(q,u,r,c,tend):
    dt=.1;nt=round(tend/dt);stride=5
    Q=np.empty((nt//stride+1,len(q)));U=np.empty_like(Q)
    Q[0]=q;U[0]=u;k=1
    for j in range(nt):
        qd,ud=deterministic_rhs(q,u,r,c)
        qmid=q+dt*.5*qd;umid=u+dt*.5*ud
        qd,ud=deterministic_rhs(qmid,umid,r,c)
        q=q+dt*qd;u=u+dt*ud
        if (j+1)%stride==0:
            Q[k]=q;U[k]=u;k+=1
    return Q,U

def stationary_states(r):
    wave.N=N;wave.DX=DX;wave.x=np.arange(N)*DX
    source=json.loads((SOURCE/SOURCES[r]).read_text())
    result={}
    for name in ['puff','edge']:
        raw=source[name];q0=np.asarray(raw['q']);u0=np.asarray(raw['u'])
        # Saved edge-scan JSON already uses 160 equally spaced spatial samples.
        if len(q0)!=N:raise ValueError('Incompatible saved edge-scan data')
        sol=wave.newton(q0,u0,r,c0=source['c_'+name],maxit=35)
        if not sol['success']:raise RuntimeError(f'Newton failed for {name}, r={r}: residual={sol["err"]}')
        result[name]=sol
    return result

def unstable_right(edge,r):
    lam,vec=eigs(wave.jac(edge['q'],edge['u'],edge['c'],r),k=5,sigma=.14,tol=1e-10)
    real=np.where(abs(lam.imag)<1e-7,lam.real,-np.inf);k=int(np.argmax(real))
    if real[k]<=0:raise RuntimeError('No positive real edge eigenmode')
    v=vec[:,k].real;v*=np.sign(v[:N].sum()) or 1.;v/=max(1e-15,max(abs(v[:N])))
    return float(lam[k].real),v

def reference_q(r,states,kind,mode):
    qp=states['puff']['q'];qe=states['edge']['q']
    s=np.linspace(0,1,NT+1)
    if kind=='linear':
        smooth=s*s*(3-2*s)
        return (1-smooth[:,None])*qp+smooth[:,None]*qe
    edge=states['edge'];v=mode
    amp=.05*edge['q'].max()
    Q,U=evolve_edge(edge['q']+amp*v[:N],edge['u']+amp*v[N:],r,edge['c'],PAUSE[r])
    xs=np.arange(len(Q))*.5
    target=np.linspace(PAUSE[r],0,NT+1)
    indices=np.interp(target,xs,np.arange(len(Q)))
    qs=[]
    idx0=int(np.argmax(qp))
    for ind in indices:
        i=min(len(Q)-2,int(np.floor(ind)));f=ind-i
        q=(1-f)*Q[i]+f*Q[i+1]
        # Reference only: quotient by translation when reversing this curve.
        qs.append(np.roll(q,idx0-int(np.argmax(q))))
    qs=np.asarray(qs)
    qref=qs+(1-s)[:,None]*(qp-qs[0])+s[:,None]*(qe-qs[-1])
    return np.maximum(qref,1e-12)

def inverse_seed(r,states,qref,scale=.05):
    u=np.copy(states['puff']['u']);c=states['puff']['c']
    ctrl=np.zeros((NT,N))
    for k in range(NT):
        q=qref[k];qnext=qref[k+1];qm=.5*(q+qnext)
        _,ud=deterministic_rhs(q,u,r,c)
        um=u+.5*DT*ud
        _,udmid=deterministic_rhs(qm,um,r,c)
        u=u+DT*udmid
        fq,_=deterministic_rhs(qm,um,r,c)
        a=((qnext-q)/DT-fq)/np.maximum(qm,1e-3)
        a[qm<.004]=0.
        ctrl[k]=np.clip(a,-2.5,2.5)*scale
    return ctrl

def model(r,rho):
    modes=jnp.arange(N//2+1)
    angles=2*jnp.pi*jnp.arange(N)/N
    def rhs(q,u,c,a):
        qx=(3*q-4*jnp.roll(q,1)+jnp.roll(q,2))/(2*DX)
        ux=(3*u-4*jnp.roll(u,1)+jnp.roll(u,2))/(2*DX)
        dq=-(u-.8-c)*qx+.5*(jnp.roll(q,-1)+jnp.roll(q,1)-2*q)/(DX*DX)+q*(r+u-2-(r+.1)*(q-1)**2)+q*a
        du=-(u-c)*ux+.1*(2-u+2*(1-u)*q)
        return dq,du
    def step(carry,a):
        q,u,c=carry
        dq,du=rhs(q,u,c,a)
        qq,uu=q+DT*dq,u+DT*du
        dq1,du1=rhs(qq,uu,c,a)
        return (q+DT*.5*(dq+dq1),u+DT*.5*(du+du1),c),None
    def rollout(a,qp,up,c):
        (q,u,_),_=jax.lax.scan(step,(qp,up,c),a)
        return q,u
    def shift_edge(q,qe,ue):
        phase=lambda z:jnp.arctan2(jnp.sum(z*jnp.sin(angles)),jnp.sum(z*jnp.cos(angles)))
        ang=phase(q)-phase(qe)
        index=jnp.arctan2(jnp.sin(ang),jnp.cos(ang))*N/(2*jnp.pi)
        mult=jnp.exp(-2j*jnp.pi*modes*index/N)
        return jnp.fft.irfft(jnp.fft.rfft(qe)*mult,n=N),jnp.fft.irfft(jnp.fft.rfft(ue)*mult,n=N)
    def loss(a,qp,up,c,qe,ue):
        q,u=rollout(a,qp,up,c)
        qe,ue=shift_edge(q,qe,ue)
        action=.5*DT*DX*jnp.sum(a*a)
        E=.5*DX*jnp.sum((q-qe)**2+3*(u-ue)**2)
        return action+rho*E,(action,E)
    return jax.jit(jax.value_and_grad(loss,has_aux=True))

def optimize(r,states,start,weight,maxiter=350):
    fun=model(r,weight)
    P=tuple(jnp.asarray(x) for x in [states['puff']['q'],states['puff']['u'],states['puff']['c'],states['edge']['q'],states['edge']['u']])
    def fg(z):
        (value,(_,_)),grad=fun(jnp.asarray(z.reshape(NT,N)),*P)
        return float(value),np.asarray(grad).ravel()
    trace=[]
    def callback(z):
        n=len(trace)
        if n%10==0:
            (value,(A,E)),g=fun(jnp.asarray(z.reshape(NT,N)),*P)
            trace.append((float(value),float(A),float(E)))
        else:trace.append(None)
    sol=minimize(fg,start.ravel(),jac=True,method='L-BFGS-B',bounds=[(-4,4)]*(N*NT),callback=callback,
                 options={'maxiter':maxiter,'gtol':1e-5,'ftol':1e-11,'maxls':15,'maxcor':10})
    ctrl=np.asarray(sol.x).reshape(NT,N)
    (value,(action,error)),grad=fun(jnp.asarray(ctrl),*P)
    return ctrl,{'nIterations':int(sol.nit),'objective':float(value),'action':float(action),
           'endpointDistance':float(jnp.sqrt(2*error)),'gradientNorm':float(jnp.linalg.norm(grad)),
           'optimizerReportedSuccess':bool(sol.success),'history':[z for z in trace if z is not None]}

def trajectories(r,states,ctrl):
    # Reference Heun rollout; intentionally no recentering in dynamics.
    @jax.jit
    def simulate(a):
        qp=jnp.asarray(states['puff']['q']);up=jnp.asarray(states['puff']['u']);c=states['puff']['c']
        def rhs(q,u,av):
            qx=(3*q-4*jnp.roll(q,1)+jnp.roll(q,2))/(2*DX);ux=(3*u-4*jnp.roll(u,1)+jnp.roll(u,2))/(2*DX)
            return (-(u-.8-c)*qx+.5*(jnp.roll(q,-1)+jnp.roll(q,1)-2*q)+q*(r+u-2-(r+.1)*(q-1)**2)+q*av,
                -(u-c)*ux+.1*(2-u+2*(1-u)*q))
        def step(carry,act):
            q,u=carry;dq,du=rhs(q,u,act);q1,u1=q+DT*dq,u+DT*du;dq1,du1=rhs(q1,u1,act)
            end=(q+DT*.5*(dq+dq1),u+DT*.5*(du+du1))
            return end,end
        _,(q,u)=jax.lax.scan(step,(qp,up),a)
        return jnp.concatenate((qp[None,:],q)),jnp.concatenate((up[None,:],u))
    qq,uu=simulate(jnp.asarray(ctrl));qq=np.asarray(qq);uu=np.asarray(uu)
    rates=.5*DX*(ctrl*ctrl).sum(axis=1);cum=np.r_[0,np.cumsum(DT*rates)]
    qe,ue=states['edge']['q'],states['edge']['u']
    theta=2*np.pi*np.arange(N)/N
    def phase(q):return np.arctan2(np.dot(q,np.sin(theta)),np.dot(q,np.cos(theta)))
    fr=[]
    for k in STEPS:
        q,u=qq[k],uu[k];z=np.angle(np.exp(1j*(phase(q)-phase(qe))))*N/(2*np.pi)
        mult=np.exp(-2j*np.pi*np.arange(N//2+1)*z/N)
        qs=np.fft.irfft(np.fft.rfft(qe)*mult,n=N);us=np.fft.irfft(np.fft.rfft(ue)*mult,n=N)
        dist=np.sqrt(DX*np.sum((q-qs)**2+3*(u-us)**2))
        fr.append({'t':round(k*DT,3),'q':np.round(q,5).tolist(),'u':np.round(u,5).tolist(),
          'a':np.round(ctrl[min(k,NT-1)],5).tolist(),'mass':round(float(q.sum()*DX),6),
          'peak':round(float(q.max()),6),'dEdge':round(float(dist),6),'edgeShift':round(float(z*DX),5),
          'power':round(float(rates[min(k,NT-1)]),7),'cumAction':round(float(cum[k]),7)})
    return fr,float(cum[-1]),float(fr[-1]['dEdge'])

def run_one(r,maxiter):
    states=stationary_states(r);lam,v=unstable_right(states['edge'],r)
    print('r',r,'edge unstable growth',lam,flush=True)
    result={'version':2,'r':r,'label':'finite-horizon full-field FW action optimization (two starts)',
      'params':{'L':L,'N':N,'dx':DX,'T':T,'dt':DT,'steps':NT,'terminal_weight':100.,'u_weight':3.,
                'control_bounds':[-4,4],'frame':'puff co-moving',
                'spatial_alignment':'final-time first-Fourier-moment phase shift only'},
      'formula':{'action':'I=0.5 int dt dx a^2','objective':'J_rho=I+rho/2*int dx[(qT-qE_shift)^2+3(uT-uE_shift)^2]',
                 'q':'q_t=Fq+q*a','u':'u_t=Fu'},
      'puff':{'q':np.round(states['puff']['q'],6).tolist(),'u':np.round(states['puff']['u'],6).tolist(),'c':float(states['puff']['c'])},
      'edge':{'q':np.round(states['edge']['q'],6).tolist(),'u':np.round(states['edge']['u'],6).tolist(),'c':float(states['edge']['c'])},
      'seeds':{},'warnings':['This is unconstrained in the spatial shape of the gridded control, not a Gaussian pulse.',
        'Endpoint is penalized, not imposed exactly; optimizer convergence and global minimality are not guaranteed.',
        'Spatial and temporal refinement and infinite-time transition limit remain to be tested.']}
    for name in ('linear','reverse'):
        qref=reference_q(r,states,name,v);a=inverse_seed(r,states,qref)
        seed={'label':'Puff–edge linear interpolation' if name=='linear' else 'Reversed edge unstable branch',
              'reference':{'frames':[{'t':round(k*DT,3),'q':np.round(qref[k],5).tolist()} for k in STEPS]},
              'description':'q-reference inverted with deterministic u, then control scaled by 0.05','stages':{}}
        for stage in ['initial','10','30','100']:
            if stage=='initial':res={'nIterations':0,'objective':None,'action':None,'endpointDistance':None,'gradientNorm':None,'optimizerReportedSuccess':False,'history':[]}
            else:a,res=optimize(r,states,a,float(stage),maxiter)
            fr,action,dist=trajectories(r,states,a)
            seed['stages'][stage]={'stage':stage,'weight':float(stage) if stage!='initial' else 0,
              'action':action,'endpointDistance':dist,'objective':None if stage=='initial' else action+float(stage)*.5*dist**2,
              'nIterations':res['nIterations'],'gradNorm':res['gradientNorm'],
              'optimizerReportedSuccess':res['optimizerReportedSuccess'],'history':res['history'],'frames':fr}
            print(name,stage,'I',round(action,6),'dEdge',round(dist,6),'iters',res['nIterations'],flush=True)
        result['seeds'][name]=seed
    l=result['seeds']['linear']['stages']['100'];a=result['seeds']['reverse']['stages']['100']
    q1=np.asarray([f['q'] for f in l['frames']]);q2=np.asarray([f['q'] for f in a['frames']]);u1=np.asarray([f['u'] for f in l['frames']]);u2=np.asarray([f['u'] for f in a['frames']])
    result['comparison']={'actionGapPct':200*abs(l['action']-a['action'])/(l['action']+a['action']),
     'rmsTimeMatchedPathDistance':float(np.sqrt(DX*np.mean(np.sum((q1-q2)**2+3*(u1-u2)**2,axis=1)))),
     'terminalStateDistance':float(np.sqrt(DX*np.sum((q1[-1]-q2[-1])**2+3*(u1[-1]-u2[-1])**2))),
     'sameMinimumCertified':False,'reason':'finite iteration budget; soft endpoint and finite horizon'}
    (OUT/f'r_{r:.1f}.json').write_text(json.dumps(result,separators=(',',':')))
    return {'r':r,'costLinear':l['action'],'costReverse':a['action'],
      'endpointLinear':l['endpointDistance'],'endpointReverse':a['endpointDistance'],**result['comparison']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--r',type=float,choices=[.5,.6],default=None)
    p.add_argument('--maxiter',type=int,default=350)
    args=p.parse_args()
    rs=[args.r] if args.r is not None else [.5,.6]
    rows=[run_one(r,args.maxiter) for r in rs]
    if len(rs)==2:(OUT/'summary.json').write_text(json.dumps({'version':2,'data':rows},indent=2))