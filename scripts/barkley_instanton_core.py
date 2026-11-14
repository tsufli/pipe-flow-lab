"""Barkley stochastic action restricted to finite Gaussian control pulses.
Autonomous deterministic PDE in puff's co-moving frame, multiplicative control q*a.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from numba import njit
from math import pi
from pathlib import Path
U0=2.;UB=1.;ZETA=.8;D=.5;DELTA=.1;EPS=.1;KAPPA=2.
DX=.5;L=160.;N=round(L/DX)
@njit(cache=True)
def rhs(q,u,r,c,a):
 n=len(q); dq=np.empty(n);du=np.empty(n)
 for i in range(n):
  im=(i-1)%n;imm=(i-2)%n;ip=(i+1)%n
  qi=q[i];ui=u[i]
  qx=(3*qi-4*q[im]+q[imm])/(2*DX)
  ux=(3*ui-4*u[im]+u[imm])/(2*DX)
  dq[i]=-(ui-ZETA-c)*qx+D*(q[ip]+q[im]-2*qi)/(DX*DX)+qi*(r+ui-U0-(r+DELTA)*(qi-1)**2)+qi*a[i]
  du[i]=-(ui-c)*ux+EPS*(U0-ui+KAPPA*(UB-ui)*qi)
 return dq,du
@njit(cache=True)
def step(q,u,r,c,dt,a):
 a1,b1=rhs(q,u,r,c,a)
 a2,b2=rhs(q+dt*a1/2,u+dt*b1/2,r,c,a)
 a3,b3=rhs(q+dt*a2/2,u+dt*b2/2,r,c,a)
 a4,b4=rhs(q+dt*a3,u+dt*b3,r,c,a)
 return q+dt*(a1+2*a2+2*a3+a4)/6,u+dt*(b1+2*b2+2*b3+b4)/6
@njit(cache=True)
def run(q0,u0,r,c,shape,amp,tau,dt=.12,T=320.,save_times=np.array([0.,10.,20.,30.,40.,60.,90.,130.,180.,250.,320.])):
 q=q0.copy();u=u0.copy();A=np.empty(N);times=np.zeros(len(save_times));snap_q=np.zeros((len(save_times),N));snap_u=np.zeros((len(save_times),N));saved=0
 nsteps=round(T/dt);dt=T/nsteps
 for k in range(nsteps+1):
  t=k*dt
  if saved<len(save_times) and t+dt/2>=save_times[saved]:
   times[saved]=t;snap_q[saved,:]=q;snap_u[saved,:]=u;saved+=1
  if k==nsteps:break
  mid=t+dt/2
  scale=-amp*np.sin(pi*mid/tau)**2 if mid<tau else 0.
  for j in range(N):A[j]=scale*shape[j]
  q,u=step(q,u,r,c,dt,A)
 return q,u,times[:saved],snap_q[:saved],snap_u[:saved]

@njit(cache=True)
def run_end(q0,u0,r,c,shape,amp,tau,dt=.18,T=180.):
 q=q0.copy();u=u0.copy();A=np.empty(N)
 nsteps=round(T/dt);dt=T/nsteps
 for k in range(nsteps):
  mid=(k+.5)*dt
  scale=-amp*np.sin(pi*mid/tau)**2 if mid<tau else 0.
  for j in range(N):A[j]=scale*shape[j]
  q,u=step(q,u,r,c,dt,A)
 return np.sum(q)*DX,float(np.max(q))

def gaussian(center, width, puff_peak):
 x=np.arange(N)*DX
 center_abs=puff_peak*DX+center
 relative=(x-center_abs+L/2)%L-L/2
 return np.exp(-.5*(relative/width)**2)

def classify(q,u,threshold=1.0):
 mass=np.sum(q)*DX
 return ('puff' if mass>threshold else 'laminar',mass,float(max(q)))

def pulse_action(shape,amp,tau):
 return .5*amp*amp*float(np.sum(shape*shape)*DX)*(3.*tau/8.)
