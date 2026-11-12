#!/usr/bin/env python3
"""Deterministic Barkley traveling-wave numerical solver for the 8-point edge scan.
Spatial semidiscretization: 2nd-order backward q and u derivatives, centered q Laplacian; periodic boundary.
Uses Newton + phase condition. These are numerical traveling waves of an approximate spatial discretization, not exact continuum eigenvalues.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np, scipy.sparse as sp, time, json, sys
from scipy.sparse.linalg import spsolve,eigs
from numba import njit
from pathlib import Path
from scipy.ndimage import shift as ndi_shift
BASE=Path(__file__).resolve().parents[1] / 'web' / 'barkley' / 'edge-scan' / 'data'; BASE.mkdir(parents=True,exist_ok=True)
U0=2.; UB=1.; ZETA=.8; D=.5; DELTA=.1; EPS=.1; KAPPA=2.
L=160.; DX=.25; N=round(L/DX); DX=L/N; x=np.arange(N)*DX
DT=.035
RS=np.linspace(.43,.72,8)
@njit(cache=True)
def rhs(q,u,r,dx):
 n=len(q); dq=np.empty(n); du=np.empty(n)
 for i in range(n):
  im=(i-1)%n;imm=(i-2)%n;ip=(i+1)%n; qi=q[i];ui=u[i]
  qx=(3*qi-4*q[im]+q[imm])/(2*dx);ux=(3*ui-4*u[im]+u[imm])/(2*dx)
  dq[i]=-(ui-ZETA)*qx+D*(q[ip]+q[im]-2*qi)/(dx*dx)+qi*(r+ui-U0-(r+DELTA)*(qi-1)**2)
  du[i]=-ui*ux+EPS*(U0-ui+KAPPA*(UB-ui)*qi)
 return dq,du
@njit(cache=True)
def step(q,u,r,dx,dt):
 q1,u1=rhs(q,u,r,dx);q2,u2=rhs(q+dt*q1/2,u+dt*u1/2,r,dx);q3,u3=rhs(q+dt*q2/2,u+dt*u2/2,r,dx);q4,u4=rhs(q+dt*q3,u+dt*u3,r,dx)
 return q+dt*(q1+2*q2+2*q3+q4)/6,u+dt*(u1+2*u2+2*u3+u4)/6
@njit(cache=True)
def evolve(q,u,r,dx,dt,n):
 for i in range(n):q,u=step(q,u,r,dx,dt)
 return q,u

def d1q(q):return (3*q-4*np.roll(q,1)+np.roll(q,2))/(2*DX)
def d1u(u):return (3*u-4*np.roll(u,1)+np.roll(u,2))/(2*DX)
def d2(q):return (np.roll(q,-1)-2*q+np.roll(q,1))/DX**2

def fun(q,u,c,r):
 f=q*(r+u-U0-(r+DELTA)*(q-1)**2)
 g=U0-u+KAPPA*(UB-u)*q
 return D*d2(q)-(u-ZETA-c)*d1q(q)+f, -(u-c)*d1u(u)+EPS*g

def jac(q,u,c,r):
 qq=d1q(q); uu=d1u(u)
 fq=(r+u-U0)-(r+DELTA)*(3*q*q-4*q+1)
 ids=np.arange(N);im=(ids-1)%N; imm=(ids-2)%N; ip=(ids+1)%N; a=u-ZETA-c
 rows=[];cols=[];vals=[]
 def put(i,j,v):
  rows.append(np.asarray(i));cols.append(np.asarray(j));vals.append(np.broadcast_to(v,(N,)))
 put(ids,ids,-2*D/DX**2-3*a/(2*DX)+fq);put(ids,im,D/DX**2+2*a/DX);put(ids,imm,-a/(2*DX));put(ids,ip,D/DX**2)
 put(ids,N+ids,q-qq)
 put(N+ids,ids,EPS*KAPPA*(UB-u))
 put(N+ids,N+ids,-3*(u-c)/(2*DX)-uu-EPS*(1+KAPPA*q))
 put(N+ids,N+im,2*(u-c)/DX)
 put(N+ids,N+imm,-(u-c)/(2*DX))
 return sp.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(2*N,2*N)).tocsr()

def align(q,u,loc=.38):
 i=np.argmax(q); shift=round(loc*N)-i
 return np.roll(q,shift),np.roll(u,shift)

def newton(q0,u0,r,c0=None,maxit=17):
 q0,u0=align(q0,u0)
 if c0 is None:
  dq,du=fun(q0,u0,0,r); tangent=np.r_[d1q(q0),d1u(u0)]; c0=-np.dot(np.r_[dq,du],tangent)/max(1e-15,np.dot(tangent,tangent))
 ref=np.r_[np.gradient(q0,DX),np.gradient(u0,DX)]; ref/=np.linalg.norm(ref)+1e-30
 y=np.r_[q0,u0,c0]
 def residual(y):
  q,u,c=y[:N],y[N:2*N],y[-1]; a,b=fun(q,u,c,r)
  return np.r_[a,b,np.dot(ref,y[:2*N]-np.r_[q0,u0])]
 errors=[]
 for k in range(maxit):
  err=np.linalg.norm(residual(y))/np.sqrt(N);errors.append(err)
  if err<1e-10:break
  q,u,c=y[:N],y[N:2*N],y[-1];J=jac(q,u,c,r)
  aug=sp.bmat([[J,sp.csr_matrix(np.r_[d1q(q),d1u(u)][:,None])],[sp.csr_matrix(ref[None,:]),sp.csr_matrix((1,1))]],format='csr')
  try:d=spsolve(aug,-residual(y))
  except Exception as exc:print('spsolve error',exc,flush=True);break
  fac=1.; good=False
  for j in range(16):
   ynew=y+fac*d
   if np.all(np.isfinite(ynew)) and np.linalg.norm(residual(ynew))/np.sqrt(N)<err:
    y=ynew;good=True;break
   fac*=.5
  if not good:break
 err=np.linalg.norm(residual(y))/np.sqrt(N)
 return {'q':y[:N], 'u':y[N:2*N], 'c':float(y[-1]),'err':float(err),'errors':errors,'success':err<2e-8, 'r':r}
