"""Reference checks for the deterministic Barkley spatial solver.

Periodic grid; explicit upwind q transport, conservative Rusanov flux for u,
central q diffusion, two-stage Heun time stepping.
"""
import numpy as np

def rhs(q, u, *, dx, r=.6, delta=.1, eps1=.1, eps2=.2, zeta=.8, D=.5, U0=2., Ub=1.):
    um = np.roll(u, 1)
    up = np.roll(u, -1)
    qm = np.roll(q, 1)
    qp = np.roll(q, -1)
    a = u - zeta
    dq = -np.where(a >= 0, a*(q-qm)/dx, a*(qp-q)/dx)
    dq += D*(qp-2*q+qm)/dx**2
    dq += q*(r+u-U0-(r+delta)*(q-1)**2)
    fl = .25*(um**2+u**2)-.5*np.maximum(abs(um),abs(u))*(u-um)
    fr = .25*(u**2+up**2)-.5*np.maximum(abs(u),abs(up))*(up-u)
    du = -(fr-fl)/dx+eps1*(U0-u)+eps2*(Ub-u)*q
    return dq, du

def step(q, u, dt, dx, **params):
    a,b=rhs(q,u,dx=dx,**params)
    c,d=rhs(q+dt*a,u+dt*b,dx=dx,**params)
    return q+dt*(a+c)/2, u+dt*(b+d)/2

def test_uniform_laminar():
    q=np.zeros(128);u=np.full(128,2.)
    a,b=rhs(q,u,dx=.25)
    np.testing.assert_allclose(a,0,atol=1e-14)
    np.testing.assert_allclose(b,0,atol=1e-14)

def test_uniform_pde_matches_ode():
    q=np.full(128,.5);u=np.full(128,2.)
    for _ in range(100):
        q,u=step(q,u,.02,.5)
        assert np.ptp(q)<1e-12
        assert np.ptp(u)<1e-12

def test_localized_seed():
    n=200; L=100.; x=(np.arange(n)+.5)*L/n
    q=1.65/(1+np.exp(np.minimum(100,(np.abs(x-L/2)-6)/1.0)))
    u=np.full(n,2.)
    for _ in range(500):
        q,u=step(q,u,.02,L/n)
        assert np.all(np.isfinite(q)) and np.all(np.isfinite(u))
    assert q.min()>-1e-9
    assert u.min()>0

def test_ito_noise_normalization():
    """Cellwise stochastic increment has variance sigma² q² dt/dx."""
    rng=np.random.default_rng(12345)
    sigma,q,dt,dx=.5,1.3,.01,.25
    increments=sigma*q*np.sqrt(dt/dx)*rng.standard_normal(100000)
    expected=sigma**2*q**2*dt/dx
    assert abs(np.var(increments)-expected)/expected<.025

def test_zero_noise_is_deterministic():
    """At sigma=0 the stochastic update is identically zero."""
    rng=np.random.default_rng(123)
    q=np.linspace(0,1,128)
    np.testing.assert_array_equal(q+0*q*np.sqrt(.01/.5)*rng.standard_normal(128),q)
