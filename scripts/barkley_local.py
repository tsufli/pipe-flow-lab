"""Numerical reference and sanity checks for Barkley (2016) local system.

This is the spatially homogeneous deterministic reduction of equations (8.14).
Integration uses scipy.integrate.solve_ivp(DOP853) with tight tolerances.
"""
import numpy as np
from scipy.integrate import solve_ivp

DEFAULTS = dict(r=0.6, delta=0.1, eps1=0.1, eps2=0.2, U0=2.0, Ub=1.0)

def rhs(t, state, *, r=0.6, delta=0.1, eps1=0.1, eps2=0.2, U0=2.0, Ub=1.0):
    u, q = state
    dq = q * (r + u - U0 - (r + delta) * (q - 1) ** 2)
    du = eps1 * (U0 - u) + eps2 * (Ub - u) * q
    return np.array([du, dq])

def integrate(u0=2.0, q0=0.55, t_end=80, **parameters):
    p = DEFAULTS | parameters
    return solve_ivp(lambda t, y: rhs(t, y, **p), (0, t_end), [u0, q0],
                     method="DOP853", rtol=1e-11, atol=1e-13,
                     dense_output=True, max_step=0.15)

def rk4(u0, q0, dt, t_end, **parameters):
    p = DEFAULTS | parameters
    n = int(np.ceil(t_end / dt))
    h = t_end / n
    y = np.array([u0, q0], dtype=float)
    for _ in range(n):
        k1 = rhs(0, y, **p)
        k2 = rhs(0, y + h*k1/2, **p)
        k3 = rhs(0, y + h*k2/2, **p)
        k4 = rhs(0, y + h*k3, **p)
        y = y + h*(k1+2*k2+2*k3+k4)/6
    return y

def test_laminar_equilibrium():
    np.testing.assert_allclose(rhs(0, [2,0]), [0,0], atol=1e-14)

def test_excitable_relaxation():
    s=integrate()
    assert s.success
    assert s.y[1].max()>1.0, "large perturbation should cause an excursion"
    assert abs(s.y[1,-1])<2e-3
    assert abs(s.y[0,-1]-2)<2e-3

def test_bistable_turbulent_equilibrium():
    s=integrate(q0=0.65,r=0.9,t_end=160)
    assert s.success
    assert s.y[1,-1]>0.5, "higher r should support turbulence"
    np.testing.assert_allclose(rhs(0,s.y[:,-1],r=.9),[0,0],atol=1e-5)

def test_rk4_against_scipy():
    for r in (0.6,0.9):
        ref=integrate(r=r,t_end=40).y[:,-1]
        candidate=rk4(2,.55,.035,40,r=r)
        np.testing.assert_allclose(candidate,ref,atol=3e-5,rtol=3e-5)

def test_positivity():
    for r in (.3,.6,.9):
        s=integrate(q0=.5,r=r,t_end=100)
        assert s.success
        assert s.y[1].min()>-1e-10
