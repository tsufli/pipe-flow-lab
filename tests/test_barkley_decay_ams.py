"""Fast regression checks for Barkley (2016) stochastic solver."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from barkley_decay_ams import step,evolve

class TestBarkleyDecayAMS(unittest.TestCase):
    def test_laminar_absorbing(self):
        q=np.zeros(100);u=np.full(100,2.)
        for _ in range(8):q,u=step(q,u,.62,.5,.25,.01)
        np.testing.assert_array_equal(q,0.)
        np.testing.assert_allclose(u,2.)
    def test_stochastic_cell_noise_variance(self):
        # Uniform q eliminates advection/diffusion.  The Ito update at each
        # cell is sigma*q*sqrt(dt/dx)*N(0,1), independent across cells.
        n=20000;dt=.004;dx=.1;sigma=.5
        q=np.ones(n);u=np.full(n,2.)
        det,_=step(q,u,.62,0.,dx,dt)
        noisy,_=step(q,u,.62,sigma,dx,dt)
        observed=float(np.var(noisy-det))
        expected=sigma**2*dt/dx
        self.assertLess(abs(observed/expected-1.),.04)
    def test_trajectory_indices_and_absorption(self):
        q=np.zeros(80);u=np.full(80,2.)
        qh,uh,m,td=evolve(q,u,.62,.5,.01,.25,20,5,123,.15)
        self.assertEqual(qh.shape,(5,80))
        self.assertEqual(uh.shape,(5,80))
        self.assertEqual(td,0)
        np.testing.assert_array_equal(m,0.)
    def test_positive_barkley_reaction(self):
        q=np.ones(100);u=np.full(100,2.)
        q1,u1=step(q,u,.62,0.,.25,.01)
        self.assertTrue(np.all(q1>1.))
        self.assertTrue(np.all(np.isfinite(u1)))
if __name__=='__main__':unittest.main()
