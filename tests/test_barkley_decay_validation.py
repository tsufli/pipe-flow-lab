"""Numerical validation smoke tests for offline Barkley AMS and direct sampling."""
import sys, unittest, math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import numpy as np
    from barkley_decay_validation import step, components, bootstrap_prob, direct_trajectories
    AVAILABLE = True
except ModuleNotFoundError as e:
    if e.name not in ('numba', 'numpy'): raise
    AVAILABLE = False

@unittest.skipUnless(AVAILABLE, "Scientific dependencies are installed in numerical workflow")
class StochasticValidationTests(unittest.TestCase):
    def test_laminar_absorbing(self):
        q=np.zeros(100);u=np.full(100,2.)
        for _ in range(5):
            q,u=step(q,u,.62,.5,.1,.004)
        np.testing.assert_array_equal(q,np.zeros_like(q))
        np.testing.assert_allclose(u,2.)
    def test_cellwise_ito_variance(self):
        n=18000
        q=np.ones(n);u=np.full(n,2.)
        deterministic,_=step(q,u,.62,0.,.1,.004)
        noisy,_=step(q,u,.62,.5,.1,.004)
        variance=np.var(noisy-deterministic)
        self.assertLess(abs(variance/(.25*.004/.1)-1),.05)
    def test_component_classifier(self):
        q=np.zeros(100);q[10:25]=2.;q[55:70]=2.
        self.assertEqual(components(q,.25),2)
        q[55:70]=0
        self.assertEqual(components(q,.25),1)
    def test_direct_laminar_absorption(self):
        n=6; q=np.zeros((n,100));u=np.full((n,100),2.)
        td=direct_trajectories(q,u,.62,.5,.25,.01,
                              np.arange(102,102+n),20,5,.15)
        np.testing.assert_allclose(td,.05)
    def test_bootstrap_interval(self):
        ci=bootstrap_prob([.03,.07,.04,.06])
        self.assertLess(ci[0],.05)
        self.assertGreater(ci[1],.05)

if __name__=='__main__':unittest.main()
