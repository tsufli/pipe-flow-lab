"""Unit checks for Barkley 100-run front tracking and drift identities."""
import pathlib, sys, unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"scripts"))
import numpy as np
from barkley_front_tracking import PARAMS,rhs,regions,analytical_length,estimate,tracking

class FrontTrackingTest(unittest.TestCase):
    def test_laminar_stationary(self):
        p=PARAMS
        q=np.zeros(50);u=np.full(50,2.)
        a,b=rhs(q,u,.3,p["r"],p["delta"],p["eps1"],p["eps2"],p["zeta"],p["D"],p["U0"],p["Ub"])
        np.testing.assert_allclose(a,0,atol=1e-13)
        np.testing.assert_allclose(b,0,atol=1e-13)

    def test_threshold_is_between_fronts(self):
        p=PARAMS; x=(np.arange(p["N"])+.5)*(p["L"]/p["N"])
        q=1.65/(1+np.exp(np.minimum(100,(np.abs(x-50)-4))))
        groups=regions(q,p["L"]/p["N"],.5,1.6)
        self.assertEqual(len(groups),1)
        self.assertTrue(7<groups[0][0]<12)
        self.assertAlmostEqual((groups[0][1]+groups[0][2])/2,50,delta=1)

    def test_periodic_unwrapping(self):
        p=dict(PARAMS)
        x=(np.arange(p["N"])+.5)*(p["L"]/p["N"])
        q0=1.5*np.exp(-.5*((x-99+50)%100-50)**2/9)
        q1=np.roll(q0,5)
        tr=tracking(np.array([q0,q1]),p)
        self.assertTrue(np.all(np.isfinite(tr[:,:3])))
        self.assertLess(abs(tr[1,0]-tr[0,0]),3)
        self.assertAlmostEqual(tr[0,2],tr[1,2],delta=.001)

    def test_asymptotics(self):
        a=analytical_length(PARAMS)
        self.assertAlmostEqual(a["speed"],.4564350094,places=6)
        self.assertAlmostEqual(a["u_down"],1.4272322176,places=6)
        self.assertAlmostEqual(a["bulk_length"],4.2506743893,places=5)
        self.assertGreater(a["q_down"],.5)

    def test_length_diffusion_identity(self):
        rng=np.random.default_rng(4)
        p=dict(PARAMS)
        tracks=np.zeros((30,61,5))
        for i in range(30):
            xu=np.cumsum(np.r_[0,rng.normal(.1,.12,60)])+2
            ld=6+np.cumsum(np.r_[0,rng.normal(0,.13,60)])
            tracks[i,:,0]=xu
            tracks[i,:,1]=xu+ld
            tracks[i,:,2]=ld
            tracks[i,:,3]=0
        est=estimate(tracks,1,p)
        data=[r for r in est["rows"] if "D_l" in r]
        self.assertTrue(len(data)>0)
        for r in data:
            self.assertAlmostEqual(r["A_l"],r["A_d"]-r["A_u"],places=11)
            self.assertAlmostEqual(r["D_l"],r["D_dd"]+r["D_uu"]-2*r["D_ud"],places=11)
            self.assertGreaterEqual(r["D_l"],0)

if __name__=="__main__":
    unittest.main()
