"""Short-puff ensemble: tracking, censoring, and sufficient-statistics tests."""
import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"scripts"))
import numpy as np
from barkley_shortlength import tracked,fit,BINS,compute_coeff,bin_sums,find_roots,edge_reference
from barkley_front_tracking import PARAMS

class TestShortLength(unittest.TestCase):
    def test_tracker_thresholds(self):
        p=dict(PARAMS,save_dt=.2)
        x=(np.arange(p["N"])+.5)*p["L"]/p["N"]
        q=.3*np.exp(-.5*((x-50)/1.3)**2)
        self.assertFalse(np.isfinite(tracked(q[None,:],p,.5)[0,2]))
        self.assertTrue(np.isfinite(tracked(q[None,:],p,.15)[0,2]))
    def test_edge_reference(self):
        edge=edge_reference(dict(PARAMS))
        self.assertGreater(edge["peak"],.15)
        self.assertLess(edge["peak"],.5)
        self.assertIsNone(edge["width_at_threshold"]["0.5"])
        self.assertGreater(edge["width_at_threshold"]["0.15"],0)
    def test_grouped_diffusion_identity(self):
        rng=np.random.default_rng(0)
        p=dict(PARAMS,save_dt=.2)
        tr=[];ids=[]
        for i in range(20):
            for branch in range(3):
                x=np.r_[0,np.cumsum(rng.normal(.08,.09,61))]
                l=2.5+np.r_[0,np.cumsum(rng.normal(.0,.05,61))]
                z=np.zeros((62,5))
                z[:,0]=x;z[:,1]=x+l;z[:,2]=l;z[:,3]=0;z[:,4]=1.3
                tr.append(z);ids.append(i)
        est=fit(tr,2,p,group_ids=ids,bootstrap=20)
        self.assertEqual(est["independent_ancestors"],20)
        vals=[r for r in est["rows"] if "A_l" in r]
        self.assertTrue(vals)
        for r in vals:
            self.assertAlmostEqual(r["A_l"],r["A_d"]-r["A_u"])
            self.assertAlmostEqual(r["D_l"],r["D_dd"]+r["D_uu"]-2*r["D_ud"])
    def test_no_intervening_invalid(self):
        p=dict(PARAMS,save_dt=.2)
        z=np.zeros((1,12,5))
        z[0,:,2]=3;z[0,:,3]=0
        z[0,:,0]=np.arange(12)*.2;z[0,:,1]=z[0,:,0]+3
        z[0,4,:3]=np.nan
        arr,_=bin_sums([z[0]],2,p)
        self.assertEqual(int(arr[:,:,0].sum()),7)
    def test_roots_with_half_width_bins(self):
        rows=[dict(lo=1,hi=1.5,ell=1.25,A_l=1),dict(lo=1.5,hi=2,ell=1.75,A_l=-1)]
        self.assertAlmostEqual(find_roots(rows)[0]["ell"],1.5)
if __name__=="__main__":
    unittest.main()
