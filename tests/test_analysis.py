"""Fast, solver-independent physical smoke tests."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('pipe_analysis',Path(__file__).parents[1]/'scripts'/'analyze.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class Tests(unittest.TestCase):
    def test_friction_laminar(self):
        re=2500
        s=a.derive_stats({'t':np.array([0,1]), 'D':np.array([2/re,2/re]),
                          "E'":np.array([0,0]),"-dPdz'":np.array([0,0])},re)
        np.testing.assert_allclose(s['D_over_laminar'],1)
        np.testing.assert_allclose(s['friction'],64/re)

    def test_friction_pressure_correction(self):
        re=2000; extra=0.004
        s=a.derive_stats({'t':np.array([0]),'D':np.array([3/re]),"-dPdz'":np.array([extra])},re)
        self.assertAlmostEqual(s['friction'][0],64/re+16*extra)

    def test_laminar_intensity_zero(self):
        r=np.linspace(.01,1,12); th=np.arange(16)*2*np.pi/16; z=np.arange(20)*8/20
        u=np.broadcast_to((1-r*r)[:,None,None],(12,16,20))
        zeros=np.zeros_like(u)
        self.assertLess(a.turbulence_intensity(u,zeros,zeros,r),1e-14)
        q,om=a.cylindrical_operators(u,zeros,zeros,r,th,z)
        self.assertLess(np.max(np.abs(q)),1e-10)
        self.assertGreater(np.max(om),1)

if __name__=='__main__': unittest.main()
