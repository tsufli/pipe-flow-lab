"""Cheap checks for committed numerical experiment datasets (no JAX needed)."""
import json,math,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]/'web'/'barkley'/'instanton'/'data'
class TestCommittedCases(unittest.TestCase):
 def test_two_starts_and_stages(self):
  for r in ['0.5','0.6']:
   with self.subTest(r=r):
    d=json.loads((BASE/f'r_{r}.json').read_text())
    self.assertEqual(d['version'],2)
    self.assertEqual(d['params']['N'],160)
    self.assertEqual(d['params']['steps'],320)
    self.assertEqual(set(d['seeds']),{'linear','reverse'})
    for seed in d['seeds'].values():
     self.assertEqual(set(seed['stages']),{'initial','10','30','100'})
     self.assertEqual(len(seed['reference']['frames']),41)
     for name,z in seed['stages'].items():
      self.assertEqual(len(z['frames']),41)
      self.assertTrue(math.isfinite(z['action']))
      self.assertGreaterEqual(z['action'],0.)
      self.assertTrue(math.isfinite(z['endpointDistance']))
      self.assertGreaterEqual(z['endpointDistance'],0.)
      self.assertEqual(z['frames'][0]['t'],0.)
      self.assertEqual(z['frames'][-1]['t'],80.)
      self.assertAlmostEqual(z['frames'][-1]['cumAction'],z['action'],delta=.00005)
      for f in z['frames']:
       for field in ['q','u','a']:
        self.assertEqual(len(f[field]),160)
        self.assertTrue(all(math.isfinite(v) for v in f[field]))
       self.assertTrue(math.isfinite(f['dEdge']))
       self.assertGreaterEqual(f['cumAction'],-1e-9)
      self.assertTrue(all(z['frames'][j+1]['cumAction']>=z['frames'][j]['cumAction']-2e-6 for j in range(40)))
    l=d['seeds']['linear']['stages']['100'];v=d['seeds']['reverse']['stages']['100']
    self.assertTrue(l['endpointDistance']<.15)
    self.assertTrue(v['endpointDistance']<.15)
 def test_profile_view(self):
  page=(BASE.parent/'index.html').read_text()
  script=(BASE.parent/'profiles.js').read_text()
  for key in ['id="initialPlot"','id="optimizedPlot"',
              'id="initialMovie"','id="optimizedMovie"',
              'id="initialMovieTime"','id="optimizedMovieTime"',
              'id="curveCount"','id="initialKind"','id="timeSlider"']:
   self.assertIn(key,page)
  self.assertIn('src="./profiles.js"',page)
  self.assertNotIn('id="deformPlot"',page)
  self.assertNotIn('id="heat"',page)
  self.assertNotIn('id="compare"',page)
  self.assertIn('function drawProfiles()',script)
  self.assertIn('function updateMovingProfiles()',script)
  self.assertIn('function staticTraces(kind)',script)
  self.assertIn('function movieTraces(kind)',script)
  # Playback must update only the movie plots, never the upper overlays.
  moving=script.split('function updateMovingProfiles(){',1)[1]
  self.assertIn('Plotly.restyle($("initialMovie")',moving)
  self.assertIn('Plotly.restyle($("optimizedMovie")',moving)
  self.assertNotIn('Plotly.restyle($("initialPlot")',moving)
  self.assertNotIn('Plotly.restyle($("optimizedPlot")',moving)
  self.assertIn('dash:"solid"',script)
  self.assertNotIn('dash:"dot"',script)
  self.assertNotIn('dash:"dash"',script)
  self.assertIn("current.seeds[id].reference.frames",script)
  self.assertIn("current.seeds[id].stages.initial.frames",script)
  self.assertIn("current.seeds[id].stages[stage].frames",script)
 def test_summary(self):
  d=json.loads((BASE/'summary.json').read_text())
  self.assertEqual(len(d['data']),2)
  self.assertEqual([x['r'] for x in d['data']],[.5,.6])
if __name__=='__main__':unittest.main()