"""Fast, standard-library smoke tests for the precomputed Barkley instanton-explorer archive."""
import json
import math
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'web'/'barkley'/'instanton'/'data'
SITE=ROOT/'web'/'barkley'/'instanton'/'index.html'

class ArchiveTest(unittest.TestCase):
    def test_site_references_and_data(self):
        html=SITE.read_text()
        self.assertIn('id="stage"',html)
        self.assertIn('id="timeRange"',html)
        self.assertIn('id="compare"',html)
        self.assertNotIn('id="pathheat"',html)
        for r in (0.5,0.6):
            doc=json.loads((DATA/f'r_{r:.1f}.json').read_text())
            self.assertEqual(doc['r'],r)
            self.assertEqual(doc['raster']['N_full'],320)
            self.assertEqual(doc['raster']['N_display'],160)
            self.assertLess(doc['edge']['residual'],1e-8)
            self.assertLess(doc['puff']['residual'],1e-8)
            self.assertGreater(doc['edgeMode']['lambda'],0)
            self.assertEqual(set(doc['candidates']),{'best','baseline'})
            self.assertLess(doc['candidates']['best']['action'],doc['candidates']['baseline']['action'])
            for candidate in doc['candidates'].values():
                self.assertLess(candidate['min_edge_distance'],0.06)
                self.assertGreater(candidate['action'],0)
                self.assertLess(candidate['amplitude_bracket_width'],1e-8)
                frames=candidate['frames']
                self.assertGreaterEqual(len(frames),30)
                self.assertTrue(all(len(f['q'])==160 and len(f['u'])==160 for f in frames))
                self.assertTrue(all(frames[i]['cumI'] <= frames[i+1]['cumI']+1e-6 for i in range(len(frames)-1)))
                self.assertAlmostEqual(frames[-1]['cumI'],candidate['action'],places=5)
                i=candidate['edge_frame_index']
                self.assertEqual(frames[i]['d_edge'],candidate['min_edge_distance'])
            self.assertGreater(len(doc['reverseDeterministic']),15)

if __name__=='__main__':unittest.main()