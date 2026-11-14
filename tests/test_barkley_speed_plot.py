"""Check the eight-point Barkley propagation-speed chart and its theory overlay."""
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "web/barkley/edge-scan/data/summary.json"
PAGE = ROOT / "web/barkley/edge-scan/index.html"

class TestBarkleySpeedPlot(unittest.TestCase):
    def test_eight_numerical_cases(self):
        data = json.loads(DATA.read_text())
        scan = data["scan"]
        self.assertEqual(len(scan), 8)
        self.assertEqual(sorted(x["r"] for x in scan), [x["r"] for x in scan])
        for item in scan:
            self.assertGreater(item["r"], 0)
            self.assertTrue(math.isfinite(item["puff_c"]))
            self.assertTrue(math.isfinite(item["edge_c"]))

    def test_theoretical_curve(self):
        data = json.loads(DATA.read_text())
        p = data["params"]
        def predicted(r):
            return p["U0"] - p["zeta"] + math.sqrt(p["D"]/2) * (
                math.sqrt(r+p["delta"]) - 3 * math.sqrt(r)
            )
        rs = [a["r"] for a in data["scan"]]
        expected = [predicted(r) for r in rs]
        self.assertTrue(all(math.isfinite(c) for c in expected))
        self.assertTrue(all(a>b for a,b in zip(expected, expected[1:])))
        # The asymptotic prediction is not forced to match the finite-epsilon data.
        self.assertGreater(abs(predicted(rs[0])-data["scan"][0]["puff_c"]), .01)

    def test_site_has_three_distinct_traces(self):
        html = PAGE.read_text()
        self.assertIn('id="speedplot"', html)
        self.assertIn('function theoreticalPuffSpeed(r)', html)
        self.assertIn('function renderSpeedTrend()', html)
        self.assertIn('"Puff · numerical (8 points)"', html)
        self.assertIn('"Decay edge · numerical (8 points)"', html)
        self.assertIn('"Puff · theory (ε → 0)"', html)
        self.assertIn('renderSelected();renderSpeedTrend();', html)
        self.assertIn("predictionR.map(theoreticalPuffSpeed)", html)
        self.assertIn('id="speedDeviation"', html)

if __name__ == "__main__":
    unittest.main()
