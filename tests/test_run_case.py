"""Regression tests for the dnsjax wall-clock stop parameter."""
import importlib.util
from datetime import timedelta
from pathlib import Path
import unittest

from pydantic import TypeAdapter

spec = importlib.util.spec_from_file_location(
    "run_case", Path(__file__).resolve().parents[1] / "scripts" / "run_case.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class WallDurationTests(unittest.TestCase):
    def test_iso_8601_duration_is_accepted_by_pydantic(self):
        adapter = TypeAdapter(timedelta)
        for minutes in [1, 15, 42, 44]:
            with self.subTest(minutes=minutes):
                value = launcher.iso_wall_duration(minutes)
                self.assertEqual(value, f"PT{minutes}M")
                self.assertEqual(adapter.validate_python(value), timedelta(minutes=minutes))

    def test_invalid_wall_budgets_rejected(self):
        for minutes in [0, 45, 100]:
            with self.subTest(minutes=minutes):
                with self.assertRaises(ValueError):
                    launcher.iso_wall_duration(minutes)


if __name__ == "__main__":
    unittest.main()
