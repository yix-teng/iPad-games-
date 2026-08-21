import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from uradata import cli

from .helpers import project, sale


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        transactions = [
            sale(1_680_000, 108.05, contract="1025", floor="01-05"),
            sale(1_900_000, 108.05, contract="0526", floor="06-10"),
            sale(1_908_000, 108.05, contract="0526", floor="11-15"),
            sale(1_570_000, 89.0, contract="0326", floor="06-10"),
            sale(1_580_000, 89.0, contract="0426", floor="11-15"),
            sale(1_440_000, 84.0, contract="0226", floor="06-10"),
            sale(1_450_000, 84.0, contract="0126", floor="11-15"),
            sale(1_750_000, 100.0, contract="0126", floor="01-05"),
            sale(1_800_000, 100.0, contract="0626", floor="11-15"),
            sale(2_000_000, 110.0, contract="0626", floor="06-10"),
        ]
        json.dump([project(name="RIVERCOVE RESIDENCES", transactions=transactions)], self.tmp)
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def run_cli(self, *argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["--cache", self.tmp.name, *argv])
        return code, buffer.getvalue()

    def test_benchmark_lists_transactions(self):
        code, out = self.run_cli("benchmark", "RIVERCOVE")
        self.assertEqual(code, 0)
        self.assertIn("RIVERCOVE RESIDENCES", out)
        self.assertIn("psf by month", out)

    def test_benchmark_unknown_project_exits_nonzero(self):
        code, out = self.run_cli("benchmark", "NOWHERE")
        self.assertEqual(code, 1)
        self.assertIn("No transactions", out)

    def test_offer_reports_a_verdict(self):
        code, out = self.run_cli("offer", "RIVERCOVE", "--price", "1988000", "--sqft", "1163")
        self.assertEqual(code, 0)
        self.assertIn("VERDICT", out)
        self.assertIn("percentile", out)

    def test_offer_with_floor_runs_the_hedonic_model(self):
        code, out = self.run_cli("offer", "RIVERCOVE", "--price", "1988000",
                                 "--sqft", "1163", "--floor", "12")
        self.assertEqual(code, 0)
        self.assertIn("Hedonic model", out)
        self.assertIn("fair value", out)

    def test_parser_requires_a_command(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.build_parser().parse_args([])


class TestCLIWindowing(TestCLI):
    def test_warns_about_a_dormant_period(self):
        code, out = self.run_cli("offer", "RIVERCOVE", "--price", "1988000",
                                 "--sqft", "1163", "--floor", "12")
        # The fixture has a clean run of months, so no warning is expected.
        self.assertEqual(code, 0)
        self.assertNotIn("WARNING", out)

    def test_since_restricts_the_window(self):
        code, out = self.run_cli("offer", "RIVERCOVE", "--price", "1988000",
                                 "--sqft", "1163", "--since", "2026-05")
        self.assertEqual(code, 0)
        self.assertIn("Window: 2026-05 onwards", out)

    def test_empty_window_exits_nonzero(self):
        code, out = self.run_cli("offer", "RIVERCOVE", "--price", "1988000",
                                 "--sqft", "1163", "--since", "2030-01")
        self.assertEqual(code, 1)
