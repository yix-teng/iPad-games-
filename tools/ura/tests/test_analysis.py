import unittest

from uradata.analysis import (absorption, assess_offer, monthly_medians,
                              privatisation_event_study, project_benchmark, size_cohort)
from uradata.model import flatten

from .helpers import build, project, sale


class TestProjectBenchmark(unittest.TestCase):
    def setUp(self):
        self.rows = flatten([
            project(name="RIVERCOVE RESIDENCES", transactions=[
                sale(1_900_000, 108.05, contract="0526"),
                sale(1_680_000, 108.05, contract="1025"),
                sale(2_000_000, 110.0, contract="0326", sale_type="1"),
            ]),
            project(name="THE VALES", transactions=[sale(1_450_000, 85.0)]),
        ])

    def test_matches_case_insensitively_and_partially(self):
        self.assertEqual(len(project_benchmark(self.rows, "rivercove")), 2)

    def test_excludes_developer_sales_by_default(self):
        self.assertTrue(all(t.is_resale() for t in project_benchmark(self.rows, "RIVERCOVE")))
        self.assertEqual(len(project_benchmark(self.rows, "RIVERCOVE", resale_only=False)), 3)

    def test_sorted_oldest_first(self):
        months = [t.year_month for t in project_benchmark(self.rows, "RIVERCOVE")]
        self.assertEqual(months, sorted(months))

    def test_unknown_project_is_empty_not_an_error(self):
        self.assertEqual(project_benchmark(self.rows, "NO SUCH PLACE"), [])


class TestSizeCohort(unittest.TestCase):
    def test_selects_within_tolerance(self):
        rows = build([sale(1, 100.0), sale(1, 102.0), sale(1, 120.0)])
        target = rows[0].sqft
        self.assertEqual(len(size_cohort(rows, target, tolerance=0.03)), 2)
        self.assertEqual(len(size_cohort(rows, target, tolerance=0.0)), 1)


class TestAssessOffer(unittest.TestCase):
    def _rows(self):
        return build([
            sale(1_900_000, 108.05, contract="0526", floor="06-10"),
            sale(1_908_000, 108.05, contract="0526", floor="11-15"),
            sale(1_680_000, 108.05, contract="1025", floor="01-05"),
        ])

    def test_offer_above_comparables_reads_strong(self):
        rows = self._rows()
        result = assess_offer(rows, 1_988_000, rows[0].sqft)
        self.assertEqual(result.cohort_n, 3)
        self.assertIn("strong", result.verdict)
        self.assertGreater(result.psf_percentile, 50)

    def test_offer_below_comparables_reads_low(self):
        rows = self._rows()
        result = assess_offer(rows, 1_500_000, rows[0].sqft)
        self.assertIn("low", result.verdict)

    def test_offer_at_median_reads_fair(self):
        rows = self._rows()
        result = assess_offer(rows, 1_900_000, rows[0].sqft)
        self.assertIn("fair", result.verdict)

    def test_explicit_fair_value_overrides_cohort_median(self):
        rows = self._rows()
        # Against a much higher fair value the same offer must read as low.
        result = assess_offer(rows, 1_900_000, rows[0].sqft, fair_value=2_200_000)
        self.assertIn("low", result.verdict)

    def test_offer_psf_is_derived_from_size(self):
        rows = self._rows()
        result = assess_offer(rows, 1_988_000, 1163.0)
        self.assertAlmostEqual(result.offer_psf, 1_988_000 / 1163.0, places=6)


class TestAbsorption(unittest.TestCase):
    def test_counts_trailing_window(self):
        rows = build([sale(1, 100, contract=f"{m:02d}26") for m in range(1, 7)]
                     + [sale(1, 100, contract="0125")])
        result = absorption(rows, months=6, total_units=600)
        self.assertEqual(result["sales"], 6)
        self.assertAlmostEqual(result["per_month"], 1.0)
        self.assertAlmostEqual(result["turnover_pct"], 1.0)

    def test_empty_input(self):
        self.assertEqual(absorption([])["sales"], 0)


class TestMonthlyMedians(unittest.TestCase):
    def test_groups_by_month(self):
        rows = build([sale(1_000_000, 100, contract="0126"),
                      sale(1_200_000, 100, contract="0126"),
                      sale(2_000_000, 100, contract="0226")])
        medians = monthly_medians(rows)
        self.assertEqual(len(medians), 2)

    def test_min_count_filters_thin_months(self):
        rows = build([sale(1_000_000, 100, contract="0126"),
                      sale(1_200_000, 100, contract="0126"),
                      sale(2_000_000, 100, contract="0226")])
        self.assertEqual(len(monthly_medians(rows, min_count=2)), 1)


class TestPrivatisationEventStudy(unittest.TestCase):
    def _ec_universe(self, post_multiplier):
        """Two EC projects: a small one privatising mid-window, and a large
        control supplying the market trend. Lease 2011 + 4 + 10 => 2025.

        The control has to dominate the monthly median, otherwise the event
        project detrends against its own price move and the lift cancels out.
        """
        projects = []
        for name, lease, per_month in (("EVENT EC", 2011, 5), ("CONTROL EC", 2018, 20)):
            transactions = []
            for year in (24, 25):
                for month in range(1, 13):
                    # 2025 is the post-privatisation window, for EVENT EC only.
                    multiplier = post_multiplier if (year == 25 and name == "EVENT EC") else 1.0
                    for _ in range(per_month):
                        transactions.append(sale(
                            round(1_000_000 * multiplier), 100.0,
                            contract=f"{month:02d}{year}",
                            tenure=f"99 yrs lease commencing from {lease}"))
            projects.append(project(name=name, transactions=transactions))
        return flatten(projects)

    def test_detects_a_real_step_up(self):
        result = privatisation_event_study(self._ec_universe(1.10), min_pre=8, min_post=8)
        self.assertEqual(result["n_events"], 1)
        self.assertAlmostEqual(result["median_lift"], 0.10, places=2)
        self.assertEqual(result["positive"], 1)

    def test_reports_no_lift_when_there_is_none(self):
        result = privatisation_event_study(self._ec_universe(1.0), min_pre=8, min_post=8)
        self.assertAlmostEqual(result["median_lift"], 0.0, places=6)
        self.assertEqual(result["positive"], 0)

    def test_min_post_filters_thin_events(self):
        result = privatisation_event_study(self._ec_universe(1.10), min_pre=8, min_post=1000)
        self.assertEqual(result["n_events"], 0)
        self.assertIsNone(result["median_lift"])

    def test_ignores_non_ec_and_non_resale(self):
        rows = build([sale(1_000_000, 100, property_type="Condominium")] * 50)
        self.assertEqual(privatisation_event_study(rows)["n_events"], 0)


class TestLargestGap(unittest.TestCase):
    def test_finds_the_dormant_period(self):
        from uradata.analysis import largest_gap
        rows = build([sale(1, 100, contract="0722"),
                      sale(1, 100, contract="1025"),
                      sale(1, 100, contract="1125")])
        gap, resume = largest_gap(rows)
        self.assertEqual(gap, 39)                       # Jul 2022 -> Oct 2025
        self.assertEqual(resume, 2025 * 12 + 9)

    def test_needs_two_points(self):
        from uradata.analysis import largest_gap
        self.assertIsNone(largest_gap([]))
        self.assertIsNone(largest_gap(build([sale(1, 100)])))


class TestSinceMonth(unittest.TestCase):
    def test_filters_inclusively(self):
        from uradata.analysis import since_month
        rows = build([sale(1, 100, contract="0925"),
                      sale(1, 100, contract="1025"),
                      sale(1, 100, contract="0126")])
        self.assertEqual(len(since_month(rows, "2025-10")), 2)
        self.assertEqual(len(since_month(rows, "2025-09")), 3)
        self.assertEqual(len(since_month(rows, "2027-01")), 0)
