import math
import unittest

from uradata.hedonic import HedonicModel, Valuation

from .helpers import build, sale


def synthetic(n_floors=4, n_months=12):
    """Prices generated from a known hedonic surface, so the fit is checkable.

    log(price) = 12 + 0.9*log(sqft) + 0.004*floor + 0.002*month
    """
    rows = []
    for month in range(n_months):
        for floor_index in range(n_floors):
            floor_range = ["01-05", "06-10", "11-15", "16-20"][floor_index]
            floor = [3, 8, 13, 18][floor_index]
            for sqm in (84.0, 90.0, 108.0):
                sqft = sqm * 10.7639104
                log_price = 12 + 0.9 * math.log(sqft) + 0.004 * floor + 0.002 * month
                contract = f"{month % 12 + 1:02d}{25 + month // 12:02d}"
                rows.append(sale(round(math.exp(log_price)), sqm,
                                 contract=contract, floor=floor_range))
    return build(rows)


class TestHedonicModel(unittest.TestCase):
    def setUp(self):
        self.transactions = synthetic()
        self.model = HedonicModel(self.transactions)

    def test_recovers_known_coefficients(self):
        self.assertAlmostEqual(self.model.size_elasticity, 0.9, places=3)
        self.assertAlmostEqual(self.model.floor_premium, math.exp(0.004) - 1, places=5)
        self.assertAlmostEqual(self.model.monthly_drift, math.exp(0.002) - 1, places=5)
        self.assertGreater(self.model.r2, 0.999)

    def test_values_an_unobserved_unit(self):
        sqft = 1163.0
        month = max(t.month for t in self.transactions)
        valuation = self.model.value(sqft, 12, month)
        base = min(t.month for t in self.transactions)
        expected = math.exp(12 + 0.9 * math.log(sqft) + 0.004 * 12
                            + 0.002 * (month - base))
        self.assertAlmostEqual(valuation.price / expected, 1.0, places=3)
        self.assertAlmostEqual(valuation.psf, valuation.price / sqft, places=6)

    def test_range_brackets_the_point_estimate(self):
        valuation = self.model.value(1163, 12, max(t.month for t in self.transactions))
        self.assertLessEqual(valuation.low, valuation.price)
        self.assertLessEqual(valuation.price, valuation.high)

    def test_assess_classifies_offers(self):
        valuation = Valuation(price=2_000_000, psf=1_720, low=1_900_000, high=2_100_000,
                              n=50, r2=0.9, residual_sd=0.05)
        self.assertEqual(valuation.assess(2_150_000), "above the model's range")
        self.assertEqual(valuation.assess(2_050_000), "above fair value, inside the range")
        self.assertEqual(valuation.assess(1_950_000), "below fair value, inside the range")
        self.assertEqual(valuation.assess(1_800_000), "below the model's range")

    def test_perfect_fit_collapses_the_range_to_a_point(self):
        # Synthetic data is noiseless, so low == price == high. Guard against a
        # future change that would silently widen it.
        valuation = self.model.value(1163, 12, max(t.month for t in self.transactions))
        self.assertAlmostEqual(valuation.high / valuation.low, 1.0, places=6)

    def test_higher_floor_is_worth_more(self):
        month = max(t.month for t in self.transactions)
        low = self.model.value(1163, 3, month).price
        high = self.model.value(1163, 18, month).price
        self.assertGreater(high, low)

    def test_requires_enough_usable_data(self):
        with self.assertRaises(ValueError):
            HedonicModel(build([sale(1_000_000, 100)]))

    def test_ignores_rows_without_a_floor(self):
        # Landed rows carry floorRange '-' and must not poison the fit.
        rows = synthetic() + build([sale(9_000_000, 300, floor="-")])
        model = HedonicModel(rows)
        self.assertEqual(model.n, len(synthetic()))
