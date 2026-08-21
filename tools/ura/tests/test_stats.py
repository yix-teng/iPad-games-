import unittest

from uradata.stats import median, ols, percentile, predict, r_squared, rank_of, residual_sd


class TestPercentile(unittest.TestCase):
    def test_known_values(self):
        data = [1, 2, 3, 4, 5]
        self.assertEqual(percentile(data, 0.0), 1)
        self.assertEqual(percentile(data, 0.5), 3)
        self.assertEqual(percentile(data, 1.0), 5)

    def test_interpolates_between_points(self):
        self.assertEqual(percentile([0, 10], 0.25), 2.5)

    def test_even_length_median(self):
        self.assertEqual(median([1, 2, 3, 4]), 2.5)

    def test_does_not_require_sorted_input(self):
        self.assertEqual(median([5, 1, 3]), 3)

    def test_edge_cases(self):
        self.assertIsNone(percentile([], 0.5))
        self.assertEqual(percentile([7], 0.9), 7.0)
        with self.assertRaises(ValueError):
            percentile([1, 2], 1.5)

    def test_rank_of(self):
        self.assertEqual(rank_of(3, [1, 2, 3, 4]), 50.0)
        self.assertEqual(rank_of(0, [1, 2]), 0.0)
        self.assertEqual(rank_of(9, [1, 2]), 100.0)
        self.assertIsNone(rank_of(1, []))


class TestOLS(unittest.TestCase):
    def test_recovers_exact_linear_relationship(self):
        # y = 2 + 3*x1 - 1*x2
        design, target = [], []
        for x1 in range(5):
            for x2 in range(5):
                design.append([1.0, float(x1), float(x2)])
                target.append(2 + 3 * x1 - x2)
        coefficients = ols(design, target)
        for got, want in zip(coefficients, [2, 3, -1]):
            self.assertAlmostEqual(got, want, places=6)
        self.assertAlmostEqual(r_squared(design, target, coefficients), 1.0, places=9)
        self.assertAlmostEqual(residual_sd(design, target, coefficients), 0.0, places=6)

    def test_fits_noisy_data_close_to_truth(self):
        noise = [0.1, -0.1, 0.05, -0.05, 0.0] * 4
        design = [[1.0, float(i)] for i in range(20)]
        target = [1 + 2 * i + noise[i] for i in range(20)]
        intercept, slope = ols(design, target)
        self.assertAlmostEqual(slope, 2.0, places=2)
        self.assertAlmostEqual(intercept, 1.0, places=1)

    def test_rejects_singular_system(self):
        # Second column is an exact copy of the first.
        design = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        with self.assertRaises(ValueError):
            ols(design, [1.0, 2.0, 3.0])

    def test_rejects_malformed_input(self):
        with self.assertRaises(ValueError):
            ols([], [])
        with self.assertRaises(ValueError):
            ols([[1.0, 2.0]], [1.0])            # fewer rows than features
        with self.assertRaises(ValueError):
            ols([[1.0], [1.0]], [1.0])          # length mismatch

    def test_predict(self):
        self.assertAlmostEqual(predict([1.0, 2.0], [3.0, 4.0]), 11.0)
