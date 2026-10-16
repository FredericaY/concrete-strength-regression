import unittest

import numpy as np

from data import load_concrete_data, split_train_test, standardize_train_test
from gradient_descent import LinearRegressionGD, _loss_and_gradient
from metrics import regression_metrics


class GradientDescentTests(unittest.TestCase):
    def test_gradient_matches_finite_differences_including_intercept(self):
        design = np.array([[1, -2, 3], [1, 0.5, -1], [1, 4, 2]], dtype=float)
        y = np.array([3.0, -2.0, 5.0])
        theta = np.array([0.7, -0.3, 1.2])
        _, actual = _loss_and_gradient(design, y, theta)
        expected = np.zeros_like(theta)
        for j in range(len(theta)):
            step = np.zeros_like(theta)
            step[j] = 1e-6
            plus = np.mean((design @ (theta + step) - y) ** 2)
            minus = np.mean((design @ (theta - step) - y) ** 2)
            expected[j] = (plus - minus) / (2e-6)
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)

    def test_univariate_recovers_known_slope_and_intercept(self):
        X = np.arange(-3.0, 4.0)
        y = 2.5 * X - 4.0
        model = LinearRegressionGD(learning_rate=0.1, tol=1e-10).fit(X, y)
        np.testing.assert_allclose(model.coef_, [2.5], atol=1e-8)
        self.assertAlmostEqual(model.intercept_, -4.0, places=8)
        np.testing.assert_allclose(model.predict(X[:, None]), y, atol=1e-8)
        self.assertTrue(model.converged_)

    def test_multivariate_matches_independent_least_squares(self):
        rng = np.random.default_rng(5140)
        X = rng.normal(size=(120, 3))
        y = 3.0 + X @ np.array([1.5, -2.0, 0.7]) + rng.normal(0, 0.1, 120)
        reference, *_ = np.linalg.lstsq(np.column_stack((np.ones(len(X)), X)), y, rcond=None)
        model = LinearRegressionGD(tol=1e-10).fit(X, y)
        np.testing.assert_allclose(np.r_[model.intercept_, model.coef_], reference, atol=1e-8)
        self.assertTrue(model.converged_)

    def test_history_and_iteration_limit(self):
        X, y = np.arange(4.0), np.array([1.0, 3.0, 5.0, 7.0])
        model = LinearRegressionGD(learning_rate=0.01, max_iter=3, tol=0).fit(X, y)
        self.assertEqual(model.n_iter_, 3)
        self.assertEqual(len(model.loss_history_), 4)
        self.assertAlmostEqual(model.loss_history_[0], np.mean(y ** 2))
        self.assertAlmostEqual(model.loss_history_[-1], np.mean((model.predict(X) - y) ** 2))
        self.assertTrue(np.all(np.diff(model.loss_history_) < 0))
        self.assertFalse(model.converged_)

    def test_invalid_inputs_and_prediction_shape(self):
        with self.assertRaises(ValueError):
            LinearRegressionGD(learning_rate=0)
        with self.assertRaises(RuntimeError):
            LinearRegressionGD().predict([1, 2])
        with self.assertRaises(ValueError):
            LinearRegressionGD().fit([1, np.nan], [2, 3])
        model = LinearRegressionGD().fit([1, 2], [2, 3])
        with self.assertRaises(ValueError):
            model.predict([[1, 2]])

    def test_numerical_divergence_is_reported(self):
        with self.assertRaisesRegex(FloatingPointError, "learning rate"):
            LinearRegressionGD(learning_rate=1e100, max_iter=10).fit([1, 2], [2, 3])


class DataAndMetricsTests(unittest.TestCase):
    def test_original_dataset_and_exact_split_boundaries(self):
        data = load_concrete_data()
        train, test = split_train_test(data)
        self.assertEqual(data.shape, (1030, 9))
        self.assertEqual(len(train), 900)
        self.assertEqual(test.index.tolist(), list(range(501, 631)))
        self.assertEqual(train.index.tolist(), list(range(1, 501)) + list(range(631, 1031)))
        self.assertEqual(data.loc[501, "cement"], 491.0)
        self.assertEqual(data.loc[630, "age"], 7)

    def test_scaling_uses_training_statistics_only(self):
        train = np.array([[1.0, 4.0], [3.0, 4.0]])
        test = np.array([[101.0, 8.0]])
        scaled_train, scaled_test = standardize_train_test(train, test)
        np.testing.assert_allclose(scaled_train, [[-1, 0], [1, 0]])
        np.testing.assert_allclose(scaled_test, [[99, 4]])
        np.testing.assert_allclose(train, [[1, 4], [3, 4]])

    def test_metrics_use_evaluated_set_population_variance(self):
        scores = regression_metrics([1, 2, 3], [1, 2, 2])
        self.assertAlmostEqual(scores["mse"], 1 / 3)
        self.assertAlmostEqual(scores["r2"], 0.5)
        self.assertTrue(np.isnan(regression_metrics([2, 2], [2, 2])["r2"]))


if __name__ == "__main__":
    unittest.main()
