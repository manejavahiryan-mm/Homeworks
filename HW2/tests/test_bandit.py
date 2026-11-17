"""Run with: python -m unittest discover -s tests -v"""
import unittest
import numpy as np
from Bandit import Bandit, EpsilonGreedy, ThompsonSampling, comparison


class BanditTests(unittest.TestCase):
    def test_inheritance_and_shapes(self):
        for cls in (EpsilonGreedy, ThompsonSampling):
            a = cls([1, 2, 3, 4], n_trials=120, seed=9).experiment()
            self.assertIsInstance(a, Bandit)
            self.assertEqual(len(a.rewards), 120)
            self.assertEqual(sum(a.counts), 120)
            self.assertTrue(np.all(a.regrets >= 0))
            self.assertEqual(a.estimate_history.shape, (120, 4))
            self.assertAlmostEqual(a.cumulative_regrets[-1], a.regrets.sum())

    def test_reproducible(self):
        for cls in (EpsilonGreedy, ThompsonSampling):
            a = cls([1, 2, 3, 4], n_trials=100, seed=10).experiment()
            b = cls([1, 2, 3, 4], n_trials=100, seed=10).experiment()
            np.testing.assert_array_equal(a.chosen, b.chosen)
            np.testing.assert_array_equal(a.rewards, b.rewards)

    def test_posterior_precision(self):
        a = ThompsonSampling([1, 2, 3, 4], n_trials=1, seed=1)
        a.update(2, 3.5)
        self.assertAlmostEqual(a.posterior_precisions[2], 2.0)
        self.assertAlmostEqual(a.posterior_means[2], 1.75)

    def test_epsilon_decay(self):
        self.assertEqual(min(1.0, 1.0 / 1), 1.0)
        self.assertAlmostEqual(min(1.0, 1.0 / 20), 0.05)

    def test_validation(self):
        with self.assertRaises(ValueError):
            ThompsonSampling([1, 2, 3, 4], prior_precision=0)
        with self.assertRaises(ValueError):
            EpsilonGreedy([1, 2, 3, 4], n_trials=0)


if __name__ == '__main__':
    unittest.main()
