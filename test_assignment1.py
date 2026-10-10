"""Local verification, not an official course grading suite.

Run: python -m unittest -v test_assignment1.py
"""
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import Assignment1 as a


class AssignmentChecks(unittest.TestCase):
    def test_matrix_defaults(self):
        expected = ([[3, 2], [2, 3]], [[7, 6], [-2, -1]],
                    [[4, 2], [6, 0]], [[11, -4], [4, -1]], 4, 6)
        for actual, wanted in zip(a.matrix_operations(), expected):
            np.testing.assert_allclose(actual, wanted)
        np.testing.assert_allclose(a.matrix_inverse(), [[.6, -.7], [-.2, .4]])

    def test_custom_matrices_and_repeated_eigenvalues(self):
        rng = np.random.default_rng(43)
        A, B = rng.normal(size=(2, 4, 4))
        original = A.copy()
        out = a.matrix_operations(A, B)
        np.testing.assert_allclose(out[0], A @ B)
        np.testing.assert_allclose(out[4], np.trace(A))
        np.testing.assert_array_equal(A, original)
        for M in (np.eye(4) * 3, A + A.T, np.array([[2, -1], [-1, 2]])):
            vals, vecs = a.eigensystem(M)
            np.testing.assert_allclose(M @ vecs, vecs * vals, atol=1e-12)
            np.testing.assert_allclose(vecs.T @ vecs, np.eye(len(vals)), atol=1e-12)
        C = A.T @ A + np.eye(4)
        np.testing.assert_allclose(C @ a.matrix_inverse(C), np.eye(4), atol=1e-12)

    def test_probability_defaults(self):
        np.testing.assert_allclose(a.distribution_moments(), [np.exp(-3), 3, 3])
        np.testing.assert_allclose(a.event_probabilities(), [.18, .36, .62])
        np.testing.assert_allclose(a.coin_estimates(), [.6, 2/3])
        np.testing.assert_allclose(a.gaussian_mle(), [4, 5])
        self.assertAlmostEqual(a.gaussian_map(), 1.)

    def test_coin_boundary_modes_and_grid_maxima(self):
        grid = np.linspace(0, 1, 10001)
        for n in (1, 5, 19):
            for h in range(n + 1):
                mle, map_ = a.coin_estimates(h, n)
                likelihood = grid**h * (1-grid)**(n-h)
                posterior = grid * likelihood
                self.assertLessEqual(abs(grid[np.argmax(likelihood)] - mle), .00011)
                self.assertLessEqual(abs(grid[np.argmax(posterior)] - map_), .00011)

    def test_general_estimates(self):
        np.testing.assert_allclose(a.distribution_moments(.7), [np.exp(-.7), .7, .7])
        np.testing.assert_allclose(a.event_probabilities(.4, .5, .2), [.1, .25, .8])
        np.testing.assert_allclose(a.gaussian_mle([2, 4]), [3, 1])
        self.assertAlmostEqual(a.gaussian_map(8, 2, 2, 6), 6.5)

    def test_generator_reproducibility_and_no_mutation(self):
        mean = np.array([4., 5.]); cov = np.array([[2., .5], [.5, 1.]])
        mean_copy, cov_copy = mean.copy(), cov.copy()
        np.random.seed(29)
        state_before = np.random.get_state()
        X = a.generate_data(50, 8, mean, cov)
        state_after = np.random.get_state()
        np.testing.assert_array_equal(state_before[1], state_after[1])
        self.assertEqual(state_before[2:], state_after[2:])
        self.assertEqual(X.shape, (50, 2))
        np.testing.assert_array_equal(X, a.generate_data(50, 8, mean, cov))
        np.testing.assert_array_equal(mean, mean_copy)
        np.testing.assert_array_equal(cov, cov_copy)

    def test_pca_geometry_and_degenerate_cases(self):
        datasets = [a.generate_data(), np.array([[1, 2], [2, 4], [3, 6]]),
                    np.ones((4, 2)), np.array([[1,0],[-1,0],[0,1],[0,-1]]),
                    np.array([[1,2],[3,7]])]
        for X in datasets:
            before = X.copy()
            r = a.pca(X)
            np.testing.assert_array_equal(X, before)
            np.testing.assert_allclose(r['centered'].mean(0), 0, atol=1e-12)
            np.testing.assert_allclose(r['covariance'], np.cov(X, rowvar=False))
            U, vals, Z = r['components'], r['eigenvalues'], r['centered']
            self.assertTrue(np.all(np.diff(vals) <= 0))
            np.testing.assert_allclose(U.T @ U, np.eye(2), atol=1e-12)
            np.testing.assert_allclose(r['covariance'] @ U, U * vals, atol=1e-12)
            np.testing.assert_allclose(r['scores'] @ U.T, Z, atol=1e-12)
            np.testing.assert_allclose(np.var(r['scores'], axis=0, ddof=1), vals, atol=1e-12)
            reconstruction = r['scores'][:, :1] @ U[:, :1].T
            self.assertAlmostEqual(np.sum((Z-reconstruction)**2), (len(X)-1)*vals[1])

    def test_variance_fractions_and_plots(self):
        values = np.array([[3., 1.]])
        np.testing.assert_allclose(a.explained_variance(values), [[.75, .25]])
        np.testing.assert_array_equal(values, [[3, 1]])
        for X in (a.generate_data(), np.ones((3, 2))):
            fig = a.make_plots(X, a.pca(X))
            self.assertEqual(len(fig.axes), 3)
            fig.canvas.draw()
            for ax in fig.axes:
                self.assertTrue(ax.get_xlabel() and ax.get_ylabel())
            plt.close(fig)


if __name__ == '__main__':
    unittest.main(verbosity=2)
