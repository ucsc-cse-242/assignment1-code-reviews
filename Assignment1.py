```python
"""
CSE 242 Assignment 1 — Assignment1.py

Solutions for matrix operations, probability, MLE/MAP, covariance,
and PCA using NumPy and Matplotlib.

Functions do not mutate input arrays, prompt for input, or perform
network I/O. Only the main guard saves and displays figures.
"""

import numpy as np


# ============================================================
# PART 1 (20 points)
# ============================================================

# Q1: Matrix operations
#
# For A = [[3, 4], [2, 1]] and B = [[1, 2], [0, -1]]:
# AB = [[3, 2], [2, 3]]
# BA = [[7, 6], [-2, -1]]
# (A+B).T = [[4, 2], [6, 0]]
# A@B.T = [[11, -4], [4, 3]]
# trace(A) = 4
# trace(AB) = 6


def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB)."""
    if A is None:
        A = np.array([[3, 4], [2, 1]])
    if B is None:
        B = np.array([[1, 2], [0, -1]])

    A = np.asarray(A)
    B = np.asarray(B)

    AB = A @ B
    BA = B @ A
    sum_transpose = (A + B).T
    AB_transpose = A @ B.T
    trace_A = np.trace(A)
    trace_AB = np.trace(AB)

    return AB, BA, sum_transpose, AB_transpose, trace_A, trace_AB


# Q2: Eigenvalues and eigenvectors
#
# For M = [[2, -1], [-1, 2]]:
# The eigenvalues are 1 and 3.
# Corresponding eigenvectors can be [1, 1] and [1, -1].
# Each eigenpair satisfies Mv = lambda*v.


def eigensystem(M=None):
    """Return eigenvalues and eigenvectors as columns."""
    if M is None:
        M = np.array([[2, -1], [-1, 2]])

    M = np.asarray(M)
    values, vectors = np.linalg.eigh(M)
    order = np.argsort(values)
    return values[order], vectors[:, order]


# Q3: Prove trace(AB) = trace(BA).
#
# For A of shape (m,n) and B of shape (n,m):
#
# trace(AB) = sum_i sum_j A[i,j]B[j,i].
#
# Reordering the sums gives:
# sum_j sum_i B[j,i]A[i,j] = trace(BA).
#
# Therefore, trace(AB) = trace(BA).


# Q4: Matrix inverse
#
# C = [[4, 7], [2, 6]]
# det(C) = 4*6 - 7*2 = 10.
# Since det(C) is nonzero, C is invertible.
#
# C^(-1) = (1/det(C)) * [[6, -7], [-2, 4]]
#        = [[0.6, -0.7], [-0.2, 0.4]].


def matrix_inverse(C=None):
    """Return the inverse of a nonsingular square matrix."""
    if C is None:
        C = np.array([[4, 7], [2, 6]])

    return np.linalg.inv(np.asarray(C))


# ============================================================
# PART 2 (50 points)
# ============================================================

# Q1: Distribution moments
#
# Sum(rate^k/k!, k=0 to infinity) = exp(rate).
# Normalization requires c*exp(rate) = 1.
# Therefore c = exp(-rate).
#
# This is a Poisson distribution:
# E[X] = rate and Var(X) = rate.


def distribution_moments(rate=3.):
    """Return normalization constant, mean, and variance."""
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be positive and finite")

    c = np.exp(-rate)
    return c, rate, rate


# Q2: Event probabilities
#
# P(A intersect B) = P(A|B)P(B).
# P(B|A) = P(A intersect B)/P(A).
# P(A union B) = P(A) + P(B) - P(A intersect B).


def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return intersection, conditional, and union probabilities."""
    if not (0 < p_a <= 1 and 0 < p_b <= 1):
        raise ValueError("p_a and p_b must be in (0, 1]")
    if not (0 <= p_a_given_b <= 1):
        raise ValueError("p_a_given_b must be in [0, 1]")

    intersection = p_a_given_b * p_b

    if intersection > p_a:
        raise ValueError("Invalid probability distribution")

    conditional = intersection / p_a
    union = p_a + p_b - intersection

    if union > 1:
        raise ValueError("Invalid probability distribution")

    return intersection, conditional, union


# Q3: MLE and MAP for coin tosses
#
# Likelihood:
# L(theta) proportional to theta^heads * (1-theta)^tails.
#
# The MLE is heads/tosses.
#
# Prior density: p(theta) = 2*theta.
# Posterior is proportional to:
# theta^(heads+1) * (1-theta)^tails.
#
# For heads < tosses, the MAP is (heads+1)/(tosses+1).
# If all tosses are heads, the posterior is maximized at theta=1.


def coin_estimates(heads=3, tosses=5):
    """Return MLE and MAP using the prior density 2*theta."""
    if not isinstance(heads, (int, np.integer)):
        raise ValueError("heads must be an integer")
    if not isinstance(tosses, (int, np.integer)) or tosses <= 0:
        raise ValueError("tosses must be a positive integer")
    if heads < 0 or heads > tosses:
        raise ValueError("Require 0 <= heads <= tosses")

    mle = heads / tosses
    map_estimate = (
        1.0 if heads == tosses
        else (heads + 1) / (tosses + 1)
    )

    return mle, map_estimate


# Q4: Gaussian maximum likelihood estimates
#
# Mean MLE = sum(x_i)/n.
# Variance MLE = sum((x_i - mean)^2)/n.
#
# The MLE variance uses divisor n, not n-1.


def gaussian_mle(data=None):
    """Return Gaussian mean and variance MLEs."""
    if data is None:
        data = np.array([1, 3, 5, 7])

    data = np.asarray(data, dtype=float)

    if data.ndim != 1 or len(data) < 2:
        raise ValueError("data must be a 1D sample of length >= 2")
    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite values")

    mean = np.mean(data)
    variance = np.mean((data - mean) ** 2)

    if variance == 0:
        raise ValueError("data must have nonzero variance")

    return mean, variance


# Q5: Gaussian MAP
#
# A Gaussian observation and Gaussian prior yield a Gaussian posterior.
# Its mode is its precision-weighted mean:
#
# MAP = (x/observation_variance + prior_mean/prior_variance)
#       / (1/observation_variance + 1/prior_variance).


def gaussian_map(
    x=5.,
    observation_variance=4.,
    prior_mean=0.,
    prior_variance=1.
):
    """Return the posterior mode for a Gaussian mean."""
    if not np.all(np.isfinite([
        x, observation_variance, prior_mean, prior_variance
    ])):
        raise ValueError("Inputs must be finite")
    if observation_variance <= 0 or prior_variance <= 0:
        raise ValueError("Variances must be positive")

    precision = 1 / observation_variance + 1 / prior_variance
    weighted_mean = (
        x / observation_variance + prior_mean / prior_variance
    )

    return weighted_mean / precision


# Q6: Prove Var(X) = E[X**2] - E[X]**2.
#
# By definition:
# Var(X) = E[(X - E[X])**2].
#
# Expanding:
# Var(X) = E[X**2 - 2X E[X] + E[X]**2].
#
# By linearity of expectation:
# Var(X) = E[X**2] - 2(E[X])**2 + (E[X])**2.
#
# Therefore:
# Var(X) = E[X**2] - (E[X])**2.


# ============================================================
# PART 3 (30 points): PCA
# ============================================================

# Q1: Derive Var(u.T @ X) = u.T @ Sigma @ u.
#
# Let X be a random vector and mu = E[X].
#
# Define the covariance matrix:
# Sigma = E[(X-mu)(X-mu).T].
#
# Let Y = u.T @ X be the projection onto direction u.
#
# E[Y] = E[u.T @ X] = u.T @ mu.
#
# By the definition of variance:
# Var(Y) = E[(Y-E[Y])**2].
#
# Substitute Y:
# Var(u.T @ X) = E[(u.T @ (X-mu))**2].
#
# Express the square as a matrix product:
# Var(u.T @ X)
# = E[u.T @ (X-mu) @ (X-mu).T @ u].
#
# Since u is constant, move it outside the expectation:
# Var(u.T @ X)
# = u.T @ E[(X-mu) @ (X-mu).T] @ u.
#
# Using the covariance definition:
# Var(u.T @ X) = u.T @ Sigma @ u.
#
# The unit-vector condition u.T @ u = 1 specifies a unit
# direction, but the identity also holds for non-unit vectors.


# Q2: Generate correlated Gaussian data
#
# The two-dimensional Gaussian has mean [0, 0] and covariance:
#
# Sigma = [[1.0, 0.8],
#          [0.8, 1.5]]
#
# The diagonal entries are the feature variances.
# The off-diagonal entries describe covariance between features.
# The positive covariance indicates positive correlation.
# A scatter plot should show an upward-sloping point cloud.
#
# The sample correlation is computed from the generated observations
# and may differ slightly from the population correlation.


def generate_data(n_samples=500, seed=42, mean=None, covariance=None):
    """Return reproducible Gaussian samples of shape (n_samples, 2)."""
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("n_samples must be an integer >= 2")

    if mean is None:
        mean = np.array([0., 0.])
    if covariance is None:
        covariance = np.array([
            [1.0, 0.8],
            [0.8, 1.5]
        ])

    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    if mean.shape != (2,):
        raise ValueError("mean must have shape (2,)")
    if covariance.shape != (2, 2):
        raise ValueError("covariance must have shape (2, 2)")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError("Inputs must be finite")
    if not np.allclose(covariance, covariance.T):
        raise ValueError("covariance must be symmetric")
    if np.min(np.linalg.eigvalsh(covariance)) <= 0:
        raise ValueError("covariance must be positive definite")

    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=n_samples)


# Q3: PCA from scratch
#
# Step 1: Center each feature:
# X_centered = X - mean(X).
#
# Step 2: Compute sample covariance:
# Sigma = X_centered.T @ X_centered / (n-1).
#
# This estimates the covariance matrix defined in Q1.
#
# Step 3: Compute eigenpairs:
# Sigma @ u_i = lambda_i * u_i.
#
# For each unit eigenvector u_i:
# Var(u_i.T @ X) = u_i.T @ Sigma @ u_i = lambda_i.
#
# Therefore, the eigenvalue gives variance along that component.
#
# Step 4: Sort eigenvalues in descending order.
# PC1 explains the most variance; PC2 explains the remaining
# variance in the orthogonal direction.
#
# Step 5: Project the observations:
# scores = X_centered @ components.
#
# The columns of components are orthonormal eigenvectors.
# The implementation also handles rank-deficient and tied cases.


def pca(X):
    """Fit PCA to a finite (n, 2) array without modifying X."""
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] < 2:
        raise ValueError("X must have shape (n, 2), with n >= 2")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")

    mean = np.mean(X, axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (X.shape[0] - 1)

    eigenvalues, components = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    components = components[:, order]

    scores = centered @ components

    return {
        "mean": mean,
        "centered": centered,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "components": components,
        "scores": scores
    }


# Q4: Explained variance and dimensionality reduction
#
# ratio_i = lambda_i / sum(lambda_j).
#
# The cumulative explained variance is the sum of the fractions
# for all components retained so far.
#
# Keeping only PC1 reduces two features to one dimension and saves
# storage and computation, but discards variance captured by PC2.
# Keeping both components retains all variance but does not reduce
# dimensionality. The best trade-off depends on the task.


def explained_variance(eigenvalues):
    """Return explained variance fractions for nonnegative eigenvalues."""
    values = np.asarray(eigenvalues, dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("eigenvalues must be a nonempty 1D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues must be finite")
    if np.any(values < 0) or np.sum(values) <= 0:
        raise ValueError("Eigenvalues must be nonnegative with positive sum")

    return values / np.sum(values)


# Q2/Q3/Q4: Figures
#
# Figure 1 shows the original observations and principal directions.
# Arrows begin at the sample mean and point along eigenvectors.
# Their lengths are proportional to the square roots of eigenvalues.
#
# Figure 2 shows individual and cumulative explained variance.
# make_plots returns figures without showing or saving them.


def make_plots(X, result):
    """Return Matplotlib figures without displaying or saving them."""
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=float)
    mean = result["mean"]
    eigenvalues = result["eigenvalues"]
    components = result["components"]
    fractions = explained_variance(eigenvalues)

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Data points")

    colors = ["red", "green"]

    for i in range(2):
        direction = components[:, i] * 2 * np.sqrt(eigenvalues[i])

        ax1.arrow(
            mean[0], mean[1],
            direction[0], direction[1],
            head_width=0.08,
            length_includes_head=True,
            color=colors[i]
        )
        ax1.plot([], [], color=colors[i], label=f"PC{i + 1}")

    ax1.scatter(
        mean[0], mean[1],
        color="black", marker="x", label="Mean"
    )
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.set_title("Original Data and Principal Component Directions")
    ax1.grid(True)
    ax1.axis("equal")
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    component_numbers = np.arange(1, len(eigenvalues) + 1)

    ax2.bar(
        component_numbers, fractions,
        alpha=0.7, label="Individual explained variance"
    )
    ax2.plot(
        component_numbers,
        np.cumsum(fractions),
        marker="o",
        label="Cumulative explained variance"
    )
    ax2.set_xticks(component_numbers)
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Proportion of Variance")
    ax2.set_title("Explained Variance by Principal Components")
    ax2.grid(True, axis="y")
    ax2.legend()

    return fig1, fig2


# REFLECTION
#
# I implemented matrix operations, eigenvalue calculations, probability
# formulas, MLE/MAP estimates, and PCA using NumPy.
# I used Matplotlib to visualize the Gaussian observations, principal
# component directions, and explained variance.
#
# Before submitting, update this reflection to accurately describe
# your own contributions and the external resources you actually used.


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Part 1 examples
    print("Matrix operations:")
    for value in matrix_operations():
        print(value)

    print("\nEigensystem:")
    print(eigensystem())

    print("\nMatrix inverse:")
    print(matrix_inverse())

    # Part 2 examples
    print("\nDistribution moments:", distribution_moments())
    print("Event probabilities:", event_probabilities())
    print("Coin estimates:", coin_estimates())
    print("Gaussian MLE:", gaussian_mle())
    print("Gaussian MAP:", gaussian_map())

    # Part 3: Generate data and inspect correlation
    X = generate_data()
    print("\nSample correlation matrix:")
    print(np.corrcoef(X.T))

    # Fit PCA
    result = pca(X)

    print("\nSample mean:", result["mean"])
    print("Covariance matrix:\n", result["covariance"])
    print("Eigenvalues:", result["eigenvalues"])
    print("Principal component vectors:\n", result["components"])

    ratios = explained_variance(result["eigenvalues"])
    cumulative = np.cumsum(ratios)

    print("\nExplained variance:")
    for i, (value, ratio, total) in enumerate(
        zip(result["eigenvalues"], ratios, cumulative),
        start=1
    ):
        print(
            f"PC{i}: eigenvalue={value:.4f}, "
            f"explained variance={ratio:.2%}, "
            f"cumulative variance={total:.2%}"
        )

    # Reduce the data to one dimension using PC1.
    X_reduced = result["scores"][:, :1]
    print("\nOriginal data shape:", X.shape)
    print("Reduced data shape:", X_reduced.shape)
    print("Variance retained:", f"{ratios[0]:.2%}")

    # Save and display figures only inside the main guard.
    figures = make_plots(X, result)

    for i, figure in enumerate(figures, start=1):
        figure.savefig(f"assignment1_figure{i}.png", dpi=150)

    plt.show()
```
