"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""

import numpy as np
import matplotlib.pyplot as plt


# P3 Q2: Describe the Gaussian generator and observed correlation.
# The plot shows a strong positive correlation between feature 1 and 2. While there's some scattering, a regression line would capture the positive trend quite well.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # check for None, and apply defaults
    if mean is None:
        mean = np.array([2, -1])
    if covariance is None:
        covariance = np.array(
            [
                [3, 1.2],
                [1.2, 1],
            ]
        )

    rng = np.random.default_rng(seed)

    # generate N-samples from 2d Gaussian
    X = rng.multivariate_normal(mean, covariance, size=n_samples)

    return X


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# The plot shows 2 orthogonal principal component directions. Both pass through the dataset's mean.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # get mean of each feature; center it
    mean = np.mean(X, axis=0)
    centered = X - mean

    # calc covariance matrix; get eigen
    covariance = (centered.T @ centered) / (len(X) - 1)
    ei_val, ei_vec = np.linalg.eigh(covariance)

    # sort eigenvalues, largest to smallest
    order = np.argsort(ei_val)[::-1]
    ei_val = ei_val[order]
    ei_vec = ei_vec[:, order]

    scores = centered @ ei_vec

    return {
        "mean": mean,
        "centered": centered,
        "covariance": covariance,
        "eigenvalues": ei_val,
        "components": ei_vec,
        "scores": scores,
    }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# PC1 explains approx 89%, while PC2 explains 11%. Keeping only PC1 reduces data from 2 to 1 dimension, while retaining most of its variance. However, the remaining 11% would be lost, losing some information.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    return eigenvalues / np.sum(eigenvalues)


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# The first plot shows the strongly positive correlation between Feature 1 and 2, and second plot shows proportion of total variance explained by each component.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # create 2 plots
    fig, (ax, ax_var) = plt.subplots(1, 2)

    ax.scatter(X[:, 0], X[:, 1], alpha=0.5)

    mean = result["mean"]
    ax.scatter(*mean, marker="x", label="mean")

    # plot principal components
    for i in range(2):
        direction = result["components"][:, i]  # ei_vec
        length = 2 * np.sqrt(max(result["eigenvalues"][i], 0))
        start = mean - length * direction
        end = mean + length * direction
        ax.plot([start[0], end[0]], [start[1], end[1]], label=f"PC{i + 1}")

    ax.set(xlabel="Feature 1", ylabel="Feature 2", title="PCA Directions")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()

    # ratio plot
    ratios = explained_variance(result["eigenvalues"])
    ax_var.bar(["PC1", "PC2"], ratios)
    ax_var.set(xlabel="Principal Component", ylabel="Variance Ratio", title="Explained Variance")
    ax_var.set_ylim(0, 1)

    fig.tight_layout()
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
# Implemented Gaussian data generation, PCA using numpy, and visualizations of principal components and explained variance.
# Used course lecture slides, numpy and matplotlib docs, as well as ChatGPT for conceptual explanations, coding guidance, and debugging.
# ChatGPT Log: https://chatgpt.com/share/6aca9acf-421c-83ea-b3ff-18b74bd65cfc

if __name__ == "__main__":
    X = generate_data()  # 2d Gaussian
    result = pca(X)

    fig = make_plots(X, result)

    plt.show()

    # This block does not run when the grader imports your functions.
    pass
