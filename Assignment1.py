"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""

import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# YOUR ANSWER HERE
# we use numpy's @ for the matrix products, .T for transposes, and np.trace for traces
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    if A is None:
        A = [[3, 4], [2, 1]]
    A = np.asarray(A, dtype=float)
    if B is None:
        B = [[1, 2], [0, -1]]
    B = np.asarray(B, dtype=float)

    ab_product = A @ B
    ba_product = B @ A
    aplusb_transpose = (A + B).T
    ab_product_transpose = A @ B.T
    trace_a = np.trace(A)
    trace_ab = np.trace(ab_product)
    return (
        ab_product,
        ba_product,
        aplusb_transpose,
        ab_product_transpose,
        trace_a,
        trace_ab,
    )


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
# eigenvalues solve det(M - lambda*I) = 0; M is symmetric so we use np.linalg.eigh
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE
    if M is None:
        M = [[2, -1], [-1, 2]]
    M = np.asarray(M, dtype=float)
    return np.linalg.eigh(M)


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE
# See HW PDF for the full proof.


# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
# C is invertible since its determinant is nonzero, so we use numpy to compute the inverse
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    if C is None:
        C = [[4, 7], [2, 6]]
    C = np.asarray(C, dtype=float)
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
# sum 3^k/k! = e^3, so c = e^-3 (Poisson with rate 3), so E[X] = 3 and Var(X) = 3
def distribution_moments(rate=3.0):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    mean = rate
    variance = rate
    c = np.exp(-rate)
    return (c, mean, variance)


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
# P(A and B) = P(A|B)P(B) = 0.18, P(B|A) = 0.18/0.5 = 0.36
# P(A or B) = 0.5 + 0.3 - 0.18 = 0.62
def event_probabilities(p_a=0.5, p_b=0.3, p_a_given_b=0.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    p_ab = p_a_given_b * p_b
    p_b_given_a = p_ab / p_a
    union = p_a + p_b - p_ab
    return (p_ab, p_b_given_a, union)


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
# likelihood = theta^3(1-theta)^2 -> MLE = 3/5
# posterior is proportional to theta^4(1-theta)^2 -> MAP = 4/6 = 2/3
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    MLE = heads / tosses
    MAP = (heads + 1) / (tosses + 1)
    return (MLE, MAP)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
# mu = (1+3+5+7)/4 = 4, sigma^2 = (9+1+1+9)/4 = 5 (MLE divides by n)
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    if data is None:
        data = [1, 3, 5, 7]
    mean = np.mean(data, dtype=float)
    variance = np.var(data, dtype=float)
    return (mean, variance)


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
# MAP = (x/4 + 0/1) / (1/4 + 1/1) = 1.25/1.25 = 1.0
def gaussian_map(x=5.0, observation_variance=4.0, prior_mean=0.0, prior_variance=1.0):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    return (x / observation_variance + prior_mean / prior_variance) / (
        1 / observation_variance + 1 / prior_variance
    )


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE
# See HW PDF for the full proof.

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE
# See HW PDF for the full proof.


# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
# samples from a 2D Gaussian with mean [2,-1] and covariance [[3,1.2],[1.2,1]]
# positive correlation (1.2/sqrt(3*1) = 0.69), so the points form a tilted ellipse


def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    if mean is None:
        mean = [2, -1]
    if covariance is None:
        covariance = [[3, 1.2], [1.2, 1]]
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
# we center the data, take covariance, then eigendecompose it
# PC1 is the top eigenvector, which maximizes u.T@Sigma@u from Q1, and lies along the
# long axis of the ellipse; PC2 is perpendicular to it
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # YOUR CODE HERE
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    X_centered = X - mean

    N = X_centered.shape[0]
    cov = (X_centered.T @ X_centered) / (N - 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0, None)
    eigvecs = eigvecs[:, order]

    scores = X_centered @ eigvecs
    return {
        "mean": mean,
        "centered": X_centered,
        "covariance": cov,
        "eigenvalues": eigvals,
        "components": eigvecs,
        "scores": scores,
    }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
# PC1 explains 89.4% of the variance, PC2 10.6%
# keeping only PC1 reduces 2D to 1D and keeps ~89% of the variance, losing the 10.6% along PC2
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    ev = np.asarray(eigenvalues, dtype=float)
    return ev / ev.sum()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
# left: data with each PC drawn from the mean, scaled by its std dev
# right: explained variance per PC, showing PC1 dominates
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    ax1.scatter(X[:, 0], X[:, 1], alpha=0.5)
    mean = result["mean"]
    eigvals = result["eigenvalues"]
    eigvecs = result["components"]
    for i in range(2):
        vec = eigvecs[:, i] * np.sqrt(eigvals[i])
        ax1.arrow(
            mean[0],
            mean[1],
            vec[0],
            vec[1],
            color=["red", "green"][i],
            width=0.05,
            length_includes_head=True,
            label=f"PC{i + 1}",
        )
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.set_title("Data with principal component directions")
    ax1.legend()
    ax1.axis("equal")

    explained = explained_variance(eigvals)
    ax2.bar(["PC1", "PC2"], explained, color=["red", "green"])
    ax2.set_xlabel("Principal component")
    ax2.set_ylabel("Fraction of total variance")
    ax2.set_title("Explained variance")
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE
# Tasks completed: all of Parts 1-3 (derivations in HW PDF, code in this file)

if __name__ == "__main__":
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    X = generate_data()
    result = pca(X)
    print("Explained variance:", explained_variance(result["eigenvalues"]))
    fig = make_plots(X, result)
    fig.savefig("pca.png")
