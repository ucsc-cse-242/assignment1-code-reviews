"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np
import matplotlib.pyplot as plt


# PART 1 (20 points), Q1: Show computations for the given A and B.
# YOUR ANSWER HERE
# A = [[3,4],[2,1]], B = [[1,2],[0,-1]]
# AB = [[3,2],[2,3]]
# BA = [[7,6],[-2,-1]]  (not the same as AB)
# (A+B)^T = [[4,2],[6,0]]
# A B^T = [[11,-4],[4,-1]]
# tr(A) = 4, tr(AB) = 6
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    if A is None:
        A = np.array([[3.0, 4.0], [2.0, 1.0]])
    else:
        A = np.array(A, dtype=float, copy=True)
    if B is None:
        B = np.array([[1.0, 2.0], [0.0, -1.0]])
    else:
        B = np.array(B, dtype=float, copy=True)
    AB = A @ B
    BA = B @ A
    ApB_T = (A + B).T
    ABT = A @ B.T
    return AB, BA, ApB_T, ABT, float(np.trace(A)), float(np.trace(AB))


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
# M = [[2,-1],[-1,2]]
# det(M-λI) = (2-λ)^2 - 1 = λ^2 - 4λ + 3 = (λ-3)(λ-1)
# so λ = 3 and 1
# λ=3: v = [1,-1],  M v = [3,-3] = 3v
# λ=1: v = [1,1],   M v = [1,1] = 1v
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE
    if M is None:
        M = np.array([[2.0, -1.0], [-1.0, 2.0]])
    else:
        M = np.array(M, dtype=float, copy=True)
    values, vectors = np.linalg.eigh(M)
    return values, vectors


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE
# tr(AB) = sum_i sum_k A_ik B_ki
# that's the same terms as tr(BA) = sum_k sum_i B_ki A_ik
# just added in a different order so they're equal
# (checked on Q1: both traces were 6)


# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
# C = [[4,7],[2,6]], det = 24-14 = 10 != 0 so invertible
# Cinv = (1/10) [[6,-7],[-2,4]] = [[0.6,-0.7],[-0.2,0.4]]
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    if C is None:
        C = np.array([[4.0, 7.0], [2.0, 6.0]])
    else:
        C = np.array(C, dtype=float, copy=True)
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
# sum c * 3^k / k! = c * e^3 = 1  =>  c = e^{-3}
# that's poisson(3) so mean and var are both 3
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    c = float(np.exp(-rate))
    return c, float(rate), float(rate)


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
# P(A and B) = P(A|B) P(B) = 0.6 * 0.3 = 0.18
# P(B|A) = 0.18 / 0.5 = 0.36
# P(A or B) = 0.5 + 0.3 - 0.18 = 0.62
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    p_intersect = p_a_given_b * p_b
    p_b_given_a = p_intersect / p_a
    p_union = p_a + p_b - p_intersect
    return float(p_intersect), float(p_b_given_a), float(p_union)


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
# 3 heads out of 5 flips
# L = θ^3 (1-θ)^2, set deriv to 0 => MLE = 3/5
# prior is 2θ so posterior ~ θ^4 (1-θ)^2 => MAP = 4/6 = 2/3
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    mle = heads / tosses
    mape = (heads + 1) / (tosses + 1)
    return float(mle), float(mape)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
# mean mle is just the sample mean: (1+3+5+7)/4 = 4
# var mle uses /n not /(n-1): (9+1+1+9)/4 = 5
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    if data is None:
        data = np.array([1.0, 3.0, 5.0, 7.0])
    else:
        data = np.array(data, dtype=float, copy=True).reshape(-1)
    mu = float(np.mean(data))
    var = float(np.mean((data - mu) ** 2))
    return mu, var


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
# x=5, σ^2=4, prior N(0,1)
# MAP = (5/4) / (1/4 + 1) = 1
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    like_prec = 1.0 / observation_variance
    prior_prec = 1.0 / prior_variance
    mu_map = (like_prec * x + prior_prec * prior_mean) / (like_prec + prior_prec)
    return float(mu_map)


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE
# expand (X-μ)^2 and the 2μX term cancels:
# E[(X-μ)^2] = E[X^2] - 2μ^2 + μ^2 = E[X^2] - μ^2


# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE
# Σ = E[(X-μ)(X-μ)^T]
# Var(u^T X) = E[(u^T (X-μ))^2] = u^T Σ u


# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
# made 2d gaussians with cov [[3,1.2],[1.2,1]]
# off diagonal is positive so they move together, cloud slants up-right
# ρ = 1.2/sqrt(3) ≈ 0.69
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    if mean is None:
        mean = np.array([2.0, -1.0])
    else:
        mean = np.array(mean, dtype=float, copy=True)
    if covariance is None:
        covariance = np.array([[3.0, 1.2], [1.2, 1.0]])
    else:
        covariance = np.array(covariance, dtype=float, copy=True)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
# subtract mean, get cov with n-1, eigendecompose
# biggest eigenvalue = PC1 (same u^T Σ u thing from Q1)
# arrows on the scatter go through the mean
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # YOUR CODE HERE
    X = np.array(X, dtype=float, copy=True)
    n = X.shape[0]
    mean = X.mean(axis=0)
    centered = X - mean
    covariance = (centered.T @ centered) / (n - 1)
    eigenvalues, components = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = components[:, order]
    scores = centered @ components
    return {
        "mean": mean,
        "centered": centered,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "components": components,
        "scores": scores,
    }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
# each pc gets λ_i / sum(λ)
# PC1 has most of it so dropping to 1d keeps the main shape
# you lose whatever is on PC2 though
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    eigenvalues = np.array(eigenvalues, dtype=float, copy=True)
    return eigenvalues / eigenvalues.sum()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
# left plot is the points + pc directions
# right plot is % variance on each pc
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    X = np.asarray(X, dtype=float)
    mean = np.asarray(result["mean"], dtype=float)
    components = np.asarray(result["components"], dtype=float)
    eigenvalues = np.asarray(result["eigenvalues"], dtype=float)
    fractions = explained_variance(eigenvalues)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=12, alpha=0.45, c="steelblue", label="samples")
    ax.scatter(mean[0], mean[1], c="black", s=40, zorder=3, label="mean")
    colors = ["crimson", "darkorange"]
    for i in range(components.shape[1]):
        direction = components[:, i]
        length = 2.5 * np.sqrt(max(eigenvalues[i], 0.0))
        ax.annotate(
            "",
            xy=mean + length * direction,
            xytext=mean - length * direction,
            arrowprops=dict(arrowstyle="<->", color=colors[i], lw=2),
        )
        tip = mean + length * direction
        ax.text(
            tip[0],
            tip[1],
            f"  PC{i + 1}",
            color=colors[i],
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_title("Synthetic Gaussian data and principal axes")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels = [f"PC{i + 1}" for i in range(len(fractions))]
    bars = ax.bar(labels, fractions, color=["crimson", "darkorange"][: len(fractions)])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("proportion of variance")
    ax.set_title("Variance explained by each principal component")
    for bar, frac, lam in zip(bars, fractions, eigenvalues):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{100.0 * frac:.1f}%\n(λ={lam:.3f})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE
# did the lin alg, probability/mle/map, and pca in numpy
# used the lectures and numpy docs a bit for multivariate_normal / eigh


if __name__ == "__main__":
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    X = generate_data()
    result = pca(X)
    fractions = explained_variance(result["eigenvalues"])
    print("PCA eigenvalues:", result["eigenvalues"])
    print("Explained variance fractions:", fractions)

    fig = make_plots(X, result)
    fig.savefig("pca_figures.png", dpi=150, bbox_inches="tight")
    plt.show()
