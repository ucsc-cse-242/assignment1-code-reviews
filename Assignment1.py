"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# YOUR ANSWER HERE
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    A = np.array([[3., 4.], [2., 1.]]) if A is None else np.array(A, dtype=float)
    B = np.array([[1., 2.], [0., -1.]]) if B is None else np.array(B, dtype=float)
    AB = A @ B
    return AB, B @ A, (A + B).T, A @ B.T, float(np.trace(A)), float(np.trace(AB))


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE
    M = np.array([[2., -1.], [-1., 2.]]) if M is None else np.array(M, dtype=float)
    values, vectors = np.linalg.eigh(M)
    return values, vectors


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    C = np.array([[4., 7.], [2., 6.]]) if C is None else np.array(C, dtype=float)
    return np.linalg.solve(C, np.eye(C.shape[0]))


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    rate = float(rate)
    return float(np.exp(-rate)), rate, rate


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    p_ab = p_a_given_b * p_b
    return float(p_ab), float(p_ab / p_a), float(p_a + p_b - p_ab)


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    h, n = int(heads), int(tosses)
    return h / n, (h + 1) / (n + 1)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    x = np.array([1., 3., 5., 7.]) if data is None else np.array(data, dtype=float).ravel()
    mu = x.mean()
    return float(mu), float(np.mean((x - mu) ** 2))


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    obs_prec = 1.0 / observation_variance
    prior_prec = 1.0 / prior_variance
    return float((x * obs_prec + prior_mean * prior_prec) / (obs_prec + prior_prec))


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE

# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    mean = np.array([2., -1.]) if mean is None else np.array(mean, dtype=float)
    covariance = (np.array([[3., 1.2], [1.2, 1.]]) if covariance is None
                  else np.array(covariance, dtype=float))
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=int(n_samples))


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # YOUR CODE HERE
    X = np.array(X, dtype=float)
    n = X.shape[0]
    mean = X.mean(axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (n - 1)
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    components = eigenvectors[:, order]
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
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    ev = np.array(eigenvalues, dtype=float)
    return ev / ev.sum()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=float)
    mean = np.asarray(result["mean"])
    comps = np.asarray(result["components"])
    evals = np.asarray(result["eigenvalues"])
    fracs = explained_variance(evals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    ax1.scatter(X[:, 0], X[:, 1], s=10, alpha=0.4, color="tab:blue", label="samples")
    ax1.scatter(*mean, color="black", marker="x", s=80, zorder=5, label="sample mean")
    colors = ["tab:red", "tab:orange"]
    for k in range(comps.shape[1]):
        vec = comps[:, k] * 2 * np.sqrt(max(evals[k], 0.0))
        ax1.annotate("", xy=mean + vec, xytext=mean - vec,
                     arrowprops=dict(arrowstyle="<->", color=colors[k % 2], lw=2.5))
        ax1.plot([], [], color=colors[k % 2], lw=2.5,
                 label=f"PC{k + 1} (lambda={evals[k]:.2f}, +/-2 sqrt(lambda))")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_title("Samples and principal directions")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    idx = np.arange(1, len(fracs) + 1)
    bars = ax2.bar(idx, fracs, color=colors[:len(fracs)], alpha=0.8, label="per component")
    for b, f in zip(bars, fracs):
        ax2.text(b.get_x() + b.get_width() / 2, f + 0.02, f"{f:.1%}", ha="center")
    ax2.plot(idx, np.cumsum(fracs), "ko-", label="cumulative")
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f"PC{i}" for i in idx])
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel("Principal component")
    ax2.set_ylabel("Fraction of variance explained")
    ax2.set_title("Explained variance")
    ax2.legend(loc="center right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    import matplotlib.pyplot as plt
    X = generate_data()
    result = pca(X)
    fig = make_plots(X, result)
    fig.savefig("pca_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
