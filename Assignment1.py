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
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    # raise NotImplementedError()
    return (
        A@B,
        B@A,
        (A+B).T,
        A@B.T,
        np.trace(A),
        np.trace(A@B),
    )


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE
    # raise NotImplementedError()
    val, vec = np.linalg.eig(M)
    for i in range(len(val)):
        # print(val[i], vec[:, i])
        # print(M @ vec[:, i])
        # print(val[i] * vec[:, i])
        assert (M @ vec[:, i] == val[i] * vec[:, i]).all
    return val, vec


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    # raise NotImplementedError()
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    raise NotImplementedError()


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


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
    # raise NotImplementedError()
    # raise ValueError("mean and covariance are not specified.")
    if mean is None:
        mean = [2, -1]
    mean = np.asarray(mean, dtype=float).copy()
    if mean.shape != (2,):
        raise ValueError("mean must have shape (2,).")
    if not np.isfinite(mean).all():
        raise ValueError("mean must contain only finite values.")

    if covariance is None:
        covariance = [[3, 1.2], [1.2, 1]]
    covariance = np.asarray(covariance, dtype=float).copy()
    if covariance.shape != (2, 2):
        raise ValueError("covariance must have shape (2,2).")
    if not np.isfinite(covariance).all():
        raise ValueError("covariance must contain only finite values.")

    # Main
    rng = np.random.default_rng(seed)
    data = rng.multivariate_normal(
        mean=mean,
        cov=covariance,
        size=n_samples
    )

    # Visualize
    plt.scatter(
        data[:, 0],
        data[:, 1],
        alpha=0.5
    )
    plt.title("Synthetic 2-dimensional correlated Gaussian features")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.axis("equal")
    print("sample correlation", np.corrcoef(data[:, 0], data[:, 1])[0, 1])

    return data

# data = generate_data(mean=[2,-1], covariance=[[3,1.2],[1.2,1]])


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
    # raise NotImplementedError()
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Illegal dimension of X: {X.ndim}.")
    if X.shape[0] < 2:
        raise ValueError("X must have at least 2 samples.")
    if X.shape[1] != 2:
        raise ValueError("X must have 2 features.")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values.")

    # Center the dataset
    # Links to X - E(X) in Q1, using the sample mean
    mean = np.mean(X, axis=0)
    centered = X - mean

    # Compute the sample covariance matrix
    # Links to \Sigma = E[(X-E(X)) (X-E(X))^T] in Q1, using the sample estimate
    covariance = (
        centered.T @ centered
        / (X.shape[0] - 1)
    )

    # Find eigenvalues and eigenvectors of the covariance matrix
    # (rank-deficient and tied cases are handled.)
    # Links to Var(u^T X) = u^T \Sigma u in Q1 and the extension in Q3 (see PDF);
    # finds orthonormal vectors u that successively maximize u^T S u
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Select the principal components corresponding to the highest eigenvalues
    # (Reordering here because np.linalg.eigh returns ascending order)
    reorder = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[reorder]
    components = eigenvectors[:, reorder]

    # Project centered data onto principal components
    # j-th column of scores is the projection onto j-th principal component direction
    # Links to the projection u^T (X-E(X)) in Q1
    scores = centered @ components

    # Visualize
    plt.scatter(X[:, 0], X[:, 1], s=20, alpha=0.5)
    plt.title("PCA on synthetic 2-dimensional correlated Gaussian features")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.axis("equal")
    for j, l in enumerate(eigenvalues):
        direction = components[:, j]
        start = mean
        end = mean - np.sqrt(l) * direction
        plt.plot([start[0], end[0]], [start[1], end[1]],
                 label=f"PC {j+1}", lw=2.5, color=('red', 'lime')[j])
    plt.legend()

    return {
        "mean": mean,
        "centered": centered,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "components": components,
        "scores": scores,
    }

# result = pca(data)


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    # raise NotImplementedError()
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if (eigenvalues < 0).any() or eigenvalues.sum() <= 0:
        raise ValueError("eigenvalues must be nonnegative with positive sum.")
    return eigenvalues / eigenvalues.sum()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    # raise NotImplementedError()

    # Q4 plot
    _explained_variance = np.concatenate((
        np.array([0]),
        explained_variance(result['eigenvalues'])))
    x = np.arange(len(_explained_variance))
    reduced_dim = 1 - x / (len(x) - 1)

    plt.title("trade-off between explained variance and dimensionality reduction")
    plt.xlabel("Number of components")
    plt.xticks(x)

    plt.plot(x, reduced_dim, "s--", color="tab:blue", label="reduced dimension")
    plt.ylabel("Proportion of dimensions reduced", color="tab:blue")
    plt.tick_params(axis="y", labelcolor="tab:blue")

    plt.twinx()
    plt.plot(x, np.cumsum(_explained_variance), "o-", color="tab:orange", label="explained variance")
    plt.ylabel("Cumulative explained variance", color="tab:orange")
    plt.tick_params(axis="y", labelcolor="tab:orange")

    # Q2/Q3's visualization codes are embeded in the corresponding functions for debugging

# make_plots(data, result)


# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    pass
