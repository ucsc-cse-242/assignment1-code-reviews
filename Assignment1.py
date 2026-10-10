"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# YOUR ANSWER HERE
# See PDF
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    A = A if A is not None else [[3, 4], [2, 1]]
    B = B if B is not None else [[1, 2], [0, -1]]
    A = np.array(A)
    B = np.array(B)
    return (A @ B, B @ A, (A + B).T, A @ B.T, np.trace(A), np.trace(A @ B))


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
# See PDF
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE
    M = M if M is not None else [[2, -1], [-1, 2]]
    M = np.array(M)
    eigval, eigvec = np.linalg.eigh(M)
    return (eigval, eigvec)


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE
# See PDF

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
# See PDF
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    C = C if C is not None else [[4, 7], [2, 6]]
    C = np.array(C)
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
# See PDF
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    c = np.exp(-rate)
    return (c, rate, rate)


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
# See PDF
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    p_a_and_b = p_a_given_b * p_b
    return (p_a_and_b, p_a_and_b / p_a, p_a + p_b - p_a_and_b)


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
# See PDF
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    mle = heads / tosses
    map_1 = (heads + 1) / (tosses + 1)
    return (mle, map_1)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
# See PDF
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    data = data if data is not None else [1, 3, 5, 7]
    mle = sum(data) / len(data)
    var_mle = sum([(x - mle) ** 2 for x in data]) / len(data)
    return (mle, var_mle)


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
# See PDF
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    return (x * prior_variance + prior_mean * observation_variance) / (observation_variance + prior_variance)


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE
# See PDF

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE
# See PDF

# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
# See PDF
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    mean = mean if mean is not None else [2, -1]
    covariance = covariance if covariance is not None else [[3, 1.2], [1.2, 1]]
    mean = np.array(mean)
    covariance = np.array(covariance)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
# See PDF
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    n = X.shape[0]
    mean = X.mean(axis=0)
    centered = X - mean  # centering all the points
    covariance = centered.T @ centered / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance)  # finding values that maximize variance
    eigenvalues = eigenvalues[::-1].copy()  # sort in decending order
    components = eigenvectors[:, ::-1].copy()

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
# See PDF
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    return eigenvalues / eigenvalues.sum()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
# See PDF
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    ax1.scatter(X[:, 0], X[:, 1])
    ax1.set_title("Q2 Initial Dataset Visualization")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.axis("equal")

    ax2.scatter(X[:, 0], X[:, 1])
    ax2.set_title("Q3 Initial Dataset + Principal Component Directions")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")

    mean = result["mean"]

    for i in range(2):
        vec = result["components"][:, i] * np.sqrt(result["eigenvalues"][i])
        arrow_start = mean - vec
        arrow_stop = 2 * vec
        ax2.arrow(
            arrow_start[0],
            arrow_start[1],
            arrow_stop[0],
            arrow_stop[1],
            color=["red", "green"][i],
            width=0.05,
            length_includes_head=True,
            label=f"PC{i + 1}",
        )

    ax2.legend()
    ax2.axis("equal")

    explained = explained_variance(result["eigenvalues"])
    ax3.bar(["PC1", "PC2"], explained, color=["red", "green"])
    ax3.set_title("Q4 Explained Variance")
    ax3.set_xlabel("Principal Component")
    ax3.set_ylabel("Proportion of Total Variance")

    return fig

# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE
# All the tasks are completed. Numpy was used in various tasks.
# Used lectures along with numpy and matplotlib docs to lookup functions.


if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    import matplotlib.pyplot as plt

    init_funcs = [
        matrix_operations,
        eigensystem,
        matrix_inverse,
        distribution_moments,
        event_probabilities,
        coin_estimates,
        gaussian_mle,
        gaussian_map,
    ]
    for func in init_funcs:
        ans = func()
        print(ans)

    X = generate_data()
    result = pca(X)
    fig = make_plots(X, result)
    # fig.show()
    fig.savefig("plot_1.png")
