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
# On the PDF is the answer to the sample matrixes
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    if A is None:
        A = [[3, 4], [2, 1]]
    if B is None:
        B = [[1,2],[0,-1]]

    A = np.array(A)
    B = np.array(B)

    return A @ B, B @ A, A @ B.T, np.trace(A), np.trace(B)


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
# Detailed solution on PDF
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE
    if M is None:
        M = [[2,-1],[-1,2]]

    M = np.array(M)
    values, vectors = np.linalg.eig(M)

    return values, vectors

# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE
# Solution on PDF


# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
# Detailed solution on PDF
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    if C is None:
        C = [[4, 7], [2, 6]]

    C = np.array(C)
    return np.linalg.inv(C)

# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
# On PDF
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    # On PDF
    raise NotImplementedError()


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
# Detailed solution on PDF
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    p_a_intersect_b = p_a_given_b * p_b
    p_b_given_a = p_a_intersect_b / p_a
    p_a_or_b = p_a + p_b - p_a_intersect_b
    
    return p_a_intersect_b, p_b_given_a, p_a_or_b


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
# on pdf
# MLE maximizes likelihood given occurances
# MAP maximum a posteriori
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    # from pdf using output relative to input example
    mle = heads/tosses
    mapa = (heads+1) / (tosses+1)

    return mle, mapa


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
# All on PDF
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
# All on PDF
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE
# On PDF

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE
# On PDF

# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE

def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    # defaults 
    if mean is None:
        mean = [2, -1]

    if covariance is None:
        covariance = [[3,1.2],[1.2,1]]

    # make sure they're np arrays
    mean = np.array(mean)
    covariance = np.array(covariance)

    # random generaroe
    rng = np.random.default_rng(seed)

    X = rng.multivariate_normal(mean=mean, cov=covariance, size=n_samples)

    return X


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
    X = np.asarray(X, dtype=float)

    # Xcentered = X - mean
    mean = np.mean(X, axis=0)
    centered = X - mean

    # Calculate covariance matrix
    n = X.shape[0]
    covariance = (centered.T @ centered) / (n - 1)

    # find and sort eigenvals and vctors from largest to smallest
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order]

    # Project the centered data onto the PC directions.
    scores = centered @ components

    return {
        "mean": mean,
        "centered": centered,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "components": components,
        "scores": scores
    }

# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    # goal: eigenvals / total variance

    # eigenvals
    eigenvalues = np.asanyarray(eigenvalues, dtype=float)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # variance sum
    total_variance = np.sum(eigenvalues)

    if total_variance == 0:
        return np.zeros_like(eigenvalues)

    return eigenvalues / total_variance


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
# explained on PDF
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    mean = result["mean"]
    components = result["components"]
    eigenvalues = result["eigenvalues"]

    ratios = explained_variance(eigenvalues)

    # 2 plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Data and principal component directions
    ax = axes[0]

    ax.scatter(X[:, 0], X[:, 1], alpha=0.4)

    for i in range(2):
        direction = components[:, i]
        length = 2 * np.sqrt(max(eigenvalues[i], 0))

        ax.arrow(
            mean[0],
            mean[1],
            direction[0] * length,
            direction[1] * length,
            width=0.02,
            head_width=0.15,
            length_includes_head=True
        )

        endpoint = mean + direction * length

        ax.text(
            endpoint[0],
            endpoint[1],
            f"PC{i + 1}"
        )

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title("PCA Principal Components")
    ax.axis("equal")

    # Plot 2: Explained variance
    ax = axes[1]

    labels = [f"PC{i + 1}" for i in range(len(ratios))]

    ax.bar(labels, ratios)

    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Explained Variance by Component")
    ax.set_ylim(0, 1)

    for i, ratio in enumerate(ratios):
        ax.text(
            i,
            ratio + 0.02,
            f"{ratio:.1%}",
            ha="center"
        )

    fig.tight_layout()

    return fig

# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE
"""
Overall, I learned a lot from this assignment. Even though, the topics were dense.
I used primarily ChatGPT to assist me with solving these questions on the PDF and code.
The transcript is attatched. 
"""

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    X = generate_data()
    result = pca(X)

    ratios = explained_variance(result["eigenvalues"])

    print("Eigenvalues:", result["eigenvalues"])
    print("Explained variance:", ratios)
    print("Total:", np.sum(ratios))

    fig = make_plots(X, result)
    fig.savefig("pca_results.png", dpi=200)

    plt.show()