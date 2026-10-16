"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# YOUR ANSWER HERE
# A = [[3, 4], [2, 1]] and B = [[1, 2], [0, -1]].
# AB = [[3*1+4*0, 3*2+4*(-1)], [2*1+1*0, 2*2+1*(-1)]] = [[3, 2], [2, 3]].
# BA = [[1*3+2*2, 1*4+2*1], [0*3+(-1)*2, 0*4+(-1)*1]] = [[7, 6], [-2, -1]].
# A+B = [[4, 6], [2, 0]], so (A+B)^T = [[4, 2], [6, 0]].
# B^T = [[1, 0], [2, -1]], so
# AB^T = [[3*1+4*2, 3*0+4*(-1)], [2*1+1*2, 2*0+1*(-1)]] = [[11, -4], [4, -1]].
# trace(A) = 3+1 = 4. trace(AB) = 3+3 = 6.
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
    ab = A @ B
    return ab, B @ A, (A + B).T, A @ B.T, np.trace(A), np.trace(ab)


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
# M = [[2, -1], [-1, 2]].
# det(M - λI) = (2-λ)^2 - 1 = λ^2 - 4λ + 3 = (λ-1)(λ-3), so λ = 1 or λ = 3.
# For λ = 1, (M-I)v = [[1, -1], [-1, 1]] v = 0, so v is a multiple of [1, 1].
# M[1, 1]^T = [1, 1]^T = 1*[1, 1]^T.
# For λ = 3, (M-3I)v = [[-1, -1], [-1, -1]] v = 0, so v is a multiple of [1, -1].
# M[1, -1]^T = [3, -3]^T = 3*[1, -1]^T.
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
# Let A be m by n and B be n by m. The diagonal entries of AB are
# (AB)_ii = Σ_k A_ik B_ki, so trace(AB) = Σ_i Σ_k A_ik B_ki.
# The sums are finite, so they can be swapped:
# Σ_i Σ_k A_ik B_ki = Σ_k Σ_i B_ki A_ik = Σ_k (BA)_kk = trace(BA).

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
# C = [[4, 7], [2, 6]]. det(C) = 4*6 - 7*2 = 24 - 14 = 10, which is not zero,
# so C is invertible. The cofactor matrix is [[6, -2], [-7, 4]], and its
# transpose (the adjugate) is [[6, -7], [-2, 4]]. Therefore
# C^{-1} = (1/det(C)) adj(C) = (1/10) [[6, -7], [-2, 4]].
# Multiplying gives C C^{-1} = I.
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
# P(X=k) = c * rate^k / k! for k = 0, 1, 2, ...
# Σ_k rate^k / k! = exp(rate), so the probabilities sum to one only when
# c * exp(rate) = 1, hence c = exp(-rate).
# This is the Poisson(rate) distribution, whose mean and variance are both rate:
# E[X] = rate and Var(X) = rate.
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    rate = float(rate)
    return float(np.exp(-rate)), rate, rate


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
# (a) P(A ∩ B) = P(A|B) P(B) = 0.6 * 0.3 = 0.18.
# (b) P(B|A) = P(A ∩ B) / P(A) = 0.18 / 0.5 = 0.36.
# (c) P(A ∪ B) = P(A) + P(B) - P(A ∩ B) = 0.5 + 0.3 - 0.18 = 0.62.
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    intersection = p_a_given_b * p_b
    conditional = intersection / p_a
    union = p_a + p_b - intersection
    return intersection, conditional, union


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
# The sequence H, H, T, H, T has h = 3 heads in n = 5 independent tosses.
# Likelihood: L(θ) = θ^h (1-θ)^{n-h}.
# Log-likelihood: l(θ) = h log θ + (n-h) log(1-θ).
# l'(θ) = h/θ - (n-h)/(1-θ) = 0 gives θ_MLE = h/n = 3/5 = 0.6.
# When 0 < h < n, L(0) = L(1) = 0, so this interior critical point is the maximum.
# If h = 0 or h = n the same formula returns the boundary maxima 0 and 1.
# Prior: p(θ) = 2θ on [0, 1]. The posterior density satisfies
# p(θ | data) ∝ 2θ * θ^h (1-θ)^{n-h} = 2 θ^{h+1} (1-θ)^{n-h}.
# Setting the derivative of the log posterior to zero,
# (h+1)/θ - (n-h)/(1-θ) = 0, gives θ_MAP = (h+1)/(n+1) = 4/6 = 2/3.
# If every toss is heads, n-h = 0 and the log-posterior derivative (h+1)/θ
# stays positive on (0, 1), so the maximum is the boundary θ = 1, which
# (h+1)/(n+1) also returns. If h = 0 the mode is the interior point 1/(n+1).
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    mle = heads / tosses
    map_estimate = (heads + 1) / (tosses + 1)
    return mle, map_estimate


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
# For x_1, ..., x_n i.i.d. N(μ, σ^2), the log-likelihood is
# l = -(n/2) log(2πσ^2) - (1/(2σ^2)) Σ_i (x_i - μ)^2.
# ∂l/∂μ = (1/σ^2) Σ_i (x_i - μ) = 0, so μ_MLE = (1/n) Σ_i x_i.
# For X = {1, 3, 5, 7}, μ_MLE = (1+3+5+7)/4 = 4.
# ∂l/∂σ^2 = -n/(2σ^2) + (1/(2(σ^2)^2)) Σ_i (x_i - μ)^2 = 0, so
# σ^2_MLE = (1/n) Σ_i (x_i - μ_MLE)^2.
# Squared deviations from 4 are 9, 1, 1, and 9, which sum to 20.
# σ^2_MLE = 20/4 = 5. The MLE divides by n, not by n-1.
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    if data is None:
        data = np.array([1.0, 3.0, 5.0, 7.0])
    else:
        data = np.array(data, dtype=float, copy=True).reshape(-1)
    mean = float(np.mean(data))
    variance = float(np.mean((data - mean) ** 2))
    return mean, variance


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
# One observation satisfies x | μ ~ N(μ, σ^2), and the prior is μ ~ N(μ0, τ^2).
# Up to a constant, the log posterior is
# -(μ - μ0)^2 / (2τ^2) - (x - μ)^2 / (2σ^2).
# Differentiating in μ and setting the derivative to zero gives
# -(μ - μ0)/τ^2 + (x - μ)/σ^2 = 0,
# so μ (1/τ^2 + 1/σ^2) = μ0/τ^2 + x/σ^2 and
# μ_MAP = (μ0/τ^2 + x/σ^2) / (1/τ^2 + 1/σ^2).
# With x = 5, σ^2 = 4, μ0 = 0, and τ^2 = 1,
# μ_MAP = (0/1 + 5/4) / (1 + 1/4) = 1.25 / 1.25 = 1.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    precision = 1.0 / prior_variance + 1.0 / observation_variance
    mode = (prior_mean / prior_variance + x / observation_variance) / precision
    return mode


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE
# By definition, Var(X) = E[(X - E[X])^2].
# Expand the square: (X - E[X])^2 = X^2 - 2 X E[X] + (E[X])^2.
# Take the expectation and use linearity:
# E[X^2 - 2 X E[X] + (E[X])^2] = E[X^2] - 2 E[X] E[X] + (E[X])^2
# = E[X^2] - 2 (E[X])^2 + (E[X])^2
# = E[X^2] - (E[X])^2.

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE
# Let μ = E[X]. The covariance matrix is Σ = E[(X - μ)(X - μ)^T],
# so the entry Σ_ij is Cov(X_i, X_j).
# For a unit vector u, the projected scalar is u^T X and E[u^T X] = u^T μ.
# Var(u^T X) = E[(u^T (X - μ))^2]
# = E[u^T (X - μ) (X - μ)^T u]
# = u^T E[(X - μ)(X - μ)^T] u
# = u^T Σ u.

# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
# The generator draws independent samples from a 2D Gaussian with
# mean [2, -1] and covariance [[3, 1.2], [1.2, 1]], using a seeded
# NumPy Generator so the same seed repeats the same sample.
# The covariance entry 1.2 is positive, so the two features move together.
# The scatter is an ellipse tilted with positive slope, stretched farther
# along the first feature because its variance, 3, is larger than 1.
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
# Subtracting the sample mean centers the cloud at the origin used in Q1.
# The sample covariance with divisor n-1 estimates Σ = E[(X-μ)(X-μ)^T].
# If Σu = λu and u is a unit vector, then u^T Σ u = λ, so each eigenvalue is
# the variance of the projection onto its eigenvector. Sorting eigenvalues
# descending puts the largest projected variance first. The plot draws those
# eigenvectors through the sample mean; the longer axis is the leading one.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # YOUR CODE HERE
    data = np.array(X, dtype=float, copy=True)
    mean = np.mean(data, axis=0)
    centered = data - mean
    n_samples = data.shape[0]
    covariance = (centered.T @ centered) / (n_samples - 1)
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
# The fraction λ_i / Σ_j λ_j is the share of total variance along principal
# component i, since λ_i = u_i^T Σ u_i. For this covariance the first
# component carries most of the variance and the second carries the rest.
# Keeping only the first component reduces the data from two coordinates to
# one and discards only the smaller fraction. The tradeoff is a shorter
# representation against the variance that projection throws away.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    eigenvalues = np.array(eigenvalues, dtype=float, copy=True)
    return eigenvalues / np.sum(eigenvalues)


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
# The left panel is the sample cloud with both principal directions drawn
# through the mean and scaled by the square root of the matching eigenvalue.
# The longer line is the direction of largest variance derived in Q1.
# The right panel shows the two explained-variance fractions. Their heights
# show how much variance a one-dimensional summary would keep or drop.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    import matplotlib.pyplot as plt

    samples = np.asarray(X)
    mean = np.asarray(result["mean"])
    components = np.asarray(result["components"])
    eigenvalues = np.asarray(result["eigenvalues"])
    fractions = explained_variance(eigenvalues)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(samples[:, 0], samples[:, 1], s=12, alpha=0.7)
    for index in range(components.shape[1]):
        direction = components[:, index] * np.sqrt(max(eigenvalues[index], 0.0))
        axes[0].plot(
            [mean[0] - direction[0], mean[0] + direction[0]],
            [mean[1] - direction[1], mean[1] + direction[1]],
            linewidth=2,
            label=f"PC{index + 1}",
        )
    axes[0].set_xlabel("feature 1")
    axes[0].set_ylabel("feature 2")
    axes[0].set_title("Data and principal directions")
    axes[0].legend()

    labels = [f"PC{index + 1}" for index in range(len(fractions))]
    axes[1].bar(labels, fractions)
    axes[1].set_xlabel("principal component")
    axes[1].set_ylabel("fraction of variance")
    axes[1].set_title("Explained variance")
    axes[1].set_ylim(0, 1)
    figure.tight_layout()
    return figure


# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE
# Completed every marked derivation and function in this file: matrix products,
# eigenpairs, the trace identity, the inverse, Poisson moments, event
# probabilities, coin MLE/MAP, Gaussian MLE/MAP, the variance identity, and
# PCA from centering through explained variance and the figures.
# External resources: the assignment README and the course lecture slides
# named there (Lecture1_intro and Lecture2_review), plus NumPy and Matplotlib
# documentation for linalg.eigh, Generator.multivariate_normal, and pyplot.

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    samples = generate_data()
    result = pca(samples)
    fractions = explained_variance(result["eigenvalues"])
    figure = make_plots(samples, result)
    figure.savefig("pca_figures.png")
    print("explained variance fractions:", fractions)
