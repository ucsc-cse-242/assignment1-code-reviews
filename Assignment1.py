"""
CSE 242 Assignment 1 - Akhilesh Nidamanuri.
Work is showed on a separate handwritten PDF file.
"""
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PART 1 (20 points), Q1: Show computations for the given A and B.
# =============================================================================

def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    if A is None:
        A = np.array([[3, 4], [2, 1]], dtype=float)
    else:
        A = np.asarray(A, dtype=float)
    if B is None:
        B = np.array([[1, 2], [0, -1]], dtype=float)
    else:
        B = np.asarray(B, dtype=float)

    AB = A @ B
    BA = B @ A
    sumT = (A + B).T
    ABT = A @ B.T
    trace_A = np.trace(A)
    trace_AB = np.trace(AB)

    return AB, BA, sumT, ABT, trace_A, trace_AB


# =============================================================================
# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# =============================================================================

def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    if M is None:
        M = np.array([[2, -1], [-1, 2]], dtype=float)
    else:
        M = np.asarray(M, dtype=float).copy()

    values, vectors = np.linalg.eigh(M)
    values = values[::-1]
    vectors = vectors[:, ::-1]

    return values, vectors


# =============================================================================
# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# =============================================================================
# Paper submission

# =============================================================================
# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# =============================================================================
# Paper submission

def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    if C is None:
        C = np.array([[4, 7], [2, 6]], dtype=float)
    else:
        C = np.asarray(C, dtype=float)
    return np.linalg.inv(C)


# =============================================================================
# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# =============================================================================
# Paper submission

def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    c = np.exp(-rate)
    mean = rate
    variance = rate
    return c, mean, variance


# =============================================================================
# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# =============================================================================
# Paper submission

def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    p_intersect = p_a_given_b * p_b
    p_b_given_a = p_intersect / p_a
    p_union = p_a + p_b - p_intersect
    return p_intersect, p_b_given_a, p_union


# =============================================================================
# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# =============================================================================
# Paper submission

def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    mle = heads / tosses
    map_est = (heads + 1) / (tosses + 1)

    return float(mle), float(map_est)


# =============================================================================
# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# =============================================================================

def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    if data is None:
        data = np.array([1, 3, 5, 7], dtype=float)
    else:
        data = np.asarray(data, dtype=float)
    n = len(data)
    mu_hat = np.mean(data)
    # MLE variance divides by n (not n-1)
    sigma2_hat = np.sum((data - mu_hat) ** 2) / n
    return mu_hat, sigma2_hat


# =============================================================================
# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# =============================================================================

def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    precision_obs = 1.0 / observation_variance
    precision_prior = 1.0 / prior_variance
    mu_map = (x * precision_obs + prior_mean * precision_prior) / (precision_obs + precision_prior)
    return mu_map


# =============================================================================
# P3 Q2: Describe the Gaussian generator and observed correlation.
# =============================================================================

def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    if mean is None:
        mean = np.array([2., -1.])
    else:
        mean = np.asarray(mean, dtype=float).copy()
    if covariance is None:
        covariance = np.array([[3., 1.2], [1.2, 1.]])
    else:
        covariance = np.asarray(covariance, dtype=float).copy()

    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mean, covariance, size=n_samples)
    return samples


# =============================================================================
# P3 Q3: Link centering/covariance/eigendecomposition to the derivation.
# =============================================================================

def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    # Step 1: Center the data (mean subtraction)
    # The Q1 derivation is built on centered variables, so we remove
    # the mean. Now the data has zero mean, which lets the next step
    # estimate the covariance matrix correctly.
    mean_vec = np.mean(X, axis=0)
    centered = X - mean_vec

    # Step 2: Compute sample covariance matrix (divisor n-1)
    # This sample version of the covariance matrix is unbiased from Q1.
    # (i, j) entry is the covariance between feature i and feature j.
    # the diagonal holds the feature variances. We then divide by n-1 to get the sample covariance.
    cov_mat = (centered.T @ centered) / (n - 1)

    # Step 3: Eigendecomposition
    # The covariance matrix is symmetric so its eigenvectors are 
    # orthogonal and orthonormal. The eigenvector with the largest
    # eigenvalue is the direction of maximum variance from Q1. 
    # 
    # eigh returns the eigenvalues in ascending order, so we reverse them to get descending order.
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
    eigenvalues = eigenvalues[::-1]
    components = eigenvectors[:, ::-1]

    # Step 4: Project data (scores)
    # each score is how far a point lies along the principal component directions
    # The columns are uncorelated and have variance equal to the corresponding eigenvalue.
    scores = centered @ components

    result = {
        "mean": mean_vec,
        "centered": centered,
        "covariance": cov_mat,
        "eigenvalues": eigenvalues,
        "components": components,
        "scores": scores,
    }
    return result


# =============================================================================
# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# =============================================================================
# PC1 has the largest eigenvalue, so it always explains the largest fraction 
# of the total variance. PC2 explains the remaining variance.
#
# For the data, PC1 explains about 89% of the variance and PC2 about 11%. This
# reflects the strong positive correlation from Q2, where most of the spread
# lies on one diagonal direction. So the first direction captures most of the info.
#
# Dimensionality reduction means keeping only the top k components and dropping
# the rest. Here, keeping just PC1 reduces the data from 2D to 1D but it still
# retains ~89% of the variance.
#
# The main trade off is that keeping more components preserves more variance
# but keeps the data higher dimensional and complex. Keeping fewer components
# gives a simpler and more compact representation but loses the variance and 
# information of the dropped components. Choosing k is a decision about how much
# information loss is fine for the reduction in dimensionality we want. 
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    total = np.sum(eigenvalues)
    return eigenvalues / total


# =============================================================================
# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# =============================================================================
# Attached to the handwritten PDF.
# The plot has three subplots.
# 1. This shows the original raw data with the two principal directions drawn
#    from the mean. The arrow length scales with the variance in that direction.
#    PC1 runs along the length of the correlated cloud and PC2 sits perpendicular to it.
# 2. This shows the data in principle component space, where teh centering and 
#    rotating into the PC axes has removed the correlation and aligns the data with the axes.
# 3. The bar graph shows the fractions from Q4. Shows how PC1 captures roughly 89.39% 
#    of the spread and PC2 captures the remaining 10.61%. This is why a 1D projection
#    retains most of the information and is a good reduction in dimensionality.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    X = np.asarray(X, dtype=float)
    mean_vec = result["mean"]
    components = result["components"]
    eigenvalues = result["eigenvalues"]
    scores = result["scores"]
    ev_frac = result["eigenvalues"] / np.sum(result["eigenvalues"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Subplot 1: Original data with principal component directions ---
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, s=10, label="Samples")
    ax.scatter(mean_vec[0], mean_vec[1], color="red", s=80, marker="X",
               zorder=5, label="Mean")

    colors = ["blue", "orange"]
    for i in range(2):
        scale = np.sqrt(eigenvalues[i]) * 2
        v = components[:, i] * scale
        ax.arrow(mean_vec[0], mean_vec[1], v[0], v[1],
                 head_width=0.15, head_length=0.2, fc=colors[i], ec=colors[i],
                 linewidth=2, label=f"PC{i+1}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Original Data with Principal Components")
    ax.legend(fontsize="small")
    ax.set_aspect("equal")

    # --- Subplot 2: Data projected onto principal components (scores) ---
    ax = axes[1]
    ax.scatter(scores[:, 0], scores[:, 1], alpha=0.5, s=10)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Data in Principal Component Space")
    ax.set_aspect("equal")

    # --- Subplot 3: Explained variance ---
    ax = axes[2]
    bars = ax.bar(["PC 1", "PC 2"], ev_frac, color=["blue", "orange"], alpha=0.7)
    for bar, frac in zip(bars, ev_frac):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{frac:.2%}", ha="center", fontsize=11)
    ax.set_ylabel("Proportion of Variance Explained")
    ax.set_title("Explained Variance by Principal Component")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


# =============================================================================
# REFLECTION: contributions, tasks completed, and external resources used.
# =============================================================================
# NumPy documentation, lecture slides (Lecture1_intro, Lecture2_review)



if __name__ == '__main__':
    # --- Part 1 ---
    print("=" * 60)
    print("PART 1: Linear Algebra")
    print("=" * 60)

    AB, BA, AplusB_T, ABT, trA, trAB = matrix_operations()
    print("Q1: Matrix Operations")
    print(f"  AB       =\n{AB}")
    print(f"  BA       =\n{BA}")
    print(f"  (A+B)^T  =\n{AplusB_T}")
    print(f"  A@B.T    =\n{ABT}")
    print(f"  trace(A) = {trA}")
    print(f"  trace(AB)= {trAB}")

    vals, vecs = eigensystem()
    print("\nQ2: Eigenvalues and Eigenvectors")
    for i in range(len(vals)):
        print(f"  λ{i+1} = {vals[i]:.4f}, v{i+1} = {vecs[:, i]}")
        # Verify Mv = λv
        M = np.array([[2, -1], [-1, 2]], dtype=float)
        print(f"    M*v{i+1} = {M @ vecs[:, i]},  λ{i+1}*v{i+1} = {vals[i] * vecs[:, i]}")

    print("\nQ3: Proof that trace(AB) = trace(BA)")
    print("  (See written derivation — verified with Q1: trace(AB)=6, trace(BA)=6)")

    Cinv = matrix_inverse()
    print("\nQ4: Matrix Inverse")
    print(f"  C inverse =\n{Cinv}")
    C = np.array([[4, 7], [2, 6]], dtype=float)
    print(f"  Verification C·C⁻¹ =\n{C @ Cinv}")

    # --- Part 2 ---
    print("\n" + "=" * 60)
    print("PART 2: Probability, MLE, and MAP")
    print("=" * 60)

    c, mean, var = distribution_moments()
    print(f"Q1: c = {c:.6f}, E[X] = {mean}, Var(X) = {var}")

    p_int, p_bga, p_union = event_probabilities()
    print(f"Q2: P(A∩B) = {p_int}, P(B|A) = {p_bga}, P(A∪B) = {p_union}")

    mle, map_est = coin_estimates()
    print(f"Q3: θ_MLE = {mle:.4f}, θ_MAP = {map_est:.4f}")

    mu_mle, sig2_mle = gaussian_mle()
    print(f"Q4: μ̂_MLE = {mu_mle}, σ̂²_MLE = {sig2_mle}")

    mu_map = gaussian_map()
    print(f"Q5: μ̂_MAP = {mu_map}")

    print("Q6: Var(X) = E[X²] - (E[X])² (see written proof)")

    # --- Part 3 ---
    print("\n" + "=" * 60)
    print("PART 3: PCA")
    print("=" * 60)

    print("Q1: Derivation of Var(uᵀX) = uᵀΣu (see written proof)")

    data = generate_data(seed=0)
    print(f"Q2: Generated {data.shape[0]} samples, shape={data.shape}")
    sample_corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
    print(f"  Sample correlation: {sample_corr:.4f}")

    result = pca(data)
    print(f"Q3: PCA fitted")
    print(f"  Mean:          {result['mean']}")
    print(f"  Eigenvalues:   {result['eigenvalues']}")
    print(f"  PC1 direction: {result['components'][:, 0]}")
    print(f"  PC2 direction: {result['components'][:, 1]}")

    ev = explained_variance(result["eigenvalues"])
    print(f"Q4: Explained variance: PC1={ev[0]:.2%}, PC2={ev[1]:.2%}")
    print("  Interpretation: PC1 captures the majority of variance;")
    print("  the data can be well-approximated in 1D with minimal loss.")

    fig = make_plots(data, result)
    fig.savefig("Assignment1_PCA_plots.png", dpi=150)
    print("\nFigures saved to Assignment1_PCA_plots.png")
    plt.close(fig)