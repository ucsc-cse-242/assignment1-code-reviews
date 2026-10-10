"""CSE 242 Assignment 1: linear algebra, probability and estimation, and PCA.

Where each answer is:
- Handwritten derivations: Part 1 (Q1-Q4), Part 2 (Q1-Q6), and Part 3 Q1.
- Assignment1_Part3_Discussion.pdf: Part 3 Q2-Q4 discussion and figures.
- This file: every required function. Run it with `python Assignment1.py`
  to print all numerical answers and save the figure pca_plots.png.

Structure note: I would normally split a program like this into several
modules (for example linear algebra, estimation, and PCA with plotting).
I kept everything in one file to follow the provided code outline, which
makes it easier to grade and lets the whole assignment run at once, even
though a single file is not the best structure for a larger project.
"""
import numpy as np
from matplotlib.figure import Figure


# Helpers
def _trace(M):
    """Return the sum of the diagonal entries of a square matrix."""
    return float(sum(M[i, i] for i in range(len(M))))


def _symmetric_eigen(S, tol=1e-12, max_sweeps=100):
    """Return eigenvalues (largest first) and orthonormal eigenvector columns.

    Uses the Jacobi method: rotate S until its off-diagonal entries vanish.
    The diagonal becomes the eigenvalues and the combined rotations the
    eigenvectors, which also covers repeated eigenvalues.
    """
    # Step 1: work on a symmetric copy; V collects the rotations.
    S = np.array(S, dtype=float)
    S = (S + S.T) / 2
    n = len(S)
    V = np.eye(n)
    threshold = tol * np.abs(S).max()

    # Step 2: zero the off-diagonal entries one at a time until S is diagonal.
    for _ in range(max_sweeps):
        if np.abs(np.triu(S, 1)).max() <= threshold:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(S[p, q]) <= threshold:
                    continue
                # Rotation that zeroes S[p, q], using the numerically stable angle.
                theta = (S[q, q] - S[p, p]) / (2 * S[p, q])
                sign = 1.0 if theta >= 0 else -1.0
                t = sign / (abs(theta) + np.hypot(theta, 1.0))
                c = 1 / np.hypot(t, 1.0)
                R = np.array([[c, t * c], [-t * c, c]])
                S[:, [p, q]] = S[:, [p, q]] @ R
                S[[p, q], :] = R.T @ S[[p, q], :]
                S[p, q] = S[q, p] = 0.0
                V[:, [p, q]] = V[:, [p, q]] @ R

    # Step 3: sort the eigenpairs from largest to smallest eigenvalue.
    values = S.diagonal()
    order = np.argsort(-values, kind="stable")
    return values[order], V[:, order]


# PART 1 (20 points), Q1: Show computations for the given A and B.
# Derived by hand: see the handwritten derivations.
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    A = np.array([[3, 4], [2, 1]] if A is None else A, dtype=float)
    B = np.array([[1, 2], [0, -1]] if B is None else B, dtype=float)
    AB = A @ B
    return AB, B @ A, (A + B).T, A @ B.T, _trace(A), _trace(AB)


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# Derived by hand: see the handwritten derivations.
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    return _symmetric_eigen([[2, -1], [-1, 2]] if M is None else M)


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# Derived by hand: see the handwritten derivations.

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# Derived by hand: see the handwritten derivations.
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    C = np.array([[4, 7], [2, 6]] if C is None else C, dtype=float)
    n = len(C)

    # Step 1: build the augmented matrix [C | I].
    augmented = np.hstack([C, np.eye(n)])

    # Step 2: Gauss-Jordan elimination, one column at a time.
    for col in range(n):
        # Partial pivoting: move the row with the largest entry into place.
        pivot = col + np.argmax(np.abs(augmented[col:, col]))
        if augmented[pivot, col] == 0:
            raise ValueError("C is singular, so it has no inverse")
        augmented[[col, pivot]] = augmented[[pivot, col]]

        # Scale the pivot row so the pivot is 1, then clear the column.
        augmented[col] /= augmented[col, col]
        for row in range(n):
            if row != col:
                augmented[row] -= augmented[row, col] * augmented[col]

    # Step 3: the left half is now I, so the right half is the inverse.
    return augmented[:, n:].copy()


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# Derived by hand: see the handwritten derivations.
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # Normalizing gives a Poisson distribution, whose mean and variance
    # both equal the rate.
    c = float(np.exp(-rate))
    return c, float(rate), float(rate)


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# Derived by hand: see the handwritten derivations.
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    p_both = p_a_given_b * p_b  # multiplication rule
    p_b_given_a = p_both / p_a  # definition of conditional probability
    p_either = p_a + p_b - p_both  # inclusion-exclusion
    return p_both, p_b_given_a, p_either


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# Derived by hand: see the handwritten derivations.
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # MLE is the fraction of heads; the 2*theta prior acts like one extra head.
    # Both formulas also hold at the boundaries (no heads or all heads).
    mle = heads / tosses
    map_estimate = (heads + 1) / (tosses + 1)
    return mle, map_estimate


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# Derived by hand: see the handwritten derivations.
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    x = np.asarray([1, 3, 5, 7] if data is None else data, dtype=float)
    mean = x.sum() / x.size
    variance = ((x - mean) ** 2).sum() / x.size  # MLE divides by n, not n - 1
    return float(mean), float(variance)


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# Derived by hand: see the handwritten derivations.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # Precision-weighted average of the observation and the prior mean.
    return ((prior_variance * x + observation_variance * prior_mean)
            / (observation_variance + prior_variance))


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# Derived by hand: see the handwritten derivations.

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# Derived by hand: see the handwritten derivations.

# P3 Q2: Describe the Gaussian generator and observed correlation.
# See Assignment1_Part3_Discussion.pdf, section Q2.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    mean = np.array([2, -1] if mean is None else mean, dtype=float)
    if covariance is None:
        covariance = [[3, 1.2], [1.2, 1]]

    # Step 1: find A with A A^T = covariance (eigenvectors scaled by the
    # square roots of the eigenvalues).
    values, vectors = _symmetric_eigen(covariance)
    A = vectors * np.sqrt(values)

    # Step 2: draw independent standard normal samples from the seeded generator.
    z = np.random.default_rng(seed).standard_normal((n_samples, mean.size))

    # Step 3: transform them to have the requested mean and covariance.
    return mean + z @ A.T


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# See Assignment1_Part3_Discussion.pdf, section Q3.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)

    # Step 1: center the data. The sample mean estimates E[X] from Q1.
    mean = X.sum(axis=0) / n
    centered = X - mean

    # Step 2: sample covariance with divisor n - 1. This estimates Sigma from Q1.
    covariance = centered.T @ centered / (n - 1)

    # Step 3: eigendecomposition. By Q1, the variance along a unit vector u is
    # u^T Sigma u, which is maximized by the top eigenvector; each eigenvalue
    # is the variance along its eigenvector. Clip tiny negative rounding errors.
    eigenvalues, components = _symmetric_eigen(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Step 4: components are sorted by variance, so the leading columns are the
    # principal components; project the data onto them.
    scores = centered @ components

    return {"mean": mean, "centered": centered, "covariance": covariance,
            "eigenvalues": eigenvalues, "components": components,
            "scores": scores}


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# See Assignment1_Part3_Discussion.pdf, section Q4.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return eigenvalues / eigenvalues.sum()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# Figure: pca_plots.png, panels (a) to (c), saved when this file runs.
# Discussion: Assignment1_Part3_Discussion.pdf.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    X = np.asarray(X, dtype=float)
    mean, eigenvalues = result["mean"], result["eigenvalues"]
    components = result["components"]
    fractions = explained_variance(eigenvalues)
    names = [f"PC{j + 1}" for j in range(len(fractions))]
    colors = [f"C{j + 1}" for j in range(len(fractions))]

    fig = Figure(figsize=(17, 5.5), layout="constrained")
    fig.suptitle("PCA of the synthetic 2-D Gaussian data")
    ax_samples, ax_pcs, ax_var = fig.subplots(1, 3)

    # Step 1: panels (a) and (b) both show the samples and their mean.
    for ax in (ax_samples, ax_pcs):
        ax.scatter(X[:, 0], X[:, 1], s=12, alpha=0.4, color="C0",
                   label=f"samples (n = {len(X)})")
        ax.plot(mean[0], mean[1], "k*", ms=14, zorder=3, label="sample mean")
        ax.set(xlabel="$x_1$", ylabel="$x_2$")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    ax_samples.set_title("(a) Samples")

    # Step 2: panel (b) adds each principal axis through the mean, two
    # standard deviations (2 sqrt(eigenvalue)) in each direction.
    for j, name in enumerate(names):
        half = 2 * np.sqrt(eigenvalues[j]) * components[:, j]
        ends = np.array([mean - half, mean + half])
        ax_pcs.plot(ends[:, 0], ends[:, 1], color=colors[j], lw=3,
                    label=f"{name} axis, $\\pm$2 SD ({fractions[j]:.1%} of variance)")
    ax_pcs.set_title("(b) Principal directions")
    for ax in (ax_samples, ax_pcs):
        ax.legend(loc="upper left", fontsize="small")

    # Step 3: panel (c) shows the variance fraction per component, plus the
    # cumulative total.
    bars = ax_var.bar(names, fractions, color=colors, alpha=0.85)
    ax_var.bar_label(bars, labels=[f"{f:.1%}" for f in fractions],
                     label_type="center", color="white", fontweight="bold")
    ax_var.plot(names, np.cumsum(fractions), "ko--", label="cumulative")
    ax_var.set(title="(c) Explained variance", xlabel="Principal component",
               ylabel="Fraction of total variance", ylim=(0, 1.1))
    ax_var.grid(axis="y", alpha=0.3)
    ax_var.legend(loc="center right")
    return fig


def main():
    """Print every numerical answer, then save the PCA figure."""
    np.set_printoptions(precision=4, suppress=True)

    # Part 1
    labels = ["AB", "BA", "(A+B)^T", "A B^T", "trace(A)", "trace(AB)"]
    for label, value in zip(labels, matrix_operations()):
        print(f"P1 Q1 {label}:\n{value}")
    values, vectors = eigensystem()
    print(f"P1 Q2 eigenvalues: {values}\neigenvectors (columns):\n{vectors}")
    print(f"P1 Q4 C^-1:\n{matrix_inverse()}")

    # Part 2
    print("P2 Q1 c, E[X], Var(X):", np.round(distribution_moments(), 4))
    print("P2 Q2 P(A and B), P(B|A), P(A or B):",
          np.round(event_probabilities(), 4))
    print("P2 Q3 MLE, MAP:", np.round(coin_estimates(), 4))
    print("P2 Q4 mu_MLE, sigma^2_MLE:", gaussian_mle())
    print("P2 Q5 mu_MAP:", gaussian_map())

    # Part 3
    X = generate_data()
    result = pca(X)
    S = result["covariance"]
    print("P3 Q2 sample mean:", result["mean"], "sample correlation:",
          round(S[0, 1] / np.sqrt(S[0, 0] * S[1, 1]), 4))
    print(f"P3 Q3 covariance:\n{S}\neigenvalues: {result['eigenvalues']}\n"
          f"components (columns):\n{result['components']}")
    fractions = explained_variance(result["eigenvalues"])
    print("P3 Q4 explained variance fractions:", fractions)

    # Reduce to one dimension: the variance lost equals the PC2 eigenvalue.
    pc1 = result["components"][:, :1]
    reconstruction = result["mean"] + result["scores"][:, :1] @ pc1.T
    lost = ((X - reconstruction) ** 2).sum() / (len(X) - 1)
    print(f"Keeping PC1 retains {fractions[0]:.1%} of the variance; "
          f"reconstruction error {lost:.4f} = lambda_2")

    make_plots(X, result).savefig("pca_plots.png", dpi=150)
    print("Saved pca_plots.png")


# REFLECTION: contributions, tasks completed, and external resources used.
# Contributions: individual submission.
# Tasks completed: all parts. Part 1, Part 2 and Part 3 Q1 are derived by
# hand; Part 3 Q2-Q4 are discussed in Assignment1_Part3_Discussion.pdf; this
# file implements every function and produces all numbers and figures.
# Resources: Claude (Anthropic) assisted with the code and the Part 3
# write-up. The conversation log is submitted on Canvas, per the course AI
# policy.

if __name__ == '__main__':
    main()
