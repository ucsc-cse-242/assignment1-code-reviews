"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np
import math
import matplotlib.pyplot as plt


def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """

    if A is None:
            A = np.array([[3, 4], [2, 1]], dtype=float)
    if B is None:
            B = np.array([[1, 2], [0, -1]], dtype=float)
    # YOUR CODE HERE
    def T(M):
        return np.transpose(M)
    
    return (A @ B, B @ A, T(A+B), A@T(B), np.trace(A), np.trace(A @ B))


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE

    if M is None:
        M = np.array([[2, -1], [-1, 2]], dtype=float)

    eigenvals, eigenvecs = np.linalg.eig(M)

    for i in range(len(eigenvals)):
        v = eigenvecs[:, i]                       # column i, not row i
        if not np.allclose(M @ v, eigenvals[i] * v):
            raise ValueError(f"Eigenpair {i} failed verification")


    return (eigenvals, eigenvecs)
    


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    return  np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    
    return (np.e**(-rate), rate, rate)


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    return(p_a_given_b*p_b, p_a_given_b * (p_b/p_a), p_a+p_b-(p_a_given_b * p_b))


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    return (heads/tosses, (heads+1)/(tosses+1))


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    if data is None:
        data = [1, 3, 5, 7]
    x = np.asarray(data, dtype=float)
    mu_hat = x.mean()
    var_hat = np.mean((x - mu_hat) ** 2)
    return float(mu_hat), float(var_hat)


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    data_precision = 1.0 / observation_variance
    prior_precision = 1.0 / prior_variance
    return (data_precision * x + prior_precision * prior_mean) / (data_precision + prior_precision)


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

    if mean is None:
        mean = [2.0, -1.0]
    if covariance is None:
        covariance = [[3.0, 1.2], [1.2, 1.0]]

    mean = np.array(mean, dtype=float)
    covariance = np.array(covariance, dtype=float)

    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=n_samples)

def show_plot_q2():
    X = generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.6)
    plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
    plt.title("Correlated Gaussian data")
    plt.axis("equal")
    plt.show()

    print(np.corrcoef(X.T)[0, 1])   # should be near 0.69
    print(np.cov(X.T))     


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
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    # Step 1: center the data (matches the (X - mu) in the definition of Sigma)
    mean = X.mean(axis=0)
    Xc = X - mean

    # Step 2: sample covariance matrix, Sigma_hat = Xc^T Xc / (n-1)
    cov = Xc.T @ Xc / (n - 1)

    # Step 3: eigendecomposition (eigh: symmetric matrix, real eigenvalues,
    # orthonormal eigenvectors, valid for tied eigenvalues too)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Step 4: sort descending by eigenvalue
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Rank-deficient case: a covariance matrix is positive semi-definite, so
    # any negative eigenvalue (e.g. -1e-17) is floating-point error. Clip to 0.
    eigvals = np.clip(eigvals, 0.0, None)

    # Sign convention: make the largest-magnitude entry of each component positive
    for j in range(d):
        if eigvecs[np.argmax(np.abs(eigvecs[:, j])), j] < 0:
            eigvecs[:, j] *= -1

    components = eigvecs
    scores = Xc @ components               # projections u^T (x - mean)

    # Explained variance; if all points are identical, total variance is 0
    total = eigvals.sum()
    explained_ratio = eigvals / total if total > 0 else np.zeros_like(eigvals)

    return {
        "mean": mean,
        "centered": Xc,
        "covariance": cov,
        "eigenvalues": eigvals,
        "components": components,
        "scores": scores,
        "explained_ratio": explained_ratio,
    }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return eigenvalues / eigenvalues.sum()


def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    X = np.asarray(X, dtype=float)
    mean = np.asarray(result["mean"], dtype=float)
    comps = np.asarray(result["components"], dtype=float)      # columns = PCs
    eigvals = np.asarray(result["eigenvalues"], dtype=float)
    ratio = np.asarray(result["explained_ratio"], dtype=float)

    n_pc = comps.shape[1]
    scores = (X - mean) @ comps                                # projections onto PCs
    cum = np.cumsum(ratio)
    colors = ["tab:red", "tab:green", "tab:purple", "tab:orange"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1: original data with principal directions through the mean
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=12, alpha=0.4, color="tab:blue", label="samples")
    ax.scatter(mean[0], mean[1], c="k", marker="x", s=100, zorder=3, label="mean")
    for j in range(min(n_pc, 2)):
        # full line through the mean (faint) plus an arrow of length 2*std
        v = comps[:, j]
        length = 2 * np.sqrt(max(eigvals[j], 0))
        c = colors[j % len(colors)]
        ax.plot(*zip(mean - 1.5 * length * v, mean + 1.5 * length * v),
                color=c, alpha=0.3, lw=1)
        ax.annotate("", xy=mean + length * v, xytext=mean,
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.5))
        ax.plot([], [], color=c, lw=2.5,
                label=f"PC{j+1} ({ratio[j]:.1%} of variance)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Data with principal directions")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: data in principal-component coordinates
    ax = axes[1]
    if n_pc >= 2:
        ax.scatter(scores[:, 0], scores[:, 1], s=12, alpha=0.4, color="tab:blue")
        ax.set_ylabel("PC2 score")
    else:
        ax.scatter(scores[:, 0], np.zeros(len(scores)), s=12, alpha=0.4, color="tab:blue")
        ax.set_ylabel("")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("PC1 score")
    ax.set_title("Data in PC coordinates (uncorrelated)")
    ax.grid(alpha=0.3)

    # Panel 3: explained variance fractions (bars) and cumulative (line)
    ax = axes[2]
    idx = np.arange(1, len(ratio) + 1)
    bars = ax.bar(idx, ratio, color="tab:blue", alpha=0.8, label="Per component")
    ax.plot(idx, cum, "o-", color="tab:red", label="Cumulative")
    for b, r in zip(bars, ratio):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.015, f"{r:.1%}",
                ha="center", fontsize=9)
    ax.set_xticks(idx)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Proportion of variance explained")
    ax.set_title("Explained variance")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("PCA of correlated 2D Gaussian data", fontsize=14)
    fig.tight_layout()
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
"""
    all tasks completed
    used claude
"""
if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    # ---------- Part 1 ----------
    AB, BA, ABt_sum_T, A_Bt, trA, trAB = matrix_operations()
    print("AB =\n", AB)
    print("BA =\n", BA)
    print("(A+B)^T =\n", ABt_sum_T)
    print("A B^T =\n", A_Bt)
    print("trace(A) =", trA, " trace(AB) =", trAB)

    vals, vecs = eigensystem()
    print("Eigenvalues:", vals)
    print("Eigenvectors (columns):\n", vecs)

    C = np.array([[4, 7], [2, 6]], dtype=float)
    print("Inverse of C =\n", matrix_inverse(C))

    # ---------- Part 2 ----------
    print("Poisson (c, mean, variance):", distribution_moments())
    print("Events (P(A&B), P(B|A), P(AuB)):", event_probabilities())
    print("Coin (MLE, MAP):", coin_estimates())
    print("Gaussian MLE (mean, var):", gaussian_mle())
    print("Gaussian MAP:", gaussian_map())

    # ---------- Part 3 ----------
    X = generate_data()
    print("Sample correlation:", np.corrcoef(X.T)[0, 1])
    print("Sample covariance:\n", np.cov(X.T))

    # Q2: scatter plot of the raw data
    fig_q2, ax_q2 = plt.subplots(figsize=(6, 5))
    ax_q2.scatter(X[:, 0], X[:, 1], s=10, alpha=0.6)
    ax_q2.set_xlabel("Feature 1")
    ax_q2.set_ylabel("Feature 2")
    ax_q2.set_title("Correlated Gaussian data")
    ax_q2.set_aspect("equal", adjustable="datalim")
    ax_q2.grid(alpha=0.3)
    fig_q2.tight_layout()
    fig_q2.savefig("q2_scatter.png", dpi=150)

    # Q3/Q4: PCA, explained variance, and the combined figure
    res = pca(X)
    print("PCA eigenvalues:", res["eigenvalues"])
    print("PCA components (columns):\n", res["components"])
    print("Explained variance:", explained_variance(res["eigenvalues"]))

    fig = make_plots(X, res)
    fig.savefig("pca_plots.png", dpi=150)

    plt.show()
