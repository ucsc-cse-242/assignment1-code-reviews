"""CSE 242 Assignment 1 — submit as Assignment1.py.

Reasoning is referenced in Assignment1_revised.pdf; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# PDF Part 1, Q1(a)-(f)
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    if A is None:
        A = [[3, 4], [2, 1]]
    if B is None:
        B = [[1, 2], [0, -1]]
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    AB = A @ B
    return AB, B @ A, (A + B).T, A @ B.T, np.trace(A), np.trace(AB)


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# PDF Part 1, Q2(a)-(c)
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    if M is None:
        M = [[2, -1], [-1, 2]]
    M = np.asarray(M, dtype=float)
    return np.linalg.eigh(M)  # all orthonormal eigenvectors, even repeated roots


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# PDF Part 1, Q3

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# PDF Part 1, Q4(a)-(c)
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    if C is None:
        C = [[4, 7], [2, 6]]
    return np.linalg.inv(np.asarray(C, dtype=float))


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# PDF Part 2, Q1(a)-(b)
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    rate = float(rate)
    return float(np.exp(-rate)), rate, rate


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# PDF Part 2, Q2(a)-(c)
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    intersection = p_a_given_b * p_b
    return intersection, intersection / p_a, p_a + p_b - intersection


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# PDF Part 2, Q3(a)-(b)
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    return heads / tosses, (heads + 1) / (tosses + 1)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# PDF Part 2, Q4
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    if data is None:
        data = [1, 3, 5, 7]
    values = np.asarray(data, dtype=float)
    return float(np.mean(values)), float(np.var(values, ddof=0))


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# PDF Part 2, Q5
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    precision = 1.0 / observation_variance + 1.0 / prior_variance
    return float((x / observation_variance + prior_mean / prior_variance) / precision)


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# PDF Part 2, Q6

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# PDF Part 3, Q1

# P3 Q2: Describe the Gaussian generator and observed correlation.
# PDF Part 3, Q2
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    if mean is None:
        mean = [2.0, -1.0]
    if covariance is None:
        covariance = [[3.0, 1.2], [1.2, 1.0]]
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.asarray(mean, dtype=float),
                                   np.asarray(covariance, dtype=float),
                                   size=n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# PDF Part 3, Q3(a)-(d)
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0)
    centered = X - mean
    covariance = (centered.T @ centered) / (X.shape[0] - 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    # Remove harmless floating-point negative zeros for PSD covariance.
    values = np.maximum(values, 0.0)
    return {'mean': mean, 'centered': centered, 'covariance': covariance,
            'eigenvalues': values, 'components': vectors,
            'scores': centered @ vectors}


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# PDF Part 3, Q4
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    values = np.asarray(eigenvalues, dtype=float)
    return values / np.sum(values)


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# PDF Part 3, Figures for Q2/Q3/Q4
#   Q2: gaussian_scatter.png; Q3: pca_overview.png;
#   Q4: explained_variance.png.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    import matplotlib.pyplot as plt
    X = np.asarray(X, dtype=float)
    mu = np.asarray(result['mean'])
    directions = np.asarray(result['components'])
    eigenvalues = np.asarray(result['eigenvalues'])
    ratio = explained_variance(eigenvalues)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.45, label='Samples')
    ax.scatter([mu[0]], [mu[1]], c='black', marker='x', s=80, label='Mean')
    for i in range(2):
        v = directions[:, i] * (2 * np.sqrt(eigenvalues[i]))
        ax.plot([mu[0] - v[0], mu[0] + v[0]],
                [mu[1] - v[1], mu[1] + v[1]], linewidth=2.5,
                label=f'PC{i+1} (variance={eigenvalues[i]:.2f})')
    ax.set(xlabel='Feature 1', ylabel='Feature 2',
           title='Gaussian samples and principal directions')
    ax.axis('equal')
    ax.legend(fontsize=8)
    axes[1].bar(['PC1', 'PC2'], ratio)
    axes[1].set(xlabel='Principal component', ylabel='Explained variance fraction',
                title='Explained variance', ylim=(0, 1))
    for i, value in enumerate(ratio):
        axes[1].text(i, min(float(value) + 0.025, 0.96), f'{value:.1%}', ha='center')
    fig.tight_layout()
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
# None

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pathlib import Path

    X = generate_data()
    result = pca(X)
    ratios = explained_variance(result['eigenvalues'])
    output_dir = Path(__file__).resolve().parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: original samples with principal directions and variance fractions.
    fig = make_plots(X, result)
    fig.savefig(output_dir / 'pca_overview.png', dpi=200, bbox_inches='tight')

    # Figure 2: original Gaussian data scatter plot.
    fig2, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], s=13, alpha=0.55)
    ax.set(xlabel='Feature 1', ylabel='Feature 2',
           title='Correlated Gaussian samples (n=500)')
    ax.axis('equal')
    fig2.tight_layout()
    fig2.savefig(output_dir / 'gaussian_scatter.png', dpi=200, bbox_inches='tight')

    # Figure 3: observations in rotated principal-component coordinates.
    fig3, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(result['scores'][:, 0], result['scores'][:, 1], s=13, alpha=0.55)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.axvline(0, color='grey', linewidth=0.8)
    ax.set(xlabel='PC1 score', ylabel='PC2 score', title='Data projected onto PCs')
    ax.axis('equal')
    fig3.tight_layout()
    fig3.savefig(output_dir / 'pca_scores.png', dpi=200, bbox_inches='tight')

    # Figure 4: retain PC1 only and reconstruct in the original coordinates.
    reconstruction = (result['mean']
                      + result['scores'][:, [0]] @ result['components'][:, [0]].T)
    fig4, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], s=13, alpha=0.24, label='Original')
    ax.scatter(reconstruction[:, 0], reconstruction[:, 1], s=12, alpha=0.52,
               label='PC1 reconstruction')
    ax.set(xlabel='Feature 1', ylabel='Feature 2', title='One-component PCA reconstruction')
    ax.axis('equal')
    ax.legend()
    fig4.tight_layout()
    fig4.savefig(output_dir / 'pca_reconstruction.png', dpi=200, bbox_inches='tight')

    # Figure 5: explained variance for the two components.
    fig5, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['PC1', 'PC2'], ratios)
    ax.set(xlabel='Principal component', ylabel='Fraction of total variance',
           title='PCA explained variance', ylim=(0, 1))
    for j, fraction in enumerate(ratios):
        ax.text(j, min(float(fraction) + 0.025, 0.96), f'{fraction:.1%}', ha='center')
    fig5.tight_layout()
    fig5.savefig(output_dir / 'explained_variance.png', dpi=200, bbox_inches='tight')

    print('Saved five figures in:', output_dir)
    plt.show()
