import numpy as np


# P1 Q1. Matrix operations.
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults are A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    A = np.asarray([[3, 4], [2, 1]] if A is None else A, dtype=float)
    B = np.asarray([[1, 2], [0, -1]] if B is None else B, dtype=float)
    AB = A @ B
    return AB, B @ A, (A + B).T, A @ B.T, np.trace(A), np.trace(AB)


# P1 Q2. Eigenvalues are 1 and 3, with directions (1,1) and (1,-1).
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    M = np.asarray([[2, -1], [-1, 2]] if M is None else M, dtype=float)
    return np.linalg.eigh(M)


# P1 Q3. trace(AB) = sum_ij A_ij B_ji = trace(BA).

# P1 Q4. det(C)=10, so C is invertible.
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    C = np.asarray([[4, 7], [2, 6]] if C is None else C, dtype=float)
    return np.linalg.inv(C)


# P2 Q1. Poisson normalization and moments.
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    return np.exp(-rate), float(rate), float(rate)


# P2 Q2. Conditional probability and the addition rule.
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    intersection = p_a_given_b * p_b
    return intersection, intersection / p_a, p_a + p_b - intersection


# P2 Q3. The prior adds one to the exponent of theta.
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    return heads / tosses, (heads + 1) / (tosses + 1)


# P2 Q4. The variance MLE uses divisor n.
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    data = np.asarray([1, 3, 5, 7] if data is None else data, dtype=float)
    return float(data.mean()), float(data.var(ddof=0))


# P2 Q5. Combine the observation and prior using their precisions.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    precision = 1 / observation_variance + 1 / prior_variance
    return (x / observation_variance + prior_mean / prior_variance) / precision


# P2 Q6. Var(X) = E[X**2] - E[X]**2.

# P3 Q1. Var(u.T@X) = u.T@Sigma@u.

# P3 Q2. Population correlation is approximately 0.693.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults are mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    mean = np.asarray([2, -1] if mean is None else mean, dtype=float)
    covariance = np.asarray(
        [[3, 1.2], [1.2, 1]] if covariance is None else covariance, dtype=float
    )
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=n_samples)


# P3 Q3. Maximize projection variance using covariance eigenvectors.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return a dictionary containing mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    X = np.asarray(X, dtype=float)
    # Center the features.
    mean = X.mean(axis=0)
    centered = X - mean
    # Sample covariance uses n - 1.
    covariance = centered.T @ centered / (X.shape[0] - 1)
    # Order components by decreasing variance.
    eigenvalues, components = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    # Remove negative roundoff.
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = components[:, order]
    # Project onto the principal directions.
    scores = centered @ components
    return dict(
        mean=mean,
        centered=centered,
        covariance=covariance,
        eigenvalues=eigenvalues,
        components=components,
        scores=scores,
    )


# P3 Q4. Variance fractions are normalized eigenvalues.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return eigenvalues / eigenvalues.sum()


# P3 Q2/Q3/Q4. Plot data, PCA scores, and variance fractions.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=float)
    eigenvalues = result['eigenvalues']
    components = result['components']
    mean = result['mean']
    # Variance fractions are undefined for constant data.
    fractions = (
        explained_variance(eigenvalues) if eigenvalues.sum() > 0 else None
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=14, alpha=0.4, label='Samples')
    colors = ['tab:red', 'tab:green']
    for i, color in enumerate(colors):
        direction = 2 * np.sqrt(eigenvalues[i]) * components[:, i]
        endpoints = np.vstack((mean - direction, mean + direction))
        ax.plot(endpoints[:, 0], endpoints[:, 1], color=color, linewidth=2,
                label=f'PC{i + 1}')
    ax.scatter(*mean, color='black', marker='x', s=60, label='Mean')
    ax.set(title='Data and principal directions', xlabel='Feature 1', ylabel='Feature 2')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend()

    ax = axes[1]
    scores = result['scores']
    ax.scatter(scores[:, 0], scores[:, 1], s=14, alpha=0.4)
    ax.axhline(0, color='tab:red', linewidth=1)
    ax.set(title='Data in principal coordinates', xlabel='PC1 score', ylabel='PC2 score')
    ax.set_aspect('equal', adjustable='datalim')

    ax = axes[2]
    if fractions is not None:
        bars = ax.bar(['PC1', 'PC2'], fractions, color=colors)
        ax.bar_label(bars, labels=[f'{value:.2%}' for value in fractions], padding=4)
    else:
        ax.text(0.5, 0.5, 'Undefined for zero total variance',
                ha='center', va='center', transform=ax.transAxes)
    ax.set(title='Explained variance', xlabel='Principal component',
           ylabel='Fraction of total variance', ylim=(0, 1.08))
    return fig


# REFLECTION. Implemented with AI assistance using NumPy and Matplotlib.

if __name__ == '__main__':
    from pathlib import Path
    import matplotlib.pyplot as plt

    X = generate_data()
    result = pca(X)
    fractions = explained_variance(result['eigenvalues'])
    # Reconstruct using PC1 only.
    reduced = result['scores'][:, :1]
    reconstructed = reduced @ result['components'][:, :1].T + result['mean']
    residual_variance = np.sum((X - reconstructed) ** 2) / (len(X) - 1)
    print('Sample mean', result['mean'])
    print('Sample correlation', np.corrcoef(X, rowvar=False)[0, 1])
    print('Sample covariance', result['covariance'])
    print('Eigenvalues', result['eigenvalues'])
    print('Principal components as columns', result['components'])
    print('Explained variance fractions', fractions)
    print('Retained dimensions', reduced.shape[1])
    print('Discarded variance', residual_variance)

    figure = make_plots(X, result)
    output_dir = Path(__file__).resolve().parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    for extension in ('png', 'pdf'):
        output_path = output_dir / f'pca_results.{extension}'
        figure.savefig(output_path, dpi=180)
        print('Saved', output_path)
    plt.close(figure)
