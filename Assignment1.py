"""CSE 242 Assignment 1 — submit as Assignment1.py.

Part 1 and Part 2 derivations are in the separately submitted handwritten PDF.
Part 3 code and comments are supplemented by Assignment1_PCA_Analysis.pdf.
Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# See the separately submitted handwritten derivations PDF for this question.
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    A = np.asarray([[3, 4], [2, 1]] if A is None else A, dtype=float)
    B = np.asarray([[1, 2], [0, -1]] if B is None else B, dtype=float)
    return A @ B, B @ A, (A + B).T, A @ B.T, np.trace(A), np.trace(A @ B)


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# See the separately submitted handwritten derivations PDF for this question.
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    M = np.asarray([[2, -1], [-1, 2]] if M is None else M, dtype=float)
    return np.linalg.eigh(M)


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# See the separately submitted handwritten derivations PDF for this question.

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# See the separately submitted handwritten derivations PDF for this question.
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    C = np.asarray([[4, 7], [2, 6]] if C is None else C, dtype=float)
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# See the separately submitted handwritten derivations PDF for this question.
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    return float(np.exp(-rate)), float(rate), float(rate)


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# See the separately submitted handwritten derivations PDF for this question.
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    intersection = p_a_given_b * p_b
    return intersection, intersection / p_a, p_a + p_b - intersection


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# See the separately submitted handwritten derivations PDF for this question.
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    return heads / tosses, (heads + 1) / (tosses + 1)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# See the separately submitted handwritten derivations PDF for this question.
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    data = np.asarray([1, 3, 5, 7] if data is None else data, dtype=float)
    mean = np.mean(data)
    return float(mean), float(np.mean((data - mean) ** 2))


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# See the separately submitted handwritten derivations PDF for this question.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    return (x / observation_variance + prior_mean / prior_variance) / (
        1 / observation_variance + 1 / prior_variance
    )


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# See the separately submitted handwritten derivations PDF for this question.

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# Let X be a d-dimensional random column vector with mean
# mu=E[X], and let u be a fixed unit vector (u^T u=1).
# The scalar projected variable is Y=u^T X, with
# E[Y]=E[u^T X]=u^T E[X]=u^T mu.
# Define the covariance matrix
# Sigma=Cov(X)=E[(X-mu)(X-mu)^T].
# Its (i,j) entry is Cov(X_i,X_j)=E[(X_i-mu_i)(X_j-mu_j)].
# Equivalently, Sigma=E[XX^T]-mu*mu^T, because
# E[(X-mu)(X-mu)^T]
# =E[XX^T]-E[X]mu^T-mu E[X]^T+mu mu^T
# =E[XX^T]-mu mu^T.
# Start with the scalar variance definition:
# Var(u^T X)=E[(u^T X-E[u^T X])^2]
#            =E[(u^T(X-mu))^2].
# The scalar square is a product with its transpose:
# (u^T(X-mu))^2 = u^T(X-mu)(X-mu)^T u.
# Since u is deterministic, linearity of expectation gives
# Var(u^T X)=u^T E[(X-mu)(X-mu)^T] u = u^T Sigma u.
# Note: unit length is needed for comparing directions at fixed
# scale, not for the covariance identity itself.

# P3 Q2: Describe the Gaussian generator and observed correlation.
# NumPy's default generator draws n=500 independent samples from
# N(mu,Sigma), with mu=(2,-1)^T and
# Sigma=[[3,1.2],[1.2,1]]. Variances are 3 and 1;
# covariance is +1.2, so theoretical correlation is
# rho=1.2/sqrt(3*1)=0.6928 (approximately).
# The scatter should form a tilted elliptical cloud: as feature 1
# increases, feature 2 tends to increase. The random sample
# correlation may differ somewhat from the population value.
# Setting seed=0 makes the generated sample reproducible.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    mean = np.asarray([2, -1] if mean is None else mean, dtype=float)
    covariance = np.asarray(
        [[3, 1.2], [1.2, 1]] if covariance is None else covariance, dtype=float
    )
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# Given n observations (rows) x_i in R^2:
# 1. Sample mean mu_hat=(1/n)sum_i x_i. Center: z_i=x_i-mu_hat.
#    In matrix form X_centered=X-1*mu_hat^T; each column mean is 0.
# 2. Sample covariance S=(1/(n-1))sum_i z_i z_i^T
#    =X_centered^T X_centered/(n-1). This is symmetric PSD.
# 3. For a unit direction u, the sample projected variance is
#    u^T S u, the sample counterpart of Part 3 Q1.
#    To maximize it subject to u^T u=1, consider
#    L(u,lambda)=u^T S u-lambda(u^T u-1).
#    dL/du=2Su-2lambda*u=0, so Su=lambda*u.
#    Also u^T S u=lambda for normalized eigenvector u.
# 4. Use np.linalg.eigh(S) (for symmetric matrices), sort eigenvalues
#    in descending order and reorder matching eigenvectors (columns).
#    The eigenvector of the largest eigenvalue is PC1: it maximizes
#    retained variance. The orthogonal PC2 captures residual variance.
# 5. Project every centered row onto both components with
#    scores=X_centered@components. PC score axes are uncorrelated:
#    their covariance is diagonal with the sorted eigenvalues.
# In the figure, arrows begin at the sample mean and point along
# the two principal directions (their signs are arbitrary).
# The elongated direction of the point cloud should match PC1.
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
    covariance = centered.T @ centered / (X.shape[0] - 1)
    eigenvalues, components = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    # Clamp tiny negative roundoff values from a positive semidefinite matrix.
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = components[:, order]
    scores = centered @ components
    return dict(mean=mean, centered=centered, covariance=covariance,
                eigenvalues=eigenvalues, components=components, scores=scores)


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# If eigenvalues are lambda_1>=lambda_2>=0, total variance equals
# trace(S)=lambda_1+lambda_2, and the fraction explained by PCj is
# ratio_j=lambda_j/(lambda_1+lambda_2).
# The first k PCs retain cumulative fraction sum_{j<=k} ratio_j.
# On the seeded default data, PC1 accounts for most variability;
# its score axis follows the longer direction of the point cloud.
# Keeping only PC1 maps each two-feature observation to one score,
# reducing storage/model input dimensionality while discarding PC2's
# variability. The reconstruction using one PC is
# x_hat_i=mu_hat+score_{i,1}*u_1^T (row-vector convention).
# The average squared Euclidean reconstruction error per observation
# equals the sum of omitted sample eigenvalues times (n-1)/n.
# Greater compression (fewer PCs) loses more variance and may
# discard useful information; keeping both PCs retains 100% and
# performs no dimensionality reduction. Explained variance alone
# does not establish usefulness for any downstream prediction task.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return eigenvalues / np.sum(eigenvalues)


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# The three panels visualize (left) the data and the orthogonal
# eigenvector directions through the mean; (middle) uncorrelated
# PC scores after centering and rotation; and (right) individual and
# cumulative explained-variance ratios. Arrow length is scaled by
# sqrt(eigenvalue) to reflect the spread in each direction.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=float)
    values = result['eigenvalues']
    total = np.sum(values)
    # A constant dataset has no variance to explain; display zeros in that case.
    fractions = explained_variance(values) if total > 0 else np.zeros_like(values)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=12, alpha=0.4, label='Samples')
    mean = result['mean']
    ax.scatter(*mean, color='black', s=40, label='Sample mean', zorder=5)
    for j, color in enumerate(('tab:red', 'tab:green')):
        vector = 2 * np.sqrt(values[j]) * result['components'][:, j]
        ax.plot([mean[0]-vector[0], mean[0]+vector[0]],
                [mean[1]-vector[1], mean[1]+vector[1]],
                color=color, linewidth=2, label=f'PC{j+1} direction')
        ax.annotate('', xy=mean+vector, xytext=mean,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
    ax.set(xlabel='Feature 1', ylabel='Feature 2', title='Data and principal directions')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize=8)
    scores = result['scores']
    axes[1].scatter(scores[:, 0], scores[:, 1], s=12, alpha=0.4)
    axes[1].axhline(0, color='gray', lw=0.7)
    axes[1].axvline(0, color='gray', lw=0.7)
    axes[1].set(xlabel='PC1 score', ylabel='PC2 score', title='Centered data in PC coordinates')
    axes[1].set_aspect('equal', adjustable='datalim')
    axes[2].bar([1, 2], fractions, color=['tab:red', 'tab:green'], label='Individual')
    axes[2].plot([1, 2], np.cumsum(fractions), 'o--', color='black', label='Cumulative')
    for j, fraction in enumerate(fractions):
        axes[2].text(j+1, fraction+0.025, f'{fraction:.2%}', ha='center')
    axes[2].set(xlabel='Principal component', ylabel='Fraction of total variance',
                title='Explained variance', ylim=(0, 1.15), xticks=[1, 2])
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
# Reflection / work summary:
# - Part 1 and Part 2 handwritten derivations are submitted separately.
# - Part 3 uses NumPy for correlated Gaussian sampling and PCA,
#   and Matplotlib for three labeled plots.
# - NumPy and Matplotlib are the software dependencies. No external
#   dataset or fitted PCA library is required.
# - Drafting assistance: OpenAI ChatGPT was used to prepare the
#   code and written explanations; check course rules for disclosure.

if __name__ == '__main__':
    from pathlib import Path
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=6, suppress=True)
    print('P1 matrix operations:', matrix_operations())
    print('P1 eigenpairs:', eigensystem())
    print('P1 inverse:', matrix_inverse())
    print('P2 distribution moments:', distribution_moments())
    print('P2 event probabilities:', event_probabilities())
    print('P2 coin MLE/MAP:', coin_estimates())
    print('P2 Gaussian MLE:', gaussian_mle())
    print('P2 Gaussian MAP:', gaussian_map())
    X = generate_data()
    result = pca(X)
    fractions = explained_variance(result['eigenvalues'])
    print('P3 sample correlation:', np.corrcoef(X, rowvar=False)[0, 1])
    print('P3 sample mean:', result['mean'])
    print('P3 sample covariance:\n', result['covariance'])
    print('P3 eigenvalues:', result['eigenvalues'])
    print('P3 components (columns):\n', result['components'])
    print('P3 explained variance fractions:', fractions)
    reconstructed = result['scores'][:, :1] @ result['components'][:, :1].T + result['mean']
    mse = np.mean(np.sum((X - reconstructed) ** 2, axis=1))
    print(f'Keeping PC1 retains {fractions[0]:.2%} of sample variance, '
          f'discards {fractions[1]:.2%}, and reduces 2 dimensions to 1.')
    print(f'Mean squared Euclidean reconstruction error: {mse:.6f}')
    fig = make_plots(X, result)
    output = Path(__file__).resolve().with_name('Assignment1_PCA.png')
    fig.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved PCA figure: {output}')
