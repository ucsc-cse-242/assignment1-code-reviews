"""CSE 242 Assignment 1 - submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import math

import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# YOUR ANSWER HERE
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    A = np.array([[3., 4.], [2., 1.]]) if A is None else np.array(A, dtype=float)
    B = np.array([[1., 2.], [0., -1.]]) if B is None else np.array(B, dtype=float)
    AB = A @ B
    BA = B @ A
    sum_T = (A + B).T
    ABt = A @ B.T
    return AB, BA, sum_T, ABt, float(np.trace(A)), float(np.trace(AB))


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE
    M = np.array([[2., -1.], [-1., 2.]]) if M is None else np.array(M, dtype=float)
    values, vectors = np.linalg.eigh(M)
    return values, vectors


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE


# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    C = np.array([[4., 7.], [2., 6.]]) if C is None else np.array(C, dtype=float)
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    rate = float(rate)
    return math.exp(-rate), rate, rate


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    p_ab = p_a_given_b * p_b
    p_b_given_a = p_ab / p_a
    p_union = p_a + p_b - p_ab
    return p_ab, p_b_given_a, p_union


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    mle = heads / tosses
    map_est = (heads + 1) / (tosses + 1)
    return float(mle), float(map_est)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    x = np.array([1., 3., 5., 7.]) if data is None else np.array(data, dtype=float).ravel()
    mu = x.mean()
    var = np.mean((x - mu) ** 2)  # divide by n (MLE), not n-1
    return float(mu), float(var)


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    precision_obs = 1.0 / observation_variance
    precision_prior = 1.0 / prior_variance
    return float((x * precision_obs + prior_mean * precision_prior)
                 / (precision_obs + precision_prior))


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE

# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
# Data are drawn from a 2-D Gaussian N(mean, Sigma) with mean = [2, -1] and
# Sigma = [[3, 1.2], [1.2, 1]], using NumPy's seeded Generator (np.random.default_rng(seed))
# so results are reproducible.
# The off-diagonal 1.2 makes the features positively correlated. The population correlation is
#   rho = 1.2 / sqrt(3 * 1) ≈ 0.69.
# With seed 0 and n = 500, the sample correlation is ≈ 0.70 and the sample covariance is
# ≈ [[2.92, 1.17], [1.17, 0.95]] (printed in main), close to the true Sigma.
# In the scatter plot the cloud is a tilted ellipse rising to the right: when feature 1 is high,
# feature 2 tends to be high too. The ellipse is stretched along x1 because Var(x1) = 3 > Var(x2) = 1.
# Its long axis is not parallel to either coordinate axis, so the features share information.
# That shared information is what PCA exploits.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    mean = np.array([2., -1.]) if mean is None else np.array(mean, dtype=float)
    covariance = (np.array([[3., 1.2], [1.2, 1.]]) if covariance is None
                  else np.array(covariance, dtype=float))
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
# Step 1: Center: subtract the sample mean x_bar from each row. This puts mu = 0 in the
#          derivation of Q1, so (X - mu) becomes the centered data X_c.
# Step 2: Covariance: S = X_c^T X_c / (n - 1), the sample estimate of Sigma = E[(X-mu)(X-mu)^T].
# Step 3: Eigendecomposition: solve S u = lambda u (np.linalg.eigh, since S is symmetric).
#          By Q1, each eigenvalue lambda is the variance of the data projected onto its eigenvector u.
# Step 4: Select components: sort eigenvalues in descending order and reorder eigenvector columns
#          to match. PC1 is the direction of maximum variance; PC2 is orthogonal to it.
#          Scores = X_c @ components gives each sample's coordinates in the PC basis. Projecting
#          onto only the first k columns gives the reduced representation.
# Results for seed 0: eigenvalues ≈ [3.46, 0.41]. PC1 ≈ ±[0.91, 0.42] (about 25° above the x1-axis)
# and PC2 is perpendicular to it.
# Plot: PC arrows are drawn from the mean, scaled by 2*sqrt(lambda) (two standard deviations).
# PC1 runs along the long axis of the tilted ellipse, and PC2 along the short axis.
# Edge cases: if the data are rank-deficient (e.g. points on a line), one eigenvalue is 0. If the
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # YOUR CODE HERE
    X = np.array(X, dtype=float)  
    n = X.shape[0]
    mean = X.mean(axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (n - 1)
    covariance = (covariance + covariance.T) / 2  
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)  
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    scores = centered @ components
    return {
        'mean': mean,
        'centered': centered,
        'covariance': covariance,
        'eigenvalues': eigenvalues,
        'components': components,
        'scores': scores,
    }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
# Fraction of variance explained by PC_j = lambda_j / sum_k lambda_k. The total variance is
# trace(S) = sum of eigenvalues, which is the same in the original and PC coordinates (rotation).
# Results for seed 0: PC1 ≈ 0.894 (89.4%) and PC2 ≈ 0.106 (10.6%).
# (Population values: eigenvalues 2 ± sqrt(2.44) ≈ 3.56 and 0.44, i.e. 89% / 11%.)
# Interpretation: because the two features are strongly correlated, almost all of the spread lies
# along one direction. Projecting onto PC1 alone cuts the data from 2-D to 1-D (half the storage)
# and still keeps ~89% of the variance. The ~11% that is lost is the spread perpendicular to PC1,
# which is often mostly noise.
# Significance of dimensionality reduction: it means less computation and memory, easier
# visualization, less redundancy between correlated features, and some noise filtering.
# It also helps against the curse of dimensionality for downstream models.
# Trade-off: each component you drop loses its share of the variance (information), so
# reconstructions get less accurate. Keeping more components preserves more information but gives
# up compression. A common rule is to choose the smallest k whose cumulative explained variance
# passes a threshold (e.g. 90 to 95%), or to look for an "elbow" in the scree plot.
# Here k = 1 gives ~89%. That is a good trade if a 1-D summary is enough, but a 95% threshold would
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    ev = np.array(eigenvalues, dtype=float)
    return ev / ev.sum()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
# YOUR ANSWER HERE
# Figure, left panel: a scatter of the 500 samples (the Q2 visualization). The tilted elliptical
# cloud shows the positive correlation. PC1 (long arrow) and PC2 (short arrow) are drawn through
# the mean, scaled to 2 standard deviations (2*sqrt(lambda)), so arrow length reflects the spread
# in that direction. The axes use equal aspect, so the arrows look perpendicular as they truly are.
# Figure, right panel: a bar chart of the fraction of variance per component, with a cumulative
# line (Q4). PC1 dominates at ~89%, which is the visual argument for reducing 2-D -> 1-D.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=float)
    mean = np.asarray(result['mean'])
    eigenvalues = np.asarray(result['eigenvalues'])
    components = np.asarray(result['components'])
    fractions = explained_variance(eigenvalues)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={'width_ratios': [1.4, 1]})


    ax1.scatter(X[:, 0], X[:, 1], s=12, alpha=0.45, color='#4C72B0',
                edgecolors='none', label='samples')
    colors = ['#C44E52', '#DD8452']
    for j in range(components.shape[1]):
        vec = components[:, j] * 2 * np.sqrt(eigenvalues[j])
        ax1.annotate('', xy=mean + vec, xytext=mean,
                     arrowprops=dict(arrowstyle='-|>', color=colors[j % 2], lw=2.5))
        ax1.plot([], [], color=colors[j % 2], lw=2.5,
                 label=f'PC{j + 1} (λ={eigenvalues[j]:.2f}, {fractions[j]:.0%})')
    ax1.scatter(*mean, color='black', s=40, zorder=5, label='mean')
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.set_xlabel('Feature 1 ($x_1$)')
    ax1.set_ylabel('Feature 2 ($x_2$)')
    corr = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
    ax1.set_title(f'Correlated 2-D Gaussian data (r = {corr:.2f}) with principal directions')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(alpha=0.3)

    
    idx = np.arange(1, len(fractions) + 1)
    bars = ax2.bar(idx, fractions, color='#4C72B0', width=0.6, label='per component')
    ax2.plot(idx, np.cumsum(fractions), 'o-', color='#C44E52', label='cumulative')
    for b, f in zip(bars, fractions):
        ax2.text(b.get_x() + b.get_width() / 2, f + 0.02, f'{f:.1%}',
                 ha='center', va='bottom', fontsize=10)
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f'PC{i}' for i in idx])
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel('Principal component')
    ax2.set_ylabel('Fraction of total variance')
    ax2.set_title('Explained variance')
    ax2.legend(loc='center right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE
# # I completed all parts of the assignment individually (Part 1 linear algebra,
# Part 2 probability/MLE/MAP, Part 3 PCA derivation, implementation, and plots).
# Resources used: Lecture 1 (Intro) and Lecture 2 (Review) slides

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    np.set_printoptions(precision=4, suppress=True)

    print('=== Part 1 ===')
    AB, BA, ApBT, ABt, trA, trAB = matrix_operations()
    print('AB =\n', AB, '\nBA =\n', BA, '\n(A+B)^T =\n', ApBT, '\nAB^T =\n', ABt)
    print('trace(A) =', trA, ' trace(AB) =', trAB)
    vals, vecs = eigensystem()
    print('eigenvalues:', vals, '\neigenvectors (columns):\n', vecs)
    M = np.array([[2., -1.], [-1., 2.]])
    for i in range(len(vals)):
        print(f'  M v{i + 1} = {M @ vecs[:, i]},  lambda v{i + 1} = {vals[i] * vecs[:, i]}')
    C_inv = matrix_inverse()
    print('C^-1 =\n', C_inv, '\nC C^-1 =\n', np.array([[4., 7.], [2., 6.]]) @ C_inv)

    print('\n=== Part 2 ===')
    print('(c, E[X], Var[X]) =', distribution_moments())
    print('(P(A∩B), P(B|A), P(A∪B)) =', event_probabilities())
    print('(MLE, MAP) coin =', coin_estimates())
    print('(mu_MLE, sigma2_MLE) =', gaussian_mle())
    print('mu_MAP =', gaussian_map())

    print('\n=== Part 3 ===')
    X = generate_data()
    print('sample correlation:', np.corrcoef(X.T)[0, 1])
    res = pca(X)
    print('mean:', res['mean'])
    print('covariance:\n', res['covariance'])
    print('eigenvalues:', res['eigenvalues'])
    print('components (columns):\n', res['components'])
    print('explained variance:', explained_variance(res['eigenvalues']))

    import matplotlib.pyplot as plt
    fig = make_plots(X, res)
    fig.savefig('pca_results.png', dpi=150)
    print('Saved figure to pca_results.png')
    plt.show()
