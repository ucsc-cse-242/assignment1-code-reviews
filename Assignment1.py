"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# YOUR ANSWER HERE
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    raise NotImplementedError()


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    raise NotImplementedError()


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE

# P3 Q2: Describe the Gaussian generator and observed correlation.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # The default mean and covariance are the ones given in the assignment.
    if mean is None:
        mean = np.array([2.0, -1.0])
    if covariance is None:
        covariance = np.array([[3.0, 1.2], [1.2, 1.0]])
    # The np.random.default_rng(seed) creates a reproducible random number generator.
    rng = np.random.default_rng(seed)
    # The rng.multivariate_normal function draws samples from a multivariate normal distribution with the specified mean and covariance.
    X = rng.multivariate_normal(mean=mean, cov=covariance, size=n_samples)
    return X


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # Calculate the mean of the data along each feature.
    mean = X.mean(axis=0)
    # Center the data by subtracting the mean from each observation.
    centered = X - mean
    # The covariance matrix is computed as the dot product of the transposed centered data and the centered data, divided by (n-1), where n is the number of samples.
    covariance = (centered.T @ centered) / (X.shape[0] - 1)

    # Compute the eigenvalues and eigenvectors of the covariance matrix using NumPy's linear algebra module.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # Sort the eigenvalues in descending order and reorder the eigenvectors accordingly.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    # Calculate the principal components by reordering the eigenvectors according to the sorted eigenvalues.
    components = eigenvectors[:, order]
    # Calculate the scores by projecting the centered data onto the principal components.
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
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # Calculate the explained variance fractions by dividing each eigenvalue by the sum of all eigenvalues.
    return eigenvalues / eigenvalues.sum()

import matplotlib.pyplot as plt

# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    mean = result['mean']
    components = result['components']
    eigenvalues = result['eigenvalues']

    # Figure 1: scatter plot of the generated Gaussian samples (Q2).
    fig_scatter, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], s=16, alpha=0.7, color='#096BFF')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'Scatter plot of the Gaussian samples (correlation = {correlation:.3f})')
    ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.8, alpha=0.5)

    # Figure 2: the data with the principal component directions (Q3).
    fig_pca, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X[:, 0], X[:, 1], s=16, alpha=0.7, color='#096BFF', label='samples')

    for i in range(len(eigenvalues)):
        direction = components[:, i] * np.sqrt(max(eigenvalues[i], 0.0))
        ax.arrow(
            mean[0] - direction[0], mean[1] - direction[1],
            2 * direction[0], 2 * direction[1],
            color='darkorange', width=0.015, head_width=0.12, length_includes_head=True,
            label=f'PC{i + 1}' if i == 0 else None
        )

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Original data with principal component directions')
    ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.8, alpha=0.5)
    ax.legend(loc='best', frameon=False)

    # Figure 3: explained variance fractions (Q4).
    fractions = explained_variance(eigenvalues)
    labels = [f'PC{i + 1}' for i in range(len(fractions))]
    fig_var, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, fractions, color='#096BFF')
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Fraction of total variance')
    ax.set_title('Explained variance by principal component')
    ax.set_ylim(0, 1)

    return (fig_scatter, fig_pca, fig_var)


# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE

if __name__ == '__main__':
    # Generate the data and inspect its correlation structure.
    X = generate_data()
    correlation = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
    print(f"Correlation: {correlation}")

    # Run PCA on the generated data and store the results in a dictionary.
    result = pca(X)
    print(f"Eigenvalues: {result['eigenvalues']}")
    print(f"Explained variance fractions: {explained_variance(result['eigenvalues'])}")

    # Build and display the requested figures.
    figures = make_plots(X, result)
    for name, figure in zip(('p3_scatter', 'p3_components', 'p3_explained_variance'), figures):
        figure.savefig(f'{name}.png', dpi=150, bbox_inches='tight')
    plt.show()
