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
    if mean is None:
        mean = [2., -1.]
    if covariance is None:
        covariance = [[3., 1.2], [1.2, 1.]]
    
    set_seed = np.random.default_rng(seed)
    data = set_seed.multivariate_normal(mean=mean, cov=covariance, size=n_samples)
    return data

# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # numpy PCA
    mean = np.mean(X, axis=0)
    # x - uT, from our expected value expression
    centered = X - mean
    # divisor to adjust the covariance
    divisor = X.shape[0] - 1
    cov = (centered.T.dot(centered)) / divisor # (centered.T * centered) transpose flipped from equation on slides
    # output matrix: 2 x 2, not 500 x 500. We go along the feature dimension 

    # simple oneliner for eigenvectors and eigenvalues. eigh is for symmetric matrices
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # IMPORTANT: eigh returns in ascending order, here we reverse
    eigenvalues = eigenvalues[::-1]
    # Flip the columns, leave the rows intact.
    # Function returns eigenvectors along columns. 
    # Typical math convention (did not know that or had forgotten it)
    eigenvectors = eigenvectors[:, ::-1]
    # PCA scores, with the corrected ordering (np.linalg.eigh sorts in ascending order)
    scores = centered.dot(eigenvectors)

    pca_return = {}
    pca_return['mean'] = mean
    pca_return['centered'] = centered
    pca_return['covariance'] = cov 
    pca_return['eigenvalues'] = eigenvalues
    pca_return['components'] = eigenvectors
    pca_return['scores'] = scores

    return pca_return


def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    total_variance = np.sum(eigenvalues)
    return eigenvalues / total_variance

# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# PC1 is the dominant component, contributing to 89.4 percent of the total variance. This implies that 
# the data is highly correlated, and can be modeled well with a single component.
# We would sacrifice 10.6 percent of the variance. In the limit, with a ton of data,
# and a constrained amount of compute, many people at banks probably make this tradeoff.
# The significance of dimensionality reduction is it allows us to see that the data
# is mostly correlated along a single component, and that the other eigenvector, the other
# "true" direction in the space, is less important here.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes[0].plot(X[:, 0], X[:, 1], '.', color="red", alpha=0.5)
    axes[0].axis('equal')
    axes[0].set_title('Data')
    mean = result['mean']
    comps = result['components']
    ratios = explained_variance(result['eigenvalues'])
    s = 2 * np.sqrt(result['eigenvalues'])   

    axes[0].quiver(mean[0], mean[1],         
                s[0] * comps[0, 0], s[0] * comps[1, 0],  
                angles='xy', scale_units='xy', scale=1,
                color='blue', label=f"PC1 ({ratios[0]*100:.1f}%)")
    axes[0].quiver(mean[0], mean[1],         
                s[1] * comps[0, 1], s[1] * comps[1, 1],  
                angles='xy', scale_units='xy', scale=1,
                color='green', label=f"PC2 ({ratios[1]*100:.1f}%)")
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')
    axes[0].legend()
    axes[0].grid()

    axes[1].bar(['PC1', 'PC2'], ratios)
    axes[1].set_xlabel('Principal component')
    axes[1].set_ylabel('Fraction of total variance')
    axes[1].set_title('Explained variance')

    fig.tight_layout()
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
# Internet, and claude for final correction. Claude caught a bugged column reversal.
# Good review from previous classes, had to use some youtube on PCA.

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.

    import matplotlib.pyplot as plt
    data = generate_data()
    pca_out = pca(data)
    fig = make_plots(data, pca_out)

    plt.show()

    data2 = generate_data(mean=[-132, 32], covariance=[[1, -10], [-1, 1]])
    pca_out = pca(data2)
    fig = make_plots(data2, pca_out)

    plt.show()
