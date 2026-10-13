"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np

# For PART 1 and 2, see handwritten PDF

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE
# See Handwritten PDF submission for work.

# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
# See Jupyter Notebook PDF for the scatter plots and the correation analysis. 
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    if mean is None:
        mean = np.array([2, -1])
    
    if covariance is None:
        covariance = np.array([[3,1.2],[1.2,1]])

    rng = np.random.default_rng(seed)
    data = rng.multivariate_normal(mean, covariance, size=n_samples)
    return data



# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
# See Jupyter Notebook PDF for the plot and analysis. 
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # YOUR CODE HERE

    # Step 1: Find the mean of each feature and center the data.
    # X_centered = X - mean
    # (X - mean) used to define the covariance matrix in Q1.
    mean = np.mean(X, axis=0)
    centered = X - mean

    # Step 2: Compute the sample covariance matrix.
    # covariance = (1 / (n - 1)) * (X_centered).T @ X_centered
    # For a sample, we estimate Sigma using:
    # S = (1 / (n - 1)) * X_centered.T @ X_centered
    n = X.shape[0]
    covariance = (centered.T @ centered) / (n - 1)

    # Step 3: Find the eigenvalues and eigenvectors (u) of the covariance matrix.
    # From Q1, Var(u^T X) = u^T Sigma u.
    # PCA finds unit directions u that maximize this projected variance.
    # These directions are the eigenvectors of the covariance matrix,
    # where Sigma*u = lambda*u.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Step 4: Sort the eigenvalues from largest to smallest and reorder the
    # corresponding eigenvectors. The eigenvector corresponding to the largest
    # eigenvalue is the first principal componen with the maximized variance.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order]

    # Project the centered observations onto the principal component directions. 
    # These are the projections u^T X discussed in Q1.
    # scores = X_centered @ components
    scores = centered @ components

    return {
        "mean": mean,
        "centered": centered,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "components": components,
        "scores": scores
    }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
# See Jupyter Notebook PDF for the analysis. 
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE

    # Each eigenvalue represents the variance explained by its
    # corresponding principal component. Divide each eigenvalue by
    # the sum of all eigenvalues to get the proportion of total
    # variance explained by each principal component.
    total_variance = np.sum(eigenvalues)
    variance_fractions = eigenvalues / total_variance

    return variance_fractions

# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    pass
