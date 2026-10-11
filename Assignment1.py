"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np
import matplotlib.pyplot as plt


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
# Looking at the results of a call to generate_data() using all default parameter values, we can observe 
# the expected behavior from the default mean vector and covariance matrix. We can see that the data does indeed
# tend to center around an x of 2 (for f1) and a y of -1 (for f2). Additionally, we can observe that the data seems
# to have a strong positive correlation, consistent with the positive of covariance given in the default matrix of 1.2. 
# Lastly, we can see that the variance for each feature seems roughly reflected in the resulting scatter plot. 
# f1, which has an input variance of 3, spreads much more than that of f2, which has a variance of only 2. 
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    if not mean:
        mean = [2,-1]
    if not covariance:
        covariance = [[3,1.2],[1.2,1]]
    random = np.random.default_rng(seed)
    dataset = random.multivariate_normal(mean, covariance, size=n_samples)
    plt.scatter(dataset[:, 0], dataset[:, 1], alpha=0.5)
    plt.xlabel("gaussian f1")
    plt.ylabel("gaussian f2")
    plt.title("gaussian feature correlation")
    plt.show()

    return dataset


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# Centering the data around zero makes it easier to see how much the data varies in different directions. The covariance
# is calculated in order to find sigma, which is then used to find the principal components. To find the principal components,
# which are vectors in which the variance is greatest, we can use the derivation from P3 Q1 that maximizes u.T@Sigma@u. Since 
# we know @Sigma@ to be the covariance matrix, our earlier calculation comes in handy. By definition, an eigenvector u of the
# covariance matrix @Sigma@ satisfies @Sigma@*u = lambda*u. Because of this, as long as we can find the eigenvectors of the 
# highest eigenvalue, we'll be able to find the eigenvectors of the highest variance, by sorting from highest eigenvalue. The plot
# shows the data transformed to be plotted along the vectors that produces the highest variances and produce zero covariance. We 
# can see this demonstrated in the produced plot, as the plot shows and blob centered at zero without an obvious trend or correlation
# between the axes. 
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # Center your dataset (mean subtraction)
    mean = np.mean(X, axis=0)
    centered = X - mean

    # Compute the covariance matrix
    covariance = np.cov(centered, rowvar=False)

    # Find eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Select the principal components corresponding to the highest eigenvalues
    # sort into descending eigenvalues, then sort eigenvalues and eigenvectors in the same order, then calculate PCA scores
    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indices]
    components = eigenvectors[:, indices]
    scores = centered @ components

    # scatter plot with selected principal components
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.5)
    plt.xlabel("principal component 1")
    plt.ylabel("principal component 2")
    plt.title("data plotted against principal components 1 and 2")
    plt.show()

    return {
        "mean": mean,
        "centered": centered,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "components": components,
        "scores": scores
    }



# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# The variance fractions explain what fraction of the total variance a principal component captures. So if PC1 captured
# 90% of the variance and PC2 the other 10%, getting rid of PC2 would also get rid of the direction that accounts for
# 10% of the dataset's variation. The good thing about reducing by a dimension is that should be able to capture most of a dataset's 
# variation with fewer axes, however you do lose a bit of the variance represented in the subtracted axis. Similarly, not reducing the 
# dimensions leads to more complexity in the representation of the dataset, however you can get the full picture of the variance. 
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    return eigenvalues / np.sum(eigenvalues)


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# I had the original functions plot the figures before returning. Hopefully that's ok!
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# REFLECTION: contributions, tasks completed, and external resources used.
# I had some help from (not a class tutor, but my friend) Evan Chen to help re-teach me some of the linear algebra
# basics, however all solutions and answers for that section were done by me and only me. In part 2, all of the work
# was done by me, some Google-ing and referencing class slides were done because I forgot stuff like how to calculate
# the MLE, MAP, and what the formula for the Gaussian PDF was (it's been a while). In part 3, I used ChatGPT mostly
# to ask about how to generate datasets according to a certain distribution and also referenced its response on how to 
# calculate the covariance matrix in numpy. ChatGPT was also used for the interpretation of the process to find the PCA
# scores, since it wasn't immediately obvious to me what the link between the steps and and derivation were, as well 
# as how to interpret the variance fractions. 

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # this block wasn't used, since I wrote most of the assignment by hand. Testing functions was done through the Python
    # interactive terminal. 
    # This block does not run when the grader imports your functions.
    pass
