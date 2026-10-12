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
# PART 1 (20 points) DONE FULLY BY HAND, SEE ATTACHED PDF

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
# PART 2 (50 points) DONE FULLY BY HAND, SEE ATTACHED PDF

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
#Q1 DONE BY HAND

# P3 Q2: Describe the Gaussian generator and observed correlation.

# YOUR ANSWER HERE
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    #setting mean to the defaults if one is not given
    if mean is None:
        mean = [2.0, -1.0]
    #setting covariance to defaults if one isn't given
    if covariance is None:
        covariance = [[3.0, 1.2], [1.2, 1.0]]
    
    #getting number to make a synthetic gaussian with (reproducible)
    rng = np.random.default_rng(seed)

    #draws n samples from the gaussian given by our parameters and returns them
    return rng.multivariate_normal(mean, covariance, size=n_samples)

#describing the gaussian generator: The generator draws 500 samples from a 2D Gaussian (w/ mean [2, -1], and covariance ([3, 1.2], [1.2, 1]). It uses a seeded NumPy generator to make results reproducible.

#describing the observed correlation: The features are positively correlated (sample correlation 0.70). This means that when one is high, the other also tends to be high. This shows up in the scatter plot as a tilted ellipse rising to the right, centered near (2, -1).

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
    # YOUR CODE HERE
    #better formatting to deal w/ different types of input
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    #centering dataset
    #gets mean of the features
    mean = X.mean(axis=0)
    #subtracting mean from each row to center the dataset
    centered_data = X - mean

    #computing covariance matrix
    #sums products of the centered values for each feature pair
    sum_products = centered_data.T @ centered_data
    #dividing by n - 1 to convert sums into sample covariances
    covariance_matrix = sum_products / (n - 1)

    #finding eigenvalues and eigenvectors of covariance matrix
    #numpy function to get the eigenvalues and eigenvectors of the matrix
    result = np.linalg.eigh(covariance_matrix)  
    #getting the resulting eigenvalues from the calculation
    eigenvalues = result[0]
    #getting the resulting eigenvectors from the calculation
    eigenvectors = result[1] 

    #selecting principal components corresponding to highest eigenvalues
    #gets index order that sorts eigenvalues from highest to lowest
    order = np.argsort(eigenvalues)[::-1]
    #stores eigenvalues in the new order
    eigenvalues = eigenvalues[order]
    #gets the principal components using the ordered eigenvector
    components = eigenvectors[:, order]
    
    #handles rank deficienct edge case by clipping tiny negative values
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    #projecting our centered data onto the principal components
    scores = centered_data @ components

    return {
        'mean': mean,
        'centered': centered_data,
        'covariance': covariance_matrix,
        'eigenvalues': eigenvalues,
        'components': components,
        'scores': scores,
    }

    #Link centering/covariance/eigendecomposition to the derivation and explain the plot: Centering computes x - mu, the distance from the mean that Q1's variance and covariance formulas use. The covariance matrix is the sample version of sigma from Q1. Q1 showed the variance along a direction u is u^T*sigma*u, which is largest when u is an eigenvector of sigma. So eigenvectors are the principal directions, and their eigenvalues are the variances along those directions.


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    #converts input to a float array for better formatting
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    
    #getting total variance by summing the eigenvals
    total_var = eigenvalues.sum()
    #calculating fraction of the total variance captured by each component by dividing the eigenvalues by the total variance
    return eigenvalues / total_var

#The eigenvalues are 3.46 and 0.41 (with total variance 3.87), so PC1 explains 89.4% of the variance and PC2 explains 10.6% due to PC1 having the much larger eigenvalue. This is because the two features are correlated, causing the data to stretch mostly in the direction of PC1.
    
# Dimensionality reduction would be important because projecting onto PC1 alone reduces the data from 2 features to 1 while keeping 89.4% of the variance. This makes data simpler, faster to go through, and easier to visualize.

#The tradeoff is that 10.6% of variance that came from PC2 is lost. So while the data would gain all the improvements described before, there is a chance that some really important information was lost from PC2. In the end, it depends on how much data we are willing to lose as a tradeoff for easier to access, simpler data. The quality of the data being sacrificed is also something to take note of (noise vs quality data).

# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    import matplotlib.pyplot as plt

    #extract required data from PCA result
    X = np.asarray(X, dtype=float)
    mean = result['mean']
    eigenvalues = result['eigenvalues']
    components = result['components']
    fractions = explained_variance(eigenvalues)

    #creates the figure showcasing both graphs side by side
    fig, (ax_data, ax_var) = plt.subplots(1, 2, figsize=(12, 5))

    #the left panel, featuring data with the principal directions drawn through the mean
    ax_data.scatter(X[:, 0], X[:, 1], s=10, alpha=0.4, label='samples')
    ax_data.scatter(*mean, color='black', marker='x', s=80, zorder=3, label='mean')
    colors = ['tab:red', 'tab:green']

    #draws a line for each principal component
    for i in range(components.shape[1]):
        direction = components[:, i] * 2 * np.sqrt(eigenvalues[i])
        start = mean - direction
        end = mean + direction
        ax_data.plot([start[0], end[0]], [start[1], end[1]], color=colors[i], lw=2.5,
                     label=f'PC{i + 1} ({fractions[i]:.1%} of variance)')

    #labeling axes and plot
    ax_data.set_xlabel('Feature 1 ($x_1$)')
    ax_data.set_ylabel('Feature 2 ($x_2$)')
    ax_data.set_title('Samples and principal directions')
    
    #formatting
    ax_data.set_aspect('equal')
    ax_data.legend(loc='upper left')

    #the right panel, showing the fraction of variance explained by each component
    #labeling bars
    labels = [f'PC{i + 1}' for i in range(len(fractions))]
    #drawing bars
    bars = ax_var.bar(labels, fractions, color=colors[:len(fractions)])
    #filling out % per bar
    ax_var.bar_label(bars, labels=[f'{f:.1%}' for f in fractions])
    
    #labeling axes and plot
    ax_var.set_xlabel('Principal component')
    ax_var.set_ylabel('Fraction of total variance')
    ax_var.set_title('Explained variance')
   
    #setting limit to y val
    ax_var.set_ylim(0, 1.05)
    
    #formatting
    fig.tight_layout()

    #returning figure to use in main
    return fig

#explaining plots:

#left plot: x axis is feature 1 and y axis is featrure 2. This plot shows the samples and principal directions of these samples. he data forms a tilted ellipse, showing a positive correlation of 0.7. PC1 (red) runs along the long axis where the data spreads most, and PC2 (green) runs across it at a right angle.

#right plot: Each bar is one principle component, with the y axis representing how much of the variance they explain. PC1 explains 89.4% of the variance and PC2 only 10.6%, so keeping just PC1 preserves most of the information.


# REFLECTION: contributions, tasks completed, and external resources used.
#claude used for assistance, transcript attached in submission

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    #generating synthetic data for gaussian point
    X = generate_data()
    #running PCA on data
    res = pca(X)

    #printing results
    print("Sample correlation:", np.corrcoef(X.T)[0, 1])
    print("Eigenvalues:", res['eigenvalues'])
    print("Explained variance:", explained_variance(res['eigenvalues']))

    #building the plots
    fig = make_plots(X, res)
    #saving figure
    fig.savefig('pca_plots.png', dpi=150)
    #displaying figure
    plt.show()
    # This block does not run when the grader imports your functions.
    pass
