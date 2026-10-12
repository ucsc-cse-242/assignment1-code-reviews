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
# YOUR ANSWER HERE
# When visualized in a scatter plot, the mean is the cordinates for the center of the distribution.
# If the identity matrix is used for the covarience, the distrubution is an even 'circle' shape, and
# scalar multiples of the identity matrix produce more or less variation. 
# Given a covariance matrix [[a,b], [c,d]] a and d represent independent variation and b and c represent
# co-variation. Some covarience matricies only produce a distribution over a single line. 
# [[1,1], [0,0]] produces a distribution over a horizontal line,
# and [[1,1], [2,2]] produces a distribution along a line with a line with a 2x slope.
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

    rng = np.random.default_rng(seed=seed) # set seed for repeatable generation
    # generate n samples on 2 dimensional distribution centered on mean, from covariance matrix 
    s = rng.multivariate_normal(mean, covariance, n_samples)

    return s


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
# Here, the first step is to esrimate the mean from the data and then use that to center the data.
# As shown in Q1, this eliminates a term and simplifies the equation for finding the covariance.
# The covarience matrix is determined by fist finding the varience in each column, then
# finding the covariance by taking the dot product of the two columns and dividing it by 
# the number of data points - 1.
# A covariance matrix can be constructed from these values, and the eigenvectors of this matrix
# can be used to project the data into a new coordinate system where each component is independent. 

# The returned "scores" matrix contains the full centered dataset projected in this new system
# The eigenvalues are reordered to be in decending order, and the eigenvector matrixc columns are 
# reordered to match the eigenvalues. Because of this, the "scores" matrix will always have the most
# variance along the first axis.

def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # YOUR CODE HERE
    # Assumes 2 dimensional dataset 
    n = len(X[:, 0])
    mean = ( np.mean(X[:, 0]), np.mean(X[:, 1]) ) # Find mean
    Xcentered = np.zeros((n,2))
    for i in range(n):
        Xcentered[i] = X[i] - mean # Populate new matrix with centered data

    var0 = np.var(Xcentered[:, 0]) # Get variance along each axis
    var1 = np.var(Xcentered[:, 1])

    covar = np.dot(Xcentered[:, 0], Xcentered[:, 1])
    covar = covar / (n-1)

    covariance = [ [var0, covar], [covar, var1]] # Reconstruct covariance matrix

    eigh_results = np.linalg.eigh(covariance) # get eigenvalues/vectors from covariance matrix
    eigenvalues = eigh_results.eigenvalues
    eigenvectors = eigh_results.eigenvectors

    id_sort = np.argsort(eigenvalues)[::-1] # eigenvalues in descending order
    eigenvalues = eigenvalues[id_sort]
    eigenvectors = eigenvectors[:, id_sort] # also sort eigenvactor columns to match
    scores = np.matmul(Xcentered, eigenvectors) # reorient the data to the new system

        
    return {
            "mean": mean,
            "centered": Xcentered,
            "covariance": covariance,
            "eigenvalues": eigenvalues,
            "components": eigenvectors,
            "scores": scores
            }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
# The explained variance ratios represent the proportion of the total varience 
# contained by each component. For example, for the default data set here the ratios are
# 0.89431527, 0.10568473. The ratios sum to 1 and so they can be though of as percents. 
# So 89% of the variance in this data is explained by the first component, and 11% by the second.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    ratios = np.zeros(len(eigenvalues))
    total = np.sum(eigenvalues)

    for i in range(len(eigenvalues)):
        ratios[i] = eigenvalues[i] / total

    return ratios


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
def make_plots(X=None, result=None):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    d = generate_data()
    plt.scatter(d[:, 0], d[:, 1 ], alpha=0.5)
    #plt.axis([-5,5,-5,5])
    plt.title("Gaussian Distribution (Assignment Default)")
    plt.show()

    d1 = generate_data(mean=[-8, 6], covariance=[[1, 0], [0, 1]])
    plt.scatter(d1[:, 0], d1[:, 1 ], alpha=0.5)
    #plt.axis([-5,5,-5,5])
    plt.title("Gaussian Distribution (Identity Matrix)")
    plt.show()

    d2 = generate_data(mean=[0,0], covariance=[[1, 1], [2, 2]])
    plt.scatter(d2[:, 0], d2[:, 1 ], alpha=0.5)
    plt.axis([-5,5,-5,5])
    plt.title("Gaussian Distribution (Improper)")
    plt.show()


    s = pca(d)
    print(s["components"])
    print(s["covariance"])

    plt.scatter(s["centered"][:, 0], s["centered"][:, 1 ], alpha=0.5)
    plt.axis([-5,5,-5,5])
    plt.title("Gaussian Distribution (Centered)")
    plt.show()



    plt.scatter(s["scores"][:, 0], s["scores"][:, 1 ], alpha=0.5)
    plt.axis([-5,5,-5,5])
    plt.title("Scores with Components")

    #swap = s["centered"][:, [1, 0]]
    n = len(s["centered"][:, 0])
    s0 = np.matmul(s["centered"], s["components"][:, 0])
    plt.scatter(s0, np.zeros(n), alpha=0.5, color="blue")
    s1 = np.matmul(s["centered"], s["components"][:, 1])
    plt.scatter(np.zeros(n), s1, alpha=0.5, color="green")
    plt.legend(["Scores", "First Component", "Second Component"])
    plt.show()

    return


# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    import matplotlib.pyplot as plt
    #s = generate_data(mean=[-100, 5], covariance=[[-0.2, 1.5], [1.5, 7]])
    make_plots()


    # s = generate_data()

    # d = pca(s)
    # print(d["mean"])
    # print(d["covariance"])
    # print(" ")
    # print(d["components"])

    # print(explained_variance(d["eigenvalues"]))
    # # e = pca(d["scores"]) # demonstrates no more covarience
    # # print(e["covariance"])


    # plt.scatter(d["centered"][:, 0], d["centered"][:, 1 ], alpha=0.5)
    # plt.axis([-5,5,-5,5])
    # plt.title("Gaussian Distribution")
    # plt.show()


    # s = d["scores"]
    # plt.scatter(s[:, 0], s[:, 1 ], alpha=0.5)
    # plt.axis([-5,5,-5,5])


    # s0 = np.matmul(d["centered"], d["components"][0])
    # plt.scatter(s0, np.zeros(len(s[:, 0])), alpha=0.5, color="blue")
    # #plt.axis([-5,5,-5,5])
    # #plt.title("Gaussian Distribution altered")
    # s1 = np.matmul(d["centered"], d["components"][1])
    # plt.scatter(np.zeros(len(s[:, 0])), s1, alpha=0.5, color="red")
    # plt.show()
    pass
