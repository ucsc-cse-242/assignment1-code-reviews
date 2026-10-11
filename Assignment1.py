"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np
import matplotlib.pyplot as plt

# All questions coded for have also been done by hand in the pdf (SEE PDF) 
# PART 1 (20 points), Q1: Show computations for the given A and B.
# YOUR ANSWER HERE
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    if A is None:
        A = [[3,4], [2,1]]
    if B is None:
        B = [[1,2], [0,-1]]

    A = np.array(A, dtype = float)
    B = np.array(B, dtype = float)
    AB = A @ B
    return AB, B @ A, (A+B).T, A @ B.T, np.trace(A), np.trace(AB)
    #raise NotImplementedError()


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE
    if M is None:
        M = [[2, -1], [-1, 2]]
    M = np.array(M, dtype = float)
    #using numpy's eigen solver
    values, vectors = np.linalg.eigh(M)
    return values, vectors 
    #raise NotImplementedError()


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE
#SEE PDF


# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    if C is None: 
        C = [[4, 7], [2, 6]]
    C = np.array(C, dtype = float)
    return np.linalg.inv(C)
    #raise NotImplementedError()


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    c = np.exp(-rate) # from sum of rate^k/k! = e^rate
    mean = rate 
    variance = rate 
    return c, mean, variance
    #raise NotImplementedError()


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    p_ab = p_a_given_b * p_b #p(A and B) = P(A|B) P(B)
    p_b_given_a = p_ab / p_a # p(B|A) = P(A and B) / P(A) 
    p_union = p_a + p_b - p_ab #p(A or B) = P(A) + P(B) -P(A and B)
    return p_ab, p_b_given_a, p_union
    #raise NotImplementedError()


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    mle = heads/tosses 
    #posterior = theta ^(heads +1) * (1-theta)^(tosses-heads))
    #setting its derivative to 0 
    map_est = (heads + 1) / (tosses + 1)
    return mle, map_est
    #raise NotImplementedError()


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    if data is None:
        data = [1, 3, 5, 7]
    x = np.array(data, dtype=float)
    mean = np.mean(x)
    variance = np.mean((x - mean) ** 2)   # divide by n (the MLE), not n-1
    return mean, variance
    #raise NotImplementedError()


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    data_weight = 1 / observation_variance
    prior_weight = 1 / prior_variance
    return (data_weight * x + prior_weight * prior_mean) / (data_weight + prior_weight)
    #raise NotImplementedError()


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE
# SEE PDF

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE
# SEE PDF

# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
# Description - The scatter plot shows a clear positive correlation between Feature 1 and Feature 2, 
# as the features tend to increase together. This produces an elongated diagonal point cloud. 
# The observed pattern is consistent with the positive covariance used to generate the synthetic data.

def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    if mean is None:
        mean = [2, -1]
    if covariance is None: 
        covariance = [[3, 1.2], [1.2, 1]]
    mean = np.array(mean, dtype = float)
    covariance = np.array(covariance, dtype = float)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size = n_samples)
    #raise NotImplementedError()


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
# After centering the data and computing the covariance matrix, the eigenvectors of the covariance
# matrix give the principal component directions. PC1 lies along the direction of greatest variation in the
# data, while PC2 is perpendicular to PC1 and captures the remaining variation. The principal component
# directions are shown in the left panel of Figure 1.

def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # YOUR CODE HERE
    X = np.array(X, dtype = float) 
    n = X.shape[0]

    #1. center the data 
    mean = X.mean(axis = 0)
    centered = X - mean 

    #2. covariance matrix 
    covariance = centered.T @ centered / (n - 1)

    #3. eigenvectors 
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    #4. eigh sorts smallest first, so flip 
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0, None)
    components = eigenvectors[:, order]

    #5. project the data 
    scores = centered @ components

    return {"mean" : mean, "centered" : centered, "covariance" : covariance, "eigenvalues" : eigenvalues, "components" : components, "scores" : scores}
    #raise NotImplementedError()


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
# PC1 explains approximately 89.4% of the total variance, while PC2 explains approximately 10.6%.
# Therefore, the dataset could be reduced from two dimensions to one by retaining only PC1 while
# preserving most of the variance. This demonstrates the trade-off in dimensionality reduction: using
# fewer dimensions simplifies the representation but results in some loss of information, with
# approximately 10.6% of the variance discarded in this case.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    ev = np.array(eigenvalues, dtype = float)
    return ev / ev.sum()
    #raise NotImplementedError()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    X = np.asarray(X)
    mean = result["mean"]
    fractions = explained_variance(result["eigenvalues"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    #left data with principal directions through the mean 
    ax1.scatter(X[:, 0], X[:, 1], s=10, alpha=0.4, label="samples")
    colors = ["tab:red", "tab:green"]
    for i in range(2):
        direction = result["components"][:, i]
        length = 2 * np.sqrt(result["eigenvalues"][i])
        ends = np.array([mean - length * direction, mean + length * direction])
        ax1.plot(ends[:, 0], ends[:, 1], color=colors[i], lw=3,
                 label=f"PC{i+1} (eigenvalue {result['eigenvalues'][i]:.2f})")
    ax1.scatter(*mean, color = "black", zorder = 3, label = "mean")
    ax1.set_aspect("equal")
    ax1.set_xlabel("Feature 1 (x1)")
    ax1.set_ylabel("Feature 2 (x2)")
    ax1.set_title("Data with principal component directions")
    ax1.legend()

    #right explained variance bar chart 
    bars = ax2.bar(["PC1", "PC2"], fractions, color=colors)
    for bar, f in zip(bars, fractions):
        ax2.text(bar.get_x() + bar.get_width() / 2, f + 0.02, f"{f:.1%}", ha="center")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Fraction of total variance")
    ax2.set_title("Explained variance by component")

    fig.tight_layout()
    return fig
    #raise NotImplementedError()


# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE
# Tasks completed: all of Parts 1 to 3. Derivations and proofs are in PDF, all functions are implemented in this file, 
# figures are saved as: pca_plots.png.
# Collaborations : no classmates 
# AI: per course policy; conversation log submitted to Canvas, I used claude to walk through the derivations step by step, explain steps I didn't
# understand (e.g. summation index notation, the E[X^2] calculation), and provide basic process to approach functions for this file 
# understand, and the code for the functions in this file, which I implemented, debugged, and checked against my hand calculations.
# Other resources: CSE 242 Lecture 1-3 slides, NumPy and Matplotlib docs.

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    print(matrix_operations())
    vals, vecs = eigensystem()
    print(vals)
    print(vecs)
    print(matrix_inverse())
    print(distribution_moments())
    print(event_probabilities())
    print(coin_estimates())
    print(gaussian_mle())
    print(gaussian_map())
    X = generate_data()
    result = pca(X)
    print(result["covariance"])
    print(result["eigenvalues"])
    print(explained_variance(result["eigenvalues"]))
    fig = make_plots(X, result)
    fig.savefig("pca_plots.png", dpi = 150)
    # This block does not run when the grader imports your functions.
    #pass