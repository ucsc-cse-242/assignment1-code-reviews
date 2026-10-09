"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# YOUR ANSWER HERE
#calculations also in submitted pdf
def matrix_operations(A:np.array, B:np.array):
    """Returns AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Assumes that inputs are square real matrices of matching size.
    """

    assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray)
    assert A.shape[0] == A.shape[1] and B.shape[0] == B.shape[1] and A.shape[0] == B.shape[0]

    return A @ B, B @ A, (A+B).T, A @ B.T, np.trace(A), np.trace(A @ B)


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
# calculations also submitted in pdf
def eigensystem(M:np.array):
    """
    Return (eigenvalues, eigenvectors) for real symmetric M.
    """

    # YOUR CODE HERE
    eigenvalues, eigenvectors =  np.linalg.eig(M)
    return (eigenvalues, eigenvectors)

# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE
# check submitted pdf for proof

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
# check pdf for full derivation
def matrix_inverse(C:np.array):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    assert np.linalg.det(C) != 0 # is invertable?
    assert C.shape[0] == C.shape[1] # is square?

    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
#poisson distribution the rate is the mean and variance. 
# check submitted pdf for full derivation
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    
    return np.exp(-rate), rate, rate


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
# these are the simple formulas 
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    P_a_intersect_b = p_a_given_b * p_b
    P_b_given_a = (p_a_given_b * p_b) / p_a
    P_a_union_B = p_a + p_b - p_a_given_b * p_b
    return P_a_intersect_b, P_b_given_a, P_a_union_B


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """

    # this was also solved by hand with the full derivation

    # first create variables for theta and 1-theta
    theta = np.polynomial.Polynomial([0,1])
    one_minus = np.polynomial.Polynomial([1,-1])

    # our MLE function is writen as
    MLE_func = theta**heads * one_minus**(tosses-heads) 

    # our MAP function is the MLE function times our prior
    MAP_func = MLE_func * np.polynomial.Polynomial([0, 2]) # prior is 2*theta

    # calculate the derivative of a function and find the max value
    def maximize(func):
        derv = func.deriv().roots()
        cand = [0.0, 1.0] + [r for r in derv.real if 0 <= r <= 1] 
        cand = np.array(cand)
        return float(cand[np.argmax(func(cand))])

    # return the max value of the MLE and MAP functions
    return maximize(MLE_func), maximize(MAP_func)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
def gaussian_mle(data=[1,3,5,7]):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    # this was also solved by hand with the full derivation
    return (np.mean(data), np.var(data))


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    # this was solved by hand in the pdf
    raise NotImplementedError()

# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE
# check submitted pdf for proof

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE
# check submitted pdf for proof


# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
# this function generates a random points with the mean x = 2 y = -1. The points
# generated distribution is a 2-d gaussian the majority of the variance is in the X 
# direction as its variance is 3 and in the y direction the variance is 1. There
# is a moderately strong positive linear relationship 
def generate_data(n_samples=500, seed=0, mean=[2,-1], covariance=[[3,1.2],[1.2,1]]):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    rng = np.random.default_rng(seed=seed)
    samples = rng.multivariate_normal(mean=mean, cov=covariance, size=n_samples)
    return samples


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
    n = X.shape[0]

    # center the data
    mean = np.mean(X, axis=0)
    centered = X - mean

    # calculate the covariance matrix
    cov = (1/(n-1)) * (centered.T @ centered) # no need to worry about subtracting the mean the data is already centered

    # calculate eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(cov)

    # order the eigenvalues and eigenvectors by the eigenvalue magintude
    order = np.argsort(evals, axis=0)[::-1]

    evals = evals[order]
    components = evecs[:,order]

    scores = centered@components

    return {"mean":mean,
            "centered":centered,
            "covariance":cov,
            "eigenvalues":evals,
            "components":components,
            "scores":scores}

# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return eigenvalues / np.sum(eigenvalues)


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
# the figures show the direction captured by PC1 and PC2
# PC1 captures the most variance and is a vector pointing towards the origin
# PC2 captures the remaining variance and is a vector pointing towards the x-axis-ish
# it can be observed the PCA is doing its job, PC1 captures the most variance ~90% and by visual inspection this relationship is true, the majority of the variance of the data is clearly 
# in the direction of PC1
# PC2 explain the rest of the variance in the data it by visual inspection the direction of PC2 clearly explains the remaining variance in the data 
def make_plots(X, result):
    """Returns a sequence of Matplotlib Figures.

    Plots samples and principal directions through their mean, and explained
    variance fractions.
    """
    # YOUR CODE HERE
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=float)
    mean = result["mean"]
    components = result["components"]
    eigenvalues = result["eigenvalues"]
    fractions = eigenvalues / eigenvalues.sum()
    N = len(eigenvalues)

    # Figure 1: samples + principal directions through the mean
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.scatter(X[:, 0], X[:, 1], s=10, alpha=0.4, label="samples")
    colors = ["tab:red", "tab:green"]
    for i in range(N):
        d = components[:, i] * 2 * np.sqrt(max(eigenvalues[i], 0))
        ax1.quiver(mean[0], mean[1], d[0], d[1],
                   angles="xy", scale_units="xy", scale=1,
                   color=colors[i], label=f"PC{i+1} ({fractions[i]:.1%})")
    ax1.scatter(mean[0], mean[1], color="black", marker="x", s=60, label="mean")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_title("Samples and principal directions")
    ax1.set_aspect("equal")
    ax1.legend()

    # Figure 2: explained variance fractions
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.bar(np.arange(1, N + 1), fractions)
    ax2.set_xticks(np.arange(1, N + 1))
    ax2.set_xlabel("Principal component")
    ax2.set_ylabel("Fraction of variance explained")
    ax2.set_ylim(0, 1)
    ax2.set_title("Explained variance")

    return [fig1, fig2]

# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE
"""In this assignment we went through the basics of linear algebra and probability. Then finally combined these concepts to create the PCA algorithm.
External resources were used in this assignment to help correct code errors and explain questions. All chat logs can be found at these links


https://claude.ai/share/134b3ae0-47bc-4142-80a1-3270006f2bb9
https://claude.ai/share/ace6dfd2-1169-4f27-918a-a0879b66413d
https://claude.ai/share/ace6dfd2-1169-4f27-918a-a0879b66413d
https://claude.ai/share/20fc0ebe-7f92-40a4-8245-48d9289c6bed
"""

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    
    import matplotlib.pyplot as plt

    # PART 1
    # Q1
    A = np.array([[3, 4], [2, 1]])
    B = np.array([[1, 2], [0, -1]])
    AB, BA, ApB_T, ABT, trA, trAB = matrix_operations(A, B)
    print("P1 Q1")
    print("AB =\n", AB)
    print("BA =\n", BA)
    print("(A+B)^T =\n", ApB_T)
    print("AB^T =\n", ABT)
    print("trace(A) =", trA)
    print("trace(AB) =", trAB)

    # Q2
    M = np.array([[2, -1], [-1, 2]])
    evals, evecs = eigensystem(M)
    print("\nP1 Q2")
    print("eigenvalues =", evals)
    print("eigenvectors (columns) =\n", evecs)
    for i in range(len(evals)):
        v = evecs[:, i]
        print(f"lambda={evals[i]:.4f}: Mv =", M @ v, " lambda*v =", evals[i] * v,
              " equal:", np.allclose(M @ v, evals[i] * v))

    # Q4
    C = np.array([[4, 7], [2, 6]])
    print("\nP1 Q4")
    print("det(C) =", np.linalg.det(C))
    C_inv = matrix_inverse(C)
    print("C^-1 =\n", C_inv)
    print("C @ C^-1 = I:", np.allclose(C @ C_inv, np.eye(2)))

    # PART 2
    # Q1
    c, mean, var = distribution_moments(3.)
    print("\nP2 Q1")
    print(f"c = {c:.6f}, E[X] = {mean}, Var(X) = {var}")

    # Q2
    p_int, p_b_given_a, p_union = event_probabilities(.5, .3, .6)
    print("\nP2 Q2")
    print(f"P(A and B) = {p_int:.4f}, P(B|A) = {p_b_given_a:.4f}, P(A or B) = {p_union:.4f}")

    # Q3
    mle, map_est = coin_estimates(heads=3, tosses=5)
    print("\nP2 Q3")
    print(f"MLE theta = {mle:.4f}, MAP theta = {map_est:.4f}")

    # Q4
    mu_mle, var_mle = gaussian_mle([1, 3, 5, 7])
    print("\nP2 Q4")
    print(f"mu MLE = {mu_mle}, sigma^2 MLE = {var_mle}")

    # PART 3
    # Q2
    X = generate_data()
    print("\nP3 Q2")
    print("sample correlation =", np.corrcoef(X.T)[0, 1])

    # Q3
    results = pca(X)
    print("\nP3 Q3")
    print("covariance =\n", results["covariance"])
    print("eigenvalues =", results["eigenvalues"])
    print("components (columns) =\n", results["components"])

    # Q4
    print("\nP3 Q4")
    print("explained variance fractions =", explained_variance(results["eigenvalues"]))

    plots = make_plots(X, results)
    for i, plot in enumerate(plots):
        plot.savefig(f"figure_{i}.png")
    plt.show()