"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""

import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# YOUR ANSWER HERE
# P1 Q1: Row-by-column multiplication gives
# AB=[[3,2],[2,3]], BA=[[7,6],[-2,-1]], (A+B).T=[[4,2],[6,0]],
# A@B.T=[[11,-4],[4,-1]]. The diagonal sums are 4 and 6.
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    A = np.asarray([[3, 4], [2, 1]] if A is None else A, dtype=float)
    B = np.asarray([[1, 2], [0, -1]] if B is None else B, dtype=float)
    AB = A @ B
    return AB, B @ A, (A + B).T, A @ B.T, np.trace(A), np.trace(AB)


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
# P1 Q2: det(M-lambda*I)=(2-lambda)**2-1=(lambda-1)(lambda-3).
# lambda=1 gives x=y; lambda=3 gives x=-y.
# M@[1,1]=[1,1] and M@[1,-1]=[3,-3]=3*[1,-1].
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE
    M = np.asarray([[2, -1], [-1, 2]] if M is None else M, dtype=float)
    return np.linalg.eigh(M)


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE
# P1 Q3: For A of shape (m,n) and B of shape (n,m),
# trace(AB)=sum_i sum_j a_ij*b_ji=sum_j sum_i b_ji*a_ij=trace(BA).


# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
# P1 Q4: det(C)=4*6-7*2=10 != 0, so C is invertible.
# [[a,b],[c,d]]^-1=[[d,-b],[-c,a]]/(ad-bc).
# C^-1=[[6,-7],[-2,4]]/10; C@C^-1=[[10,0],[0,10]]/10=I.
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    C = np.asarray([[4, 7], [2, 6]] if C is None else C, dtype=float)
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
# P2 Q1: 1=c*sum_k rate**k/k!=c*exp(rate), so c=exp(-rate).
# E[X]=c*rate*sum_j rate**j/j!=rate.
# E[X(X-1)]=rate**2; E[X**2]=rate**2+rate; Var(X)=rate.
# At rate=3, (c, mean, variance)=(exp(-3), 3, 3).
def distribution_moments(rate=3.0):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    return np.exp(-rate), float(rate), float(rate)


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
# P2 Q2: P(A intersect B)=P(A|B)*P(B)=0.6*0.3=0.18.
# P(B|A)=0.18/0.5=0.36; P(A union B)=0.5+0.3-0.18=0.62.
def event_probabilities(p_a=0.5, p_b=0.3, p_a_given_b=0.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    intersection = p_a_given_b * p_b
    return intersection, intersection / p_a, p_a + p_b - intersection


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
# P2 Q3: For h heads in n independent flips, L(theta)=theta**h*(1-theta)**(n-h).
# log L=h*log(theta)+(n-h)*log(1-theta).
# Its derivative h/theta-(n-h)/(1-theta)=0 gives MLE=h/n.
# Prior p(theta)=2*theta gives posterior proportional to
# theta**(h+1)*(1-theta)**(n-h). Its log derivative gives MAP=(h+1)/(n+1).
# Interior log second derivatives are negative. For h=0 the MLE is 0;
# for h=n both estimates are 1. Otherwise the MAP is interior.
# For h=3, n=5: MLE=3/5 and MAP=4/6=2/3.
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    return heads / tosses, (heads + 1) / (tosses + 1)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
# P2 Q4: Write v=sigma**2. For independent Gaussian observations,
# L=(2*pi*v)**(-n/2)*exp(-sum_i (x_i-mu)**2/(2*v)).
# log L=-n/2*log(2*pi*v)-sum_i (x_i-mu)**2/(2*v).
# d(log L)/dmu=sum_i (x_i-mu)/v=0 gives mu_hat=mean(x).
# d(log L)/dv=-n/(2*v)+sum_i (x_i-mu)**2/(2*v**2)=0
# gives v_hat=sum_i (x_i-mu_hat)**2/n.
# For [1,3,5,7], mu_hat=4 and v_hat=(9+1+1+9)/4=5.
# The mean second derivative is -n/v<0; the variance derivative changes
# from positive to negative at v_hat, establishing the maximum.
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    data = np.asarray([1, 3, 5, 7] if data is None else data, dtype=float)
    return float(data.mean()), float(data.var(ddof=0))


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
# P2 Q5: log posterior=-(x-mu)**2/(2*v)-(mu-m0)**2/(2*v0)+constant.
# Its derivative (x-mu)/v-(mu-m0)/v0=0 gives
# mu_MAP=(x/v+m0/v0)/(1/v+1/v0). Second derivative=-1/v-1/v0<0.
# For x=5, v=4, m0=0, v0=1: mu_MAP=(5/4)/(1/4+1)=1.
def gaussian_map(x=5.0, observation_variance=4.0, prior_mean=0.0, prior_variance=1.0):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    precision = 1 / observation_variance + 1 / prior_variance
    return (x / observation_variance + prior_mean / prior_variance) / precision


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE
# P2 Q6: Let m=E[X], with finite E[X**2]. Then
# Var(X)=E[(X-m)**2]=E[X**2]-2*m*E[X]+m**2=E[X**2]-E[X]**2.

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE
# P3 Q1: Let mu=E[X], Sigma=E[(X-mu)(X-mu).T],
# so Sigma_ij=Cov(X_i,X_j). For a fixed unit vector u, E[u.T@X]=u.T@mu.
# Var(u.T@X)=E[(u.T@(X-mu))**2]
# =E[u.T@(X-mu)@(X-mu).T@u]
# =u.T@E[(X-mu)@(X-mu).T]@u=u.T@Sigma@u.


# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
# P3 Q2: Generate independent rows from N([2,-1], [[3,1.2],[1.2,1]]).
# Population correlation=1.2/sqrt(3*1)=0.6928. With default seed and size,
# sample correlation is about 0.6992; the scatter slopes upward.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    mean = np.asarray([2, -1] if mean is None else mean, dtype=float)
    covariance = np.asarray(
        [[3, 1.2], [1.2, 1]] if covariance is None else covariance, dtype=float
    )
    return np.random.default_rng(seed).multivariate_normal(mean, covariance, n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
# P3 Q3: Center rows: Z=X-mean(X). Sample covariance S=Z.T@Z/(n-1).
# Q1 gives projected sample variance u.T@S@u. For unit u it is maximized
# by the leading eigenvector; subsequent directions are orthogonal.
# Sort eigenpairs by descending eigenvalue; scores=Z@components.
# In the scatter, PC1 follows the long direction and PC2 the perpendicular one.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # YOUR CODE HERE
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (len(X) - 1)
    eigenvalues, components = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = components[:, order]
    return {
        "mean": mean,
        "centered": centered,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "components": components,
        "scores": centered @ components,
    }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
# P3 Q4: Fraction_j=lambda_j/sum_i lambda_i; total variance=trace(S).
# For the default sample, PC1 explains 89.39% and PC2 10.61%.
# Keeping only PC1 reduces two coordinates to one but discards PC2 variation.
# Both are needed for a 95% variance target. Low variance can still be useful.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return eigenvalues / eigenvalues.sum()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
# P3 plots: The scatter shows positive correlation and orthogonal PC directions
# through the mean. The bars show the variance retained by each component.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=float)
    values = result["eigenvalues"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(X[:, 0], X[:, 1], s=10, alpha=0.4)
    for j in range(2):
        direction = 2 * np.sqrt(values[j]) * result["components"][:, j]
        endpoints = result["mean"] + np.array([-1, 1])[:, None] * direction
        axes[0].plot(endpoints[:, 0], endpoints[:, 1], label=f"PC{j + 1}")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend()

    if values.sum() > 0:
        fractions = explained_variance(values)
    else:
        fractions = np.zeros_like(values)
        axes[1].text(
            0.5, 0.5, "Total variance is zero", ha="center", transform=axes[1].transAxes
        )
    axes[1].bar(["PC1", "PC2"], 100 * fractions)
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Explained variance (%)")
    axes[1].set_ylim(0, 100)
    fig.tight_layout()
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE
# Reflection: Completed the derivations, NumPy functions, and simple plots.
# Assistance: OpenAI Codex. Packages: NumPy and Matplotlib.
# No external references were consulted.

if __name__ == "__main__":
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    from pathlib import Path
    import matplotlib.pyplot as plt

    X = generate_data()
    result = pca(X)
    print("Sample correlation:", np.corrcoef(X, rowvar=False)[0, 1])
    print("Explained variance:", explained_variance(result["eigenvalues"]))
    figure = make_plots(X, result)
    output = Path(__file__).with_name("plots.png")
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print("Saved:", output)
