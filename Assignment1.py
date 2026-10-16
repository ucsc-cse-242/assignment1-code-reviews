"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# PART 1 (20 points), Q1: Show computations for the given A and B.
# On paper too
# A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Each product entry is a row-column dot product.
# AB=[[3*1+4*0, 3*2+4*(-1)], [2*1+1*0, 2*2+1*(-1)]]=[[3,2],[2,3]].
# BA=[[1*3+2*2, 1*4+2*1], [0*3-1*2, 0*4-1*1]]=[[7,6],[-2,-1]].
# A+B=[[4,6],[2,0]], so (A+B).T=[[4,2],[6,0]].
# B.T=[[1,0],[2,-1]], so A@B.T=[[11,-4],[4,-1]].
# trace(A)=3+1=4; trace(AB)=3+3=6.

def matrix_operations(A=None, B=None):
    """
    Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    A = np.asarray([[3, 4], [2, 1]] if A is None else A, dtype=float)
    B = np.asarray([[1, 2], [0, -1]] if B is None else B, dtype=float)

    AB = A @ B

    return AB, B @ A, (A + B).T, A @ B.T, float(np.trace(A)), float(np.trace(AB))

# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# det(M-lambda*I)=(2-lambda)^2-1=(lambda-1)(lambda-3)=0.
# For lambda=1, (M-I)v=0 gives v1-v2=0, so v=(1,1)/sqrt(2).
# For lambda=3, (M-3I)v=0 gives -v1-v2=0, so v=(1,-1)/sqrt(2).
# Verification: M(1,1).T=(1,1).T; M(1,-1).T=(3,-3).T=3(1,-1).T.
# scalign these vectors does not change the eigenpair equations.
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    M = np.asarray([[2, -1], [-1, 2]] if M is None else M, dtype=float)

    return np.linalg.eigh(M)

# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# Let A be m-by-n and B be n-by-m. Both AB and BA are square.
# trace(AB)=sum_i=1^m (AB)_ii=sum_i=1^m sum_j=1^n A_ij B_ji.
# Interchanging these finite sums and commuting scalar multiplication gives
# sum_j=1^n sum_i=1^m B_ji A_ij=sum_j=1^n (BA)_jj=trace(BA).
# QED (on paper version is a bit more elaborate)

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# det(C)=4*6-7*2=24-14=10 != 0, so C-1 exists.
# Standard formula is swap diagonal entries, then negate off-diagonal entries, and divide by determinant:
# C^-1=(1/10)*[[6,-7],[-2,4]]=[[0.6,-0.7],[-0.2,0.4]].
# C*[[6,-7],[-2,4]]=[[24-14,-28+28],[12-12,-14+24]]=10I.
# [[6,-7],[-2,4]]*C=[[24-14,42-42],[-8+8,-14+24]]=10I.
# thereforee both C C^-1 and C^-1 C equal I.
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    C = np.asarray([[4, 7], [2, 6]] if C is None else C, dtype=float)
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# With r=3: c=exp(-3), E[X]=3, E[X^2]=12, Var(X)=3 (Poisson distribution).
# Normalization: 1=c*sum_k=0^infinity r^k/k!=c*exp(r), so c=exp(-r).
# E[X]=exp(-r)*sum_k=1^infinity k*r^k/k!
#     =r*exp(-r)*sum_j=0^infinity r^j/j!=r.
# E[X(X-1)]=exp(-r)*sum_k=2^infinity k(k-1)*r^k/k!
#           =r^2*exp(-r)*sum_j=0^infinity r^j/j!=r^2.
# Since X^2=X(X-1)+X, E[X^2]=r^2+r and Var(X)=r^2+r-r^2=r.
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    rate = float(rate)
    return float(np.exp(-rate)), rate, rate

# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# P(A intersect B)=P(A|B)P(B)=0.6*0.3=0.18.
# P(B|A)=P(A intersect B)/P(A)=0.18/0.5=0.36.
# P(A union B)=P(A)+P(B)-P(A intersect B)=0.5+0.3-0.18=0.62.
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    intersection = p_a_given_b * p_b
    return intersection, intersection / p_a, p_a + p_b - intersection


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# For the exact observed sequence H,H,T,H,T, L(theta)=theta^3(1-theta)^2.
# More generally, with h heads and t=n-h tails, L=theta^h(1-theta)^t.
# Interior log-likelihood: ell=h*log(theta)+t*log(1-theta).
# ell'=h/theta-t/(1-theta)=0 implies h(1-theta)=t*theta -> theta=h/n.
# ell''=-h/theta^2-t/(1-theta)^2<0 when h,t>0, so this is the maximum.
# L(0)=L(1)=0 for mixed outcomes. All tails gives MLE=0, all heads MLE=1.
# Default MLE=3/5=0.6.
# Prior density p(theta)=2theta on [0,1] is valid: integral_0^1 2theta dtheta=1.
# Bayes: p(theta|D) proportional to L(theta)p(theta)=2theta^(h+1)(1-theta)^t.
# The normalization and factor 2 do not depend on theta, so MAP maximizes
# (h+1)log(theta)+t log(1-theta).
# Derivative (h+1)/theta-t/(1-theta)=0 gives theta=(h+1)/(n+1).
# Second derivative is negative in the interior. If t>0, both endpoint
# densities are zero, so the interior maximum is global. If t=0, theta^(n+1)
# increases on [0,1], giving boundary MAP=1. This same formula covers t=0.
# Default posterior: p(theta|D)=105theta^4(1-theta)^2 (Beta(5,3)); MAP=4/6=2/3.
# Normalizer: integral_0^1 2theta^4(1-theta)^2 dtheta
# =2(1/5-2/6+1/7)=2/105, giving the coefficient 105 above.
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    return heads / tosses, (heads + 1) / (tosses + 1)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# asssume independent N(mu,v) observations, v=sigma^2>0.
# L(mu,v)=(2piv)^(-n/2)*exp[-sum_i (x_i-mu)^2/(2v)].
# ell=-n/2*log(2*pi)-n/2*log(v)-sum_i(x_i-mu)^2/(2v).
# d ell/d mu=sum_i(x_i-mu)/v=0 -> mu_hat=sum_i x_i/n.
# the squared-error sum decomposes as sum_i(x_i-mu)^2
# =sum_i(x_i-xbar)^2+n(mu-xbar)^2, so xbar globally maximizes for fixed v.
# d ell/d v=-n/(2v)+sum_i(x_i-mu)^2/(2v^2)=0
# -> v_hat=sum_i(x_i-mu_hat)^2/n. With nonzero sample variance, the profiled
# derivative changes from positive to negative at this value; hence a maximum.
# [1,3,5,7]: mu_hat=16/4=4; squared deviations=9,1,1,9; v_hat=20/4=5.
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    data = np.asarray([1, 3, 5, 7] if data is None else data, dtype=float)

    mean = float(np.mean(data))

    return mean, float(np.mean((data - mean) ** 2))


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# x|mu ~ N(mu,s^2), mu ~ N(m0,tau^2).
# Log posterior, omitting mu-independent constants:
# g(mu)=-(x-mu)^2/(2s^2)-(mu-m0)^2/(2tau^2).
# g'=(x-mu)/s^2-(mu-m0)/tau^2=0.
# -> mu*(1/s^2+1/tau^2)=x/s^2+m0/tau^2.
# mu_MAP=(x/s^2+m0/tau^2)/(1/s^2+1/tau^2).
# g''=-1/s^2-1/tau^2<0, so this unique mode is the global maximum.
# Default: (5/4+0/1)/(1/4+1)=1; posterior variance=(1/4+1)^-1=0.8.
# Thus posterior is N(1,0.8); the prior pulls the estimate toward 0.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    return (x / observation_variance + prior_mean / prior_variance) / (
        1. / observation_variance + 1. / prior_variance
    )


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# Var(X)=E[(X-m)^2]=E[X^2-2mX+m^2]
#       =E[X^2]-2mE[X]+m^2
#       =E[X^2]-2m^2+m^2
#       =E[X^2]-(E[X])^2.

# This uses linearity of expectation, not independence.
# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# Let X be a column random vector with finite second moments, m=E[X], and
# Sigma=E[(X-m)(X-m).T]. Entry Sigma_ij=Cov(X_i,X_j).
# Let u be a fixed unit column vector (u.T u=1), and Y=u.T X a scalar.
# E[Y]=u.T m, so Y-E[Y]=u.T(X-m).
# Var(Y)=E[(u.T(X-m))^2]
#       =E[u.T(X-m)(X-m).T u]  (a scalar equals its transpose)
#       =u.T E[(X-m)(X-m).T] u=u.T Sigma u.
# fixed u moves outside expectation. The identity also holds for non-unit u;
# unit length makes it a directional projection without extra scaling.

# P3 Q2: Describe the Gaussian generator and observed correlation.
# generate rows as independent samples from N([2,-1], [[3,1.2],[1.2,1]]).
# this covariance is symmetric positive definite: first leading minor=3>0,
# determinant=3*1-1.2^2=1.56>0. Population correlation is
# 1.2/sqrt(3*1)=approximately 0.693, so the cloud slopes upward and is elongated.
# a local default_rng(seed) makes repeated calls reproducible without changing
# numpy's global RNG state. Sample moments fluctuate around population values.
# the main guard prints the actual observed correlation for this generated data.
# (seed = 42 if you're cool)
def generate_data(n_samples=500, seed=42, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    mean = np.asarray([2., -1.] if mean is None else mean, dtype=float)
    covariance = np.asarray([[3., 1.2], [1.2, 1.]] if covariance is None else covariance, dtype=float)

    rng = np.random.default_rng(seed)

    return rng.multivariate_normal(mean, covariance, size=n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# With samples as rows, mean is the sample analogue of m=E[X]. Subtract it to
# form Z, the analogue of X-m. The sample covariance is S=Z.T Z/(n-1).
# from Q1, variance in direction u is u.T S u. Since S is symmetric, write
# S=V diag(lambda) V.T. For unit u, w=V.T u has sum_j w_j^2=1; hence
# u.T S u=sum_j lambda_j*w_j^2 <= largest eigenvalue.
# equality is attained by a top eigenvector: it is the first principal direction.
# Remaining eigenvectors give orthogonal directions of decreasing variance.
# eigh returns eigenvectors as columns; reorder columns with descending values.
# Scores=Z V are coordinates in this basis; their covariance is diag(lambda).
# Rank deficiency permits zero eigenvalues. Tied eigenvalues admit multiple
# equally valid orthonormal bases. Tiny negative roundoff eigenvalues are clipped.
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
    covariance = centered.T @ centered / (X.shape[0] - 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    eigenvalues = np.maximum(values[order], 0.)
    components = vectors[:, order]
    scores = centered @ components

    return {"mean": mean, "centered": centered, "covariance": covariance, "eigenvalues": eigenvalues, "components": components, "scores": scores}


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# Each lambda_j is variance along PC j, so fraction_j=lambda_j/sum_k lambda_k.
# The population eigenvalues here are (4 +/- sqrt(9.76))/2
# ~=3.56205 and 0.43795; fractions ~=89.05% and 10.95%.
# Actual sample values and fractions are printed below; do not substitute
# population fractions for measured sample results.
# Retaining PC1 reduces two coordinates to one and captures most variance.
# Reconstruction with one PC: Z_hat=scores[:, :1] @ components[:, :1].T.
# Add the mean to return to original coordinates. The discarded orthogonal
# variation cannot be recovered: sum squared residuals/(n-1)=lambda_2.
# Dimensionality reduction trades storage/simplification and potentially noise
# removal against information loss. Low variance does not imply low predictive
# importance. PCA maximizes retained variance, not supervised prediction quality.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    eigenvalues = np.asarray(eigenvalues, dtype=float)

    return eigenvalues / eigenvalues.sum()

# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# left: raw sample cloud; positive slope indicates positive correlation.
# middle: orthogonal principal axes through sample mean, each drawn to +/- one
# score standard deviation (sqrt(eigenvalue)). Equal axis scaling avoids visually
# distorting angles and lengths. PC1 follows the cloud's greatest spread.
# right: fractions quantify how much sample variance each component retains.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    X = np.asarray(X, dtype=float)
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.6), constrained_layout=True
    )

    for ax in axes[:2]:
        ax.scatter(X[:, 0], X[:, 1], s=12, alpha=0.4, color="purple")
        ax.set_xlabel("Feauture 1")
        ax.set_ylabel("Feature 2")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)

    correlation = np.corrcoef(X, rowvar=False)[0, 1]
    axes[0].set_title(
        f"Correlated Gaussian samples\nSample correlation = {correlation:.3f}"
    )

    mean = result["mean"]

    for j, color in enumerate(("red", "blue")):
        delta = result["components"][:, j] * np.sqrt(result["eigenvalues"][j])
        segment = np.vstack((mean - delta, mean + delta))
        axes[1].plot(
            segment[:, 0],
            segment[:, 1],
            color=color,
            lw=3,
            label=f"PC{j + 1}: ±1 score SD",
        )

    axes[1].scatter(*mean, color="black", s=30, label="Sample mean", zorder=4)
    axes[1].set_title("Principal component directions")
    axes[1].legend(fontsize=8)

    total = np.sum(result["eigenvalues"])

    if total > 0:
        fractions = explained_variance(result["eigenvalues"])
        axes[2].bar(["PC1", "PC2"], fractions, color=["red", "blue"])

        for j, fraction in enumerate(fractions):
            axes[2].text(j, fraction + 0.02, f"{fraction:.1%}", ha="center")
    else:
        axes[2].text(
            0.5,
            0.5,
            "No variance: fractions undefined",
            ha="center",
            transform=axes[2].transAxes,
        )

    axes[2].set_ylim(0, 1.12)
    axes[2].set_ylabel("Fraction of total sample variance")
    axes[2].set_title("Explained variance")

    return fig

# REFLECTION: contributions, tasks completed, and external resources used.
# All the logic etc is mine, I did use chatgippity to clean up my LaTeX and check for errors, format
# I also had forgotten some numpy stuff so ig I used their docs
# code, all within the purview of this class' rules I believe
# As far as the mathematics review goes, I'd been studying the past 2 weeks
# using flashcards with the guidance of AI to help me determine what to study on
# this was guided also by the slides and content of the assignment but I tried to only
# study near problems not exact problems to get general patterns down
if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)

    print("P1 Q1: AB, BA, (A+B).T, A@B.T, trace(A), trace(AB)")
    for answer in matrix_operations():
        print(answer)

    eigvals, eigvecs = eigensystem()
    print("P1 Q2 eigenvalues:", eigvals)
    print("Eigenvectors (columns):\n", eigvecs)
    print("P1 Q4 inverse:\n", matrix_inverse())

    print("P2 Q1: c, mean, variance =", distribution_moments())
    print("P2 Q2: intersection, conditional, union =", event_probabilities())
    print("P2 Q3: MLE, MAP =", coin_estimates())
    print("P2 Q4 Gaussian MLE:", gaussian_mle())
    print("P2 Q5 Gaussian MAP mean:", gaussian_map())

    X = generate_data()
    result = pca(X)
    fractions = explained_variance(result["eigenvalues"])

    pc1_scores = result["scores"][:, :1]
    pc1_direction = result["components"][:, :1]
    reconstructed = pc1_scores @ pc1_direction.T + result["mean"]

    residual = X - reconstructed
    residual_variance = np.sum(residual ** 2) / (len(X) - 1)
    correlation = np.corrcoef(X, rowvar=False)[0, 1]

    print("\nP3 sample mean:", result["mean"])
    print("Sample covariance:\n", result["covariance"])
    print("Sample correlation:", correlation)
    print("Eigenvalues:", result["eigenvalues"])
    print("Principal directions (columns):\n", result["components"])
    print("Explained variance:", fractions)
    print(
        f"Keeping PC1 retains {fractions[0]:.2%} of the variance; "
        f"dropping PC2 loses {fractions[1]:.2%}."
    )
    print("One-PC residual variance:", residual_variance)
    print("The cloud slopes upward, and PC1 follows its longer direction.")
    print("Using only PC1 gives a 1D representation and loses PC2 variation.")

    fig = make_plots(X, result)
    destination = Path(__file__).resolve().with_name("Assignment1_PCA.pdf")
    fig.savefig(destination, dpi=180)
    plt.close(fig)

    print("Figure saved to:", destination)