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
#AB=[[3+0,6+(-4)],[2+0,4+(-1)]] = [[3,2],[2,3]]
#BA=[[3+4,4+2],[0+(-2),0+(-1)]]= [[7,6],[-2,-1]]
#(A+B)T=> A+B=[[4,6],[2,0]] -transpose--> [[4,2],[6,0]]
#ABT=[[3,4],[2,1]] x [[1,0],[2,-1]]=[[3+8,0+(-4)],[2+2,0+(-1)]]= [[11, -4],[4,-1]]
#tr(A)=3+1=4
#tr(AB)=> AB=[[3,2],[2,3]] -->3+3=6
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    if A is None:
        A = [[3,4], [2,1]]
    if B is None:
        B=[[1,2],[0,-1]]
    A=np.array(A,dtype=float)
    B=np.array(B,dtype=float)

    AB=A@B
    BA=B@A
    transpose = (A+B).T
    ABt=A@B.T
    tr_A=np.trace(A)
    tr_AB=np.trace(AB)
    return AB, BA, transpose, ABt, tr_A, tr_AB


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
# det(M-lambda(I))=0
# M-lambda(I)=[[2-lambda,-1],[-1,2-lambda]]=(2-lambda)^2-(-1)(-1)=> (2-lambda)^2=1 
# lambda=1 or lambda=3 (these values add up to the trace of the matrix. Just an extra check)
# M-I=[[1,-1],[-1,1]]=[1,1]=v1         M-3I=[[-1,-1],[-1,-1]]=[1,-1]=v2
# lambda=1,v=(1,1): Mv=[[2,-1],[-1,2]][1,1]=[[2-1],[-1+2]]=[1,1]=1[1,1]
# lambda=3, v=(1,-1):Mv=[[2,-1],[-1,2]][1,-1]=[[2+1],[-1-2]]=[3,-3]=3[1,-1] 

def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE
    if M is None:
        M=[[2,-1],[-1,2]]
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    return eigenvalues, eigenvectors


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE
# let's say matrix A is n x m and matrix b is m x n
# trace equation: trace(A)=sum_i A_ii
# (AB)_ii=sum_{k=1..m} A_ik * B_ki
# trace(AB) = sum_{i=1..n} sum_{k=1..m} (A_ik)(B_ik) --> and since (A_ik)(B_ki) = (B_ki)(A_ik)
#trace(AB)=sum{k=1..m} sum_{i=1..n} (B_ki)(A_ik)
# so then the you get the diagonal element, which is (BA)_kk:
#  (BA)_kk = sum{i=1..n} (B_ki)(A_ik)
#   trace(AB) = sum{k=1..m} (BA)_kk = trace(BA)
# 


# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
# det(C)=(4)(6)-(7)(2)=24-14=10 --> inverse exists because determinant is non-zero
# Equation: [[a,b],[c,d]]^-1=1/(ad-bc)*[[d,-b],[-c,a]]
# ==> 1/10*[[6,-7],[-2,4]]=[[0.6,-0.7],[-0.2,0.4]]

def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    if C is None:
        C=[[4,7],[2,6]]
    return np.linalg.inv(C)
 


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
# (a)Normalization constant. 
# e^x is evaluated at x=3 so that means c * e^3 = 1 ==> c=e^-3
# (b) Expectation and Variance
# starting at k=1, k/k! becomes 1/(k-1), then to make the top and bottom match, pull out factor 3
# then where j=k-1, E[X] = 3e^-3 sum_{k=1..inf} 3j/j! = 3e^-3 * e^3 = 3
# For variance, we first compute E[X(X-1)]
# E[X(X-1)] = sum_{k=2..inf} k(k-1) *e^-3 * 3k/k! = e^-3 * sum_{k=2..inf} 3^k/(k-2)!
#Then take out 3^2 and let j=k-2: E[X(X-1)] = 9e^-3 * sum_{j=0..inf} 3j/j! = 9e^-3 * e^3 = 9
# E[X^2] = E[X(X-1)] + E[x] = 9+3=12
# Var[X] = E[X^2] -(E[X])^2 = 12-3^2 = 3

def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    c = np.exp(-rate)
    mean = rate
    variance = rate
    return c, mean, variance
 


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
# (a) P(A intersect B) = P(A|B) * P(B) = 0.6 * 0.3 = 0.18
# (b) P(B|A) = P(A intersect B)/ P(A) = 0.18/0.5 = 0.36
# (c) P(A union B) = P(A) + P(B) - P(A intersect B)= 0.5 + 0.3 - 0.18 = 0.62

def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    p_a_int_b = p_a_given_b *p_b
    p_b_given_a = p_a_int_b / p_a
    p_a_union_b = p_a + p_b - p_a_int_b
    return p_a_int_b, p_b_given_a, p_a_union_b
 


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
# (a) Derive MLE for theta : argmax_theta P(data | theta)
#     L(theta) = theta^3 *(1-theta)^2
#     L(theta) = P(z|theta) = P(H) * P(H) * P(T) * P(H) * P(T) = theta * theta * (1-theta) *theta * (1-theta) = 3theta*(1-theta)^2
#     ln(L(theta)) = 3ln(theta) + 2ln(1-theta) = l(theta)
#     deriving each part gets you ==> 3/theta - 2/(1-theta)
#     set it to 0 and solve: 3/theta = 2/(1-theta) ==> 3(1-theta) = 2theta ==>3=5theta ==> theta_MLE = 3/5
#(b) MAP: theta_MAP = argmax_theta P(theta|data)
    # From Bayes, P(data|theta) P(theta), where the P(data) can be dropped since it doesn't depend on theta
    # P(theta|data) --> theta^3(1-theta)^2 *2theta = 2theta^4 * (1-theta)^2
    # ln(P(theta|data)) = 4ln(theta)+2ln(1-theta)
    # deriving each part gets you: 4/theta - 2/(1-theta)
    # set it to 0 and solve: 4/theta = 2/(1-theta) ==> 4(1-theta) = 2theta ==> 4=6theta ==>theta_MAP = 4/6 or 2/3
     
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    MLE = heads/tosses
    MAP = (heads+1)/(tosses+1)
    return MLE, MAP
 


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
# p(x|mu, sigma^2) = prod_n 1/sqrt(2 pi sigma^2) * exp(-(xn - mu)^2 / (2sigma^2))
# fill in vals for the dataset:
# (1/sqrt(2 pi sigma^2))^4 exp(- ((1-mu)^2 + (3-mu)^2 + (5-mu)^2 + (7-mu)^2)/2sigma^2)
# ln(p(x|mu, sigma^2)) = -2ln(2 pi sigma^2) - ((1-mu)^2 + (3-mu)^2 + (5-mu)^2 + (7-mu)^2)/2sigma^2
# drop parts that don't have mu since it wouldn't affect the peak
# mu_ML = argmax_mu (- ((1-mu)^2 + (3-mu)^2 + (5-mu)^2 + (7-mu)^2)/2sigma^2)
# can also drop 1/2sigma^2 for that same reason. Finding the argmax of a negative sum of squares is the same as minimizing (argmin)
# mu_ML = argmin_mu (1-mu)^2 + (3-mu)^2 + (5-mu)^2 + (7-mu)^2
# f'(mu)= -2[(1-mu)+(3-mu)+(5-mu)+(7-mu)] ==> set to 0 and solve
# 16-4mu=0 ==> mu_ML = 4
# Plug in mu_ML val: (1-4)^2 + (3-4)^2 + (5-4)^2 + (7-4)^2 = 20
# after splitting ln(2 pi sigma^2) into ln(2pi) and ln(sigma^2)
# ln(x|4,sigma^2) = -2ln(2pi)-2lnsigma^2-20/2sigma^2 = -2ln(2pi) - 2lnsigma^2 - 10/sigma^2
# can drop -2ln(2pi)
# sigma^2_ML = argmax_sigma^2 (-2lnsigma^2 - 10/sigma^2)
# g(sigma^2) = -2lnsigma^2 - 10/sigma^2
# g'(sigma^2) = -2/sigma^2 + 10/(sigma^2)^2 
# set it to 0 and solve:
# -2sigma^2 + 10 = 0 ==> sigma^2_ML = 5


def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    if data is None:
        data = 1,3,5,7

    x= np.array(data,dtype=float)
    n=len(x)

    mean_MLE = np.sum(x)/n
    variance_MLE = np.sum((x-mean_MLE)**2)/n
    return mean_MLE, variance_MLE
 


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
# likelihood: p(x|mu) = 1/(sqrt(2pi * 4)) exp(-(5-mu)^2 / 2*4)
# prior: p(mu) = 1/sqrt(2pi * 1) exp(-(mu-0)^2 / 2*1)
# then multiply likelihood * prior:
# p(x|mu)p(mu) = exp(-(5-mu)^2 / 8 - mu^2 / 2)
# Differentiate both parts gets you:
# 5-mu /4 - mu
# Set to 0: 5=5mu ==> mu_MAP = 1
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    numerator = prior_variance * x + observation_variance * prior_mean
    denominator = prior_variance + observation_variance
    post_mode = numerator/denominator
    return post_mode
 


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE
# Var(X) = E[(X - mu)^2]
# Expanding ==> Var(X) = E[X^2 -2*mu*X +mu^2]
# Pull out constants ==> Var(X) = E[X^2] - 2*mu*E[X] + E[mu^2]
# Since E[X] = mu, - 2*mu*E[X] becomes -2*mu*mu = -2mu^2. And E[u^2] = u^2 since the expectation of a constant is just the constant
# So: Var(X) = E[X^2] - 2mu^2 + mu^2 
# Simplifies to: Var(X) = E[X^2] -mu^2
# Substituting E[X] back in as mu: Var(X) = E[X^2] - (E[X])^2

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE
# Answered on paper -- refer to PDF

# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
# generator: 500 samples from a 2D Gaussian with mean [2,-1] and covariance [[3,1.2], [1.2,1]], uses numpy's multivariate_normal
# Covariance: X1 has variance 3 and X2 has variance 1, so X1 is more spread out. the covariance 1.2 is positive, so the two features tend to increase together
# Correlation = 1.2/sqrt(3*1) ~ 0.69
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    if mean is None:
        mean = [2,-1]
    if covariance is None:
        covariance = [[3,1.2], [1.2,1]]

    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(mean, covariance, size=n_samples)
    return X
 


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
# From Q1, the vairance along a direction u is u^T sigma u
# PCA finds where this variance is the largest
# First you ceneter, subtracting the mean from each point. Then you get the covariance, which is the sample version of sigma from Q1. S = centered^T @ centered / (n-1)
# Then the eigenvectors: maximizing u^T covariance u with u^T u=1 gives S u = lambda u. So the best directions are the eigenvectors of S, and the variance along each one is its eigenvalue lambda
# Then sort eigenvalues from largest to smallest. PC1 is the direction of most variance.
# scores = centered @ components is each point's coords along the new axes (prejection u^T (x-mean) from Q1)

def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # YOUR CODE HERE
    n=len(X)
    mean = np.mean(X,axis=0)
    centered = X-mean
    covariance = centered.T @ centered/(n-1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues,0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order]
    scores = centered @ components
    return {"mean":mean,"centered":centered,"covariance":covariance,"eigenvalues":eigenvalues,"components":components, "scores":scores}
 


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
# proportion: k = lambda_k / (sum of all eigenvalues) ==> where the fractions add up to 1
# PC1 explains most of the variance because the two features are strongly correlated, so the data lies along one direction
# PC2 captures the small spread across the oval. If the features were uncorrelated, wach PC would explain 50% and PCA wouldn't be the right move
# Dimentionality reduction is keeping only PC1 turns the 2D data into 1D while keeping most of the variance. 
# However, dropping components loses information.

def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    return np.array(eigenvalues) / np.sum(eigenvalues)
    # raise NotImplementedError()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
# Q2 + Q3 plot: The points form an oval centered near (2,-1) with an upward tilt to the right. When feature 1 is large, you expect feature 2 to be large too, indicating positive correlation
#          The oval is more wide than tall, with only a few outliers that sit far from the center
#          PC1 runs along the long axis of the oval. its slope is roughly 3.1/6.8 =0.46
#          PC2 runs along the short axis, and is perpendicular to PC1, which is expected
# Q4 plot: PC1 explains 89.4% of the variance and PC2 explains 10.6%.Regarding dimentionaly reduction, if you only kept PC1, you still keep 89.4% of the variance.

def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    X = np.array(X)
    mean = result["mean"]
    eigenvalues = result["eigenvalues"]
    components = result["components"]
    fractions = explained_variance(eigenvalues)

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))

    #scatter plot: samples with prim=ncipal directions through the mean
    ax1.scatter(X[:,0],X[:,1], s=10, alpha=0.4,label="Samples")
    for k in range(2):
        half = components[:,k] * 2 * np.sqrt(eigenvalues[k])
        ax1.plot([mean[0]-half[0],mean[0]+half[0]],[mean[1]-half[1],mean[1]+half[1]],linewidth=3, label=f"PC{k+1}")
    ax1.set_xlabel("feature 1")
    ax1.set_ylabel("feature 2")
    ax1.set_aspect("equal")
    ax1.legend()

    #bar plot: feaction of variance explained by each component
    ax2.bar(["PC1","PC2"], fractions)
    for k in range(2):
        ax2.text(k,fractions[k]+0.02,f"{fractions[k]:.1%}", ha="center")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Fraction of variance explained")
    ax2.set_ylim(0,1)

    fig.tight_layout()
    return fig
 


# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE
# I completed most of the code by myself. I used Claude to help with graphing options. 
# For theory, I used previous notes + lecture slides as well as Claude to explain the more confusing parts and to help when I'm stuck.

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    # Part 1 and 2 check
    # print("Matrix operations:", matrix_operations())
    # print("Eigensystem:", eigensystem())
    # print("Inverse:", matrix_inverse())
    # print("Distribution moments:", distribution_moments())
    # print("Event probabilities:", event_probabilities())
    # print("Coin estimates:", coin_estimates())
    # print("Gaussian MLE:", gaussian_mle())
    # print("Gaussian MAP:", gaussian_map())
    
    
    X = generate_data()
    result = pca(X)

    print("Mean:", result["mean"])
    print("Correlation:", np.corrcoef(X, rowvar=False)[0,1])
    print("Eigenvalues:", result["eigenvalues"])
    print("PC1:", result["components"][:,0])
    print("Explained variance:", explained_variance(result["eigenvalues"]))

    fig=make_plots(X, result)
    fig.savefig("pca_plots.pdf", dpi=150)
    plt.show()

