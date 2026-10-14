"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np

################################################################################################################
#
#                    PART 1
#
################################################################################################################

# PART 1 (20 points), Q1: Show computations for the given A and B.
# YOUR ANSWER HERE
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    if A is None:
        A = [[3,4],[2,1]]
    if B is None:
        B = [[1,2],[0,-1]]
    A = np.asarray(A)
    B = np.asarray(B)
    ab = np.matmul(A, B)
    ba = np.matmul(B, A)
    a_plus_b = A + B
    a_plus_b_T = a_plus_b.T
    ab_T = np.matmul(A,B.T)
    a_trace = np.trace(A)
    ab_trace = np.trace(ab)
    return (ab, ba, a_plus_b_T, ab_T, a_trace, ab_trace)


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    if M is None:
        M = [[2,-1],[-1,2]]
    M = np.asarray(M, dtype=float)
    eigval, eigvec = np.linalg.eigh(M)
    Mv = np.matmul(M, eigvec)
    lambda_v = eigval * eigvec
    print(f"Mv = {Mv}")
    print(f"lambda*v = {lambda_v}")
    return eigval, eigvec


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE
"""
Using the same A and B matrices used in part 1, q1. I calculated
AB and BA matrices. Used np.trace(ab) and np.trace(ba). np.trace()
method computes the sum of the diagonal of square matrices. I then
print out both values. And as shown in the terminal trace(AB)=trace(BA).
"""
def trace_proof(a, b):
    ab = a @ b
    ba = b @ a
    ab_trace = np.trace(ab)
    ba_trace = np.trace(ba)
    return ab_trace, ba_trace

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
""" 
Using the given C, to calculate a 2 x 2 we use the following:
A^-1 = 1 / determinant(C) * adjugate_matrix
A^-1 = 1 / (ad-bc) * [[d, -b], [-c, a]] 

To show that its invertible AB = BA = I, where I is the identity matrix
A is the invertible matrix, B is the inverse of A. Matrices must be 
n x n square matrix.
"""
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    if C is None:
        C = [[4,7],[2,6]]
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        print("Invalid input")
        return None

    if C.shape[0] != 2:
        return np.linalg.inv(C)

    a = C[0, 0]
    b = C[0, 1]
    c = C[1, 0]
    d = C[1, 1]
    det_C = a * d - b * c

    # Determinant cant equal to 0
    if det_C == 0:
        print("determinant = 0")
        return None
    
    C_inv = 1 / det_C * np.array([[d, -b],
                                  [-c, a]])
    C_inv_mult_C = np.matmul(C_inv, C)
    C_mult_inv_C = np.matmul(C, C_inv)
    I = np.eye(2)

    # used allclose since array_equal doesn't handle fp
    if not (np.allclose(C_inv_mult_C, I) and np.allclose(C_mult_inv_C, I)):
        print(f"Not identity \n C_mult_inv_C: {C_mult_inv_C} \n C_inv_mult_C: {C_inv_mult_C}")
        return None
    return C_inv

################################################################################################################
#
#                    PART 2
#
################################################################################################################

# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
"""
(a)
Slide 15 probability mass function p(x)
The summation p(x) must equal to 1
x in this case is k. so summation of P(X = k) is the normalization
To the get its constant c we do 1 / (summation of P(X = k))

(b)
Expectation E[X] = summation (xi * pi)
For here: E[X=k] = summation (ki * p(k)i)
With variance expansion E[X^2] - E[X]^2
Let E[X^2] = e_1 and E[X]^2 = e_2
"""

from math import factorial
def p_func(rate, k):
    return (rate**k) / factorial(k)

def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # (a)
    c = float(np.exp(-rate))

    # (b)
    expectation = float(rate)
    e_1 = rate + rate**2
    e_2 = expectation**2
    variance = float(e_1 - e_2)
    
    return (c, expectation, variance)


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
"""
(a)
P(A and B) = P(A) * P(B) iff A and B are independent, else:
P(A and B) = P(A | B) * P(B)
From whats given we can compute both.

P(B | A) = P(B and A) / P(A)
We have P(A) given. So to calculate P(B and A) its
P(B and A) = P(A and B) = P(A | B) * P(B)
With P(A | B) * P(B), vars are given so:
P(B | A) = P(A and B) / P(A)


P(A union B) = P(A) + P(B) - P(A and B)
(a) and the rest are given hence we can compute directly.
"""
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # (a)
    a = p_a_given_b * p_b

    # (b)
    b = a / p_a

    # (c)
    c = p_a + p_b - a
    return (a, b, c)

# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
"""
Let v be p(H) and 1-v be p(T).
Thus p(HHTHT) = v*v*(1-v)*v*(1-v)
              = v^5-2v^4+v^3

derivative = 5v^4-8v^3+3v^2
solving for v gives us
v = 0 and v=3/5, v=1 from quadratic eq
Plug each v value back into p() and find the
biggest p(v) value, thus MLE is v = 3/5
"""
import sympy as sp
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # (a)
    theta = sp.symbols('theta')
    tails = tosses - heads
    f = theta**heads * (1 - theta)**(tails)
    drv = sp.diff(f, theta)

    # sympy solve method equal function to 0 and solves for var
    sol = sp.solve(drv, theta) 
    sol += [1, 0] # to ensure bounds
    mle = 0
    best = 0
    for v in sol:
        subs = f.subs(theta, v)
        if best < subs and (0 <= v <= 1):
            mle = v
            best = subs

    # (b)
    f_map = theta**heads * (1 - theta)**(tails) * (2 * theta)
    drv_map = sp.diff(f_map, theta)
    sol_map = sp.solve(drv_map, theta) 
    sol_map += [1, 0]
    map = 0
    best_map = 0
    for v in sol_map:
        subs = f_map.subs(theta, v)
        if best_map < subs and (0 <= v <= 1):
            map = v
            best_map = subs
    return (float(mle), float(map))

# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
"""
We want (u, sigma), where u is the mean, and sigma is the variance
From the slides expecation expansion, we have:
Var(X) = E[X^2] - E[X]^2 OR E[(X - E[X])^2]
We can sub in the u with E[X] so that its 
Var(X) = E[X^2] - u^2
For E[X^2] we can do summation of x^2 across the data.
"""
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    if data is None:
        data = [1,3,5,7]
    data = np.asarray(data, dtype=float)
    mean = np.mean(data)
    mean_pow = mean**2
    x_pow = np.power(data, 2)
    e_2 = np.mean(x_pow)
    var = e_2 - mean_pow
    return (float(mean), float(var))

# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
"""
p(x) = 1/(sqrt(2pi) * sigma) exp(-(x-u)^2 / (2 * sigma^2))
prior:      p(u)     = 1/(sqrt(2pi) * 1) exp(-(u-0)^2 / (2 * 1))
likelihood: p(x | u) = 1/(sqrt(2pi) * 2) exp(-(5-u)^2 / (2 * 4))
MAP maximizes likelihood * prior: p(x | u) p(u)
p(x | u) * p(u) = 1/(sqrt(2pi)*2) * 1/sqrt(2pi) * exp( -(5-u)^2/8 - u^2/2 )
"""
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    u = sp.symbols('u')
    prior_u = 1 / (sp.sqrt(2 * sp.pi * prior_variance)) * sp.exp(- (u - prior_mean)**2 / (2 * prior_variance))
    likelihood_x_u = 1 / (sp.sqrt(2 * sp.pi * observation_variance)) * sp.exp(- (x - u)**2 / (2 * observation_variance))

    drv = sp.diff(prior_u * likelihood_x_u, u)
    sol = sp.solve(drv, u)
    return float(sol[0])


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
def prove_var():
    print("""
    PROVE VAR(X)

    Let mu = E[X].

    Start from the definition:
    Var(X) = E[(X - mu)^2]

    Expand the square using (a - b)^2 = a^2 - 2ab + b^2:
    = E[X^2 - 2*mu*X + mu^2]

    Split the expectation
    E[A + B] = E[A] + E[B], and constants pull out, E[cA] = c*E[A]:
    = E[X^2] - 2*mu*E[X] + mu^2
    The last term is just mu^2, since the expectation of a constant is that constant.

    Substitute E[X] = mu:
    = E[X^2] - 2*mu*mu + mu^2
    = E[X^2] - 2*mu^2 + mu^2
    Thus
    = E[X^2] - mu^2
    = E[X^2] - E[X]^2
    """)
    return

################################################################################################################
#
#                    PART 3
#
################################################################################################################

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
def derive_var():
    print("""
    Covariance matrix is defined as E[(X - u)(X - u)^T]

    We use definition of Var(X) = E[(X - E[X])^2]
    Var(u^T @ X) = E[(u^T @ X - E[u^T @ X])^2]
    The inner E term becomes u^T @ mu, where mu is the mean.
        = E[(u^T @ X - u^T @ mu)^2]

    Pull out u^T so its = E[(u^T (X - mu)^2]
    Expanding u^T(X - mu)^2 gives us u^T (X - mu) u^T (X - mu)

    For one of the term we can transpose it
    (u^T (X - mu))^T so that (u^T)T (X - mu)^T

    Thus (u^T)T = u then, 
    E[u (X - mu)^T u^T (X - mu)]

    We can pull out u and u^T then u u^T E[(X - mu)^T (X - mu)]
    And E[u (X - mu)^T u^T (X - mu)] is the covariance matrix summation symbol

    so its u^T (summation symbol) u
    """)

# P3 Q2: Describe the Gaussian generator and observed correlation.
import matplotlib.pyplot as plt

def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    if mean is None:
        mean = [2, -1]
    if covariance is None:
        covariance = [[3, 1.2], [1.2, 1]]
    data = np.random.default_rng(seed).multivariate_normal(mean, covariance, size=n_samples)
    return data

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
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0)
    centered = X - mean
    n = X.shape[0]
    covariance = (centered.T @ centered) / (n - 1)
    eigenval, eigvec = np.linalg.eigh(covariance)
    order = np.argsort(eigenval)[::-1]
    eigenval = np.clip(eigenval[order], 0, None)
    components = eigvec[:, order]
    scores = centered @ components
    return {
        "mean" : mean, 
        "centered" : centered,
        "covariance" : covariance,
        "eigenvalues" : eigenval,
        "components" : components,
        "scores" : scores
    }

# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
"""
explained variance = eigenvalues / sum(eigenvalues)
"""
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    explained_var = eigenvalues / eigenvalues.sum()
    return explained_var


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    X = np.asarray(X, dtype=float)
    mean = result["mean"]
    components = result["components"]
    eigenvalues = result["eigenvalues"]
    fractions = explained_variance(eigenvalues)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(X[:, 0], X[:, 1], alpha=0.4, s=12, label="samples")
    ax1.scatter(*mean, color="black", marker="x", s=80, label="mean")
    colors = ["tab:red", "tab:green"]
    for k in range(2):
        direction = components[:, k] * 2 * np.sqrt(eigenvalues[k])
        ax1.annotate("", xy=mean + direction, xytext=mean,
                     arrowprops=dict(arrowstyle="->", color=colors[k], lw=2.5))
        ax1.plot([], [], color=colors[k], lw=2.5,
                 label=f"PC{k+1} ({fractions[k]:.1%} of variance)")  # legend entry
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_title("Samples with principal directions (arrows = 2 std devs)")
    ax1.set_aspect("equal")
    ax1.legend(loc="upper left")

    labels = [f"PC{k+1}" for k in range(len(fractions))]
    bars = ax2.bar(labels, fractions, color=colors)
    ax2.bar_label(bars, labels=[f"{f:.1%}" for f in fractions])
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Principal component")
    ax2.set_ylabel("Fraction of total variance")
    ax2.set_title("Explained variance")

    fig.tight_layout()
    return fig 
# REFLECTION: contributions, tasks completed, and external resources used.
"""
all parts 1-3 completed.
"""

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    
    ################ PART 1 ################

    # Q1 
    A=[[3,4],[2,1]] 
    B=[[1,2],[0,-1]]
    q1_res = matrix_operations(np.array(A), np.array(B))
    for i, t in enumerate(q1_res): print(f"op{i}:\n {t}", end="\n")

    # Q2
    M=[[2,-1],[-1,2]]
    eigval, eigvec = eigensystem(np.array(M))

    # Q3
    ab_trace, ba_trace = trace_proof(np.array(A), np.array(B))
    print(f"ab trace {ab_trace}")
    print(f"ba trace {ba_trace}")

    # Q4
    C=[[4,7],[2,6]]
    C_inv = matrix_inverse(np.array(C))
    print(f"C_inv = {C_inv}")

    ################ PART 2 ################

    # Q1
    (c, mean, variance) = distribution_moments()
    print(f"c: {c}")
    print(f"mean: {mean}")
    print(f"variance: {variance}")

    # Q2
    q2_p2_res = event_probabilities()
    for i, t in enumerate(q2_p2_res):
        print(f"p{i}: {t}")

    # Q3
    print("(MLE, MAP):", coin_estimates())

    # Q4
    gmle = gaussian_mle(data=[1,3,5,7])
    print(gmle)

    # Q5
    sol1 = gaussian_map()
    print(f"Posterior Mode for One Observation: {sol1}")

    # Q6
    prove_var()    

    ################ PART 3 ################
    
    # Q1
    derive_var()

    # Q2
    X = generate_data(mean=[2, -1], covariance=[[3,1.2],[1.2,1]])

    # Q3
    pca_dict = pca(X)
    # print(pca_dict)

    # Q4
    e_v = explained_variance(pca_dict["eigenvalues"])
    print(e_v)

    # make plots
    make_plots(X, pca_dict)
    plt.show()