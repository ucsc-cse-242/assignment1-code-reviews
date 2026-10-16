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
# A = [[3, 4], [2, 1]]
# B = [[1, 2], [0, -1]]
#
# AB = [[3*1 + 4*0, 3*2 + 4*(-1)],
#       [2*1 + 1*0, 2*2 + 1*(-1)]]
#    = [[3, 2], [2, 3]]
#
# BA = [[1*3 + 2*2, 1*4 + 2*1],
#       [0*3 + (-1)*2, 0*4 + (-1)*1]]
#    = [[7, 6], [-2, -1]]
#
# (A+B).T = [[4, 6], [2, 0]].T
#         = [[4, 2], [6, 0]]
#
# A @ B.T = [[3, 4], [2, 1]] @ [[1, 0], [2, -1]]
#         = [[11, -4], [4, -1]]
#
# trace(A) = 3 + 1 = 4
# trace(AB) = 3 + 3 = 6
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    if A is None:
        A = np.array([[3, 4], [2, 1]])
    else:
        A = np.asarray(A)

    if B is None:
        B = np.array([[1, 2], [0, -1]])
    else:
        B = np.asarray(B)

    AB = A @ B
    BA = B @ A
    sum_transpose = (A + B).T
    AB_transpose = A @ B.T
    trace_A = np.trace(A)
    trace_AB = np.trace(AB)

    return AB, BA, sum_transpose, AB_transpose, trace_A, trace_AB


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
# M = [[2, -1],
#      [-1, 2]]
#
# Step 1: Find eigenvalues using det(M - lambda*I) = 0
#
# det([[2-lambda, -1],
#      [-1, 2-lambda]]) = 0
#
# (2-lambda)^2 - 1 = 0
# lambda^2 - 4*lambda + 3 = 0
# (lambda - 1)(lambda - 3) = 0
#
# Eigenvalues: lambda_1 = 1, lambda_2 = 3
#
# Step 2: Find eigenvectors using (M - lambda*I)v = 0
#
# For lambda_1 = 1:
# [[1, -1], [-1, 1]] @ [x, y] = [0, 0]
# x - y = 0, so x = y
# v_1 = [1, 1]
#
# For lambda_2 = 3:
# [[-1, -1], [-1, -1]] @ [x, y] = [0, 0]
# x + y = 0, so x = -y
# v_2 = [1, -1]
#
# Step 3: Verify Mv = lambda*v
#
# M @ [1, 1] = [1, 1] = 1 * [1, 1]
# M @ [1, -1] = [3, -3] = 3 * [1, -1]
#
# Both eigenpairs satisfy Mv = lambda*v.
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE
        if M is None:
        M = np.array([[2, -1],
                      [-1, 2]])
    else:
        M = np.asarray(M)

    # Compute eigenvalues and eigenvectors
    values, vectors = np.linalg.eigh(M)

    return values, vectors



# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE
#
# Let A be an m x n matrix and B be an n x m matrix.
# Then AB is m x m and BA is n x n.
#
# By definition, the trace is the sum of diagonal elements:
#
# trace(AB) = sum_i (AB)_ii
#
# Since (AB)_ii = sum_j A_ij * B_ji:
#
# trace(AB) = sum_i sum_j A_ij * B_ji
#
# Rearranging the order of summation:
#
#           = sum_j sum_i B_ji * A_ij
#
# Since (BA)_jj = sum_i B_ji * A_ij:
#
#           = sum_j (BA)_jj
#           = trace(BA)
#
# Therefore, trace(AB) = trace(BA).
#
# This holds for all compatible matrices A and B.


# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
#
# C = [[4, 7],
#      [2, 6]]
#
# Step 1: Calculate the determinant.
#
# For a 2x2 matrix [[a,b],[c,d]]:
# det(C) = ad - bc
#
# det(C) = (4)(6) - (7)(2)
#        = 24 - 14
#        = 10
#
# Since det(C) != 0, C is invertible.
#
# Step 2: Apply the 2x2 inverse formula.
#
# C^-1 = (1/det(C)) * [[d, -b],
#                     [-c, a]]
#
# C^-1 = (1/10) * [[6, -7],
#                  [-2, 4]]
#
# Step 3: Simplify.
#
# C^-1 = [[0.6, -0.7],
#         [-0.2, 0.4]]
#
# Step 4: Verify C @ C^-1 = I.
#
# [[4, 7], [2, 6]] @ [[0.6, -0.7], [-0.2, 0.4]]
#
# = [[4(0.6)+7(-0.2), 4(-0.7)+7(0.4)],
#    [2(0.6)+6(-0.2), 2(-0.7)+6(0.4)]]
#
# = [[1, 0],
#    [0, 1]]
#
# Therefore, C^-1 is correct.


def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    if C is None:
        C = np.array([[4, 7],
                      [2, 6]])
    else:
        C = np.asarray(C)

    inverse = np.linalg.inv(C)

    return inverse


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
#
# Given:
# P(X=k) = c * rate^k / k!, for k >= 0
#
# Step 1: Find the normalization constant c.
#
# Since all probabilities must sum to 1:
#
# sum(k=0 to infinity) c * rate^k / k! = 1
#
# Using the Taylor series for the exponential function:
# e^rate = sum(k=0 to infinity) rate^k / k!
#
# c * e^rate = 1
# c = e^(-rate)
#
# For rate = 3:
# c = e^(-3) = 0.049787
#
# Step 2: Find the expectation E[X].
#
# E[X] = sum(k=0 to infinity) k * P(X=k)
#      = c * sum(k=1 to infinity) k * rate^k / k!
#
# Since k/k! = 1/(k-1)!:
#
# E[X] = c * rate * sum(k=1 to infinity) rate^(k-1)/(k-1)!
#      = c * rate * e^rate
#      = rate
#
# For rate = 3:
# E[X] = 3
#
# Step 3: Find the variance Var(X).
#
# Var(X) = E[X^2] - (E[X])^2
#
# Use the identity X^2 = X(X-1) + X.
#
# E[X(X-1)] = c * sum(k=2 to infinity)
#             k(k-1) * rate^k / k!
#
# Since k(k-1)/k! = 1/(k-2)!:
#
# E[X(X-1)] = c * rate^2 * e^rate
#           = rate^2
#
# Therefore:
# E[X^2] = rate^2 + rate
#
# Var(X) = (rate^2 + rate) - rate^2
#        = rate
#
# For rate = 3:
# Var(X) = 3
#
# This is a Poisson distribution with parameter rate.
# Final results:
# c = e^(-3), E[X] = 3, Var(X) = 3


def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    if rate <= 0:
        raise ValueError("rate must be positive")

    c = np.exp(-rate)
    mean = rate
    variance = rate

    return c, mean, variance




# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
#
# Given:
# P(A) = 0.5
# P(B) = 0.3
# P(A|B) = 0.6
#
# Step 1: Find P(A intersect B).
#
# Using the conditional probability formula:
# P(A|B) = P(A intersect B) / P(B)
#
# Rearranging:
# P(A intersect B) = P(A|B) * P(B)
#                  = 0.6 * 0.3
#                  = 0.18
#
# Step 2: Find P(B|A).
#
# Using conditional probability:
# P(B|A) = P(A intersect B) / P(A)
#
# Substituting:
# P(B|A) = 0.18 / 0.5
#        = 0.36
#
# Step 3: Find P(A union B).
#
# Using the addition rule:
# P(A union B) = P(A) + P(B) - P(A intersect B)
#
# Substituting:
# P(A union B) = 0.5 + 0.3 - 0.18
#              = 0.62
#
# Final results:
# P(A intersect B) = 0.18
# P(B|A) = 0.36
# P(A union B) = 0.62


def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE

    # Calculate intersection
    intersection = p_a_given_b * p_b

    # Calculate conditional probability P(B|A)
    b_given_a = intersection / p_a

    # Calculate union
    union = p_a + p_b - intersection

    return intersection, b_given_a, union



# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
#
# Given:
# Observations: H, H, T, H, T
# Number of heads = 3
# Number of tosses = 5
# Let theta = probability of getting heads.
#
# Step 1: Derive the Maximum Likelihood Estimate (MLE).
#
# The likelihood function is:
# L(theta) = theta^3 * (1-theta)^2
#
# Taking the natural logarithm:
# log L(theta) = 3*ln(theta) + 2*ln(1-theta)
#
# Differentiate with respect to theta:
# d/dtheta log L(theta) = 3/theta - 2/(1-theta)
#
# Set the derivative equal to zero:
# 3/theta - 2/(1-theta) = 0
# 3(1-theta) = 2*theta
# 3 = 5*theta
# theta_MLE = 3/5 = 0.6
#
# The second derivative is negative for 0 < theta < 1,
# confirming that this is a maximum.
#
# Step 2: Derive the Maximum A Posteriori Estimate (MAP).
#
# Given prior density:
# p(theta) = 2*theta, for 0 <= theta <= 1
#
# By Bayes' theorem:
# p(theta|data) is proportional to L(theta)*p(theta)
#
# p(theta|data) proportional to
# theta^3 * (1-theta)^2 * 2*theta
#
# = 2*theta^4 * (1-theta)^2
#
# Since the constant 2 does not affect maximization,
# maximize theta^4 * (1-theta)^2.
#
# Taking the natural logarithm:
# log posterior = 4*ln(theta) + 2*ln(1-theta) + constant
#
# Differentiate:
# d/dtheta = 4/theta - 2/(1-theta)
#
# Set equal to zero:
# 4/theta - 2/(1-theta) = 0
# 4(1-theta) = 2*theta
# 4 = 6*theta
# theta_MAP = 4/6 = 2/3 = 0.6667
#
# The second derivative is negative for 0 < theta < 1,
# confirming a maximum.
#
# General formulas:
# MLE = heads / tosses
# MAP = (heads + 1) / (tosses + 1)
#
# Boundary cases:
# If heads = 0, MLE = 0 and MAP = 1/(tosses+1).
# If heads = tosses, MLE = 1 and MAP = 1.
#
# Final results:
# MLE = 0.6
# MAP = 0.6667


def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE

    # Maximum Likelihood Estimate
    mle = heads / tosses

    # Maximum A Posteriori Estimate
    map_estimate = (heads + 1) / (tosses + 1)

    return mle, map_estimate




# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
#
# Given:
# Data = [1, 3, 5, 7]
# n = 4
#
# Assume observations are independent and identically
# distributed (i.i.d.) from N(mu, sigma^2).
#
# Step 1: Write the Gaussian likelihood function.
#
# L(mu, sigma^2) =
#   product(i=1 to n) [
#       1/sqrt(2*pi*sigma^2) *
#       exp(-(x_i-mu)^2/(2*sigma^2))
#   ]
#
# Simplifying:
# L(mu, sigma^2) =
#   (2*pi*sigma^2)^(-n/2) *
#   exp(-sum((x_i-mu)^2)/(2*sigma^2))
#
# Step 2: Take the natural logarithm.
#
# log L = -(n/2)*ln(2*pi)
#         -(n/2)*ln(sigma^2)
#         -sum((x_i-mu)^2)/(2*sigma^2)
#
# Step 3: Differentiate with respect to mu.
#
# d(log L)/d(mu) = sum(x_i-mu)/sigma^2
#
# Set equal to zero:
# sum(x_i-mu) = 0
# sum(x_i) - n*mu = 0
#
# mu_MLE = sum(x_i)/n
#
# For the given data:
# mu_MLE = (1+3+5+7)/4
#        = 16/4
#        = 4
#
# Step 4: Differentiate with respect to sigma^2.
#
# Let v = sigma^2.
#
# d(log L)/dv = -n/(2*v)
#                +sum((x_i-mu)^2)/(2*v^2)
#
# Set equal to zero:
# -n*v + sum((x_i-mu)^2) = 0
#
# v = sum((x_i-mu)^2)/n
#
# Therefore:
# sigma^2_MLE = sum((x_i-mu_MLE)^2)/n
#
# Step 5: Substitute the sample values.
#
# sigma^2_MLE = [(1-4)^2 + (3-4)^2
#                + (5-4)^2 + (7-4)^2]/4
#
#             = [9 + 1 + 1 + 9]/4
#             = 20/4
#             = 5
#
# Final results:
# Mean MLE = 4
# Variance MLE = 5
#
# Note: MLE variance divides by n, not n-1.
# The unbiased sample variance would be 20/3.


def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    if data is None:
        data = np.array([1, 3, 5, 7])
    else:
        data = np.asarray(data, dtype=float)

    # Calculate maximum likelihood estimate of mean
    mean_mle = np.mean(data)

    # Calculate maximum likelihood estimate of variance
    variance_mle = np.var(data, ddof=0)

    return mean_mle, variance_mle




# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
#
# Given:
# Observation x = 5
# Observation variance = 4
# Prior mean = 0
# Prior variance = 1
#
# Step 1: Define the likelihood and prior.
#
# Likelihood:
# x | mu ~ N(mu, observation_variance)
#
# Prior:
# mu ~ N(prior_mean, prior_variance)
#
# Step 2: Apply Bayes' theorem.
#
# P(mu|x) is proportional to P(x|mu) * P(mu)
#
# Ignoring constants independent of mu:
#
# P(mu|x) proportional to
# exp(-(x-mu)^2/(2*observation_variance)) *
# exp(-(mu-prior_mean)^2/(2*prior_variance))
#
# Step 3: Take the log posterior.
#
# log P(mu|x) =
#   -(x-mu)^2/(2*observation_variance)
#   -(mu-prior_mean)^2/(2*prior_variance)
#   + constant
#
# Step 4: Differentiate with respect to mu.
#
# d/dmu log P(mu|x) =
#   (x-mu)/observation_variance
#   -(mu-prior_mean)/prior_variance
#
# Set the derivative equal to zero:
#
# (x-mu)/observation_variance =
#   (mu-prior_mean)/prior_variance
#
# Rearranging:
#
# x/prior? No; collect terms correctly:
#
# mu * (1/observation_variance + 1/prior_variance)
#   = x/observation_variance + prior_mean/prior_variance
#
# Therefore:
#
# mu_MAP =
#   (x/observation_variance + prior_mean/prior_variance) /
#   (1/observation_variance + 1/prior_variance)
#
# Step 5: Substitute the given values.
#
# mu_MAP = (5/4 + 0/1) / (1/4 + 1/1)
#        = 1.25 / 1.25
#        = 1
#
# Step 6: Verify this is a maximum.
#
# Second derivative:
# -1/observation_variance - 1/prior_variance
#
# Since both variances are positive, this is negative.
# Therefore, the stationary point is a maximum.
#
# Final result:
# mu_MAP = 1


def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE

    if observation_variance <= 0 or prior_variance <= 0:
        raise ValueError("Variances must be positive")

    # Calculate precision (inverse variance)
    observation_precision = 1 / observation_variance
    prior_precision = 1 / prior_variance

    # Calculate posterior mode
    map_estimate = (
        x * observation_precision + prior_mean * prior_precision
    ) / (observation_precision + prior_precision)

    return map_estimate




# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE
#
# By definition, variance measures the expected squared
# deviation from the mean.
#
# Var(X) = E[(X - E[X])^2]
#
# Step 1: Expand the squared term.
#
# (X - E[X])^2 = X^2 - 2*X*E[X] + E[X]^2
#
# Step 2: Apply the expectation operator.
#
# Var(X) = E[X^2 - 2*X*E[X] + E[X]^2]
#
# Using linearity of expectation:
#
# Var(X) = E[X^2] - 2*E[X]*E[X] + E[X]^2
#
# Step 3: Simplify.
#
# Var(X) = E[X^2] - 2*E[X]^2 + E[X]^2
#
# Therefore:
#
# Var(X) = E[X^2] - E[X]^2
#
# This identity holds whenever the second moment is finite.


# PART 3 (30 points), Q1:
# Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE
#
# Step 1: Define covariance.
#
# Cov(X_i, X_j) = E[(X_i - E[X_i]) * (X_j - E[X_j])]
#
# Covariance measures how two random variables vary together.
#
# Positive covariance: variables tend to increase together.
# Negative covariance: one tends to increase as the other decreases.
# Zero covariance: no linear relationship is present.
#
# Step 2: Define the covariance matrix.
#
# Let X be a random vector with mean mu = E[X].
#
# Sigma = E[(X - mu)(X - mu).T]
#
# The diagonal entries represent individual variances.
# The off-diagonal entries represent pairwise covariances.
#
# Step 3: Define a linear combination of X.
#
# Let u be a fixed vector and Y = u.T @ X.
#
# We want to derive Var(Y).
#
# Step 4: Apply the definition of variance.
#
# Var(Y) = E[(Y - E[Y])^2]
#
# Since E[Y] = E[u.T @ X] = u.T @ mu:
#
# Var(Y) = E[(u.T @ X - u.T @ mu)^2]
#
#        = E[(u.T @ (X - mu))^2]
#
# Step 5: Express the square as matrix multiplication.
#
# Var(Y) = E[u.T @ (X-mu) @ (X-mu).T @ u]
#
# Step 6: Move the constant vectors outside expectation.
#
# Var(Y) = u.T @ E[(X-mu) @ (X-mu).T] @ u
#
# By definition:
#
# E[(X-mu) @ (X-mu).T] = Sigma
#
# Therefore:
#
# Var(u.T @ X) = u.T @ Sigma @ u
#
# This formula gives the variance of any linear combination
# of random variables using their covariance matrix.


# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
#
# Step 1: Define the bivariate Gaussian distribution.
#
# X ~ N(mu, Sigma)
#
# Default mean vector:
# mu = [2, -1]
#
# Default covariance matrix:
# Sigma = [[3, 1.2],
#          [1.2, 1]]
#
# Step 2: Interpret the covariance matrix.
#
# Var(X1) = 3
# Var(X2) = 1
# Cov(X1, X2) = 1.2
#
# Step 3: Calculate the theoretical correlation.
#
# Correlation formula:
#
# rho = Cov(X1,X2) / sqrt(Var(X1)*Var(X2))
#
# rho = 1.2 / sqrt(3*1)
#     = 1.2 / sqrt(3)
#     = 0.6928
#
# The positive correlation indicates that larger X1 values
# tend to be associated with larger X2 values.
#
# Step 4: Generate random samples.
#
# Use NumPy's multivariate_normal() to sample 500 points
# from the specified bivariate Gaussian distribution.
#
# A fixed seed ensures reproducibility.
#
# Step 5: Describe the expected observations.
#
# The samples should form an elliptical point cloud centered
# approximately around (2, -1).
#
# Because covariance is positive, the ellipse should be
# tilted upward from left to right.
#
# The empirical correlation should be close to 0.6928,
# but will generally differ due to random sampling.
#
# Increasing the sample size generally improves the agreement
# between empirical and theoretical correlation.


def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE

    if mean is None:
        mean = np.array([2, -1])
    else:
        mean = np.asarray(mean, dtype=float)

    if covariance is None:
        covariance = np.array([[3, 1.2],
                               [1.2, 1]])
    else:
        covariance = np.asarray(covariance, dtype=float)

    # Initialize random number generator with fixed seed
    rng = np.random.default_rng(seed)

    # Generate samples from multivariate Gaussian distribution
    samples = rng.multivariate_normal(
        mean=mean,
        cov=covariance,
        size=n_samples
    )

    return samples




# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
#
# Step 1: Center the data.
#
# Given a data matrix X with n observations and 2 features,
# calculate the mean of each feature:
#
# mu = (1/n) * sum(X_i)
#
# Subtract the mean from every observation:
#
# X_centered = X - mu
#
# Centering ensures that each feature has a mean of zero.
#
# Step 2: Compute the covariance matrix.
#
# Sigma = (1/(n-1)) * X_centered.T @ X_centered
#
# For two features:
#
# Sigma = [[Var(X1), Cov(X1,X2)],
#          [Cov(X1,X2), Var(X2)]]
#
# The covariance matrix describes the spread of the data
# and the relationship between the two features.
#
# Step 3: Perform eigendecomposition.
#
# Solve:
# Sigma @ v = lambda * v
#
# Eigenvectors represent directions in feature space.
# Eigenvalues represent variance along those directions.
#
# Step 4: Derive the PCA objective.
#
# For a unit vector u, the projected data is:
#
# Y = X_centered @ u
#
# From Part 3 Q1:
#
# Var(Y) = u.T @ Sigma @ u
#
# PCA seeks the direction maximizing this variance:
#
# maximize u.T @ Sigma @ u
# subject to u.T @ u = 1
#
# Using a Lagrange multiplier:
#
# L(u,lambda) = u.T @ Sigma @ u - lambda*(u.T @ u - 1)
#
# Taking the derivative with respect to u:
#
# 2*Sigma @ u - 2*lambda*u = 0
#
# Therefore:
#
# Sigma @ u = lambda*u
#
# This is the eigenvalue equation.
#
# The eigenvector with the largest eigenvalue is PC1.
# The eigenvector with the second largest eigenvalue is PC2.
#
# Step 5: Project the centered data.
#
# Let V contain the eigenvectors as columns.
#
# scores = X_centered @ V
#
# The scores represent observations in PCA coordinates.
#
# Step 6: Explain the PCA plot.
#
# PC1 is the direction of greatest variance in the dataset.
# PC2 is perpendicular to PC1 and captures the remaining variance.
#
# For the Gaussian data generated in Q2, the positive
# covariance produces an upward-tilted elliptical cloud.
#
# PCA rotates the coordinate system to align with the
# major and minor axes of that ellipse.
#
# In a PCA score plot, the transformed data is centered
# near (0,0), with zero sample covariance between PCs.
#
# If PC1 explains most of the total variance, we can
# reduce the data from 2D to 1D while retaining much
# of the original variation.
#
# Explained variance ratio:
#
# ratio_i = eigenvalue_i / sum(eigenvalues)
#
# Larger ratios indicate more variance captured by that PC.
#
# PCA is an unsupervised dimensionality reduction method.


def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # YOUR CODE HERE

    # Convert input into a NumPy array without modifying original
    X = np.asarray(X, dtype=float)

    # Step 1: Calculate feature means
    mean = np.mean(X, axis=0)

    # Step 2: Center the data
    centered = X - mean

    # Step 3: Compute covariance matrix
    covariance = (centered.T @ centered) / (X.shape[0] - 1)

    # Step 4: Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Step 5: Sort eigenvalues in descending order
    indices = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[indices]
    components = eigenvectors[:, indices]

    # Step 6: Project centered data onto principal components
    scores = centered @ components

    # Return requested results
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
#
# Step 1: Calculate the explained variance ratio.
#
# Each eigenvalue represents the variance captured by
# its corresponding principal component.
#
# The total variance is the sum of all eigenvalues:
#
# Total variance = sum(lambda_i)
#
# The explained variance ratio for component i is:
#
# ratio_i = lambda_i / sum(lambda_i)
#
# Step 2: Interpret the variance fractions.
#
# Using the theoretical covariance matrix from Q2:
#
# Sigma = [[3, 1.2],
#          [1.2, 1]]
#
# The eigenvalues are approximately:
# lambda_1 = 3.562
# lambda_2 = 0.438
#
# Total variance = 3.562 + 0.438 = 4
#
# PC1 explained variance = 3.562 / 4 = 0.8905 (89.05%)
# PC2 explained variance = 0.438 / 4 = 0.1095 (10.95%)
#
# Therefore, PC1 captures approximately 89% of the
# variance, while PC2 captures approximately 11%.
#
# Step 3: Explain dimensionality reduction.
#
# PCA allows us to reduce the number of dimensions
# by keeping only the principal components that
# explain the most variance.
#
# For example, reducing this dataset from 2D to 1D
# by keeping only PC1 retains approximately 89%
# of the total variance.
#
# However, the remaining 11% of variance is lost.
#
# Step 4: Discuss the tradeoff.
#
# Advantages:
# - Reduces data dimensionality and computational cost.
# - Makes high-dimensional datasets easier to visualize.
# - Removes correlations between retained components.
# - Can reduce noise if low-variance directions are noisy.
#
# Disadvantages:
# - Some information is lost when components are removed.
# - Different observations may overlap after projection.
# - Low-variance components may contain useful information.
# - Principal components may be harder to interpret than
#   the original features.
#
# For example, when projecting 2D data onto a single
# principal component, points that differ along PC2
# may become indistinguishable.
#
# Conclusion:
# PCA provides a tradeoff between dimensionality reduction
# and information retention. Keeping fewer components
# simplifies the dataset but discards some variance.
#
# For this dataset, retaining only PC1 is reasonable
# if preserving approximately 89% of variance is sufficient.


def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE

    # Convert eigenvalues to NumPy array
    eigenvalues = np.asarray(eigenvalues, dtype=float)

    # Calculate total variance
    total_variance = np.sum(eigenvalues)

    # Calculate fraction explained by each component
    variance_ratios = eigenvalues / total_variance

    return variance_ratios




# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
#
# Plot 1: Gaussian samples and principal components
#
# The scatter plot shows the distribution of the generated
# Gaussian samples in two dimensions.
#
# PC1 points in the direction of greatest variance.
# PC2 is perpendicular to PC1 and captures the remaining
# variance.
#
# The principal component arrows originate at the sample
# mean and are scaled by the square root of their eigenvalues
# to illustrate the relative standard deviations.
#
# Since the covariance between X1 and X2 is positive,
# the data should form an upward-tilted elliptical cloud.
#
# Plot 2: Explained variance ratios
#
# Each bar represents the proportion of total variance
# explained by a principal component.
#
# For the theoretical covariance matrix from Q2,
# PC1 explains approximately 89% of the total variance,
# while PC2 explains approximately 11%.
#
# Keeping only PC1 reduces the data from 2D to 1D while
# retaining most of the variance.
#
# However, information along PC2 is lost, meaning some
# distinct observations may overlap after projection.


def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    import matplotlib.pyplot as plt

    X = np.asarray(X)

    # Extract PCA results
    mean = result["mean"]
    components = result["components"]
    eigenvalues = result["eigenvalues"]

    # Calculate explained variance fractions
    ratios = explained_variance(eigenvalues)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Gaussian samples and principal directions
    ax1 = axes[0]

    ax1.scatter(
        X[:, 0], X[:, 1],
        alpha=0.4,
        s=15,
        label="Gaussian Samples"
    )

    # Mark the sample mean
    ax1.scatter(
        mean[0], mean[1],
        color="black",
        marker="x",
        s=100,
        label="Sample Mean"
    )

    # Draw principal component directions
    for i in range(2):
        direction = components[:, i]
        scale = 2 * np.sqrt(max(eigenvalues[i], 0))
        vector = direction * scale

        ax1.arrow(
            mean[0], mean[1],
            vector[0], vector[1],
            width=0.025,
            head_width=0.15,
            length_includes_head=True,
            color=["red", "green"][i],
            label=f"PC{i+1}"
        )

    ax1.set_title("Gaussian Samples and Principal Components")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="datalim")

    # Plot 2: Explained variance fractions
    ax2 = axes[1]

    labels = ["PC1", "PC2"]

    bars = ax2.bar(
        labels,
        ratios,
        color=["red", "green"],
        alpha=0.7
    )

    # Add percentage labels above bars
    for bar, ratio in zip(bars, ratios):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{ratio * 100:.1f}%",
            ha="center"
        )

    ax2.set_title("Explained Variance by Principal Component")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Fraction of Total Variance")
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    return fig



# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE
#Used matplotlib and numpy, code made with the assistance of AI.


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # PART 1: Linear Algebra
    print("=== PART 1: LINEAR ALGEBRA ===")

    print("\nQ1: Matrix Operations")
    print(matrix_operations())

    print("\nQ2: Eigenvalues and Eigenvectors")
    print(eigensystem())

    print("\nQ4: Matrix Inverse")
    print(matrix_inverse())

    # PART 2: Probability and Statistics
    print("\n=== PART 2: PROBABILITY AND STATISTICS ===")

    print("\nQ1: Distribution Moments")
    print(distribution_moments())

    print("\nQ2: Event Probabilities")
    print(event_probabilities())

    print("\nQ3: Coin MLE and MAP")
    print(coin_estimates())

    print("\nQ4: Gaussian MLE")
    print(gaussian_mle())

    print("\nQ5: Gaussian MAP")
    print(gaussian_map())

    # PART 3: PCA
    print("\n=== PART 3: PCA ===")

    # Generate Gaussian data
    X = generate_data(n_samples=500, seed=0)

    print("\nQ2: Generated Data")
    print("Data shape:", X.shape)
    print("Observed correlation:", np.corrcoef(X, rowvar=False)[0, 1])

    # Perform PCA
    result = pca(X)

    print("\nQ3: PCA Results")
    print("Mean:", result["mean"])
    print("Covariance:\n", result["covariance"])
    print("Eigenvalues:", result["eigenvalues"])
    print("Principal Components:\n", result["components"])

    # Calculate explained variance
    ratios = explained_variance(result["eigenvalues"])

    print("\nQ4: Explained Variance")
    print("Explained variance ratios:", ratios)
    print("Total explained variance:", np.sum(ratios))

    # Generate and save figures
    print("\nGenerating PCA figures...")

    fig = make_plots(X, result)
    fig.savefig("pca_results.png", dpi=300, bbox_inches="tight")

    # Display figures
    plt.show()

    print("Figures saved as pca_results.png")
