"""CSE 242 - Assignment 1

My solutions for the linear algebra, probability, and PCA questions.
I kept the calculations next to the functions so it is easier to see
where the answers come from.
"""
import numpy as np

# Part 1, Q1
# I multiplied each row of the first matrix by each column of the second.
# The trace is just the sum of the diagonal entries.
# AB=[[3,2],[2,3]], BA=[[7,6],[-2,-1]], (A+B)^T=[[4,2],[6,0]],
# AB^T=[[11,-4],[4,-1]], trace(A)=4, trace(AB)=6.
def matrix_operations(A=None, B=None):
    A = np.array([[3, 4], [2, 1]], dtype=float) if A is None else np.asarray(A, dtype=float)
    B = np.array([[1, 2], [0, -1]], dtype=float) if B is None else np.asarray(B, dtype=float)
    AB = A @ B
    return AB, B @ A, (A + B).T, A @ B.T, np.trace(A), np.trace(AB)

# Part 1, Q2
# To get the eigenvalues, I set det(M - lambda*I) equal to zero:
# det(M-lambda I)=(2-lambda)^2-1=(lambda-1)(lambda-3).
# lambda=1 has eigenvector [1,1], lambda=3 has eigenvector [1,-1].
# Direct multiplication verifies Mv=lambda*v for both.
def eigensystem(M=None):
    M = np.array([[2, -1], [-1, 2]], dtype=float) if M is None else np.asarray(M, dtype=float)
    return np.linalg.eigh(M)

# Part 1, Q3
# Here is the trace proof, using the definition of matrix multiplication.
# Suppose A is m x n and B is n x m:
# tr(AB) = sum_{i=1}^m (AB)_{ii}
#        = sum_{i=1}^m sum_{j=1}^n A_{ij} B_{ji}
#        = sum_{j=1}^n sum_{i=1}^m B_{ji} A_{ij}
#        = sum_{j=1}^n (BA)_{jj} = tr(BA).
# The third equality exchanges finite sums and uses commutativity of scalars.

# Part 1, Q4
# First I checked the determinant: 4*6-7*2=10.
# Since it is not zero, C has an inverse.
# C^-1=(1/10)[[6,-7],[-2,4]].
def matrix_inverse(C=None):
    C = np.array([[4, 7], [2, 6]], dtype=float) if C is None else np.asarray(C, dtype=float)
    return np.linalg.inv(C)

# Part 2, Q1
# The probabilities have to add up to 1, which gives the constant c.
# 1 = sum_{k>=0} c*rate^k/k! = c*exp(rate), so c=exp(-rate).
# E[X] = c*sum_{k>=1} k*rate^k/k!
#      = c*rate*sum_{j>=0} rate^j/j! = rate.
# E[X(X-1)] = c*rate^2*sum_{j>=0} rate^j/j! = rate^2.
# E[X^2] = rate^2+rate, hence Var(X)=E[X^2]-E[X]^2=rate.
def distribution_moments(rate=3.):
    return float(np.exp(-rate)), float(rate), float(rate)

# Part 2, Q2
# I started with the conditional probability formula.
# P(A and B)=P(A|B)P(B), P(B|A)=P(A and B)/P(A),
# P(A or B)=P(A)+P(B)-P(A and B).
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    intersection = p_a_given_b * p_b
    return intersection, intersection / p_a, p_a + p_b - intersection

# Part 2, Q3
# For MLE I maximize the likelihood of the observed heads and tails.
# For MAP I also include the given prior.
# For h heads in n tosses, L(theta)=theta^h*(1-theta)^(n-h).
# log L=h*log(theta)+(n-h)*log(1-theta).
# Setting derivative h/theta-(n-h)/(1-theta)=0 gives theta=h/n.
# The endpoint cases h=0 and h=n yield MLE 0 and 1 respectively.
# Prior p(theta)=2*theta on [0,1] gives posterior proportional to
# theta^(h+1)*(1-theta)^(n-h). Its interior log derivative is
# (h+1)/theta-(n-h)/(1-theta), giving MAP=(h+1)/(n+1).
# When h=n, the posterior increases to its maximum at theta=1;
# the same formula yields 1. For default h=3,n=5: MLE=.6, MAP=2/3.
def coin_estimates(heads=3, tosses=5):
    return heads / tosses, (heads + 1) / (tosses + 1)

# Part 2, Q4
# I used the log likelihood because it makes the derivatives easier.
# log L(mu,v)=-n/2*log(2*pi)-n/2*log(v)-sum((x_i-mu)^2)/(2*v).
# d(log L)/d(mu)=sum(x_i-mu)/v=0 -> mu_hat=sum(x_i)/n.
# d(log L)/d(v)=-n/(2*v)+sum((x_i-mu)^2)/(2*v^2)=0
# -> v_hat=sum((x_i-mu_hat)^2)/n. For [1,3,5,7], mu=4 and v=5.
# MLE divides by n, not n-1 (the latter is unbiased sample variance).
def gaussian_mle(data=None):
    x = np.array([1, 3, 5, 7], dtype=float) if data is None else np.asarray(data, dtype=float)
    return float(np.mean(x)), float(np.mean((x - np.mean(x)) ** 2))

# Part 2, Q5
# This is similar to the previous problem, except now the prior
# contributes another term to the log posterior.
# log p(mu|x)=constant-(x-mu)^2/(2*obs_var)
#                        -(mu-prior_mean)^2/(2*prior_var).
# Differentiate: (x-mu)/obs_var-(mu-prior_mean)/prior_var=0.
# Thus mu_MAP=(x/obs_var+prior_mean/prior_var)
#             /(1/obs_var+1/prior_var).
# For x=5, obs_var=4, prior N(0,1): (5/4)/(1/4+1)=1.
# Second derivative = -(1/obs_var+1/prior_var)<0.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    return (x / observation_variance + prior_mean / prior_variance) / (1 / observation_variance + 1 / prior_variance)

# Part 2, Q6
# Expanding the squared term in the variance definition gives:
# Var(X)=E[(X-E[X])^2]
# =E[X^2]-2E[X]E[X]+E[X]^2=E[X^2]-E[X]^2.

# Part 3, Q1
# The main idea is to write the projected variance in terms of
# the covariance matrix, then pull the fixed vector u outside E[ ].
# Let mu=E[X] and Sigma=E[(X-mu)(X-mu)^T]. Let ||u||_2=1.
# E[u^T X]=u^T mu by linearity.
# Var(u^T X)=E[(u^T X-u^T mu)^2]
#           =E[(u^T(X-mu))((X-mu)^T u)]
#           =u^T E[(X-mu)(X-mu)^T]u = u^T Sigma u.
# Maximizing u^T Sigma u subject to u^T u=1 leads to
# L(u,lambda)=u^T Sigma u-lambda*(u^T u-1).
# Stationarity: 2*Sigma*u-2*lambda*u=0 -> Sigma*u=lambda*u.
# Therefore the maximizing unit u is a top covariance eigenvector.

# Part 3, Q2
# I generated two Gaussian features with a positive covariance.
# The default covariance gives positive correlation
# rho=1.2/sqrt(3*1)=0.69282; expect an upward-sloping elliptical cloud.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    mean = np.array([2., -1.]) if mean is None else np.asarray(mean, dtype=float)
    covariance = np.array([[3., 1.2], [1.2, 1.]]) if covariance is None else np.asarray(covariance, dtype=float)
    return np.random.default_rng(seed).multivariate_normal(mean, covariance, size=n_samples)

# Part 3, Q3
# PCA starts by centering the data. After that I find the directions
# where the data varies the most by diagonalizing the covariance.
# Center rows, compute sample covariance Xc^T Xc/(n-1),
# diagonalize symmetric covariance, order eigenvalues descending,
# and project each centered observation onto eigenvector columns.
def pca(X):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (len(X) - 1)
    eigenvalues, components = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.)
    components = components[:, order]
    scores = centered @ components
    return {'mean': mean, 'centered': centered, 'covariance': covariance,
            'eigenvalues': eigenvalues, 'components': components, 'scores': scores}

# Part 3, Q4
# The explained variance ratio tells me how much information each
# direction keeps. Dropping PC2 makes the data smaller, but loses
# the variation that was in that direction.
# r_i=lambda_i/(lambda_1+lambda_2); r_1+r_2=1.
# The default population Sigma=[[3,1.2],[1.2,1]] has eigenvalues
# (4 +/- sqrt(9.76))/2 = 3.56205 and 0.43795, so its theoretical
# explained fractions are ~89.05% and ~10.95% respectively.
# The realized sample values vary; always report the fitted sample ratios.
# Keeping PC1 reduces 2D to 1D (50% fewer features) while retaining
# ~89% of population variance; the lost ~11% may contain useful signal.
def explained_variance(eigenvalues):
    values = np.asarray(eigenvalues, dtype=float)
    return values / np.sum(values)

# For the plots I wanted to show both what PCA does geometrically
# and how much variance each direction explains.
# Plot the data with directions through the empirical mean,
# plus the observed explained variance fractions.
def make_plots(X, result):
    import matplotlib.pyplot as plt
    X = np.asarray(X, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X[:, 0], X[:, 1], s=10, alpha=.35, label='Samples')
    center = result['mean']
    for i in range(2):
        direction = result['components'][:, i]
        length = 2 * np.sqrt(max(result['eigenvalues'][i], 0))
        endpoints = np.array([center - length * direction, center + length * direction])
        axes[0].plot(endpoints[:, 0], endpoints[:, 1], linewidth=2.5, label=f'PC{i+1}')
    axes[0].scatter(*center, marker='x', s=80, label='Sample mean')
    axes[0].set(xlabel='Feature 1', ylabel='Feature 2', title='Gaussian Data and PCA Directions')
    axes[0].axis('equal')
    axes[0].legend()
    ratios = explained_variance(result['eigenvalues'])
    bars = axes[1].bar(['PC1', 'PC2'], ratios)
    for bar, ratio in zip(bars, ratios):
        axes[1].text(bar.get_x() + bar.get_width()/2, ratio + .015, f'{ratio:.2%}', ha='center')
    axes[1].set(xlabel='Principal Component', ylabel='Fraction of Total Variance',
                title='Explained Variance', ylim=(0, 1.08))
    fig.tight_layout()
    return fig

# Reflection:
# This assignment brought together matrix operations, probability, and PCA.
# The part that connected the math to the code most clearly was PCA: the
# eigenvectors give the directions, and the eigenvalues show the variance.

if __name__ == '__main__':
    X = generate_data()
    result = pca(X)
    print('Sample mean:', result['mean'])
    print('Sample covariance:\n', result['covariance'])
    print('Eigenvalues:', result['eigenvalues'])
    print('Explained variance:', explained_variance(result['eigenvalues']))
    print('Sample correlation:', np.corrcoef(X, rowvar=False)[0, 1])
    fig = make_plots(X, result)
    fig.savefig('Assignment1_PCA.png', dpi=180)
    import matplotlib.pyplot as plt
    plt.show()
