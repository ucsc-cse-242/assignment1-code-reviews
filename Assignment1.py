"""CSE 242 Assignment 1: mathematical foundations and NumPy PCA.

Run: python Assignment1.py
Requires NumPy and Matplotlib. Figures are saved beside this file only when
executed as a script. Importing this module performs no file or network I/O.
AI assistance: OpenAI Codex generated this solution and its local checks.
Submit the actual conversation log as required by the course policy.
"""
import numpy as np


# P1 Q1: Row-by-column multiplication gives
# AB=[[3,2],[2,3]], BA=[[7,6],[-2,-1]], (A+B).T=[[4,2],[6,0]],
# A@B.T=[[11,-4],[4,-1]]. Diagonal sums give tr(A)=4, tr(AB)=6.
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB)."""
    A = np.asarray([[3, 4], [2, 1]] if A is None else A, dtype=float)
    B = np.asarray([[1, 2], [0, -1]] if B is None else B, dtype=float)
    AB = A @ B
    return AB, B @ A, (A + B).T, A @ B.T, np.trace(A), np.trace(AB)


# P1 Q2: det(M-lambda I)=(2-lambda)^2-1=(lambda-1)(lambda-3).
# lambda=1: x=y; v1=(1,1)/sqrt(2); Mv1=v1.
# lambda=3: x=-y; v2=(1,-1)/sqrt(2); Mv2=3v2.
def eigensystem(M=None):
    """Return eigenvalues and paired orthonormal columns for symmetric M."""
    M = np.asarray([[2, -1], [-1, 2]] if M is None else M, dtype=float)
    return np.linalg.eigh(M)


# P1 Q3: For A of shape (m,n), B of shape (n,m),
# tr(AB)=sum_i sum_j A_ij B_ji = sum_j sum_i B_ji A_ij=tr(BA).
# Finite sums can be interchanged, and scalar multiplication commutes.

# P1 Q4: det(C)=4*6-7*2=10 != 0, so C is invertible.
# adj(C)=[[6,-7],[-2,4]]; C^-1=adj(C)/10.
# C@adj(C)=[[10,0],[0,10]], verifying the inverse.
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C."""
    C = np.asarray([[4, 7], [2, 6]] if C is None else C, dtype=float)
    return np.linalg.inv(C)


# P2 Q1: 1=c*sum_k rate^k/k!=c*exp(rate), so c=exp(-rate).
# E[X]=c*rate*sum_{j>=0}rate^j/j!=rate.
# E[X(X-1)]=rate^2, E[X^2]=rate^2+rate, Var(X)=rate.
# At rate=3, (c,mean,variance)=(exp(-3),3,3).
def distribution_moments(rate=3.):
    """Return normalization constant, mean and variance for rate>0."""
    return np.exp(-rate), float(rate), float(rate)


# P2 Q2: P(A and B)=P(A|B)P(B)=0.6*0.3=0.18.
# P(B|A)=0.18/0.5=0.36; P(A or B)=0.5+0.3-0.18=0.62.
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return intersection, reverse conditional, and union probabilities."""
    intersection = p_a_given_b * p_b
    return intersection, intersection / p_a, p_a + p_b - intersection


# P2 Q3: For h heads in n independent tosses, L(theta)=theta^h(1-theta)^(n-h).
# For the specific ordered sequence there is no binomial coefficient.
# d log L/d theta=h/theta-(n-h)/(1-theta)=0 => MLE=h/n.
# Prior p(theta)=2theta is Beta(2,1), so posterior is Beta(h+2,n-h+1).
# Its log derivative is (h+1)/theta-(n-h)/(1-theta), giving MAP=(h+1)/(n+1).
# Interior second derivatives are negative. For all heads both modes are 1;
# for no heads MLE=0 and MAP=1/(n+1). At (h,n)=(3,5): MLE=3/5, MAP=2/3.
def coin_estimates(heads=3, tosses=5):
    """Return MLE and MAP under density 2*theta, including boundary cases."""
    return heads / tosses, (heads + 1) / (tosses + 1)


# P2 Q4: Write v=sigma^2>0. log L=-n/2 log(2*pi*v)-sum_i(x_i-mu)^2/(2v).
# d/dmu log L=sum_i(x_i-mu)/v=0 => mu_hat=mean(x).
# d/dv log L=-n/(2v)+sum_i(x_i-mu)^2/(2v^2)=0 => v_hat=SSE/n.
# [1,3,5,7] has mean 4 and SSE=9+1+1+9=20; variance MLE=20/4=5.
# This uses divisor n, unlike the unbiased sample variance with divisor n-1.
def gaussian_mle(data=None):
    """Return Gaussian mean MLE and variance MLE for nonconstant 1D samples."""
    data = np.asarray([1, 3, 5, 7] if data is None else data, dtype=float)
    mean = np.mean(data)
    return float(mean), float(np.mean((data - mean) ** 2))


# P2 Q5: log posterior=constant-(x-mu)^2/(2v)-(mu-m0)^2/(2t).
# Derivative: (x-mu)/v-(mu-m0)/t=0; second derivative=-1/v-1/t<0.
# mu_MAP=(x/v+m0/t)/(1/v+1/t). Here (5/4)/(1/4+1)=1.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode; both variance arguments must be positive."""
    return ((x / observation_variance + prior_mean / prior_variance)
            / (1 / observation_variance + 1 / prior_variance))


# P2 Q6: Assume E[X^2]<infinity, and let m=E[X].
# Var(X)=E[(X-m)^2]=E[X^2]-2mE[X]+m^2=E[X^2]-(E[X])^2.

# P3 Q1: Let mu=E[X], Sigma=E[(X-mu)(X-mu).T]. For fixed u,
# E[u.T X]=u.T mu. Var(u.T X)=E[(u.T(X-mu))^2]
# =E[u.T(X-mu)(X-mu).T u]=u.T Sigma u.
# The identity holds for any fixed u; ||u||=1 prevents arbitrary rescaling.

# P3 Q2: Generate 500 observations from N([2,-1],[[3,1.2],[1.2,1]]).
# Population correlation is 1.2/sqrt(3)=0.69282: an upward tilted ellipse.
# A local seeded Generator ensures reproducibility without changing global RNG.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible (n_samples,2) Gaussian samples without mutation."""
    mean = np.asarray([2, -1] if mean is None else mean, dtype=float)
    covariance = np.asarray([[3, 1.2], [1.2, 1]] if covariance is None
                            else covariance, dtype=float)
    return np.random.default_rng(seed).multivariate_normal(mean, covariance, n_samples)


# P3 Q3: Centering estimates X-mu. S=Z.T@Z/(n-1) estimates Sigma, so the
# sample variance along u is u.T@S@u. Maximize subject to u.T@u=1:
# gradient of u.T@S@u-lambda*(u.T@u-1) is 2*S@u-2*lambda*u=0.
# Thus S@u=lambda*u, and largest lambda yields greatest projected variance.
# eigh handles symmetric S, including repeated eigenvalues and zero rank.
def pca(X):
    """Return mean, centered, covariance, eigenvalues, components and scores.

    X has shape (n,2), n>=2. Use divisor n-1 and descending eigenvalues;
    eigenvectors are orthonormal columns. Input is not modified.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (X.shape[0] - 1)
    eigenvalues, components = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    # Covariance is PSD; remove only numerical negative roundoff.
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = components[:, order]
    return dict(mean=mean, centered=centered, covariance=covariance,
                eigenvalues=eigenvalues, components=components,
                scores=centered @ components)


# P3 Q4: Fraction_j=lambda_j/sum(lambda). With one retained component,
# reconstruction=mean+score_1*u_1.T and residual lies along PC2.
# Total squared reconstruction error=(n-1)*lambda_2. Retaining PC1 reduces
# two coordinates to one but discards all PC2 information, which may matter
# for a downstream task. Large explained variance is not a guarantee of utility.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative values with positive sum."""
    values = np.asarray(eigenvalues, dtype=float)
    return values / values.sum()


# P3 plots: Directions pass through the sample mean; arrow lengths are two
# projected standard deviations. Equal axis scales preserve their geometry.
def make_plots(X, result):
    """Return a labeled Matplotlib Figure without showing or saving it."""
    import matplotlib.pyplot as plt

    X = np.asarray(X)
    values = result['eigenvalues']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    colors = ['#d45532', '#7146a3']
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=12, alpha=.35, color='#22798b')
    mean = result['mean']
    for j, color in enumerate(colors):
        delta = 2 * np.sqrt(values[j]) * result['components'][:, j]
        ax.annotate('', xy=mean + delta, xytext=mean - delta,
                    arrowprops=dict(arrowstyle='<->', color=color, lw=2.5))
        ax.plot([], [], color=color, lw=2.5, label=f'PC{j+1}')
    ax.scatter(*mean, color='black', s=24, label='Sample mean')
    ax.set(xlabel='Feature 1', ylabel='Feature 2', title='Data and principal directions')
    ax.axis('equal')
    ax.legend(fontsize=8)

    ax = axes[1]
    if values.sum() > 0:
        fractions = explained_variance(values)
        ax.bar(['PC1', 'PC2'], fractions, color=colors)
        for j, value in enumerate(fractions):
            ax.text(j, value + .025, f'{value:.2%}', ha='center')
    else:
        ax.text(.5, .5, 'Zero total variance:\nfractions undefined',
                ha='center', transform=ax.transAxes)
    ax.set(ylim=(0, 1.08), xlabel='Principal component',
           ylabel='Fraction of total variance', title='Explained variance')

    ax = axes[2]
    reconstructed = mean + result['scores'][:, :1] @ result['components'][:, :1].T
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=.2, color='#22798b', label='Original')
    ax.scatter(reconstructed[:, 0], reconstructed[:, 1], s=10, alpha=.5,
               color=colors[0], label='One-PC reconstruction')
    ax.set(xlabel='Feature 1', ylabel='Feature 2', title='Reduction from 2D to 1D')
    ax.axis('equal')
    ax.legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
    return fig


# REFLECTION: Codex AI generated the derivations, implementation, report,
# figures and local validation. No student-authored contribution is claimed.
# Resources: course Assignment1 README and starter Assignment1.py at
# https://github.com/ucsc-cse-242/assignments/tree/main/Assignment1
# This solves the stated tasks using NumPy; no fitted library PCA is used.
# Review the solution and submit the actual AI conversation log with the work.

if __name__ == '__main__':
    import json
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(__file__).resolve().parent
    X = generate_data()
    result = pca(X)
    figure = make_plots(X, result)
    figure.savefig(output / 'pca_visualizations.png', dpi=200)
    plt.close(figure)
    reconstructed = result['mean'] + result['scores'][:, :1] @ result['components'][:, :1].T
    summary = {
        'n_samples': len(X), 'seed': 0,
        'sample_mean': result['mean'].tolist(),
        'sample_covariance': result['covariance'].tolist(),
        'sample_correlation': float(np.corrcoef(X, rowvar=False)[0, 1]),
        'eigenvalues': result['eigenvalues'].tolist(),
        'components_columns': result['components'].tolist(),
        'explained_variance_ratio': explained_variance(result['eigenvalues']).tolist(),
        'one_pc_mean_squared_euclidean_error': float(np.mean(np.sum((X-reconstructed)**2, axis=1))),
        'numpy_version': np.__version__,
    }
    (output / 'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
