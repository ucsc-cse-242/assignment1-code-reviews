## Part 1: Linear Algebra (20 points)

**Q1 (5 points).** Given matrices:

$$
A = \begin{bmatrix}3 & 4 \\ 
2 & 1\end{bmatrix}, \quad
B = \begin{bmatrix}1 & 2 \\ 
0 & -1\end{bmatrix}
$$

Compute:

$$
AB,\quad BA,\quad (A+B)^T,\quad AB^T,\quad \mathrm{trace}(A),\quad \mathrm{trace}(AB).
$$

> [!success] Solution
>$$\begin{align} AB &= \begin{bmatrix} 3 & 2 \\  2 & 3 \end{bmatrix} \\ BA &= \begin{bmatrix} 7 & 6 \\  -2 & -1 \end{bmatrix} \\ (A+B)^T &= \begin{bmatrix} 4 & 6 \\  2 & 0 \end{bmatrix} \\ AB^T &= \begin{bmatrix} 3 & 2 \\  2 & 3 \end{bmatrix} \\ \text{trace}(A) &= 4 \\ \text{trace}(AB) &= 6 \end{align}$$

**Q2 (5 points).** For the following matrix:

$$
M = \begin{bmatrix}2 & -1 \\ 
-1 & 2\end{bmatrix}
$$

Calculate the eigenvalues and eigenvectors. Verify your solution by showing $Mv = \lambda v$.

> [!success] Solution
>The eigenvalues and eigenvectors of $M$ can be computer using the characteristic polynomial. Setting this to zero and solving results in eigenvalues $\lambda = 1,3$ with eigenvectors $[-\sqrt 2 / 2, \sqrt 2 /2]^T, [\sqrt 2 / 2, \sqrt 2/2]^T$. 
>
>To verify, we find that $M \begin{bmatrix} - \sqrt 2 / 2 \\ \sqrt 2 / 2 \end{bmatrix} = \begin{bmatrix} - 3 \sqrt 2 / 2 \\ 3 \sqrt 2 / 2 \end{bmatrix} = 3 \begin{bmatrix} - \sqrt 2 / 2 \\ \sqrt 2 / 2 \end{bmatrix}$ and $M \begin{bmatrix}  \sqrt 2 / 2 \\ \sqrt 2 / 2 \end{bmatrix} = \begin{bmatrix} \sqrt 2 / 2 \\ \sqrt 2 / 2 \end{bmatrix} = \begin{bmatrix} \sqrt 2 / 2 \\ \sqrt 2 / 2 \end{bmatrix}$

**Q3 (5 points).** Prove the following property of the trace operator:

$$
\mathrm{trace}(AB)=\mathrm{trace}(BA)
$$

> [!success] Solution
>Assuming $A, B \in \mathbb R^{n \times n}$, we find: $$\begin{align} \text{trace}(AB) &= \sum_{i=1}^n (AB)_{ii}\\ &= \sum_{i=1}^n \biggr ( \sum_{k=1}^n A_{ik} B_{ki} \biggr ) \\ &= \sum_{k=1}^n \biggr ( \sum_{i=1}^n B_{ki} A_{ik} \biggr ) \\ &= \sum_{k=1}^n (BA)_{kk} \\ &= \text{trace}(BA) \end{align}$$which confirms the claim.

**Q4 (5 points).** Show step-by-step that the inverse of the matrix

$$
C = \begin{bmatrix}4 & 7 \\ 
2 & 6\end{bmatrix}
$$

exists, and compute its inverse.

> [!success] Solution
>To determine whether the inverse exists, we just need to show that the two column vectors are linearly independent. Hence, we compute the determinant of $C$ to find that $\det C = 4(6)-7(2)=10 \neq 0$. Thus, $C$ is nonsingular and we know that they are linearly independent. Now, we compute the inverse using Gaussian elimination.
>
>$$\begin{aligned}
> \left[\begin{array}{cc|cc}4&7&1&0\\2&6&0&1\end{array}\right] &\sim \left[\begin{array}{cc|cc}1&\frac74&\frac14&0\\2&6&0&1\end{array}\right]\\[6pt]&\sim \left[\begin{array}{cc|cc}1&\frac74&\frac14&0\\0&\frac52& \frac12&1\end{array}\right]\\[6pt] &\sim
> \left[\begin{array}{cc|cc}1&\frac74&\frac14&0\\0&1& \frac15&\frac25\end{array}\right]\\[6pt] &\sim
> \left[\begin{array}{cc|cc}1&0&\frac35&-\frac7{10}\\0&1& \frac15&\frac25\end{array}\right] \end{aligned}$$Therefore, $$C^{-1}=\begin{bmatrix}\frac35&-\frac7{10}\\ \frac15&\frac25\end{bmatrix}$$

## Part 2: Probability, MLE, and MAP Estimation (50 points)

**Q1 (10 points).** Suppose a random variable $X$ follows a discrete distribution:

$$
P(X=k)=c\cdot\frac{3^k}{k!},\quad k=0,1,2,\ldots
$$

(a) Find the normalization constant $c$.

> [!success] Solution
>We want to sum over the entire support of $X$ to determine what $c$ should be. Hence, we find that $$\sum_{k=0}^{\infty} P(X=k) = c \sum_{k=0}^{\infty} \frac{3^k}{k!} =ce^3 =1 \implies c = \frac{1}{e^3}$$

(b) Compute the expectation $E[X]$ and variance $Var(X)$.

> [!success] Solution
>$$\mathbb E[X] = \sum_{k=0}^{\infty} k\Pr(X=k) = e^{-3}\sum_{k=1}^{\infty}\frac{3^k}{(k-1)!} = 3e^{-3}\sum_{k=0}^{\infty}\frac{3^k}{k!} = 3e^{-3}e^3 = 3$$
>$$\begin{align} \text{Var}(X) &= \mathbb E[X^2] - (\mathbb E[X])^2 \\ &= \mathbb E [X(X-1)] + \mathbb E[X] - 9 \\ &= e^{-3} \biggr ( \sum_{k=0}^{\infty} k(k-1) \frac{3^k}{k!} + \sum_{k=0}^{\infty} k \frac{3^k}{k!} \biggr ) - 9\\ &=  e^{-3} \biggr ( 9\sum_{k=2}^{\infty} \frac{3^{k-2}}{(k-2)!} + 3\sum_{k=1}^{\infty} \frac{3^{k-1}}{(k-1)!} \biggr ) -  9 \\ &= 3e^{-3} ( 4e^3) -  9 = 3\end{align}$$

**Q2 (15 points).** Given two events $A$ and $B$ with $P(A)=0.5$, $P(B)=0.3$, and $P(A\mid B)=0.6$, compute the following:

(a) $P(A\cap B)$,

> [!success] Solution
>Using the Law of Conditional Probability, we get $$P(A \cap B) = P(A \mid B) P(B) = 0.6(0.3) = 0.18$$

(b) $P(B\mid A)$,

> [!success] Solution
>Using the part above and the Law of Conditional Probability, we get that $$P(B \mid A) = P(A \cap B) / P(A) = 0.18/0.5 = 0.36$$

(c) $P(A\cup B)$.

> [!success] Solution
>$$P(A \cup B) = P(A) + P(B) - P(A \cap B) = 0.8 - 0.18 = 0.62$$

**Q3 (10 points).** Suppose you have a biased coin with unknown probability of heads $\theta=P(H)$. You flip this coin 5 times and observe the sequence: H, H, T, H, T.

(a) Derive the Maximum Likelihood Estimate (MLE) for $\theta$, showing each step from the likelihood function.

> [!success] Solution
>Let our coin flips be denoted as samples $\{x_i \}_{i=1}^5$.We begin by deriving the likelihood function. $$\begin{align} L(\theta) &= \prod_{i=1}^n \theta^{x_i} (1-\theta)^{1-x_i} = \theta^3 (1-\theta)^2 \end{align}$$Then, we take the logarithm of this expression, $$\ln (L(\theta)) = 3 \ln \theta + 2\ln (1-\theta)$$FInally, we differentiate with respect to $\theta$, since this is the parameter we want to optimize. $$\frac{d \ln (\theta)}{d\theta} = \frac{3}{\theta} - \frac{2}{1- \theta}$$Hence, we find $$\begin{align} \frac{d \ln (\theta)}{d\theta} = 0 & \implies \frac{3-5 \theta}{\theta (1-\theta)} = 0 \implies \theta^* = \frac{3}{5} \end{align}$$

(b) Compute the Maximum a Posteriori (MAP) estimate for $\theta$, assuming a prior distribution:

$$
P(\theta)=\begin{cases}2\theta, & 0\leq\theta\leq 1 \\ 0, & \text{otherwise}\end{cases}
$$
> [!success] Solution
>Using the likelihood function from before, we find that $$\ln L(\theta) + \ln P(\theta) = 3 \ln \theta + 2 \ln (1-\theta) + \ln (2\theta)$$So, we again differentiate this with respective to $\theta$ to find $$\frac{d(\ln L(\theta) + \ln P(\theta))}{d \theta} = \frac{3}{\theta} - \frac{2}{1-\theta} + \frac{1}{\theta}$$Setting this to zero and solving for $\theta$ yields $$4-6\theta = 0 \implies \theta^* = \frac{2}{3}$$

**Q4 (5 points).** Consider a Gaussian distribution with parameters $\mu$ (mean) and $\sigma^2$ (variance). Given the dataset $X=\{1,3,5,7\}$, derive the Maximum Likelihood Estimates (MLE) for both $\mu$ and $\sigma^2$. Show each mathematical step clearly.

> [!success] Solution
>Since we are using a Gaussian distribution with parameters $\theta = (\mu, \sigma^2)$ and $X = (1,3,5,7)$, we have that $$L(X \mid \theta) = \prod_{i=1}^4 \frac{1}{\sqrt{2 \pi \sigma^2}}\exp \biggr (-\frac{(X_i-\mu)^2}{2\sigma^2} \biggr )$$Taking the logarithm makes this easier to work with and we get that $$\ln (L(X \mid \theta)) = -2 \ln (2 \pi) - 2\ln (\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^4 (X_i-\mu)^2$$We then take the partial derivatives with respect to $\theta$  and set them equal to $0$: $$\nabla_{(\mu,\sigma^2)} \ln (L(X \mid \theta)) = \begin{bmatrix} \frac{1}{\sigma^2}\sum_{i=1}^4(X_i-\mu) \\ -\frac{2}{\sigma^2}+\frac{1}{2(\sigma^2)^2}\sum_{i=1}^4(X_i-\mu)^2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$Simplifying these expressions yields $$\mu^* = \frac{16}{4} = 4, \quad \sigma^{2^*} = \frac{1}{4}\biggr ( 3^2 + 1^2 + 1^2 + 3^2 \biggr ) = 5$$which are the MLE for $\mu$ and $\sigma^2$.

**Q5 (5 points).** You observe samples drawn from a Gaussian distribution with unknown mean $\mu$ and known variance $\sigma^2=4$. Given a Gaussian prior for the mean $\mu$:

$$
\mu\sim\mathcal{N}(0,1)
$$

derive the MAP estimate of $\mu$ after observing a single data point $x=5$.

> [!success] Solution
>The Gaussian prior for $\mu \sim \mathcal N (0,1)$ is given by $$p (\mu) = \frac{1}{\sqrt{2 \pi}}\exp \biggr ( - \frac{\mu^2}{2} \biggr )$$Hence, the MAP estimate of $\mu$ with $X = (5)$ can be determined as follows. $$\ln\!\bigl(L(\mu\mid x)p(\mu)\bigr) = -\frac12\ln(2\pi) -\frac12\ln4-\frac{(5-\mu)^2}{8}-\frac12\ln(2\pi)-\frac{\mu^2}{2}$$Then, we differentiate with respect to $\mu$ and set equal to $0$ to get our final estimate. $$\frac{d\ln\!\bigl(L(\mu\mid x)p(\mu)\bigr)}{d \mu} = \frac{5 - \mu}{4}- \mu = 0 \implies \,u^* =  1$$

**Q6 (Variance Expansion Proof, 5 points).** Prove the variance expansion identity:

$$Var(X)=E(X^2)-[E(X)]^2$$

> [!success] Solution
>Using the linearity of expectation, we find $$\begin{align} Var(X) &= E[(X-\mu)^2] \\ &= E[(X-E[X])^2] \\ &= E[X^2 - 2XE[X] + E[X]^2]  \\ &= E[X^2] -2(E[X])^2 + (E[X])^2 \\ &= E[X^2] - (E[X])^2\end{align}$$

## Part 3: Principal Component Analysis (Math and Programming) (30 points)

**Q1 (10 points).** Given a dataset represented by a random vector $X$, derive the mathematical expression for the variance of the projection onto a unit vector $u$:

$$
Var(u^TX)=u^T\Sigma u
$$

Document each intermediate mathematical step and define the covariance matrix $\Sigma$.

> [!success] Solution
>Let $X, u \in \mathbb R^n$ where $X$ is a random vector and $u$ is a unit vector. We begin by defining the covariance matrix, $\Sigma \in \mathbb R^{n \times n}$, entry-wise as follows: $$\Sigma_{ij} = cov(X_i, X_j) = E[(X_i - E[X_i])(X_j - E[X_j])] = E[X_i X_j] - E[X_i]E[X_j]$$We now proceed with the derivation of the expression. $$\begin{align} Var (u^TX) &= Var (\sum_{i=1}^n u_i X_i) \\ &= E[\biggr (\sum_{i=1}^n u_i X_i \biggr )^2] - (E[\sum_{i=1}^n u_i X_i])^2 \\ &= E[\sum_{i=1}^n \sum_{j=1}^n u_i u_j X_i X_j] - \sum_{i=1}^n \sum_{j=1}^n E[u_i]E[u_j]E[X_i] E[X_j] \\ &= \sum_{i=1}^n \sum_{j=1}^n \biggr (u_iu_j\biggr(E[ X_i X_j]-E[X_i] E[X_j] \biggr) \biggr ) \\ &= \sum_{i=1}^n \sum_{j=1}^n \biggr (u_i u_j cov(X_i, X_j) \biggr) \\ &= u^T \Sigma u\end{align}$$which provides the desired statement.

**Q2 (5 points).** Generate a synthetic 2-dimensional dataset composed of two correlated Gaussian features. Use NumPy for data generation. Visualize this dataset with scatter plots and discuss the correlation structure observed.

> [!success] Solution
> ```code
> import numpy as np
> import matplotlib.pyplot as plt
>
> x = np.random.normal(0, 1, 500)
> noise = np.random.normal(0, 1, 500)
> y = 0.8 * x + 0.6 * noise
>
> X = np.column_stack((x, y)) # 500 points, two features
>
> plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
> plt.xlabel("Feature 1")
> plt.ylabel("Feature 2")
> plt.axis("equal")
> plt.show()
> ```
>
> ![[q3.png]]
>
> In the scatter plot, we can see that as Feature 1 increases, so does Feature 2. This makes sense as the way Feature 2 is generated is by simply scaling Feature 1 correspondingly and adding a little bit of noise to each entry. Hence, the line of best fit would have a positive slope with relatively little distance from it amongst the points. There are some outliers, which is a result of the normally distributed noise that was added.

**Q3 (10 points).** Implement PCA from scratch using NumPy. Follow these steps:

- Center your dataset (mean subtraction).
- Compute the covariance matrix.
- Find eigenvalues and eigenvectors of the covariance matrix.
- Select the principal components corresponding to the highest eigenvalues.

Document each step in your implementation, linking it to your mathematical derivation from Q1. Provide visualizations of your original data along with identified principal component directions.

> [!success] Solution
> ```code
> # continued from above...
>
> scores = center_X @ vec
>
> # calculate the mean
> mu = X.mean(axis=0)
>
> # center the data
> center_X = X - mu
>
> # compute the covariance matrix in correspondence with derivation abov
> covX = np.cov(center_X, rowvar=False)
>
> # eigenvalues then sort
> val, vec = npla.eigh(covX)
> order = np.argsort(val)[::-1]
> val = val[order]
> vec = vec[:, order]
>
> # Figure for principal directions and projection
> fig, axes = plt.subplots(1, 2, figsize=(11, 5))
>
> axes[0].scatter(X[:, 0], X[:, 1], alpha=0.4, s=15)
> axes[0].set(
> 	title="Principal component directions",
> 	xlabel="Feature 1",
> 	ylabel="Feature 2"
> )
>
> axes[0].set_aspect("equal", adjustable="datalim")
>
> for i, color in enumerate(["crimson", "darkorange"]):
> 	direction = 2 * np.sqrt(max(val[i], 0)) * vec[:, i]
> 	axes[0].quiver(
> 		mu[0], mu[1], direction[0], direction[1],
> 		angles="xy", scale_units="xy", scale=1,
> 		color=color, width=0.008, label=f"PC{i + 1}"
> 	)
>
> axes[0].legend()
> axes[1].scatter(scores[:, 0], np.zeros(len(X)), alpha=0.4, s=15)
> axes[1].set(
> 	title="Projection onto PC1",
> 	xlabel="PC1 score",
> 	yticks=[]
> )
>
> fig.tight_layout()
> plt.show()
>
> ```
>
>
> ![[q2.png]]

**Q4 (10 points).** Compute the proportion of total variance explained by each principal component. Discuss the results and explain the significance of dimensionality reduction. Provide insights into the trade-off between explained variance and dimensionality reduction.

> [!success] Solution
> In using dimensionality reduction, we are literally losing "directions" the data takes. Hence, the methods we use are especially important because they may favor different patterns over others. Especially with PCA, the explained variance in large part determines the order of our principal components, so performing data reduction will lose the way the data varies, but it is a lot easier to store.
>
> ```code
> # continued from above...
>
> explained_ratio = val / val.sum()
> cumulative_ratio = np.cumsum(explained_ratio)
>
> for i, ratio in enumerate(explained_ratio):
> 	print(f"PC{i + 1}: {ratio:.2%} of total variance")
>
>
> print(f"\nKeeping only PC1 retains {explained_ratio[0]:.2%} "
> 	f"and discards {1 - explained_ratio[0]:.2%} of the variance.")
>
> pcs = np.arange(1, len(val) + 1)  
>
> plt.figure(figsize=(6, 4))
> plt.bar(pcs, explained_ratio, label="Individual")
> plt.plot(pcs, cumulative_ratio, "o-", color="crimson",
> label="Cumulative")
> plt.xticks(pcs, [f"PC{i}" for i in pcs])
> plt.ylim(0, 1.05)
> plt.xlabel("Principal component")
> plt.ylabel("Proportion of total variance")
> plt.title("Variance explained by each principal component")
> plt.legend()
> plt.tight_layout()
> plt.show()
> ```
>
> ![[q4 1.png]]