import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
X = rng.multivariate_normal(
    mean=[0, 0],
    cov=[[1, 0.8], [0.8, 1]],
    size=1000
) #Recreate from Q2

#Center Data, X_centered = X - u
mean = X.mean(axis=0)
X_centered = X - mean

#Covariance Matrix by 1/(n-1) X^T centered X centered
n = X.shape[0]
C = (X_centered.T @ X_centered) / (n - 1)

#Finding Eigenvalues, covariance matrix should be symmetric
eigenvalues, eigenvectors = np.linalg.eigh(C)

#Sort Eigenvalues largest to smallest
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

k = 1
W = eigenvectors[:, :k]
Z = X_centered @ W

#Check PC1 and PC2 that Cv = Lamda(v)
for i in range(2):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]

    print(f"\nChecking PC{i+1}:")
    print("C @ v =", C @ v)
    print("lambda * v =", lam * v)
    print("Valid:", np.allclose(C @ v, lam * v))

#return true if identity matrix
print(np.allclose(eigenvectors.T @ eigenvectors, np.eye(2)))


explained_variance_ratio = eigenvalues / eigenvalues.sum()

#Print
print("Feature means:", mean)
print("\nCovariance matrix:\n", C)
print("\nEigenvalues:", eigenvalues)
print("\nPrincipal components (columns):\n", eigenvectors)
print("\nExplained variance ratios:", explained_variance_ratio)
print("Expected Sum of 1.0, The Sum:", explained_variance_ratio.sum())
print("\nReduced data shape:", Z.shape)

# Visualize the original data and principal component directions.
plt.scatter(X[:, 0], X[:, 1], alpha=0.4)

# Plot eigenvectors starting from the mean
colors = ["blue","red"]
for i in range(2):
    direction = eigenvectors[:, i]
    length = 2 * np.sqrt(eigenvalues[i])

    plt.quiver(
        mean[0], mean[1],
        direction[0] * length,
        direction[1] * length,
        angles='xy',
        scale_units='xy',
        scale=1,
        color = colors[i],
        label=f"PC{i+1}"
    )

plt.xlabel("Guassian 1")
plt.ylabel("Guassian 2")
plt.title("PCA Principal Component Directions")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.show()