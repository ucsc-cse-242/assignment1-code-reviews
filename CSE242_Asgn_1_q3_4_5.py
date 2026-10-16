import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Q2: Generate and visualize correlated Gaussian data
# ============================================================

rng = np.random.default_rng(42)

mean = np.array([0.0, 0.0])

true_cov = np.array([
    [3.0, 2.0],
    [2.0, 2.0]
])

X = rng.multivariate_normal(
    mean,
    true_cov,
    size=300
)

correlation_matrix = np.corrcoef(X.T)

print("Q2: Correlation matrix")
print(correlation_matrix)


# ------------------------------------------------------------
# Figure 1: Original correlated data
# ------------------------------------------------------------

plt.figure(1)

plt.scatter(
    X[:, 0],
    X[:, 1],
    alpha=0.6
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Q2: Correlated 2D Gaussian Data")
plt.axis("equal")


# ============================================================
# Q3: Implement PCA from scratch using NumPy
# ============================================================

# Step 1: Center the data
mean_vector = np.mean(X, axis=0)
X_centered = X - mean_vector

print("\nQ3: Mean vector")
print(mean_vector)


# Step 2: Compute covariance matrix
n = X_centered.shape[0]

cov_matrix = (
    X_centered.T @ X_centered
) / (n - 1)

print("\nQ3: Covariance matrix")
print(cov_matrix)


# Step 3: Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(
    cov_matrix
)


# Step 4: Sort eigenvalues from largest to smallest
order = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

print("\nQ3: Eigenvalues")
print(eigenvalues)

print("\nQ3: Eigenvectors")
print(eigenvectors)


# Step 5: Project centered data onto principal components
X_pca = X_centered @ eigenvectors

print("\nQ3: First five PCA coordinates")
print(X_pca[:5])


# ------------------------------------------------------------
# Figure 2: Principal component directions
# ------------------------------------------------------------

plt.figure(2)

plt.scatter(
    X[:, 0],
    X[:, 1],
    alpha=0.5
)

for i in range(2):

    direction = eigenvectors[:, i]

    scale = 2 * np.sqrt(eigenvalues[i])

    plt.arrow(
        mean_vector[0],
        mean_vector[1],
        scale * direction[0],
        scale * direction[1],
        width=0.025,
        length_includes_head=True
    )

    plt.text(
        mean_vector[0] + scale * direction[0],
        mean_vector[1] + scale * direction[1],
        f"PC{i + 1}"
    )

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Q3: Principal Component Directions")
plt.axis("equal")


# ============================================================
# Q4: Compute explained variance
# ============================================================

total_variance = np.sum(eigenvalues)

explained_variance_ratio = (
    eigenvalues / total_variance
)

print("\nQ4: Explained variance")

for i, ratio in enumerate(
    explained_variance_ratio
):
    print(
        f"PC{i + 1}: "
        f"{ratio:.4f} "
        f"({ratio * 100:.2f}%)"
    )


# ------------------------------------------------------------
# Figure 3: Explained variance
# ------------------------------------------------------------

plt.figure(3)

plt.bar(
    ["PC1", "PC2"],
    explained_variance_ratio * 100
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance (%)")
plt.title("Q4: Variance Explained by Principal Components")


# ============================================================
# Display all three figures together
# ============================================================

plt.show()