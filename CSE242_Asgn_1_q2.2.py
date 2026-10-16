import numpy as np
import matplotlib.pyplot as plt

# Set a random seed so results are reproducible
np.random.seed(42)

# Mean of the two Gaussian features
mean = [0, 0]

# Covariance matrix
# The positive off-diagonal values create positive correlation
covariance = [
    [1.0, 0.8],
    [0.8, 1.0]
]

# Generate 500 two-dimensional samples
data = np.random.multivariate_normal(mean, covariance, size=500)

# Separate the two features
x = data[:, 0]
y = data[:, 1]

# Calculate the sample correlation
correlation = np.corrcoef(x, y)[0, 1]

print(f"Correlation between Feature 1 and Feature 2: {correlation:.3f}")

# Plot the data
plt.scatter(x, y, alpha=0.6)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Correlated 2D Gaussian Dataset")

plt.grid(True)
plt.show()