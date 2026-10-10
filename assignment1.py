import numpy as np
import matplotlib.pyplot as plt

# Set up the synthetic dataset
meanValues = [0, 0]
covarianceValues = [[3, 2], [2, 2]]
data = np.random.multivariate_normal(meanValues, covarianceValues, 500)

# Scatter plot of the raw data
plt.scatter(data[:, 0], data[:, 1])
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axis("equal") # If the lines aren't equal, then the scaling might not make sense
plt.show()

# Centering the data 
dataMean = data.mean(axis=0)
centeredData = data - dataMean

# Sample the covariance matrix
numSamples = centeredData.shape[0]
covarianceMatrix = centeredData.T @ centeredData / (numSamples - 1)

# Eigen decomposition of the matrix
eigenvalues, eigenvectors = np.linalg.eigh(covarianceMatrix)

# Sort eigenpairs by eigen values
sortedIndexes = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sortedIndexes]
eigenvectors = eigenvectors[:, sortedIndexes]

firstComponent = eigenvectors[:, 0]
secondComponent = eigenvectors[:, 1]

# Visualize centered data with the principal component directions
plt.scatter(centeredData[:, 0], centeredData[:, 1])
plt.arrow(0, 0, firstComponent[0] * 3, firstComponent[1] * 3, color="red", width=0.05)
plt.arrow(0, 0, secondComponent[0] * 3, secondComponent[1] * 3, color="green", width=0.05)
plt.title("Centered Data With Principal Components")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axis("equal")
plt.show()

# Portion of variance by each component
totalVariance = eigenvalues.sum()
explainedRatio = eigenvalues / totalVariance


# Print out results
print("Eigenvalues:", eigenvalues)
print("Explained variance ratio:", explainedRatio)

projectedData = centeredData @ firstComponent
print("Variance of projection onto PC1:", projectedData.var(ddof=1))
print("Largest eigenvalue:", eigenvalues[0])
