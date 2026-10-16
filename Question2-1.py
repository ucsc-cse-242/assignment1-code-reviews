import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(50) #Replaceable seed


# Two Gaussian features with correlation 0.8

#2D Graph
data = rng.multivariate_normal(mean = [0,0], cov = [[1, 0.8], [0.8,1]], size=1000)
x, y = data.T

correlation = np.corrcoef(x, y)[0, 1]
print(f"Correlation: {correlation:.3f}")

#Plotting
plt.scatter(x, y, s=15, alpha=0.5)
plt.title(f"Correlated Gaussian Features (r = {correlation:.3f})")
plt.grid(True)
plt.axis("equal")
plt.show()