import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt

x = np.random.normal(0, 1, 500)
noise = np.random.normal(0, 1, 500)
y = 0.8 * x + 0.6 * noise

X = np.column_stack((x, y))  # 500 points, two features

mu = X.mean(axis=0)
center_X = X - mu
covX = np.cov(center_X, rowvar=False)

val, vec = npla.eigh(covX)
order = np.argsort(val)[::-1]
val = val[order]
vec = vec[:, order]

explained_ratio = val / val.sum()
cumulative_ratio = np.cumsum(explained_ratio)

for i, ratio in enumerate(explained_ratio):
    print(f"PC{i + 1}: {ratio:.2%} of total variance")

print(f"\nKeeping only PC1 retains {explained_ratio[0]:.2%} "
      f"and discards {1 - explained_ratio[0]:.2%} of the variance.")

pcs = np.arange(1, len(val) + 1)

plt.figure(figsize=(6, 4))
plt.bar(pcs, explained_ratio, label="Individual")
plt.plot(pcs, cumulative_ratio, "o-", color="crimson",
         label="Cumulative")
plt.xticks(pcs, [f"PC{i}" for i in pcs])
plt.ylim(0, 1.05)
plt.xlabel("Principal component")
plt.ylabel("Proportion of total variance")
plt.title("Variance explained by each principal component")
plt.legend()
plt.tight_layout()
plt.show()

scores = center_X @ vec

# Separate figure for the raw data
fig_raw, ax_raw = plt.subplots(figsize=(6, 5))
ax_raw.scatter(X[:, 0], X[:, 1], alpha=0.4, s=15)
ax_raw.set(
    title="Raw data",
    xlabel="Feature 1",
    ylabel="Feature 2"
)
ax_raw.set_aspect("equal", adjustable="datalim")
fig_raw.tight_layout()

# Figure for principal directions and projection
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

axes[0].scatter(X[:, 0], X[:, 1], alpha=0.4, s=15)
axes[0].set(
    title="Principal component directions",
    xlabel="Feature 1",
    ylabel="Feature 2"
)
axes[0].set_aspect("equal", adjustable="datalim")

for i, color in enumerate(["crimson", "darkorange"]):
    direction = 2 * np.sqrt(max(val[i], 0)) * vec[:, i]
    axes[0].quiver(
        mu[0], mu[1], direction[0], direction[1],
        angles="xy", scale_units="xy", scale=1,
        color=color, width=0.008, label=f"PC{i + 1}"
    )

axes[0].legend()

axes[1].scatter(scores[:, 0], np.zeros(len(X)), alpha=0.4, s=15)
axes[1].set(
    title="Projection onto PC1",
    xlabel="PC1 score",
    yticks=[]
)

fig.tight_layout()
plt.show()