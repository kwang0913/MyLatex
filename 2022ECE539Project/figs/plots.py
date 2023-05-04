# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from itertools import cycle, islice

np.random.seed(363)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)

colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    2,
                )
            )
        )

def poly(X):
    return np.vstack((X[:, 0] ** 2, X[:, 1] ** 2, np.sqrt(2)*X[:, 0] * X[:, 1])).T

# %%
X, y = noisy_circles
fig, ax = plt.subplots()
ax.scatter(
    X[:, 0], 
    X[:, 1], 
    s=10, 
    color=colors[y],
    )
ax.set(
    # xticklabels=[],
    # yticklabels=[],
    xlabel="$x_1$",
    ylabel="$x_2$",
    )
plt.savefig("../2d_poly_circle.eps", format="eps")

# %%
X = poly(X)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.scatter(
    X[:, 0], 
    X[:, 1], 
    X[:, 2], 
    s=10, color=colors[y],
    )
ax.set(
    # xticklabels=[],
    # yticklabels=[],
    # zticklabels=[],
    xlabel="$x_1$",
    ylabel="$x_2$",
    zlabel="$x_3$",
    )
plt.savefig("../3d_poly_circle.eps", format="eps")

# %%
