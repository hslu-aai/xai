import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVR
from mpl_toolkits.mplot3d import Axes3D

X, y = datasets.fetch_california_housing(return_X_y=True)

model = SVR(kernel="rbf", C=100, gamma=0.1)
model.fit(X, y)

f1, f2 = 0, 1
fixed_point = X[42].copy()

n_grid = 100
x_range = np.linspace(X[:, f1].min(), X[:, f1].max(), n_grid)
y_range = np.linspace(X[:, f2].min(), X[:, f2].max(), n_grid)
xx, yy = np.meshgrid(x_range, y_range)

X_grid = np.tile(fixed_point, (n_grid * n_grid, 1))
X_grid[:, f1] = xx.ravel()
X_grid[:, f2] = yy.ravel()

zz = model.predict(X_grid).reshape(xx.shape)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(xx, yy, zz, cmap="viridis", edgecolor="none")
ax.set_xlabel(f"Feature {f1}")
ax.set_ylabel(f"Feature {f2}")
ax.set_zlabel("Target")
plt.tight_layout()
plt.savefig(os.path.join(os.pardir, "figures", "lecture3_cp_3d.svg"))

plt.figure(figsize=(8, 6))
contour = plt.contourf(xx, yy, zz, levels=30, cmap="viridis")
plt.xlabel(f"Feature {f1}")
plt.ylabel(f"Feature {f2}")
plt.colorbar(contour, label="Target")
plt.tight_layout()
plt.savefig(os.path.join(os.pardir, "figures", "lecture3_cp_2d.svg"))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
base_point = fixed_point.copy()
x_vals = np.linspace(X[:, f1].min(), X[:, f1].max(), 200)
f2_vals = np.linspace(X[:, f2].min(), X[:, f2].max(), 50)
cmap = plt.cm.plasma
for i, val in enumerate(f2_vals):
    X_line = np.tile(base_point, (len(x_vals), 1))
    X_line[:, f1] = x_vals
    X_line[:, f2] = val
    y_pred = model.predict(X_line)
    plt.plot(x_vals, y_pred, color=cmap(i / (len(f2_vals) - 1)))
sm = plt.cm.ScalarMappable(
    cmap=cmap, norm=plt.Normalize(vmin=f2_vals.min(), vmax=f2_vals.max())
)
plt.colorbar(sm, ax=ax, label=f"Feature {f2}")
plt.xlabel(f"Feature {f1}")
plt.ylabel("Target")
plt.tight_layout()
plt.savefig(os.path.join(os.pardir, "figures", "lecture3_cp_1d.svg"))
