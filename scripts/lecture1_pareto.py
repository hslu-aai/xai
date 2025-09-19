import os
import string
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


def is_dominated(p: Sequence[float], others: Sequence[Sequence[float]]) -> bool:
    return any(
        o[0] >= p[0] and o[1] >= p[1] and (o[0] > p[0] or o[1] > p[1]) for o in others
    )


np.random.seed(42)
n_points = 12
xs = np.random.rand(n_points)
ys = np.random.rand(n_points)
labels = list(string.ascii_uppercase[:n_points])

points = list(zip(xs, ys, labels))

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Interpretability")
ax.set_ylabel("Performance")
ax.set_xticks([])
ax.set_yticks([])


for x, y, label in points:
    ax.plot(x, y, "o", color="black")
    ax.text(x, y - 0.05, label)

pareto_points = [p for p in points if not is_dominated(p, points)]
pareto_points.sort(key=lambda p: p[0])

step_x = [0]
step_y = [pareto_points[0][1]]
for i, (x, y, _) in enumerate(pareto_points):
    if i == 0:
        step_x.append(x)
        step_y.append(y)
    else:
        prev_x, prev_y = pareto_points[i - 1][:2]
        step_x.extend([prev_x, x])
        step_y.extend([y, y])
step_x += [pareto_points[-1][0]]
step_y += [0]

ax.plot(
    step_x, step_y, linestyle="dashed", linewidth=2, color="red", label="Pareto Front"
)
ax.legend()

output_path = os.path.join(os.pardir, "figures", "lecture1_pareto.svg")
fig.savefig(output_path, format="svg")
