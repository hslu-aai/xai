import os
import numpy as np
import matplotlib.pyplot as plt

x0, y0 = 0.5, 0.5

t = np.linspace(-0.2, 0.2, 200)
x = x0 + t
y = y0 + t**3

fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(x, y, linewidth=2)
ax.plot([x0], [y0], marker='o', markersize=6)
ax.set_xlabel("Feature")
ax.set_ylabel("Target")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0.1, 0.9)
ax.set_ylim(0.3, 0.7)
fig.tight_layout()
fig.savefig(os.path.join(os.pardir, "figures", "lecture3_cp_idea.svg"))
