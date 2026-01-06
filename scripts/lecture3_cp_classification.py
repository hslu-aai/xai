import os
import numpy as np
import matplotlib.pyplot as plt

x0, y0 = 0.3, 0.9

t = np.linspace(-0.2, 0.2, 200)
x = x0 + t
y = np.cos(np.pi * t) * (y0 + t)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))

line, = ax[0].plot(x, y, linewidth=2)
ax[0].plot([x0], [y0], marker='o', markersize=6, color=line.get_color())
ax[0].set_xlabel("Feature")
ax[0].set_ylabel("Probability")
ax[0].set_xticks([])
ax[0].set_yticks([0, 1])
ax[0].set_xlim(0.0, 1.0)
ax[0].set_ylim(0.0, 1.0)

ax[1].plot(x, np.log(y/(1-y)), linewidth=2)
ax[1].plot([x0], [np.log(y0/(1-y0))], marker='o', markersize=6, color=line.get_color())
ax[1].set_xlabel("Feature")
ax[1].set_ylabel("Logit")
ax[1].set_xticks([])
ax[1].set_yticks([-2, -1, 0, +1, +2])
ax[1].set_xlim(0.0, 1.0)
ax[1].set_ylim(-3.0, +3.0)

y01 = y0 / 0.9 * 0.6
y1 = y / 0.9 * 0.6
ax[2].plot(x, y1, linewidth=2)
ax[2].plot([x0], [y01], marker='o', markersize=6, color=line.get_color(), label="class 1")
y02 = 0.25
y2 = y02 + (x - x0) * x
line, = ax[2].plot(x, y2, linewidth=2)
ax[2].plot([x0], [y02], marker='o', markersize=6, color=line.get_color(), label="class 2")
y03 = 1 - y01 - y02
y3 = 1 - y1 - y2
line, = ax[2].plot(x, y3, linewidth=2)
ax[2].plot([x0], [y03], marker='o', markersize=6, color=line.get_color(), label="class 3")
ax[2].set_xlabel("Feature")
ax[2].set_ylabel("Probability")
ax[2].set_xticks([])
ax[2].set_yticks([0, 1])
ax[2].set_xlim(0.0, 1.0)
ax[2].set_ylim(0.0, 1.0)
ax[2].legend()

fig.tight_layout()
fig.savefig(os.path.join(os.pardir, "figures", "lecture3_cp_classification.svg"))
