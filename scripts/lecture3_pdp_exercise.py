import os
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0.1, 0.9, 81)
y1 = 1 / (t + 3) + 0.01 * np.cos(4 * np.pi * t)
y2 = 0.5 * t**2 + 0.3 * t + 0.1

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

ax[0].plot(t, y1, linewidth=2)
ax[0].set_xlabel("Feature")
ax[0].set_ylabel("Target")
ax[0].set_xlim(0.0, 1.0)
ax[0].set_ylim(0.0, 1.0)

ax[1].plot(t, y2, linewidth=2)
ax[1].set_xlabel("Feature")
ax[1].set_ylabel("Target")
ax[1].set_xlim(0.0, 1.0)
ax[1].set_ylim(0.0, 1.0)

fig.tight_layout()
fig.savefig(os.path.join(os.pardir, "figures", "lecture3_pdp_exercise.svg"))
