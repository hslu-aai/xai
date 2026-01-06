import os
import matplotlib.pyplot as plt
import numpy as np

x = np.array([0.0, 1.0])

yA1 = np.array([0.0, 1.0])
yA2 = np.array([1.0, 0.0])

yB1 = np.array([0.2, 0.2])
yB2 = np.array([0.8, 0.8])

pd_A = (yA1 + yA2) / 2.0
pd_B = (yB1 + yB2) / 2.0
assert np.allclose(pd_A, pd_B), "PD curves must be identical for both datasets."

def plot_pair(x, dataA, dataB, show_avg, filename):
    """Plot crossing vs non-crossing ICE side by side."""
    (yA1, yA2) = dataA
    (yB1, yB2) = dataB

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=150, sharey=True)

    for ax, (y1, y2) in zip(axes, [(yA1, yA2), (yB1, yB2)]):
        ax.plot(x, y1, marker='o', label="Instance 1")
        ax.plot(x, y2, marker='o', label="Instance 2")
        if show_avg:
            ax.plot(x, (y1 + y2) / 2.0, linestyle='--', marker='o', label="Average")
        ax.set_xlabel("Feature")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(-1.05, 1.05)
        ax.legend(loc="best", frameon=False, fontsize=8)

    axes[0].set_ylabel("Target")
    fig.tight_layout()
    fig.savefig(os.path.join(os.pardir, "figures", filename), format="svg")
    plt.close(fig)

if __name__ == "__main__":
    dataA = (yA1, yA2)
    dataB = (yB1, yB2)
    plot_pair(x, dataA, dataB, show_avg=False, filename="lecture3_crossing.svg")
    plot_pair(x, dataA, dataB, show_avg=True, filename="lecture3_crossing_avg.svg")
