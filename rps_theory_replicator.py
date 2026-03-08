"""
Deterministic replicator benchmark for the RPS payoff family used in the report.

Sources acknowledged:
- Nowak (2006), Evolutionary Dynamics.
- Hofbauer and Sigmund (1998), Evolutionary Games and Population Dynamics.

Authorship note:
- This implementation in this repository was prepared by Ameer Alhashemi.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

R, P, S = 0, 1, 2

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    }
)


def payoff_matrix(s):
    return np.array([[0.0, -s, 1.0], [1.0, 0.0, -s], [-s, 1.0, 0.0]], dtype=float)


def replicator_rhs(x, a):
    pi = a @ x
    phi = float(x @ pi)
    return x * (pi - phi)


def integrate_replicator(x0, s, dt=0.02, t_steps=5000):
    a = payoff_matrix(s)
    x = np.array(x0, dtype=float)
    x = np.clip(x, 1e-12, 1.0)
    x /= np.sum(x)

    traj = np.zeros((t_steps + 1, 3), dtype=float)
    traj[0] = x
    for t in range(1, t_steps + 1):
        x = x + dt * replicator_rhs(x, a)
        x = np.clip(x, 1e-12, 1.0)
        x /= np.sum(x)
        traj[t] = x
    return traj


def plot_replicator_benchmark():
    os.makedirs("results", exist_ok=True)
    s_values = [0.8, 1.0, 1.2]
    inits = [
        [0.36, 0.31, 0.33],
        [0.34, 0.36, 0.30],
        [0.31, 0.34, 0.35],
        [0.38, 0.29, 0.33],
        [0.29, 0.38, 0.33],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharex=True, sharey=True)
    for ax, s in zip(axes, s_values):
        for x0 in inits:
            tr = integrate_replicator(x0, s=s)
            ax.plot(tr[:, R], tr[:, P], linewidth=1.4, alpha=0.95)

        # Simplex boundary in (R, P) projection.
        ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color="black", linewidth=1.0, alpha=0.7)
        ax.scatter([1.0 / 3.0], [1.0 / 3.0], s=24, color="black")
        ax.set_title(f"s = {s}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.set_xlabel("R")
    axes[0].set_ylabel("P")

    plt.tight_layout()
    plt.savefig("results/theory_replicator_phase.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot_replicator_benchmark()
