"""
Q3 variant: RPS on a periodic lattice using pairwise local update.

Sources acknowledged:
- Module brief for March 2026 (creative lattice/network extension).
- Claussen (2016), Chapter 24, "Evolutionary Dynamics: How Payoffs and
  Global Feedback Control the Stability" for Local Update process context.

Authorship note:
- This implementation in this repository was prepared by Ameer Alhashemi.
"""

import csv
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

def payoff_one_vs_one(a, b, s):
    # payoff to player using strategy a against opponent b
    # matrix:
    # R vs P = -s, R vs S = +1
    if a == R and b == P: return -s
    if a == R and b == S: return 1
    if a == P and b == S: return -s
    if a == P and b == R: return 1
    if a == S and b == R: return -s
    if a == S and b == P: return 1
    return 0

def neighbours(i, j, L):
    # 4-neighbour von Neumann with periodic boundaries
    return [((i-1)%L, j), ((i+1)%L, j), (i, (j-1)%L), (i, (j+1)%L)]

def local_payoff(grid, i, j, s):
    L = grid.shape[0]
    strat = grid[i, j]
    total = 0.0
    for ni, nj in neighbours(i, j, L):
        total += payoff_one_vs_one(strat, grid[ni, nj], s)
    return total


def pairwise_probability(pi_a, pi_b, w, pi_max_diff):
    p = 0.5 + (w * (pi_a - pi_b)) / (8.0 * pi_max_diff)
    return float(np.clip(p, 0.0, 1.0))


def distance_to_mixed(rt, pt, st):
    n = rt + pt + st + 1e-12
    r = rt / n
    p = pt / n
    return np.sqrt((r - 1.0 / 3.0) ** 2 + (p - 1.0 / 3.0) ** 2)


def run_lattice(L=30, T=50000, s=1.0, w=0.5, seed=1):
    rng = np.random.default_rng(seed)

    # start near equilibrium: random with equal probs
    grid = rng.integers(0, 3, size=(L, L))

    Rt = np.zeros(T+1, dtype=int)
    Pt = np.zeros(T+1, dtype=int)
    St = np.zeros(T+1, dtype=int)

    counts = np.bincount(grid.ravel(), minlength=3)
    Rt[0], Pt[0], St[0] = counts[R], counts[P], counts[S]

    # Local payoff difference bound for degree-4 neighbourhood.
    pi_max_diff = 4.0 * (1.0 + s)

    for t in range(1, T+1):
        i = rng.integers(L)
        j = rng.integers(L)

        # pick a random neighbour as role model
        ni, nj = neighbours(i, j, L)[rng.integers(4)]

        strat_b = grid[i, j]
        strat_a = grid[ni, nj]

        pi_b = local_payoff(grid, i, j, s)
        pi_a = local_payoff(grid, ni, nj, s)

        p = pairwise_probability(pi_a, pi_b, w=w, pi_max_diff=pi_max_diff)

        if strat_a != strat_b and rng.random() < p:
            grid[i, j] = strat_a
            counts[strat_b] -= 1
            counts[strat_a] += 1

        Rt[t], Pt[t], St[t] = counts[R], counts[P], counts[S]

    return Rt, Pt, St, grid

def plot_q3(Rt, Pt, St, grid, name):
    os.makedirs("results_q3", exist_ok=True)

    t = np.arange(len(Rt))
    N = Rt[0] + Pt[0] + St[0]

    # time series (proportions)
    plt.figure(figsize=(10.8, 5.2))
    plt.plot(t, Rt/N, label="R proportion")
    plt.plot(t, Pt/N, label="P proportion")
    plt.plot(t, St/N, label="S proportion")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("proportion")
    plt.savefig(f"results_q3/{name}_timeseries.png")
    plt.close()

    # phase plot (R,P)
    plt.figure(figsize=(6.5, 6.2))
    plt.plot(Rt/N, Pt/N)
    plt.xlabel("R proportion")
    plt.ylabel("P proportion")
    plt.savefig(f"results_q3/{name}_phase.png")
    plt.close()

    # grid snapshot
    plt.figure(figsize=(6.5, 6.2))
    plt.imshow(grid, interpolation="nearest")
    plt.savefig(f"results_q3/{name}_grid.png")
    plt.close()

    # distance to mixed point
    dist = distance_to_mixed(Rt, Pt, St)
    plt.figure(figsize=(9.8, 4.6))
    plt.plot(t, dist)
    plt.xlabel("t")
    plt.ylabel("distance d(t) to mixed state")
    plt.tight_layout()
    plt.savefig(f"results_q3/{name}_distance.png")
    plt.close()

def summarize_lattice(L=40, T=60000, s=1.0, w=0.5, n_seeds=8):
    rows = []
    n = L * L
    for seed in range(n_seeds):
        rt, pt, st, _ = run_lattice(L=L, T=T, s=s, w=w, seed=seed)
        r = rt / n
        p = pt / n
        dist = np.sqrt((r - 1.0 / 3.0) ** 2 + (p - 1.0 / 3.0) ** 2)
        rows.append(
            {
                "seed": seed,
                "L": L,
                "N": n,
                "s": s,
                "w": w,
                "late_distance_mean": float(np.mean(dist[-5000:])),
                "final_R": float(r[-1]),
                "final_P": float(p[-1]),
                "final_S": float(1.0 - r[-1] - p[-1]),
            }
        )
    return rows


def plot_lattice_seed_summary(rows, outdir="results_q3"):
    os.makedirs(outdir, exist_ok=True)

    dvals = [float(r["late_distance_mean"]) for r in rows]
    rvals = [float(r["final_R"]) for r in rows]
    pvals = [float(r["final_P"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    axes[0].boxplot([dvals], tick_labels=["late distance"])
    axes[0].set_ylabel("distance to mixed point")

    axes[1].scatter(rvals, pvals, alpha=0.85)
    axes[1].set_xlabel("final R share")
    axes[1].set_ylabel("final P share")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lattice_seed_summary.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    Rt, Pt, St, grid = run_lattice(L=40, T=60000, s=1.0, w=0.5, seed=2)
    plot_q3(Rt, Pt, St, grid, "lattice_rps")

    rows = summarize_lattice()
    out_csv = "results_q3/lattice_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", out_csv)
    plot_lattice_seed_summary(rows)
