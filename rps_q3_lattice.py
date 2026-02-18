import numpy as np
import matplotlib.pyplot as plt
import os

R, P, S = 0, 1, 2

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

def run_lattice(L=30, T=50000, s=1.0, w=0.5, seed=1):
    rng = np.random.default_rng(seed)

    # start near equilibrium: random with equal probs
    grid = rng.integers(0, 3, size=(L, L))

    Rt = np.zeros(T+1, dtype=int)
    Pt = np.zeros(T+1, dtype=int)
    St = np.zeros(T+1, dtype=int)

    counts = np.bincount(grid.ravel(), minlength=3)
    Rt[0], Pt[0], St[0] = counts[R], counts[P], counts[S]

    # local max payoff magnitude: each neighbour gives at most max(1,s)
    pi_max = 4 * max(1.0, s)

    for t in range(1, T+1):
        i = rng.integers(L)
        j = rng.integers(L)

        # pick a random neighbour as role model
        ni, nj = neighbours(i, j, L)[rng.integers(4)]

        strat_b = grid[i, j]
        strat_a = grid[ni, nj]

        pi_b = local_payoff(grid, i, j, s)
        pi_a = local_payoff(grid, ni, nj, s)

        p = 0.5 + (w * (pi_a - pi_b)) / (8.0 * pi_max)
        p = np.clip(p, 0, 1)

        if strat_a != strat_b and rng.random() < p:
            grid[i, j] = strat_a

        counts = np.bincount(grid.ravel(), minlength=3)
        Rt[t], Pt[t], St[t] = counts[R], counts[P], counts[S]

    return Rt, Pt, St, grid

def plot_q3(Rt, Pt, St, grid, name):
    os.makedirs("results_q3", exist_ok=True)

    t = np.arange(len(Rt))
    N = Rt[0] + Pt[0] + St[0]

    # time series (proportions)
    plt.figure(figsize=(10,5))
    plt.plot(t, Rt/N, label="R proportion")
    plt.plot(t, Pt/N, label="P proportion")
    plt.plot(t, St/N, label="S proportion")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("proportion")
    plt.title(name)
    plt.savefig(f"results_q3/{name}_timeseries.png")
    plt.close()

    # phase plot (R,P)
    plt.figure(figsize=(6,6))
    plt.plot(Rt/N, Pt/N)
    plt.xlabel("R proportion")
    plt.ylabel("P proportion")
    plt.title(name)
    plt.savefig(f"results_q3/{name}_phase.png")
    plt.close()

    # grid snapshot
    plt.figure(figsize=(6,6))
    plt.imshow(grid, interpolation="nearest")
    plt.title(name + " (final grid state)")
    plt.savefig(f"results_q3/{name}_grid.png")
    plt.close()

if __name__ == "__main__":
    Rt, Pt, St, grid = run_lattice(L=40, T=60000, s=1.0, w=0.5, seed=2)
    plot_q3(Rt, Pt, St, grid, "lattice_rps")
