# rps_q3_smallworld.py
# Q3 Option C: RPS on a Watts–Strogatz small-world network
# Compare different rewiring probabilities (controls clustering).

import numpy as np
import matplotlib.pyplot as plt
import os

R, P, S = 0, 1, 2

# -----------------------------
# Payoff: non-zero-sum RPS
# -----------------------------
def payoff_one_vs_one(a, b, s):
    if a == R and b == P: return -s
    if a == R and b == S: return 1
    if a == P and b == S: return -s
    if a == P and b == R: return 1
    if a == S and b == R: return -s
    if a == S and b == P: return 1
    return 0.0

# -----------------------------
# Build Watts–Strogatz network (undirected)
# N nodes in a ring lattice with k neighbors, then rewire each edge with prob beta
# Returns adjacency list: list[set[int]]
# -----------------------------
def watts_strogatz_graph(N, k, beta, seed=0):
    if k % 2 != 0:
        raise ValueError("k must be even for Watts–Strogatz ring lattice.")
    rng = np.random.default_rng(seed)

    adj = [set() for _ in range(N)]

    # ring lattice: connect i to i±1..i±k/2
    half = k // 2
    for i in range(N):
        for d in range(1, half + 1):
            j = (i + d) % N
            adj[i].add(j)
            adj[j].add(i)

    # rewire edges (only consider i -> i+d direction to avoid double processing)
    for i in range(N):
        for d in range(1, half + 1):
            j = (i + d) % N
            if rng.random() < beta:
                # remove existing edge (i, j)
                if j in adj[i]:
                    adj[i].remove(j)
                    adj[j].remove(i)

                # choose new target m not equal to i and not already connected
                # ensure no self-loops and no multi-edges
                possible = np.setdiff1d(np.arange(N), np.array([i] + list(adj[i])), assume_unique=False)
                if possible.size == 0:
                    # fallback: restore old edge if no possible node (very rare)
                    adj[i].add(j)
                    adj[j].add(i)
                    continue
                m = int(rng.choice(possible))
                adj[i].add(m)
                adj[m].add(i)

    return adj

# -----------------------------
# Approximate clustering coefficient (global average of local clustering)
# For each node: C_i = (# edges among neighbors) / (deg*(deg-1)/2)
# -----------------------------
def clustering_coefficient(adj):
    N = len(adj)
    Cs = []
    for i in range(N):
        neigh = list(adj[i])
        d = len(neigh)
        if d < 2:
            Cs.append(0.0)
            continue
        edges_among = 0
        neigh_set = set(neigh)
        # count each neighbor-neighbor edge once
        for idx, u in enumerate(neigh):
            for v in neigh[idx+1:]:
                if v in adj[u]:
                    edges_among += 1
        Cs.append(edges_among / (d * (d - 1) / 2))
    return float(np.mean(Cs))

# -----------------------------
# Local payoff = sum vs neighbors
# -----------------------------
def local_payoff(strats, adj, node, s):
    a = strats[node]
    total = 0.0
    for nb in adj[node]:
        total += payoff_one_vs_one(a, strats[nb], s)
    return total

# -----------------------------
# Run pairwise comparison on network:
# pick B uniformly; pick A uniformly among B's neighbors; B adopts A with prob
# -----------------------------
def run_smallworld_rps(N=400, k=8, beta=0.1, T=60000, s=1.0, w=0.5, seed=1):
    rng = np.random.default_rng(seed)

    adj = watts_strogatz_graph(N, k, beta, seed=seed)
    C = clustering_coefficient(adj)

    # start near mixed equilibrium
    strats = rng.integers(0, 3, size=N)

    Rt = np.zeros(T + 1, dtype=int)
    Pt = np.zeros(T + 1, dtype=int)
    St = np.zeros(T + 1, dtype=int)

    counts = np.bincount(strats, minlength=3)
    Rt[0], Pt[0], St[0] = counts[R], counts[P], counts[S]

    # max local payoff magnitude: degree * max(1, s)
    pi_max = k * max(1.0, s)

    for t in range(1, T + 1):
        b = int(rng.integers(N))
        if not adj[b]:
            # isolated node (shouldn't happen in WS), skip
            Rt[t], Pt[t], St[t] = Rt[t-1], Pt[t-1], St[t-1]
            continue

        a = int(rng.choice(list(adj[b])))

        strat_b = int(strats[b])
        strat_a = int(strats[a])

        pi_b = local_payoff(strats, adj, b, s)
        pi_a = local_payoff(strats, adj, a, s)

        p = 0.5 + (w * (pi_a - pi_b)) / (8.0 * pi_max)
        p = float(np.clip(p, 0.0, 1.0))

        if strat_a != strat_b and rng.random() < p:
            strats[b] = strat_a

        counts = np.bincount(strats, minlength=3)
        Rt[t], Pt[t], St[t] = counts[R], counts[P], counts[S]

    return Rt, Pt, St, C

# -----------------------------
# Plotting helpers
# -----------------------------
def plot_outputs(Rt, Pt, St, title, outdir="results_q3_sw"):
    os.makedirs(outdir, exist_ok=True)

    t = np.arange(len(Rt))
    N = Rt[0] + Pt[0] + St[0]
    r = Rt / N
    p = Pt / N
    s = St / N

    # time series
    plt.figure(figsize=(10, 5))
    plt.plot(t, r, label="R proportion")
    plt.plot(t, p, label="P proportion")
    plt.plot(t, s, label="S proportion")
    plt.xlabel("t")
    plt.ylabel("proportion")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title}_timeseries.png"))
    plt.close()

    # phase plot (R,P)
    plt.figure(figsize=(6, 6))
    plt.plot(r, p, linewidth=1)
    plt.xlabel("R proportion")
    plt.ylabel("P proportion")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title}_phase.png"))
    plt.close()

def main():
    # Choose 3 rewiring values to compare clustering effects:
    # beta=0.0 => lattice-like (high clustering)
    # beta=0.1 => small-world (still clustered, some shortcuts)
    # beta=0.5 => more random (low clustering, more mixing)
    N = 40 * 40   # 1600 nodes
    k = 8
    T = 60000
    s = 1.0
    w = 0.5
    seed = 2

    for beta in [0.0, 0.1, 0.5]:
        Rt, Pt, St, C = run_smallworld_rps(N=N, k=k, beta=beta, T=T, s=s, w=w, seed=seed)
        title = f"sw_N{N}_k{k}_beta{beta}_C{C:.2f}_s{s}_w{w}"
        print("Saved:", title)
        plot_outputs(Rt, Pt, St, title)

if __name__ == "__main__":
    main()
