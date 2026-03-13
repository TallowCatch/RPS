"""
Q3 (creative): RPS on a Watts-Strogatz small-world network.

Sources acknowledged:
- Module brief for March 2026 (creative lattice/network extension).
- Watts & Strogatz (1998), Nature 393:440-442 (small-world network model).
- Claussen (2016), Chapter 24, "Evolutionary Dynamics: How Payoffs and
  Global Feedback Control the Stability" for Local Update pairwise
  process in finite populations.

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
    if a == R and b == P:
        return -s
    if a == R and b == S:
        return 1
    if a == P and b == S:
        return -s
    if a == P and b == R:
        return 1
    if a == S and b == R:
        return -s
    if a == S and b == P:
        return 1
    return 0.0


def pairwise_probability(pi_a, pi_b, w, pi_max_diff):
    p = 0.5 + (w * (pi_a - pi_b)) / (8.0 * pi_max_diff)
    return float(np.clip(p, 0.0, 1.0))


def distance_to_mixed(rt, pt, st):
    n = rt + pt + st + 1e-12
    r = rt / n
    p = pt / n
    return np.sqrt((r - 1.0 / 3.0) ** 2 + (p - 1.0 / 3.0) ** 2)


def watts_strogatz_graph(N, k, beta, seed=0):
    if k % 2 != 0:
        raise ValueError("k must be even for Watts-Strogatz ring lattice.")
    rng = np.random.default_rng(seed)
    adj = [set() for _ in range(N)]

    half = k // 2
    for i in range(N):
        for d in range(1, half + 1):
            j = (i + d) % N
            adj[i].add(j)
            adj[j].add(i)

    for i in range(N):
        for d in range(1, half + 1):
            j = (i + d) % N
            if rng.random() < beta:
                if j in adj[i]:
                    adj[i].remove(j)
                    adj[j].remove(i)

                excluded = np.array([i] + list(adj[i]), dtype=int)
                possible = np.setdiff1d(np.arange(N), excluded, assume_unique=False)
                if possible.size == 0:
                    adj[i].add(j)
                    adj[j].add(i)
                    continue
                m = int(rng.choice(possible))
                adj[i].add(m)
                adj[m].add(i)
    return adj


def clustering_coefficient(adj):
    cs = []
    for i in range(len(adj)):
        neigh = list(adj[i])
        d = len(neigh)
        if d < 2:
            cs.append(0.0)
            continue
        edges_among = 0
        for idx, u in enumerate(neigh):
            for v in neigh[idx + 1 :]:
                if v in adj[u]:
                    edges_among += 1
        cs.append(edges_among / (d * (d - 1) / 2))
    return float(np.mean(cs))


def local_payoff(strats, adj, node, s, normalized=True):
    neigh = adj[node]
    if not neigh:
        return 0.0

    a = int(strats[node])
    total = 0.0
    for nb in neigh:
        total += payoff_one_vs_one(a, int(strats[nb]), s)

    if normalized:
        return total / len(neigh)
    return total


def run_smallworld_rps(
    N=1600,
    k=8,
    beta=0.1,
    T=60_000,
    s=1.0,
    w=0.5,
    seed=1,
    normalized_payoff=True,
):
    rng = np.random.default_rng(seed)
    adj = watts_strogatz_graph(N, k, beta, seed=seed)
    c_coeff = clustering_coefficient(adj)

    strats = rng.integers(0, 3, size=N)
    rt = np.zeros(T + 1, dtype=int)
    pt = np.zeros(T + 1, dtype=int)
    st = np.zeros(T + 1, dtype=int)

    counts = np.bincount(strats, minlength=3)
    rt[0], pt[0], st[0] = counts[R], counts[P], counts[S]

    # Using normalized payoff removes degree-driven payoff-scale confounds after rewiring.
    if normalized_payoff:
        pi_max_diff = 1.0 + s
    else:
        pi_max_diff = k * (1.0 + s)

    for t in range(1, T + 1):
        b = int(rng.integers(N))
        if not adj[b]:
            rt[t], pt[t], st[t] = rt[t - 1], pt[t - 1], st[t - 1]
            continue

        a = int(rng.choice(list(adj[b])))
        strat_b = int(strats[b])
        strat_a = int(strats[a])

        pi_b = local_payoff(strats, adj, b, s=s, normalized=normalized_payoff)
        pi_a = local_payoff(strats, adj, a, s=s, normalized=normalized_payoff)
        p = pairwise_probability(pi_a, pi_b, w=w, pi_max_diff=pi_max_diff)

        if strat_a != strat_b and rng.random() < p:
            strats[b] = strat_a
            counts[strat_b] -= 1
            counts[strat_a] += 1

        rt[t], pt[t], st[t] = counts[R], counts[P], counts[S]

    return rt, pt, st, c_coeff


def plot_outputs(rt, pt, st, title, outdir="results_q3_sw"):
    os.makedirs(outdir, exist_ok=True)
    t = np.arange(len(rt))
    n = rt[0] + pt[0] + st[0]
    r = rt / n
    p = pt / n
    s = st / n

    plt.figure(figsize=(10.8, 5.2))
    plt.plot(t, r, label="R/N")
    plt.plot(t, p, label="P/N")
    plt.plot(t, s, label="S/N")
    plt.xlabel("t")
    plt.ylabel("proportion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title}_timeseries.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(6.5, 6.2))
    plt.plot(r, p, linewidth=1)
    plt.xlabel("R/N")
    plt.ylabel("P/N")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title}_phase.png"), dpi=160)
    plt.close()

    dist = distance_to_mixed(rt, pt, st)
    plt.figure(figsize=(9.8, 4.6))
    plt.plot(t, dist)
    plt.xlabel("t")
    plt.ylabel("distance d(t) to mixed state")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title}_distance.png"), dpi=160)
    plt.close()


def plot_sw_summary(rows, outdir="results_q3_sw"):
    os.makedirs(outdir, exist_ok=True)

    betas = sorted(set(float(r["beta"]) for r in rows))
    mean_d = []
    std_d = []
    mean_c = []

    for beta in betas:
        subset = [r for r in rows if float(r["beta"]) == beta]
        ds = [float(r["late_distance_mean"]) for r in subset]
        cs = [float(r["C"]) for r in subset]
        mean_d.append(float(np.mean(ds)))
        std_d.append(float(np.std(ds)))
        mean_c.append(float(np.mean(cs)))

    fig, ax1 = plt.subplots(figsize=(7.8, 5.2))
    ax1.errorbar(betas, mean_d, yerr=std_d, marker="o", capsize=4, label="late distance")
    ax1.set_xlabel("rewiring probability beta")
    ax1.set_ylabel("late distance to mixed point")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(betas, mean_c, marker="s", color="tab:orange", label="clustering C")
    ax2.set_ylabel("mean clustering coefficient")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "q3_sw_beta_distance_clustering.png"), dpi=160)
    plt.close(fig)

    plt.figure(figsize=(7.0, 5.2))
    c_vals = [float(r["C"]) for r in rows]
    d_vals = [float(r["late_distance_mean"]) for r in rows]
    beta_vals = [float(r["beta"]) for r in rows]
    scatter = plt.scatter(c_vals, d_vals, c=beta_vals, cmap="viridis", alpha=0.8)
    plt.colorbar(scatter, label="beta")
    plt.xlabel("clustering coefficient C")
    plt.ylabel("late distance to mixed point")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "q3_sw_clustering_vs_distance.png"), dpi=160)
    plt.close()


def save_sw_aggregate(rows, out_csv="results_q3_sw/q3_sw_aggregate.csv"):
    betas = sorted(set(float(r["beta"]) for r in rows))
    agg_rows = []
    for beta in betas:
        subset = [r for r in rows if float(r["beta"]) == beta]
        cs = np.array([float(r["C"]) for r in subset], dtype=float)
        ds = np.array([float(r["late_distance_mean"]) for r in subset], dtype=float)
        rs = np.array([float(r["final_R"]) for r in subset], dtype=float)
        ps = np.array([float(r["final_P"]) for r in subset], dtype=float)
        ss = np.array([float(r["final_S"]) for r in subset], dtype=float)
        agg_rows.append(
            {
                "beta": beta,
                "n_seeds": len(subset),
                "C_mean": float(np.mean(cs)),
                "C_std": float(np.std(cs)),
                "late_distance_mean": float(np.mean(ds)),
                "late_distance_std": float(np.std(ds)),
                "final_R_mean": float(np.mean(rs)),
                "final_P_mean": float(np.mean(ps)),
                "final_S_mean": float(np.mean(ss)),
            }
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)
    print("Saved:", out_csv)


def summarize_smallworld(
    N=1600,
    k=8,
    T=20_000,
    s=1.0,
    w=0.5,
    betas=(0.0, 0.1, 0.5),
    n_seeds=8,
    normalized_payoff=True,
):
    rows = []
    for beta in betas:
        for seed in range(n_seeds):
            rt, pt, st, c_coeff = run_smallworld_rps(
                N=N,
                k=k,
                beta=beta,
                T=T,
                s=s,
                w=w,
                seed=seed,
                normalized_payoff=normalized_payoff,
            )
            n = rt[0] + pt[0] + st[0]
            r = rt / n
            p = pt / n
            dist = np.sqrt((r - 1.0 / 3.0) ** 2 + (p - 1.0 / 3.0) ** 2)
            rows.append(
                {
                    "beta": beta,
                    "seed": seed,
                    "C": float(c_coeff),
                    "late_distance_mean": float(np.mean(dist[-5000:])),
                    "final_R": float(r[-1]),
                    "final_P": float(p[-1]),
                    "final_S": float(1.0 - r[-1] - p[-1]),
                    "normalized_payoff": normalized_payoff,
                }
            )
    return rows


def main():
    N = 40 * 40
    k = 8
    T = 60_000
    s = 1.0
    w = 0.5
    seed = 2

    for beta in [0.0, 0.1, 0.5]:
        rt, pt, st, c_coeff = run_smallworld_rps(
            N=N, k=k, beta=beta, T=T, s=s, w=w, seed=seed, normalized_payoff=True
        )
        title = f"sw_N{N}_k{k}_beta{beta}_C{c_coeff:.2f}_s{s}_w{w}"
        print("Saved:", title)
        plot_outputs(rt, pt, st, title)

    rows = summarize_smallworld(N=N, k=k)
    out_csv = "results_q3_sw/q3_sw_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", out_csv)

    plot_sw_summary(rows)
    save_sw_aggregate(rows)


if __name__ == "__main__":
    main()
