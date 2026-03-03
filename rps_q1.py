"""
Q1: Finite-population non-zero-sum RPS with pairwise comparison (Local Update).

Sources acknowledged:
- GT_Coursework_March2026_v01.pdf (assignment specification).
- Claussen (2016), "Evolutionary Dynamics: How Payoffs and Global Feedback..."
  for finite-population Local Update process context.
- Traulsen, Claussen, Hauert (2006), Phys. Rev. E 74, 011901
  for pairwise comparison process background.

Authorship note:
- This implementation in this repository was prepared by Ameer Alhashemi.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

R, P, S = 0, 1, 2


def payoff_from_counts(strategy, counts, s):
    n_r, n_p, n_s = counts
    if strategy == R:
        return -s * n_p + n_s
    if strategy == P:
        return n_r - s * n_s
    return -s * n_r + n_p


def pairwise_probability(pi_a, pi_b, w, pi_max_diff):
    # Coursework local-update form:
    # p_{B->A} = 1/2 + w (pi_A - pi_B) / (2 * 4 * pi_max)
    p = 0.5 + (w * (pi_a - pi_b)) / (8.0 * pi_max_diff)
    return float(np.clip(p, 0.0, 1.0))


def run_rps_pairwise(N=150, s=1.0, w=0.5, T=30_000, seed=0):
    rng = np.random.default_rng(seed)

    pop = np.array([R] * (N // 3) + [P] * (N // 3) + [S] * (N - 2 * (N // 3)))
    rng.shuffle(pop)

    counts = np.bincount(pop, minlength=3)

    # Max payoff difference between two agents in this game:
    # each pair interaction difference is bounded by (1 + s), summed over (N-1) opponents.
    pi_max_diff = (N - 1) * (1.0 + s)

    rt = np.zeros(T + 1, dtype=int)
    pt = np.zeros(T + 1, dtype=int)
    st = np.zeros(T + 1, dtype=int)
    rt[0], pt[0], st[0] = counts[R], counts[P], counts[S]

    for t in range(1, T + 1):
        b = int(rng.integers(N))
        a = int(rng.integers(N - 1))
        if a >= b:
            a += 1

        strat_b = int(pop[b])
        strat_a = int(pop[a])

        pi_b = payoff_from_counts(strat_b, counts, s)
        pi_a = payoff_from_counts(strat_a, counts, s)
        p = pairwise_probability(pi_a, pi_b, w=w, pi_max_diff=pi_max_diff)

        if strat_b != strat_a and rng.random() < p:
            pop[b] = strat_a
            counts[strat_b] -= 1
            counts[strat_a] += 1

        rt[t], pt[t], st[t] = counts[R], counts[P], counts[S]

    return rt, pt, st


def first_extinction_time(rt, pt, st):
    extinct_idx = np.where((rt == 0) | (pt == 0) | (st == 0))[0]
    if extinct_idx.size == 0:
        return None
    return int(extinct_idx[0])


def plot_results(rt, pt, name):
    t = np.arange(len(rt))

    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.plot(t, rt, label="R(t)")
    plt.plot(t, pt, label="P(t)")
    plt.xlabel("t")
    plt.ylabel("count")
    plt.legend()
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"results/{name}_timeseries.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(rt, pt, linewidth=1)
    plt.xlabel("R")
    plt.ylabel("P")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"results/{name}_phase.png", dpi=160)
    plt.close()


def run_q1_sweep(
    N=150,
    T=30_000,
    s_values=(0.8, 1.0, 1.2),
    w_values=(0.1, 0.5, 1.0),
    n_seeds=20,
):
    os.makedirs("results", exist_ok=True)
    rows = []
    for s in s_values:
        for w in w_values:
            extinction_times = []
            final_r = []
            final_p = []
            for seed in range(n_seeds):
                rt, pt, st = run_rps_pairwise(N=N, s=s, w=w, T=T, seed=seed)
                t_ext = first_extinction_time(rt, pt, st)
                extinction_times.append((T + 1) if t_ext is None else t_ext)
                final_r.append(rt[-1] / N)
                final_p.append(pt[-1] / N)

            # Representative trajectory for figures.
            rep_rt, rep_pt, _ = run_rps_pairwise(N=N, s=s, w=w, T=T, seed=1)
            name = f"N{N}_s{s}_w{w}"
            print("Running:", name)
            plot_results(rep_rt, rep_pt, name)

            row = {
                "N": N,
                "T": T,
                "s": s,
                "w": w,
                "n_seeds": n_seeds,
                "extinction_time_mean": float(np.mean(extinction_times)),
                "extinction_time_median": float(np.median(extinction_times)),
                "extinction_time_min": int(np.min(extinction_times)),
                "extinction_time_max": int(np.max(extinction_times)),
                "final_R_mean": float(np.mean(final_r)),
                "final_P_mean": float(np.mean(final_p)),
            }
            rows.append(row)

    out_csv = "results/q1_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", out_csv)


def main():
    run_q1_sweep()


if __name__ == "__main__":
    main()
