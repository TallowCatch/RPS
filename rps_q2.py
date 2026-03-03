"""
Q2: Seasonal birth-only/death-only extension with fluctuating population size.

Sources acknowledged:
- GT_Coursework_March2026_v01.pdf (assignment requirement for separate birth and death).
- Claussen (2016), Chapter 24 (finite-population stochastic local-update context).
- Lecture notes on Local Update / pairwise comparison process.

Authorship note:
- This implementation in this repository was prepared by Ameer Alhashemi.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

R, P, S = 0, 1, 2


def payoff(strategy, counts, s):
    n_r, n_p, n_s = counts
    if strategy == R:
        return -s * n_p + n_s
    if strategy == P:
        return n_r - s * n_s
    return -s * n_r + n_p


def pairwise_probability(pi_a, pi_b, w, pi_max_diff):
    p = 0.5 + (w * (pi_a - pi_b)) / (8.0 * pi_max_diff)
    return float(np.clip(p, 0.0, 1.0))


def seasonal_birth_prob(t, period=8000, base=0.50, amp=0.30):
    # Slow sinusoidal seasonality in [base-amp, base+amp].
    return base + amp * np.sin(2.0 * np.pi * t / period)


def run_q2(
    T=40_000,
    s=1.0,
    w=0.1,
    mutation_rate=0.0,
    period=8000,
    base=0.50,
    amp=0.30,
    n0=120,
    seed=1,
):
    rng = np.random.default_rng(seed)

    # Start near mixed equilibrium with finite-size rounding.
    pop = [R] * (n0 // 3) + [P] * (n0 // 3) + [S] * (n0 - 2 * (n0 // 3))

    rt, pt, st, nt = [], [], [], []

    for t in range(T):
        n = len(pop)
        if n < 2:
            break

        counts = np.bincount(pop, minlength=3)
        a, b = rng.choice(n, 2, replace=False)
        strat_a = int(pop[a])
        strat_b = int(pop[b])

        pi_a = payoff(strat_a, counts, s)
        pi_b = payoff(strat_b, counts, s)

        pi_max_diff = (n - 1) * (1.0 + s)
        p_b_to_a = pairwise_probability(pi_a, pi_b, w=w, pi_max_diff=pi_max_diff)

        if rng.random() < seasonal_birth_prob(t, period=period, base=base, amp=amp):
            # Birth-only event: one strategy replicates, opponent stays alive.
            parent_idx = a if rng.random() < p_b_to_a else b
            new_strat = int(pop[parent_idx])
            if mutation_rate > 0.0 and rng.random() < mutation_rate:
                new_strat = int(rng.integers(3))
            pop.append(new_strat)
        else:
            # Death-only event: one of the pair is removed.
            victim_idx = b if rng.random() < p_b_to_a else a
            pop.pop(victim_idx)

        counts = np.bincount(pop, minlength=3) if pop else np.array([0, 0, 0])
        rt.append(int(counts[R]))
        pt.append(int(counts[P]))
        st.append(int(counts[S]))
        nt.append(len(pop))

    return np.array(rt), np.array(pt), np.array(st), np.array(nt)


def plot_q2(rt, pt, st, nt, name):
    os.makedirs("results_q2", exist_ok=True)
    t = np.arange(len(rt))

    total = rt + pt + st + 1e-12
    r_norm = rt / total
    p_norm = pt / total

    plt.figure(figsize=(10, 5))
    plt.plot(t, r_norm, label="R/N")
    plt.plot(t, p_norm, label="P/N")
    plt.plot(t, nt / np.max(nt), label="N (scaled)", alpha=0.7)
    plt.legend()
    plt.title(name)
    plt.xlabel("t")
    plt.ylabel("normalized")
    plt.tight_layout()
    plt.savefig(f"results_q2/{name}_timeseries.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(r_norm, p_norm, linewidth=1)
    plt.xlabel("R/N")
    plt.ylabel("P/N")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"results_q2/{name}_phase.png", dpi=160)
    plt.close()


def summarize_q2_variant(
    name,
    mutation_rate,
    n_seeds=10,
    T=40_000,
    s=1.0,
    w=0.1,
    period=8000,
    base=0.50,
    amp=0.30,
    n0=120,
):
    rows = []
    for seed in range(n_seeds):
        rt, pt, st, nt = run_q2(
            T=T,
            s=s,
            w=w,
            mutation_rate=mutation_rate,
            period=period,
            base=base,
            amp=amp,
            n0=n0,
            seed=seed,
        )
        extinct_steps = int(np.sum((rt == 0) | (pt == 0) | (st == 0)))
        rows.append(
            {
                "variant": name,
                "seed": seed,
                "mutation_rate": mutation_rate,
                "N_min": int(np.min(nt)),
                "N_max": int(np.max(nt)),
                "N_last": int(nt[-1]),
                "N_mean": float(np.mean(nt)),
                "extinct_steps": extinct_steps,
            }
        )
    return rows


def main():
    os.makedirs("results_q2", exist_ok=True)

    # Coursework baseline: no mutation.
    rt, pt, st, nt = run_q2(mutation_rate=0.0, seed=1)
    plot_q2(rt, pt, st, nt, "seasonal_rps_baseline")

    # Optional robustness extension: add rare mutation.
    rt_m, pt_m, st_m, nt_m = run_q2(mutation_rate=0.01, seed=1)
    plot_q2(rt_m, pt_m, st_m, nt_m, "seasonal_rps_mutation")

    rows = []
    rows.extend(summarize_q2_variant("baseline", mutation_rate=0.0))
    rows.extend(summarize_q2_variant("mutation", mutation_rate=0.01))

    out_csv = "results_q2/q2_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
