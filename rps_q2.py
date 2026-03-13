"""
Q2: Seasonal birth-only/death-only extension with fluctuating population size.

Sources acknowledged:
- Module brief for March 2026 (requirement for separate birth and death processes).
- Claussen (2016), Chapter 24, "Evolutionary Dynamics: How Payoffs and
  Global Feedback Control the Stability" for finite-population stochastic
  Local Update context.

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


def distance_to_mixed(rt, pt, st):
    total = rt + pt + st + 1e-12
    r = rt / total
    p = pt / total
    return np.sqrt((r - 1.0 / 3.0) ** 2 + (p - 1.0 / 3.0) ** 2)


def run_q2(
    T=40_000,
    s=1.0,
    w=0.1,
    period=8000,
    base=0.50,
    amp=0.30,
    n0=120,
    seed=1,
):
    rng = np.random.default_rng(seed)

    # Start close to mixed equilibrium with finite-size rounding.
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
            # Birth-only event: one strategy replicates while opponent remains alive.
            parent_idx = a if rng.random() < p_b_to_a else b
            new_strat = int(pop[parent_idx])
            pop.append(new_strat)
        else:
            # Death-only event: one of the sampled agents is removed.
            victim_idx = b if rng.random() < p_b_to_a else a
            pop.pop(victim_idx)

        counts = np.bincount(pop, minlength=3) if pop else np.array([0, 0, 0])
        rt.append(int(counts[R]))
        pt.append(int(counts[P]))
        st.append(int(counts[S]))
        nt.append(len(pop))

    return np.array(rt), np.array(pt), np.array(st), np.array(nt)


def plot_q2(rt, pt, st, nt, name, period=8000, base=0.50, amp=0.30):
    os.makedirs("results_q2", exist_ok=True)
    t = np.arange(len(rt))

    total = rt + pt + st + 1e-12
    r_norm = rt / total
    p_norm = pt / total
    s_norm = st / total
    b_t = seasonal_birth_prob(t, period=period, base=base, amp=amp)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    axes[0].plot(t, r_norm, label="R/N")
    axes[0].plot(t, p_norm, label="P/N")
    axes[0].plot(t, s_norm, label="S/N", alpha=0.85)
    axes[0].plot(t, nt / np.max(nt), label="N (scaled)", alpha=0.7, linewidth=1.0)
    axes[0].set_ylabel("normalized share")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, b_t, color="tab:orange", label="birth probability b(t)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("b(t)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(f"results_q2/{name}_timeseries.png", dpi=160)
    plt.close(fig)

    plt.figure(figsize=(6.5, 6.2))
    plt.plot(r_norm, p_norm, linewidth=1)
    plt.xlabel("R/N")
    plt.ylabel("P/N")
    plt.tight_layout()
    plt.savefig(f"results_q2/{name}_phase.png", dpi=160)
    plt.close()


def plot_q2_diagnostics(rt, pt, st, nt, name):
    os.makedirs("results_q2", exist_ok=True)
    t = np.arange(len(rt))
    d = distance_to_mixed(rt, pt, st)

    fig, axes = plt.subplots(2, 1, figsize=(10.6, 7.4), sharex=False)
    axes[0].plot(t, d, label="distance d(t) to mixed state", color="tab:blue")
    axes[0].plot(t, nt / np.max(nt), label="N (scaled)", color="tab:green", alpha=0.8)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("value")
    axes[0].legend(loc="upper right")

    axes[1].scatter(nt, d, s=5, alpha=0.25, color="tab:purple")
    axes[1].set_xlabel("N(t)")
    axes[1].set_ylabel("distance d(t)")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"results_q2/{name}_distance_population.png", dpi=160)
    plt.close(fig)


def plot_q2_seed_summary(rows):
    os.makedirs("results_q2", exist_ok=True)

    n_min = [float(r["N_min"]) for r in rows]
    n_max = [float(r["N_max"]) for r in rows]
    n_last = [float(r["N_last"]) for r in rows]
    edge = [float(r["near_edge_fraction"]) for r in rows]
    d_late = [float(r["late_distance_mean"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    axes[0].boxplot([n_min, n_max, n_last], tick_labels=["N_min", "N_max", "N_last"])
    axes[0].set_ylabel("population size")

    axes[1].boxplot(
        [edge, d_late], tick_labels=["near-edge fraction", "late distance"]
    )
    axes[1].set_ylabel("fraction / distance")

    plt.tight_layout()
    plt.savefig("results_q2/q2_seed_summary_boxplots.png", dpi=160)
    plt.close(fig)


def summarize_q2(
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
            period=period,
            base=base,
            amp=amp,
            n0=n0,
            seed=seed,
        )
        d = distance_to_mixed(rt, pt, st)
        extinct_steps = int(np.sum((rt == 0) | (pt == 0) | (st == 0)))
        total = rt + pt + st + 1e-12
        min_share = float(np.min(np.minimum(np.minimum(rt, pt), st) / total))
        corr_n_d = float(np.corrcoef(nt, d)[0, 1]) if len(nt) > 1 else 0.0
        rows.append(
            {
                "seed": seed,
                "N_min": int(np.min(nt)),
                "N_max": int(np.max(nt)),
                "N_last": int(nt[-1]),
                "N_mean": float(np.mean(nt)),
                "extinct_steps": extinct_steps,
                "near_edge_fraction": float(extinct_steps / len(nt)),
                "min_strategy_share": min_share,
                "distance_mean": float(np.mean(d)),
                "late_distance_mean": float(np.mean(d[-5000:])),
                "corr_N_distance": corr_n_d,
            }
        )
    return rows


def main():
    os.makedirs("results_q2", exist_ok=True)

    rt, pt, st, nt = run_q2(seed=1)
    plot_q2(rt, pt, st, nt, "seasonal_rps_baseline")
    plot_q2_diagnostics(rt, pt, st, nt, "seasonal_rps_baseline")

    rows = summarize_q2()
    out_csv = "results_q2/q2_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", out_csv)
    plot_q2_seed_summary(rows)


if __name__ == "__main__":
    main()
