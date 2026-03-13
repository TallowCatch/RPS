"""
Q1: Finite-population non-zero-sum RPS with pairwise comparison (Local Update).

Sources acknowledged:
- Module brief for March 2026 (project specification).
- Claussen (2016), Chapter 24, "Evolutionary Dynamics: How Payoffs and
  Global Feedback Control the Stability" for finite-population Local
  Update process context.
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


def payoff_from_counts(strategy, counts, s):
    n_r, n_p, n_s = counts
    if strategy == R:
        return -s * n_p + n_s
    if strategy == P:
        return n_r - s * n_s
    return -s * n_r + n_p


def pairwise_probability(pi_a, pi_b, w, pi_max_diff):
    # Local-update form:
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


def distance_to_mixed(rt, pt, st):
    n = rt + pt + st + 1e-12
    r = rt / n
    p = pt / n
    return np.sqrt((r - 1.0 / 3.0) ** 2 + (p - 1.0 / 3.0) ** 2)


def plot_results(rt, pt, st, name):
    t = np.arange(len(rt))

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(8.8, 4.8))
    plt.plot(t, rt, label="R(t)")
    plt.plot(t, pt, label="P(t)")
    plt.xlabel("t")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{name}_timeseries.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.2, 5.8))
    plt.plot(rt, pt, linewidth=1)
    plt.xlabel("R")
    plt.ylabel("P")
    plt.tight_layout()
    plt.savefig(f"results/{name}_phase.png", dpi=160)
    plt.close()

    dist = distance_to_mixed(rt, pt, st)
    plt.figure(figsize=(8.8, 4.8))
    plt.plot(t, dist)
    plt.xlabel("t")
    plt.ylabel("distance d(t) to mixed state")
    plt.tight_layout()
    plt.savefig(f"results/{name}_distance.png", dpi=160)
    plt.close()


def plot_q1_summary(rows, long_rows, s_values, w_values):
    s_values = list(s_values)
    w_values = list(w_values)

    med_ext = np.zeros((len(s_values), len(w_values)))
    edge_frac = np.zeros((len(s_values), len(w_values)))
    late_dist = np.zeros((len(s_values), len(w_values)))
    for i, s in enumerate(s_values):
        for j, w in enumerate(w_values):
            row = next(
                r for r in rows if float(r["s"]) == float(s) and float(r["w"]) == float(w)
            )
            med_ext[i, j] = float(row["extinction_time_median"])
            edge_frac[i, j] = float(row["edge_contact_fraction"])
            late_dist[i, j] = float(row["late_distance_mean_mean"])

    plt.figure(figsize=(7.2, 5.0))
    im = plt.imshow(med_ext, cmap="viridis", aspect="auto", origin="lower")
    plt.colorbar(im, label="median first-extinction time")
    plt.xticks(range(len(w_values)), [str(w) for w in w_values])
    plt.yticks(range(len(s_values)), [str(s) for s in s_values])
    plt.xlabel("w")
    plt.ylabel("s")
    for i in range(len(s_values)):
        for j in range(len(w_values)):
            plt.text(
                j,
                i,
                f"{med_ext[i, j]:.0f}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )
    plt.tight_layout()
    plt.savefig("results/q1_extinction_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7.2, 5.0))
    im = plt.imshow(
        edge_frac, cmap="magma", aspect="auto", origin="lower", vmin=0.0, vmax=1.0
    )
    plt.colorbar(im, label="edge-contact fraction")
    plt.xticks(range(len(w_values)), [str(w) for w in w_values])
    plt.yticks(range(len(s_values)), [str(s) for s in s_values])
    plt.xlabel("w")
    plt.ylabel("s")
    for i in range(len(s_values)):
        for j in range(len(w_values)):
            plt.text(
                j,
                i,
                f"{edge_frac[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )
    plt.tight_layout()
    plt.savefig("results/q1_edge_contact_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    for s in s_values:
        ys = []
        for w in w_values:
            row = next(
                r for r in rows if float(r["s"]) == float(s) and float(r["w"]) == float(w)
            )
            ys.append(float(row["extinction_time_median"]))
        plt.plot(w_values, ys, marker="o", label=f"s={s}")
    plt.xlabel("w")
    plt.ylabel("median first-extinction time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/q1_median_extinction_vs_w.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)
    markers = ["o", "s", "^", "D"]
    for j, w in enumerate(w_values):
        axes[0].plot(
            s_values,
            med_ext[:, j],
            marker=markers[j % len(markers)],
            linewidth=2.0,
            label=f"w={w}",
        )
        axes[1].plot(
            s_values,
            late_dist[:, j],
            marker=markers[j % len(markers)],
            linewidth=2.0,
            label=f"w={w}",
        )

    axes[0].set_xlabel("s")
    axes[0].set_ylabel("median first-extinction time")
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("s")
    axes[1].set_ylabel("mean late-time distance")
    axes[1].grid(alpha=0.25)

    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig("results/q1_combined_s_w.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(7.0, 4.8))
    for s in s_values:
        ys = []
        for w in w_values:
            row = next(
                r for r in rows if float(r["s"]) == float(s) and float(r["w"]) == float(w)
            )
            ys.append(float(row["late_distance_mean_mean"]))
        plt.plot(w_values, ys, marker="o", label=f"s={s}")
    plt.xlabel("w")
    plt.ylabel("mean late-time distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/q1_late_distance_vs_w.png", dpi=160)
    plt.close()

    groups = []
    labels = []
    for s in s_values:
        for w in w_values:
            vals = [
                float(r["extinction_time"])
                for r in long_rows
                if float(r["s"]) == float(s) and float(r["w"]) == float(w)
            ]
            groups.append(vals)
            labels.append(f"s={s}\nw={w}")

    plt.figure(figsize=(10.8, 5.2))
    plt.boxplot(groups, showfliers=False)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=0)
    plt.ylabel("first-extinction time")
    plt.tight_layout()
    plt.savefig("results/q1_extinction_boxplot.png", dpi=160)
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
    long_rows = []
    for s in s_values:
        for w in w_values:
            extinction_times = []
            late_distances = []
            final_r = []
            final_p = []
            for seed in range(n_seeds):
                rt, pt, st = run_rps_pairwise(N=N, s=s, w=w, T=T, seed=seed)
                t_ext = first_extinction_time(rt, pt, st)
                t_ext_val = (T + 1) if t_ext is None else t_ext
                extinction_times.append(t_ext_val)

                dist = distance_to_mixed(rt, pt, st)
                late_d = float(np.mean(dist[-5000:]))
                late_distances.append(late_d)
                final_r.append(rt[-1] / N)
                final_p.append(pt[-1] / N)

                long_rows.append(
                    {
                        "N": N,
                        "T": T,
                        "s": s,
                        "w": w,
                        "seed": seed,
                        "extinction_time": t_ext_val,
                        "edge_contact": int(t_ext_val <= T),
                        "late_distance_mean": late_d,
                        "final_R": float(rt[-1] / N),
                        "final_P": float(pt[-1] / N),
                    }
                )

            # Representative trajectory for figures.
            rep_rt, rep_pt, rep_st = run_rps_pairwise(N=N, s=s, w=w, T=T, seed=1)
            name = f"N{N}_s{s}_w{w}"
            print("Running:", name)
            plot_results(rep_rt, rep_pt, rep_st, name)

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
                "extinction_time_q25": float(np.percentile(extinction_times, 25)),
                "extinction_time_q75": float(np.percentile(extinction_times, 75)),
                "edge_contact_fraction": float(np.mean(np.array(extinction_times) <= T)),
                "late_distance_mean_mean": float(np.mean(late_distances)),
                "late_distance_mean_std": float(np.std(late_distances)),
                "late_distance_q25": float(np.percentile(late_distances, 25)),
                "late_distance_q75": float(np.percentile(late_distances, 75)),
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

    out_long_csv = "results/q1_seed_metrics.csv"
    with open(out_long_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        writer.writeheader()
        writer.writerows(long_rows)
    print("Saved:", out_long_csv)

    plot_q1_summary(rows, long_rows, s_values=s_values, w_values=w_values)


def main():
    run_q1_sweep()


if __name__ == "__main__":
    main()
