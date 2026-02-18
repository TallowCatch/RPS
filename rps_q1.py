import numpy as np
import matplotlib.pyplot as plt
import os

R, P, S = 0, 1, 2

# ---------- payoff ----------
def payoff_from_counts(strategy, counts, s):
    nR, nP, nS = counts
    if strategy == R:
        return -s * nP + nS
    if strategy == P:
        return nR - s * nS
    return -s * nR + nP


# ---------- simulation ----------
def run_rps_pairwise(N=150, s=1.0, w=0.5, T=30_000, seed=0):
    rng = np.random.default_rng(seed)

    pop = np.array([R]*(N//3) + [P]*(N//3) + [S]*(N - 2*(N//3)))
    rng.shuffle(pop)

    counts = np.bincount(pop, minlength=3)

    pi_max = (N - 1) * max(1.0, s)

    Rt = np.zeros(T+1)
    Pt = np.zeros(T+1)

    Rt[0], Pt[0] = counts[R], counts[P]

    for t in range(1, T+1):

        b = rng.integers(N)
        a = rng.integers(N-1)
        if a >= b:
            a += 1

        strat_b = pop[b]
        strat_a = pop[a]

        pi_b = payoff_from_counts(strat_b, counts, s)
        pi_a = payoff_from_counts(strat_a, counts, s)

        p = 0.5 + (w * (pi_a - pi_b)) / (8.0 * pi_max)
        p = np.clip(p, 0, 1)

        if strat_b != strat_a and rng.random() < p:
            pop[b] = strat_a
            counts[strat_b] -= 1
            counts[strat_a] += 1

        Rt[t], Pt[t] = counts[R], counts[P]

    return Rt, Pt


# ---------- plotting ----------
def plot_results(Rt, Pt, name):
    t = np.arange(len(Rt))

    os.makedirs("results", exist_ok=True)

    # time series
    plt.figure()
    plt.plot(t, Rt, label="R(t)")
    plt.plot(t, Pt, label="P(t)")
    plt.xlabel("t")
    plt.ylabel("count")
    plt.legend()
    plt.title(name)
    plt.savefig(f"results/{name}_timeseries.png")
    plt.close()

    # phase plot
    plt.figure()
    plt.plot(Rt, Pt)
    plt.xlabel("R")
    plt.ylabel("P")
    plt.title(name)
    plt.savefig(f"results/{name}_phase.png")
    plt.close()


# ---------- parameter sweep (Q1 part 3.2) ----------
def main():

    N = 150
    T = 30_000

    s_values = [0.8, 1.0, 1.2]
    w_values = [0.1, 0.5, 1.0]

    for s in s_values:
        for w in w_values:
            name = f"N{N}_s{s}_w{w}"
            print("Running:", name)
            Rt, Pt = run_rps_pairwise(N=N, s=s, w=w, T=T, seed=1)
            plot_results(Rt, Pt, name)


if __name__ == "__main__":
    main()
