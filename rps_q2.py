import numpy as np
import matplotlib.pyplot as plt
import os

R, P, S = 0, 1, 2

# -----------------------------
# Payoff function
# -----------------------------
def payoff(strategy, counts, s):
    nR, nP, nS = counts
    if strategy == R:
        return -s*nP + nS
    if strategy == P:
        return nR - s*nS
    return -s*nR + nP


# -----------------------------
# Seasonal birth probability
# slow oscillation
# -----------------------------
def seasonal_birth_prob(t, period=10000, base=0.5, amp=0.4):
    return base + amp * np.sin(2*np.pi*t/period)


# -----------------------------
# Q2 simulation
# -----------------------------
def run_q2(T=40000, s=1.0, w=0.1, mutation_rate=0.01, seed=1):

    rng = np.random.default_rng(seed)

    # start near equilibrium
    pop = [R, P, S] * 20
    pop = list(pop)

    Rt, Pt, St, Nt = [], [], [], []

    for t in range(T):

        N = len(pop)
        counts = np.bincount(pop, minlength=3)

        if N < 2:
            break

        a, b = rng.choice(N, 2, replace=False)
        strat_a = pop[a]
        strat_b = pop[b]

        pi_a = payoff(strat_a, counts, s)
        pi_b = payoff(strat_b, counts, s)

        pi_max = (N-1)*max(1.0, s)

        p = 0.5 + (w*(pi_a-pi_b))/(8*pi_max)
        p = np.clip(p, 0, 1)

        birth_p = seasonal_birth_prob(t)

        # -----------------
        # birth event
        # -----------------
        if rng.random() < birth_p or N <= 3:
            if rng.random() < p:
                new_strat = strat_a

                # mutation prevents extinction
                if rng.random() < mutation_rate:
                    new_strat = rng.integers(3)

                pop.append(new_strat)

        # -----------------
        # death event
        # -----------------
        else:
            if rng.random() < p and N > 3:
                pop.pop(b)

        counts = np.bincount(pop, minlength=3)

        Rt.append(counts[R])
        Pt.append(counts[P])
        St.append(counts[S])
        Nt.append(len(pop))

    return np.array(Rt), np.array(Pt), np.array(St), np.array(Nt)


# -----------------------------
# Plotting
# -----------------------------
def plot_q2(Rt, Pt, St, Nt, name):

    os.makedirs("results_q2", exist_ok=True)

    t = np.arange(len(Rt))

    total = Rt + Pt + St + 1e-9
    r_norm = Rt / total
    p_norm = Pt / total

    # time series
    plt.figure(figsize=(10,5))
    plt.plot(t, r_norm, label="R proportion")
    plt.plot(t, p_norm, label="P proportion")
    plt.plot(t, Nt/np.max(Nt), label="N (scaled)", alpha=0.6)
    plt.legend()
    plt.title(name)
    plt.xlabel("t")
    plt.ylabel("normalized")
    plt.savefig(f"results_q2/{name}_timeseries.png")
    plt.close()

    # phase plot
    plt.figure(figsize=(6,6))
    plt.plot(r_norm, p_norm)
    plt.xlabel("R proportion")
    plt.ylabel("P proportion")
    plt.title(name)
    plt.savefig(f"results_q2/{name}_phase.png")
    plt.close()


# -----------------------------
# main run
# -----------------------------
if __name__ == "__main__":

    Rt, Pt, St, Nt = run_q2(
        s=1.0,
        w=0.1,
        mutation_rate=0.01
    )

    plot_q2(Rt, Pt, St, Nt, "seasonal_rps_mutation")
