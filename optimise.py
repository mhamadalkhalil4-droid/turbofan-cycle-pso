"""Optimise the turbofan design parameters with PSO and plot the result."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from turbofan_cycle import turbofan_cycle
from pso import pso

LB = [2.0, 1.2, 2.0, 10.0]      # alpha, pi_f, pi_LPC, pi_HPC
UB = [12.0, 2.5, 5.0, 15.0]
OPR_MAX = 60.0                  # overall pressure ratio limit (pi_f·pi_LPC·pi_HPC)
TSFC_TARGET = 17.8              # mg/(N·s)

def objective(x):
    alpha, pi_f, pi_LPC, pi_HPC = x
    if pi_f * pi_LPC * pi_HPC > OPR_MAX:
        return 1e6
    tsfc, fsp, _ = turbofan_cycle(alpha, pi_f, pi_LPC, pi_HPC)
    if not np.isfinite(tsfc):
        return 1e6
    # same objective as the final-year project: hit a TSFC target, reward specific thrust
    return (tsfc - TSFC_TARGET) ** 2 - 0.01 * fsp

if __name__ == "__main__":
    best_x, best_f, hist = pso(objective, LB, UB, n_particles=80, n_iter=200, seed=1)
    tsfc, fsp, f = turbofan_cycle(*best_x)
    print(f"Best design: BPR={best_x[0]:.2f}, FPR={best_x[1]:.3f}, pi_LPC={best_x[2]:.2f}, pi_HPC={best_x[3]:.2f}")
    print(f"OPR={np.prod(best_x[1:]):.1f}  TSFC={tsfc:.2f} mg/N·s  Fsp={fsp:.1f} N·s/kg  f={f:.4f}")

    plt.figure(figsize=(6, 4)); plt.plot(hist, lw=2); plt.xlabel("iteration"); plt.ylabel("best objective")
    plt.title("PSO convergence"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("../figures/convergence.png", dpi=130)

    alphas = np.linspace(2, 12, 60)
    res = np.array([turbofan_cycle(a, best_x[1], best_x[2], best_x[3])[:2] for a in alphas])
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(alphas, res[:, 0], "b", lw=2); ax1.set_xlabel("bypass ratio"); ax1.set_ylabel("TSFC [mg/(N·s)]", color="b")
    ax2 = ax1.twinx(); ax2.plot(alphas, res[:, 1], "r", lw=2); ax2.set_ylabel("specific thrust [N·s/kg]", color="r")
    ax1.axvline(best_x[0], ls="--", c="k", alpha=.5); ax1.grid(alpha=.3)
    plt.title("Bypass-ratio trade-off at the optimum pressure ratios"); fig.tight_layout(); plt.savefig("../figures/bpr_tradeoff.png", dpi=130)

    rng = np.random.default_rng(3)
    X = np.array(LB) + (np.array(UB) - np.array(LB)) * rng.random((3000, 4))
    X = X[np.prod(X[:, 1:], axis=1) <= OPR_MAX]
    cloud = np.array([turbofan_cycle(*x)[:2] for x in X]); cloud = cloud[np.isfinite(cloud[:, 0])]
    plt.figure(figsize=(6, 4)); plt.scatter(cloud[:, 1], cloud[:, 0], s=4, alpha=.3, label="random designs")
    plt.scatter(fsp, tsfc, c="r", s=120, marker="*", label="PSO optimum", zorder=5)
    plt.xlabel("specific thrust [N·s/kg]"); plt.ylabel("TSFC [mg/(N·s)]"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Design space"); plt.tight_layout(); plt.savefig("../figures/design_space.png", dpi=130)
