"""Minimal Particle Swarm Optimisation with velocity clamping and box constraints."""
import numpy as np

def pso(fun, lb, ub, n_particles=60, n_iter=150, w=0.7, c1=1.5, c2=1.5, vmax_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    dim = lb.size
    x = lb + (ub - lb) * rng.random((n_particles, dim))
    v = np.zeros_like(x)
    vmax = vmax_frac * (ub - lb)
    pbest_x = x.copy()
    pbest_f = np.array([fun(xi) for xi in x])
    g = np.argmin(pbest_f)
    gbest_x, gbest_f = pbest_x[g].copy(), pbest_f[g]
    history = [gbest_f]
    for _ in range(n_iter):
        r1, r2 = rng.random((n_particles, dim)), rng.random((n_particles, dim))
        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        v = np.clip(v, -vmax, vmax)
        x = np.clip(x + v, lb, ub)
        f = np.array([fun(xi) for xi in x])
        better = f < pbest_f
        pbest_x[better], pbest_f[better] = x[better], f[better]
        g = np.argmin(pbest_f)
        if pbest_f[g] < gbest_f:
            gbest_x, gbest_f = pbest_x[g].copy(), pbest_f[g]
        history.append(gbest_f)
    return gbest_x, gbest_f, np.array(history)
