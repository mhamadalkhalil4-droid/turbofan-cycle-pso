# Turbofan cycle optimisation with Particle Swarm Optimisation

Final-year project, HIAST (2025) — thermodynamic model of a two-spool separate-flow turbofan and
PSO search of its four main cycle parameters.

| Design variable | Bounds |
|---|---|
| Bypass ratio (BPR) | 2 – 12 |
| Fan pressure ratio (FPR) | 1.2 – 2.5 |
| LPC pressure ratio | 2 – 5 |
| HPC pressure ratio | 10 – 15 |
| Constraint: overall pressure ratio FPR·π_LPC·π_HPC | ≤ 60 |

Flight condition: M 0.85 at 11 km. Component efficiencies and TIT = 1450 K are fixed (see `turbofan_cycle.py`).

**Objective** — reach a TSFC target of 17.8 mg/(N·s) while keeping specific thrust as high as possible:
`J = (TSFC − 17.8)² − 0.01·F_sp`.

## Result

| | BPR | FPR | π_LPC | π_HPC | OPR | TSFC [mg/(N·s)] | F_sp [N·s/kg] |
|---|---|---|---|---|---|---|---|
| PSO optimum | 11.6 | 1.51 | 2.65 | 15.0 | 60 | **18.1** | **108** |

The optimiser pushes BPR to the upper bound and OPR onto the constraint — exactly the direction real
engines took (LEAP, GTF): propulsive efficiency rises with BPR, thermal efficiency with OPR, and
specific thrust is the price paid.

![design space](figures/design_space.png)
![BPR trade-off](figures/bpr_tradeoff.png)
![convergence](figures/convergence.png)

## Repository layout

```
matlab/            cleaned MATLAB version (F1.m cycle model, goal.m fitness, run_pso.m optimiser + animation)
matlab/original/   the code as submitted in 2025, kept for honesty
python/            Python port: turbofan_cycle.py, pso.py, optimise.py
figures/           outputs of optimise.py
```

Run the Python version:

```bash
cd python
pip install numpy matplotlib
python optimise.py
```

## Bugs found while cleaning the 2025 code

Going back to the original files for this repo, I found and fixed:

1. `F1.m` — in the bypass nozzle the static pressure `P19` was recomputed *after* the choke test,
   with a wrong exponent, so the "fully expanded" branch never took effect.
2. `plotPSO.m` — the OPR column was updated with the outer loop index (`particles(i,5)`) instead of the
   particle index `j`, so most particles kept a stale OPR.
3. `test.m` — the bound-repair `while` used a vector condition, which MATLAB evaluates with `all()`;
   it almost never triggered.
4. `PSO.m` — particles violating the OPR constraint were re-sampled uniformly every iteration,
   destroying the swarm memory. Replaced by a geometric rescale of π_LPC and π_HPC onto the bound.

The Python port gives the same TSFC as the 2025 report (18.1 mg/(N·s)) and a specific thrust of 108 N·s/kg
against 120 in the report; the gap is consistent with bug 1 (the original always used the choked-nozzle
pressure). Rerunning the fixed MATLAB to confirm is on the to-do list.

## What I would do next

Off-design operation (the model is on-design only), a turbine-inlet-temperature and fan-diameter
constraint, and a proper multi-objective front (NSGA-II) instead of a weighted scalar objective.
