"""On-design thermodynamic cycle of a two-spool, separate-flow turbofan.

Port of the MATLAB model F1.m written for my final-year project (HIAST, 2025).
Station numbering: 0 freestream, 2 fan face, 13 fan exit (bypass), 19 bypass
nozzle exit, 25 LPC exit, 3 HPC exit, 4 burner exit / HPT inlet, 45 HPT exit,
5 LPT exit, 9 core nozzle exit.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class FlightCondition:
    M0: float = 0.85
    P0: float = 22632.0   # Pa (11 km ISA)
    T0: float = 216.65    # K

@dataclass
class ComponentEfficiencies:
    pi_d: float = 0.98
    eta_f: float = 0.89
    eta_LPC: float = 0.88
    eta_HPC: float = 0.86
    pi_b: float = 0.96
    eta_b: float = 0.99
    eta_HPT: float = 0.91
    eta_LPT: float = 0.92
    eta_mH: float = 0.993
    eta_mL: float = 0.997
    pi_np: float = 0.99
    pi_ns: float = 0.99
    Tt4: float = 1450.0   # turbine inlet temperature, K
    h: float = 43e6       # fuel LHV, J/kg
    gamma_c: float = 1.4
    gamma_t: float = 1.3
    R: float = 287.0

def _nozzle(Pt, Tt, P0, gamma, R):
    """Convergent nozzle: returns exit velocity U, static T, static P, Mach."""
    pr_crit = ((gamma + 1) / 2) ** (gamma / (gamma - 1))
    if Pt / P0 >= pr_crit:            # choked
        M = 1.0
        P = Pt / pr_crit
    else:                              # fully expanded
        M = np.sqrt(2 / (gamma - 1) * ((Pt / P0) ** ((gamma - 1) / gamma) - 1))
        P = P0
    T = Tt / (1 + (gamma - 1) / 2 * M**2)
    U = M * np.sqrt(gamma * R * T)
    return U, T, P, M

def turbofan_cycle(alpha, pi_f, pi_LPC, pi_HPC, fc=FlightCondition(), ce=ComponentEfficiencies()):
    """Return (TSFC [mg/(N·s)], specific thrust [N·s/kg of total air], fuel-air ratio f)."""
    gc, gt, R = ce.gamma_c, ce.gamma_t, ce.R
    Cp_c = gc * R / (gc - 1)
    Cp_t = gt * R / (gt - 1)

    U0 = fc.M0 * np.sqrt(gc * R * fc.T0)
    tau_r = 1 + (gc - 1) / 2 * fc.M0**2
    pi_r = tau_r ** (gc / (gc - 1))
    Tt2 = fc.T0 * tau_r
    Pt2 = fc.P0 * pi_r * ce.pi_d

    # fan + bypass nozzle
    tau_f = 1 + (pi_f ** ((gc - 1) / gc) - 1) / ce.eta_f
    Tt13 = tau_f * Tt2
    Pt19 = Pt2 * pi_f * ce.pi_ns
    U19, T19, P19, _ = _nozzle(Pt19, Tt13, fc.P0, gc, R)
    F_bypass = alpha * ((U19 - U0) + (R * T19 / U19) * (1 - fc.P0 / P19))

    # core compressors
    tau_LPC = 1 + (pi_LPC ** ((gc - 1) / gc) - 1) / ce.eta_LPC
    tau_HPC = 1 + (pi_HPC ** ((gc - 1) / gc) - 1) / ce.eta_HPC
    Tt3 = Tt2 * tau_LPC * tau_HPC
    Pt3 = Pt2 * pi_LPC * pi_HPC

    # burner
    f = (Cp_t * ce.Tt4 - Cp_c * Tt3) / (ce.eta_b * ce.h - Cp_t * ce.Tt4)
    Pt4 = Pt3 * ce.pi_b

    # HPT drives HPC; LPT drives LPC + fan (work balance)
    tau_HPT = 1 - Cp_c * Tt2 * tau_LPC * (tau_HPC - 1) / (ce.eta_mH * (1 + f) * Cp_t * ce.Tt4)
    pi_HPT = (1 - (1 - tau_HPT) / ce.eta_HPT) ** (gt / (gt - 1))
    tau_LPT = 1 - (Cp_c * Tt2 * (tau_LPC - 1) + alpha * Cp_c * Tt2 * (tau_f - 1)) / (ce.eta_mL * (1 + f) * Cp_t * ce.Tt4 * tau_HPT)
    pi_LPT = (1 - (1 - tau_LPT) / ce.eta_LPT) ** (gt / (gt - 1))
    Tt5 = ce.Tt4 * tau_HPT * tau_LPT
    Pt9 = Pt4 * pi_HPT * pi_LPT * ce.pi_np

    if tau_HPT <= 0 or tau_LPT <= 0 or Pt9 <= fc.P0 or f <= 0:
        return np.nan, np.nan, np.nan   # physically infeasible cycle

    U9, T9, P9, _ = _nozzle(Pt9, Tt5, fc.P0, gt, R)
    F_core = (1 + f) * U9 - U0 + (1 + f) * (R * T9 / U9) * (1 - fc.P0 / P9)

    F_sp = (F_core + F_bypass) / (1 + alpha)          # N·s per kg of total air
    TSFC = f / (1 + alpha) / F_sp * 1e6               # mg/(N·s)
    return TSFC, F_sp, f

if __name__ == "__main__":
    for a in (5, 8, 11):
        tsfc, fsp, f = turbofan_cycle(a, 1.6, 3.0, 12.0)
        print(f"BPR={a:2d}  TSFC={tsfc:6.2f} mg/N·s  Fsp={fsp:6.1f} N·s/kg  f={f:.4f}")
