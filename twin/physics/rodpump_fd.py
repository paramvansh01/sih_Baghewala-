"""Finite-difference (lumped-mass) sucker-rod string + pump-chamber model -> dynamometer cards.

Rod string: N elements, node masses rho_r A_r dx, axial springs E A_r / dx (the discrete
form of the damped wave equation, Gibbs 1963). Distributed forces per node: buoyant
weight, viscous drag c(z) v with c = 2 pi mu / ln(D_t/D_r) per unit length (same law as
srp_analytic.py). Top node follows the prescribed polished-rod motion (harmonic half
strokes, independent up/down durations). Integration: symplectic Euler for the elastic
forces (time step limited by the CFL condition and the stiffest pump-chamber fluid spring);
viscous drag is integrated implicitly (v <- v / (1 + dt c / m), unconditionally stable for
the very large drag of 10^4 cP oil) and Coulomb plunger friction with a stick/slip update.

Pump chamber (at the bottom node): contents = liquid (bulk modulus K) + ideal gas at
constant temperature (p V_g = C). Pressure from the volume constraint
    V_l (1 - (p - p_i)/K) + C/p = V_c      (quadratic in p)
with a vapour-pressure floor (void -> fluid pound). Valves are ideal check valves:
standing valve opens when p < p_intake (inflow: liquid limited by the per-stroke liquid
budget `fill`, remainder gas at intake pressure; free-gas fraction `gvf`), travelling valve
opens when p > p_discharge (homogeneous outflow). Leaks: laminar, proportional to pressure
difference, scaled so `leak_frac` = fraction of the stroke volume passed per upstroke at the
full pressure difference. Unseated pump: no fluid load is transferred to the rods.
Force on the rods at the pump: A_p (p_discharge - p_chamber) + Coulomb + viscous plunger drag.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from ..units import G


@dataclass
class PumpCondition:
    fill: float = 1.0  # liquid available per stroke / pump displacement
    gvf: float = 0.0  # free-gas volume fraction of intake flow
    tv_leak: float = 0.0
    sv_leak: float = 0.0
    unseated: bool = False


@njit(cache=True)
def _chamber_pressure(v_l, c_g, v_c, p_i, k_bulk, p_vap):
    a = v_l / k_bulk
    b = v_c - v_l * (1.0 + p_i / k_bulk)
    if a <= 0.0:
        p = c_g / v_c if v_c > 0 else p_vap
    elif b > 0.0:
        p = 2.0 * c_g / (b + math.sqrt(b * b + 4.0 * a * c_g)) if c_g > 0 else 0.0
    else:
        p = (-b + math.sqrt(b * b + 4.0 * a * c_g)) / (2.0 * a)
    return max(p, p_vap)


@njit(cache=True)
def _simulate(n, dx, a_r, rho_r, e_r, rho_f, c_node, a_p, c_pl, f_c, m_pl,
              stroke, t_up, t_dn, n_strokes, dt, p_i, p_d, k_bulk, p_vap, v_dead,
              fill, gvf, k_tv, k_sv, unseated, n_out):
    k_s = e_r * a_r / dx
    m = np.full(n + 1, rho_r * a_r * dx)
    m[0] *= 0.5
    m[n] = 0.5 * m[n] + m_pl
    w = np.full(n + 1, (rho_r - rho_f) * G * a_r * dx)
    w[0] *= 0.5
    w[n] *= 0.5
    # static equilibrium hanging from the polished rod at y=0, fluid load on standing valve
    u = np.zeros(n + 1)
    for i in range(1, n + 1):
        below = 0.0
        for j in range(i, n + 1):
            below += w[j]
        u[i] = u[i - 1] - below / k_s
    v = np.zeros(n + 1)
    # pump spaced for the loaded rod string: bottom-of-stroke reference is the static
    # position with the full fluid load on the plunger (extra stretch F_o L / (E A_r))
    u_ref = u[n] - a_p * (p_d - p_i) * (n * dx) / (e_r * a_r)
    v_l = v_dead / (1.0 - (p_d - p_i) / k_bulk)  # liquid-full dead volume at p_d
    c_g = 0.0
    p = p_d
    period = t_up + t_dn
    n_steps = int(math.ceil(n_strokes * period / dt))
    dt = n_strokes * period / n_steps
    steps_per = n_steps // n_strokes
    last_start = n_steps - steps_per
    out_t = np.empty(n_out)
    out_y = np.empty(n_out)
    out_prl = np.empty(n_out)
    out_up = np.empty(n_out)
    out_fp = np.empty(n_out)
    out_p = np.empty(n_out)
    next_out = 0
    liq_budget = fill * a_p * stroke
    liq_left = liq_budget
    in_up_prev = False
    min_prl = 1e300
    f_fric = 0.0
    for step in range(n_steps):
        t = (step + 1) * dt
        tc = t % period
        if tc < t_up:
            y = 0.5 * stroke * (1.0 - math.cos(math.pi * tc / t_up))
            vy = 0.5 * stroke * math.pi / t_up * math.sin(math.pi * tc / t_up)
            ay = 0.5 * stroke * (math.pi / t_up) ** 2 * math.cos(math.pi * tc / t_up)
            in_up = True
        else:
            td = tc - t_up
            y = 0.5 * stroke * (1.0 + math.cos(math.pi * td / t_dn))
            vy = -0.5 * stroke * math.pi / t_dn * math.sin(math.pi * td / t_dn)
            ay = -0.5 * stroke * (math.pi / t_dn) ** 2 * math.cos(math.pi * td / t_dn)
            in_up = False
        if in_up and not in_up_prev:
            liq_left = liq_budget  # new stroke: annulus liquid available again
        in_up_prev = in_up
        # pump fluid force on bottom node (downward positive)
        f_fluid = 0.0 if unseated else a_p * (p_d - p)
        # velocity update: elastic + gravity + fluid (explicit), drag (implicit), friction
        for i in range(1, n + 1):
            t_above = k_s * (u[i - 1] - u[i])
            t_below = k_s * (u[i] - u[i + 1]) if i < n else 0.0
            f = t_above - t_below - w[i]
            c_i = c_node[i]
            if i == n:
                f -= f_fluid
                c_i += c_pl
            vi = (v[i] + dt * f / m[i]) / (1.0 + dt * c_i / m[i])
            if i == n and f_c > 0.0:
                dv = f_c * dt / m[i]
                if abs(vi) <= dv:
                    fr = vi * m[i] / dt  # static friction holds the plunger
                    vi = 0.0
                else:
                    fr = f_c if vi > 0 else -f_c
                    vi -= dv if vi > 0 else -dv
                f_fric = fr
            v[i] = vi
        for i in range(1, n + 1):
            u[i] += dt * v[i]
        u[0] = y
        v[0] = vy
        # chamber
        v_c = max(v_dead + a_p * (u[n] - u_ref), 0.05 * v_dead)
        if not unseated:
            p = _chamber_pressure(v_l, c_g, v_c, p_i, k_bulk, p_vap)
            if p < p_d and k_tv > 0.0:
                v_l += k_tv * (p_d - p) * dt
            if p > p_i and k_sv > 0.0:
                v_l = max(v_l - k_sv * (p - p_i) * dt, 0.0)
            p = _chamber_pressure(v_l, c_g, v_c, p_i, k_bulk, p_vap)
            if p < p_i:
                # standing valve opens: fill to p_i
                need = v_c - v_l * (1.0 - 0.0 / k_bulk) - c_g / p_i
                if need > 0.0:
                    liq = min(need * (1.0 - gvf), liq_left)
                    liq_left -= liq
                    v_l += liq
                    c_g += p_i * (need - liq)
                p = p_i
            elif p > p_d:
                denom = v_l * (1.0 - (p_d - p_i) / k_bulk) + c_g / p_d
                r = v_c / denom
                v_l *= r
                c_g *= r
                p = p_d
        else:
            p = p_i
        prl = k_s * (u[0] - u[1]) + w[0] + m[0] * ay + c_node[0] * vy
        if step >= last_start:
            if prl < min_prl:
                min_prl = prl
            k_out = (step - last_start) * n_out // steps_per
            if k_out >= next_out and next_out < n_out:
                out_t[next_out] = tc
                out_y[next_out] = y
                out_prl[next_out] = prl
                out_up[next_out] = u[n] - u_ref
                out_fp[next_out] = f_fluid + f_fric + c_pl * v[n]
                out_p[next_out] = p
                next_out += 1
    return out_t, out_y, out_prl, out_up, out_fp, out_p, min_prl, dt


def simulate_card(design, spm, ratio, mu_nodes, rho_fluid, p_intake, p_discharge, cond: PumpCondition,
                  k_bulk, plunger_mass, n_strokes=4, n_out=128, cfl=0.8, p_vap=5e3):
    """Simulate one steady pumping condition; returns dict with the last stroke's card.

    mu_nodes: viscosity (Pa.s) at the n+1 rod nodes from surface (index 0) to pump (index n).
    """
    n = len(mu_nodes) - 1
    L = design.pump_depth
    dx = L / n
    a_r = design.a_rod
    c_per_len = 2.0 * np.pi * np.asarray(mu_nodes, float) / np.log(design.d_tubing / design.d_rod)
    c_node = c_per_len * dx
    c_node[0] *= 0.5
    c_node[-1] *= 0.5
    mu_pump = float(mu_nodes[-1])
    c_pl = np.pi * design.d_plunger * design.l_plunger * mu_pump / design.clearance
    period = 60.0 / spm
    t_up = period / (1.0 + ratio)
    t_dn = period - t_up
    a_p = design.a_plunger
    v_dead = a_p * design.dead_length
    # time step: CFL and stiffest fluid spring at the bottom node
    wave = math.sqrt(design.e_rod / design.rho_rod)
    dt_cfl = cfl * dx / wave
    k_s = design.e_rod * a_r / dx
    m_n = 0.5 * design.rho_rod * a_r * dx + plunger_mass
    k_fluid = k_bulk * a_p**2 / v_dead
    dt_fluid = 0.5 * 2.0 / math.sqrt((k_s + k_fluid) / m_n)
    dt = min(dt_cfl, dt_fluid)
    dp = p_discharge - p_intake
    stroke_vol = a_p * design.stroke
    k_tv = cond.tv_leak * stroke_vol / (dp * t_up) if cond.tv_leak > 0 else 0.0
    k_sv = cond.sv_leak * stroke_vol / (dp * t_up) if cond.sv_leak > 0 else 0.0
    out = _simulate(n, dx, a_r, design.rho_rod, design.e_rod, rho_fluid, c_node, a_p, c_pl,
                    design.f_coulomb, plunger_mass, design.stroke, t_up, t_dn, n_strokes, dt,
                    p_intake, p_discharge, k_bulk, p_vap, v_dead, cond.fill, cond.gvf, k_tv, k_sv,
                    bool(cond.unseated), n_out)
    t, y, prl, up, fp, pc, min_prl, dt_used = out
    return dict(t=t, position=y, load=prl, plunger=up, pump_force=fp, chamber_p=pc,
                min_load=min_prl, dt=dt_used,
                w_air=design.rho_rod * G * a_r * L,
                w_buoyant=(design.rho_rod - rho_fluid) * G * a_r * L,
                fluid_load=a_p * dp)
