"""Fast analytic sucker-rod-pump model used inside the daily reservoir loop.

Assumptions (all stated, all testable against the finite-difference model in
rodpump_fd.py):
  * Polished-rod motion is harmonic within each half-stroke; a VFD/hydraulic unit can
    make the downstroke last `ratio` times longer than the upstroke.
  * Rod string treated as rigid for loads (inertia + viscous drag + Coulomb friction);
    elastic stretch only used for plunger-stroke loss (static fluid-load stretch).
  * Viscous rod drag per unit length for a rod moving in stationary Newtonian fluid in
    concentric tubing (annular Couette flow):  c = 2 pi mu / ln(D_t / D_r)   [N s/m^2]
    (the flow of produced fluid is neglected - a simplification of Lea, 1991).
  * Plunger viscous drag (Couette in the plunger/barrel clearance delta):
        c_p = pi D_p L_p mu / delta
  * Plunger slippage (Poiseuille through a narrow concentric annulus):
        q_slip = pi D_p delta^3 dp / (12 mu L_p)
  * Rod floating: the minimum polished-rod load on the downstroke drops below zero,
    i.e. the rods cannot fall as fast as the carrier bar (rigid-rod estimate):
        PRL_min = W_b - F_c - sqrt((C v_d)^2 + (m a_d)^2)
    where v_d = pi S / (2 t_d), a_d = (S/2)(pi/t_d)^2 (harmonic motion amplitudes).
  * Modified Goodman (API RP 11BR):  S_a = (T/4 + 0.5625 S_min) SF;
    loading = (S_max - S_min) / (S_a - S_min).
"""
from dataclasses import dataclass

import numpy as np

from ..units import G


@dataclass
class SRPDesign:
    pump_depth: float
    stroke: float
    d_plunger: float
    l_plunger: float
    clearance: float  # radial clearance delta, m
    f_coulomb: float
    dead_length: float
    d_rod: float
    rho_rod: float
    e_rod: float
    tensile: float
    service_factor: float
    d_tubing: float

    @property
    def a_plunger(self):
        return np.pi / 4 * self.d_plunger**2

    @property
    def a_rod(self):
        return np.pi / 4 * self.d_rod**2

    @property
    def rod_mass(self):
        return self.rho_rod * self.a_rod * self.pump_depth

    @property
    def rod_weight_air(self):
        return self.rod_mass * G

    @classmethod
    def from_config(cls, cfg, reservoir_depth, stroke):
        return cls(
            pump_depth=reservoir_depth - cfg.value("srp.pump_setting_offset"),
            stroke=stroke,
            d_plunger=cfg.value("srp.plunger_diameter"),
            l_plunger=cfg.value("srp.plunger_length"),
            clearance=cfg.value("srp.plunger_clearance") / 2.0,
            f_coulomb=cfg.value("srp.pump_coulomb_friction"),
            dead_length=cfg.value("srp.pump_dead_volume_length"),
            d_rod=cfg.value("srp.rod_diameter"),
            rho_rod=cfg.value("srp.rod_density"),
            e_rod=cfg.value("srp.rod_youngs_modulus"),
            tensile=cfg.value("srp.rod_tensile_strength"),
            service_factor=cfg.value("srp.goodman_service_factor"),
            d_tubing=cfg.value("wellbore.tubing_inner_diameter"),
        )


def rod_drag_coefficient(mu, d_tubing, d_rod):
    """N s / m^2 (force per unit length per unit velocity)."""
    return 2.0 * np.pi * mu / np.log(d_tubing / d_rod)


def plunger_drag_coefficient(mu, d_p, l_p, delta):
    return np.pi * d_p * l_p * mu / delta


def kinematics(stroke, spm, ratio):
    """Harmonic half-strokes. Returns dict of arrays broadcast over spm, ratio."""
    spm = np.asarray(spm, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    period = 60.0 / spm
    t_up = period / (1.0 + ratio)
    t_dn = period * ratio / (1.0 + ratio)
    half = stroke / 2.0
    v_up, v_dn = half * np.pi / t_up, half * np.pi / t_dn
    a_up, a_dn = half * (np.pi / t_up) ** 2, half * (np.pi / t_dn) ** 2
    # time-mean of v^2 : each harmonic half has mean v^2 = V^2/2
    mean_v2 = (t_up * v_up**2 / 2 + t_dn * v_dn**2 / 2) / period
    mean_abs_v = 2.0 * stroke / period
    return dict(period=period, t_up=t_up, t_dn=t_dn, v_up=v_up, v_dn=v_dn,
                a_up=a_up, a_dn=a_dn, mean_v2=mean_v2, mean_abs_v=mean_abs_v)


def operating_point(d: SRPDesign, mu_profile, dz, mu_pump, rho_fluid, p_intake, p_wellhead,
                    spm, ratio, surface_efficiency):
    """Evaluate SRP performance for (spm, ratio) - arrays broadcast.

    mu_profile: viscosity at tubing nodes (Pa.s), dz: node spacing (m).
    Returns dict of arrays: capacity (m3/s), prl_max, prl_min (N), floating (bool),
    goodman (fraction), friction_power (W), and fixed quantities.
    """
    k = kinematics(d.stroke, spm, ratio)
    L = d.pump_depth
    p_discharge = p_wellhead + rho_fluid * G * L
    dp = max(p_discharge - p_intake, 0.0)
    f_o = d.a_plunger * dp  # fluid load on plunger (upstroke)
    c_rod = float(np.sum(rod_drag_coefficient(np.asarray(mu_profile), d.d_tubing, d.d_rod)) * dz)
    c_pl = plunger_drag_coefficient(mu_pump, d.d_plunger, d.l_plunger, d.clearance)
    c_tot = c_rod + c_pl
    w_b = (d.rho_rod - rho_fluid) * G * d.a_rod * L
    m = d.rod_mass

    prl_max = w_b + f_o + d.f_coulomb + np.sqrt((c_tot * k["v_up"]) ** 2 + (m * k["a_up"]) ** 2)
    prl_min = w_b - d.f_coulomb - np.sqrt((c_tot * k["v_dn"]) ** 2 + (m * k["a_dn"]) ** 2)

    s_max = prl_max / d.a_rod
    s_min = np.maximum(prl_min, 0.0) / d.a_rod
    s_allow = (d.tensile / 4.0 + 0.5625 * s_min) * d.service_factor
    goodman = (s_max - s_min) / np.maximum(s_allow - s_min, 1.0)

    stretch = f_o * L / (d.e_rod * d.a_rod)
    s_plunger = max(d.stroke - stretch, 0.0)
    gross = d.a_plunger * s_plunger * np.asarray(spm) / 60.0
    q_slip = np.pi * d.d_plunger * d.clearance**3 * dp / (12.0 * mu_pump * d.l_plunger)
    capacity = np.maximum(gross - q_slip * k["t_up"] / k["period"], 0.0)

    friction_power = c_tot * k["mean_v2"] + d.f_coulomb * k["mean_abs_v"]
    return dict(capacity=capacity, prl_max=prl_max, prl_min=prl_min, floating=prl_min < 0.0,
                goodman=goodman, friction_power=friction_power, p_discharge=p_discharge,
                fluid_load=f_o, buoyant_weight=w_b, c_rod=c_rod, c_plunger=c_pl,
                plunger_stroke=s_plunger, surface_efficiency=surface_efficiency)


def motor_power(op, q_liquid, p_intake):
    """Electrical power (W): hydraulic + friction, divided by surface efficiency."""
    hydraulic = q_liquid * max(op["p_discharge"] - p_intake, 0.0)
    return (hydraulic + op["friction_power"]) / op["surface_efficiency"]
