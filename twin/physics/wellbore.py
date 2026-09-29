"""Wellbore heat transfer (Ramey 1962).

Transient formation heat-conduction function, Ramey's long-time approximation
(valid for t > ~1 week):   f(t) = ln(2 sqrt(alpha_e t) / r_wb) - 0.29

Thermal resistance per unit length between fluid and undisturbed earth:
    R' = 1 / (2 pi r_to U_to) + f(t) / (2 pi k_e)          [K m / W]

Steam injection: heat lost over depth L at constant steam temperature Ts
    Q_loss = integral_0^L (Ts - Te(z)) / R' dz
Production: fluid temperature rising through the tubing (z measured upward from the
pump, geothermal gradient g, relaxation distance A = m_dot c_p R'):
    Tf(y) = Te(y) + g A + (Tf0 - Te0 - g A) exp(-y / A)
    (exact solution of m c dTf/dy = -(Tf - Te)/R', Te(y) = Te0 - g y).
"""
import numpy as np


def ramey_f(t, alpha_e, r_wb):
    t = np.maximum(t, 7 * 86400.0)  # long-time approximation floor (1 week)
    return np.log(2.0 * np.sqrt(alpha_e * t) / r_wb) - 0.29


def resistance_per_length(t, r_to, u_to, k_e, alpha_e, r_wb):
    return 1.0 / (2 * np.pi * r_to * u_to) + ramey_f(t, alpha_e, r_wb) / (2 * np.pi * k_e)


def earth_temperature(z, t_surface, grad):
    return t_surface + grad * z


def steam_heat_loss_rate(t_steam, depth, t_surface, grad, r_prime):
    """W lost between surface and depth for steam at uniform t_steam (analytic integral)."""
    # integral_0^L (Ts - Tsurf - g z) dz = (Ts - Tsurf) L - g L^2 / 2
    return ((t_steam - t_surface) * depth - 0.5 * grad * depth**2) / r_prime


def production_temperature_profile(t_bottom, y, depth_bottom, t_surface, grad, m_dot_cp, r_prime):
    """Fluid temperature at heights y (m above the pump) for fluid entering at t_bottom."""
    y = np.asarray(y, dtype=float)
    te0 = earth_temperature(depth_bottom, t_surface, grad)
    if m_dot_cp <= 0:
        return earth_temperature(depth_bottom - y, t_surface, grad)
    a = m_dot_cp * r_prime
    te = te0 - grad * y
    return te + grad * a + (t_bottom - te0 - grad * a) * np.exp(-y / a)
