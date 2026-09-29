"""Heated-zone thermal models for cyclic steam stimulation.

Marx & Langenheim (1959): heated area of a layer of thickness h receiving a constant
heat rate Q while losing heat by conduction to semi-infinite cap- and base-rock:

    A(t)  = Q * alpha_c * M_R * h / (4 * lam_c**2 * dT) * G(tD)
    tD    = 4 * lam_c**2 * t / (alpha_c * M_R**2 * h**2)
    G(tD) = exp(tD) * erfc(sqrt(tD)) + 2*sqrt(tD/pi) - 1

(derivation: Laplace transform of the energy balance
 Q = M_R h dT dA/dt + 2 * integral lam dT / sqrt(pi alpha (t - tau)) dA(tau);
 verified numerically in tests/test_thermal.py).

After injection the heated zone is treated as a lumped volume whose excess energy
decays through conduction (cap/base rock + radial boundary, same 1/sqrt(pi alpha t)
semi-infinite kernel) and convection with produced fluids. This is a lumped
simplification in the spirit of Boberg & Lantz (1966), not a reproduction of it.
"""
import numpy as np
from scipy.special import erfcx


def ml_dimensionless_time(t, h, m_r, lam_c, alpha_c):
    return 4.0 * lam_c**2 * t / (alpha_c * m_r**2 * h**2)


def ml_G(t_d):
    t_d = np.asarray(t_d, dtype=float)
    s = np.sqrt(t_d)
    return erfcx(s) + 2.0 * s / np.sqrt(np.pi) - 1.0


def marx_langenheim_area(q_heat, t, h, m_r, lam_c, alpha_c, d_t):
    """Heated area (m2) after injecting heat rate q_heat (W) for time t (s)."""
    t_d = ml_dimensionless_time(t, h, m_r, lam_c, alpha_c)
    return q_heat * alpha_c * m_r * h / (4.0 * lam_c**2 * d_t) * ml_G(t_d)


def conduction_conductances(r_h, h, t_since_heating, lam_c, alpha_c, lam_r, alpha_r):
    """Conductances (W/K) of a cylindrical heated zone of radius r_h, thickness h:
    (top+bottom faces -> cap/base rock, lateral face -> adjacent cold reservoir),
    each with the semi-infinite-solid kernel lam / sqrt(pi alpha t).

    Heat through the top/bottom leaves the reservoir. Heat through the lateral face stays
    in the reservoir: the simulator uses it to grow the heated zone (energy conserving).
    """
    t = max(t_since_heating, 1.0)
    area_tb = 2.0 * np.pi * r_h**2
    area_side = 2.0 * np.pi * r_h * h
    return (area_tb * lam_c / np.sqrt(np.pi * alpha_c * t),
            area_side * lam_r / np.sqrt(np.pi * alpha_r * t))
