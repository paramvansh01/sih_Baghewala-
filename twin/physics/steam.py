"""Water/steam properties from IAPWS-IF97 (via the `iapws` package).

All inputs/outputs SI: pressure Pa, enthalpy J/kg, temperature degC, viscosity Pa.s.
"""
from functools import lru_cache

import numpy as np
from iapws import IAPWS97


def _mpa(p_pa):
    return round(float(p_pa) / 1e6, 6)


@lru_cache(maxsize=4096)
def _sat(p_mpa):
    liq = IAPWS97(P=p_mpa, x=0.0)
    vap = IAPWS97(P=p_mpa, x=1.0)
    return liq.T - 273.15, liq.h * 1e3, vap.h * 1e3, liq.rho


def saturation_temperature(p_pa):
    return _sat(_mpa(p_pa))[0]


def h_liquid_sat(p_pa):
    return _sat(_mpa(p_pa))[1]


def h_vapor_sat(p_pa):
    return _sat(_mpa(p_pa))[2]


def latent_heat(p_pa):
    _, hf, hg, _ = _sat(_mpa(p_pa))
    return hg - hf


@lru_cache(maxsize=4096)
def _h_liquid(t_c, p_mpa):
    return IAPWS97(P=p_mpa, T=t_c + 273.15).h * 1e3


def h_liquid(t_c, p_pa):
    """Enthalpy of compressed liquid water at (T, p). Requires T below saturation at p."""
    if t_c >= saturation_temperature(p_pa):
        raise ValueError("h_liquid called above saturation temperature")
    return _h_liquid(round(float(t_c), 3), _mpa(p_pa))


class WaterProps:
    """Saturated-liquid water viscosity, density, cp vs temperature, tabulated from IAPWS once."""

    def __init__(self, t_min=5.0, t_max=350.0, n=200):
        self.t = np.linspace(t_min, t_max, n)
        states = [IAPWS97(T=t + 273.15, x=0.0) for t in self.t]
        self.mu = np.array([w.mu for w in states])
        self.rho = np.array([w.rho for w in states])
        self._cp = np.array([w.cp * 1e3 for w in states])

    def viscosity(self, t_c):
        return np.interp(t_c, self.t, self.mu)

    def density(self, t_c):
        return np.interp(t_c, self.t, self.rho)

    def cp(self, t_c):
        return np.interp(t_c, self.t, self._cp)
