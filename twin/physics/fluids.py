"""Oil PVT: density vs temperature and ASTM D341 (Walther) viscosity-temperature law.

ASTM D341:  log10(log10(nu + 0.7)) = A - B * log10(T[K]),   nu in cSt (valid nu >= 2 cSt)
A is anchored so the curve passes exactly through the published viscosity point
(10,000-13,000 cP at 50 C for Baghewala); B (slope) is an assumption until lab
viscosity-temperature data are supplied.
"""
import numpy as np

from ..units import to_kelvin

WATER_DENSITY_60F = 999.016  # kg/m3, water at 60 F (API gravity reference)
T_60F_C = 15.5556


class OilPVT:
    def __init__(self, api, mu_ref, t_ref_c, walther_b, beta, cp):
        self.api = api
        self.rho_sc = 141.5 / (131.5 + api) * WATER_DENSITY_60F
        self.beta = beta
        self.cp = cp
        self.walther_b = walther_b
        rho_ref = self.density(t_ref_c)
        nu_ref = mu_ref / rho_ref * 1e6  # m2/s -> cSt
        self.walther_a = np.log10(np.log10(nu_ref + 0.7)) + walther_b * np.log10(to_kelvin(t_ref_c))

    def density(self, t_c):
        return self.rho_sc * (1.0 - self.beta * (np.asarray(t_c) - T_60F_C))

    def kinematic_cst(self, t_c):
        tk = to_kelvin(np.asarray(t_c, dtype=float))
        return 10.0 ** (10.0 ** (self.walther_a - self.walther_b * np.log10(tk))) - 0.7

    def viscosity(self, t_c):
        """Dynamic viscosity, Pa.s"""
        return self.kinematic_cst(t_c) * 1e-6 * self.density(t_c)

    @classmethod
    def from_config(cls, cfg, api, mu_ref, walther_b):
        return cls(api=api, mu_ref=mu_ref, t_ref_c=cfg.value("oil.viscosity_ref_temperature"),
                   walther_b=walther_b, beta=cfg.value("oil.thermal_expansion"),
                   cp=cfg.value("oil.specific_heat"))
