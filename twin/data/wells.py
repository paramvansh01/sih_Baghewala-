"""Sample a synthetic well population from the ranges in config/field.yaml."""
import numpy as np

from ..physics.reservoir import Well


def sample_well(cfg, rng: np.random.Generator, well_id: str) -> Well:
    s = lambda k: cfg.sample(k, rng)  # noqa: E731
    return Well(well_id=well_id, k=s("reservoir.permeability"), h=s("reservoir.net_pay"),
                phi=s("reservoir.porosity"), s_oi=s("reservoir.initial_oil_saturation"),
                r_e=s("reservoir.drainage_radius"), skin=s("reservoir.skin_cold"),
                p_i=s("reservoir.initial_pressure"), t_r=s("reservoir.temperature"),
                api=s("oil.api_gravity"), mu_ref=s("oil.viscosity_at_ref"),
                walther_b=s("oil.walther_slope_B"), stroke=s("srp.stroke_length"))


def central_well(cfg, well_id="BGW-central") -> Well:
    m = cfg.mid
    return Well(well_id=well_id, k=m("reservoir.permeability"), h=m("reservoir.net_pay"),
                phi=m("reservoir.porosity"), s_oi=m("reservoir.initial_oil_saturation"),
                r_e=m("reservoir.drainage_radius"), skin=m("reservoir.skin_cold"),
                p_i=m("reservoir.initial_pressure"), t_r=m("reservoir.temperature"),
                api=m("oil.api_gravity"), mu_ref=m("oil.viscosity_at_ref"),
                walther_b=m("oil.walther_slope_B"), stroke=m("srp.stroke_length"))
