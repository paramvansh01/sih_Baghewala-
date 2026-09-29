import numpy as np
import pytest

from twin.physics import steam as stm
from twin.physics.fluids import OilPVT


def test_iapws_saturation_reference_values():
    # IAPWS-IF97 / standard steam tables
    assert stm.saturation_temperature(1.0e6) == pytest.approx(179.88, abs=0.02)
    assert stm.saturation_temperature(0.101325e6) == pytest.approx(99.97, abs=0.02)
    assert stm.latent_heat(1.0e6) / 1e3 == pytest.approx(2014.6, abs=1.0)
    assert stm.h_liquid_sat(1.0e6) / 1e3 == pytest.approx(762.7, abs=0.5)


def test_walther_passes_through_published_anchor_and_decreases():
    oil = OilPVT(api=18, mu_ref=11.5, t_ref_c=50.0, walther_b=3.5, beta=7e-4, cp=2000)
    assert oil.viscosity(50.0) == pytest.approx(11.5, rel=1e-9)
    t = np.linspace(30, 300, 50)
    mu = oil.viscosity(t)
    assert np.all(np.diff(mu) < 0)


def test_api_density():
    oil = OilPVT(api=10.0, mu_ref=1.0, t_ref_c=50.0, walther_b=3.5, beta=0.0, cp=2000)
    # API 10 has specific gravity 1.0 by definition
    assert oil.density(15.5556) == pytest.approx(999.016, rel=1e-9)


def test_water_props_table():
    w = stm.WaterProps()
    assert float(w.density(25.0)) == pytest.approx(997.0, abs=1.0)
    assert float(w.viscosity(25.0)) == pytest.approx(0.00089, rel=0.02)
