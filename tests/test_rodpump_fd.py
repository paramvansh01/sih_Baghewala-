import numpy as np
import pytest

from twin.config import FieldConfig
from twin.physics.rodpump_fd import PumpCondition, simulate_card
from twin.physics.srp_analytic import SRPDesign, operating_point
from twin.units import G

CFG = FieldConfig()
K, MP = CFG.value("oil.bulk_modulus"), CFG.value("srp.plunger_mass")


def _run(d, spm, ratio, mu, cond=PumpCondition(), rho=950.0, p_i=0.5e6, p_wh=1e6):
    p_d = p_wh + rho * G * d.pump_depth
    return simulate_card(d, spm, ratio, np.full(41, mu), rho, p_i, p_d, cond, K, MP, n_out=256)


def test_quasi_static_loads_match_analytic():
    d = SRPDesign.from_config(CFG, 1150.0, 3.0)
    o = _run(d, 0.2, 1.0, 1e-5)
    w_b = o["w_buoyant"]
    assert o["load"].max() == pytest.approx(w_b + o["fluid_load"] + d.f_coulomb, rel=0.02)
    assert o["min_load"] == pytest.approx(w_b - d.f_coulomb, rel=0.03)


def test_unseated_pump_carries_no_fluid_load():
    d = SRPDesign.from_config(CFG, 1150.0, 3.0)
    o = _run(d, 3.0, 1.0, 0.01, PumpCondition(unseated=True))
    assert o["load"].max() < o["w_buoyant"] + 0.2 * o["fluid_load"]
    assert o["load"].min() > o["w_buoyant"] - 0.2 * o["fluid_load"]


def test_slower_downstroke_reduces_floating():
    d = SRPDesign.from_config(CFG, 1150.0, 3.0)
    fast = _run(d, 3.0, 1.0, 10.0)
    slow = _run(d, 3.0, 3.0, 10.0)
    assert fast["min_load"] < 0 < fast["load"].max()
    assert slow["min_load"] > fast["min_load"]


def test_analytic_floating_check_agrees_with_fd_model():
    """The daily simulator uses the rigid-rod analytic floating check; measure its agreement
    with the finite-difference model over random conditions (reported, and must be >= 85%)."""
    rng = np.random.default_rng(0)
    agree, n = 0, 120
    for _ in range(n):
        d = SRPDesign.from_config(CFG, 1150.0, rng.uniform(2.5, 4.3))
        mu = float(np.exp(rng.uniform(np.log(0.05), np.log(30))))
        spm, ratio = rng.uniform(1, 8), rng.uniform(1, 3)
        o = _run(d, spm, ratio, mu)
        an = operating_point(d, np.full(40, mu), d.pump_depth / 40, mu, 950.0, 0.5e6, 1e6, spm, ratio, 1.0)
        agree += bool(an["floating"]) == bool(o["min_load"] < 0)
    assert agree / n >= 0.85, agree / n
