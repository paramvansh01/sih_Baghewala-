import numpy as np
import pytest

from twin.config import FieldConfig
from twin.physics.srp_analytic import SRPDesign, kinematics, operating_point
from twin.units import G


@pytest.fixture
def design():
    return SRPDesign.from_config(FieldConfig(), 1150.0, 3.0)


def test_kinematics_means_match_numerical_average():
    S, spm, r = 3.0, 4.0, 2.0
    k = kinematics(S, spm, r)
    t = np.linspace(0, k["period"], 200001)
    t_up = k["t_up"]
    y = np.where(t <= t_up, S / 2 * (1 - np.cos(np.pi * t / t_up)),
                 S / 2 * (1 + np.cos(np.pi * (t - t_up) / k["t_dn"])))
    v = np.gradient(y, t)
    assert np.mean(v**2) == pytest.approx(float(k["mean_v2"]), rel=1e-3)
    assert np.mean(np.abs(v)) == pytest.approx(float(k["mean_abs_v"]), rel=1e-3)
    assert v.max() == pytest.approx(float(k["v_up"]), rel=1e-3)


def test_static_limit_loads(design):
    rho_f, p_in, p_wh = 900.0, 0.5e6, 1.0e6
    mu = np.full(40, 1e-6)  # ~no drag
    op = operating_point(design, mu, design.pump_depth / 40, 1e-6, rho_f, p_in, p_wh,
                         spm=0.01, ratio=1.0, surface_efficiency=1.0)
    w_b = (design.rho_rod - rho_f) * G * design.a_rod * design.pump_depth
    f_o = design.a_plunger * (p_wh + rho_f * G * design.pump_depth - p_in)
    assert float(op["prl_max"]) == pytest.approx(w_b + f_o + design.f_coulomb, rel=1e-4)
    assert float(op["prl_min"]) == pytest.approx(w_b - design.f_coulomb, rel=1e-4)
    # plunger stroke = stroke - static fluid-load stretch
    assert op["plunger_stroke"] == pytest.approx(
        design.stroke - f_o * design.pump_depth / (design.e_rod * design.a_rod))


def test_floating_appears_with_viscosity_and_speed(design):
    args = dict(rho_fluid=950.0, p_intake=0.5e6, p_wellhead=1e6, surface_efficiency=0.6)
    dz = design.pump_depth / 40
    low = operating_point(design, np.full(40, 0.01), dz, 0.01, spm=4.0, ratio=1.0, **args)
    high = operating_point(design, np.full(40, 20.0), dz, 20.0, spm=4.0, ratio=1.0, **args)
    slow_dn = operating_point(design, np.full(40, 20.0), dz, 20.0, spm=1.0, ratio=3.0, **args)
    assert not bool(low["floating"]) and bool(high["floating"])
    assert float(slow_dn["prl_min"]) > float(high["prl_min"])


def test_goodman_definition(design):
    op = operating_point(design, np.full(40, 0.5), design.pump_depth / 40, 0.5, 950.0, 0.5e6, 1e6,
                         spm=3.0, ratio=1.0, surface_efficiency=0.6)
    s_max = float(op["prl_max"]) / design.a_rod
    s_min = max(float(op["prl_min"]), 0) / design.a_rod
    s_a = (design.tensile / 4 + 0.5625 * s_min) * design.service_factor
    assert float(op["goodman"]) == pytest.approx((s_max - s_min) / (s_a - s_min))
