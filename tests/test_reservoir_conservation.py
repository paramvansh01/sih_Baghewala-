import copy

import numpy as np
import pytest

from twin.config import FieldConfig, load_pipeline
from twin.data.wells import central_well, sample_well
from twin.physics.reservoir import CSSWellSimulator, CycleControls
from twin.physics.steam import WaterProps

CFG, PL, WATER = FieldConfig(), load_pipeline(), WaterProps()


def _sim(well, policy="manual"):
    sim = CSSWellSimulator(CFG, well, PL["simulation"], WATER, np.random.default_rng(0))
    st = sim.initial_state()
    sim.produce(st, CycleControls(0, 0, 0, 0, 0.0, 30, policy, spm_init=1.0), phase="cold",
                min_days=10**9)
    return sim, st


def _controls(policy="manual", **kw):
    c = dict(steam_mass=1400e3, inj_pressure=10e6, quality_surface=0.75, soak_days=5,
             cutoff_rate=0.6 / 86400, max_prod_days=365, spm_policy=policy, spm_init=4.0,
             min_prod_days=60)
    c.update(kw)
    return CycleControls(**c)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_energy_and_mass_conservation_over_cycles(seed):
    well = sample_well(CFG, np.random.default_rng(seed), "T")
    sim, st = _sim(well)
    n0 = sim.v_p * well.s_oi
    oil_before = st.cum_oil
    for _ in range(3):
        sim.run_cycle(st, _controls())
    # energy: stored by injection = remaining + lost to cap/base rock + carried by fluids
    assert st.heat_in == pytest.approx(st.e_h + st.heat_cond_loss + st.heat_produced, rel=1e-9)
    # injected water: in bank + moved outward + leaked/dispersed + produced back
    assert st.water_injected == pytest.approx(st.w_in + st.w_out + st.leaked_water + st.water_prod_mass,
                                              rel=1e-9)
    # oil: reservoir volume removed = stock-tank production * Bo
    assert n0 - st.n_rem == pytest.approx(st.cum_oil * sim.b_o, rel=1e-9)
    assert st.cum_oil > oil_before


def test_pressure_solver_steady_state_equals_series_resistance():
    sim, st = _sim(central_well(CFG))
    sim.run_cycle(st, _controls())
    t_h = sim.heated_temperature(st)
    g = sim.regions(st, t_h)
    g = dict(g, v_in=0.0, v_out=1e15)  # no inner storage, constant outer pressure -> steady state
    p_wf = sim.p_wf_min
    _, _, q, _ = sim._solve_pressures(st, g, 86400.0, p_wf=p_wf)
    tr = 2 * np.pi * sim.w.k * sim.w.h
    assert q == pytest.approx(tr * (st.p_e - p_wf) / (g["R_in"] + g["R_out"]), rel=1e-6)


def test_rate_never_exceeds_pump_capacity_or_potential():
    sim, st = _sim(central_well(CFG))
    for policy in ("manual", "twin"):
        recs, _ = sim.run_cycle(st, _controls(policy))
        for r in recs:
            if r["phase"] == "production":
                q = r["oil_m3d_true"] * sim.b_o + r["water_m3d_true"]
                assert q <= r["capacity_m3d"] * (1 + 1e-9) + 1e-12
                assert q <= r["potential_m3d"] * (1 + 1e-9) + 1e-12


def test_more_steam_gives_more_oil_same_start_state():
    sim, st0 = _sim(central_well(CFG))
    oils = []
    for m in (800e3, 1400e3, 2000e3):
        st = copy.deepcopy(st0)
        sim.rng = np.random.default_rng(5)
        _, s = sim.run_cycle(st, _controls(steam_mass=m))
        oils.append(s["oil_m3_true"])
    assert oils[0] < oils[1] < oils[2]


def test_injection_pressure_bounds_enforced():
    sim, st = _sim(central_well(CFG))
    with pytest.raises(ValueError):
        sim.run_cycle(st, _controls(inj_pressure=25e6))  # above fracture pressure
