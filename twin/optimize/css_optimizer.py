"""CSS cycle optimiser: surrogate-based search + physics-twin verification.

For a well at the end of its history:
 1. Build the next-cycle feature row (static data + the well's own history).
 2. Sample candidate controls uniformly inside the envelope seen in the training data
    (the surrogate is only trusted where it has data; bounds come from the training set,
    not from hand-set numbers).
 3. Score candidates by predicted net value per cycle-day:
      (oil * price - fuel(steam, pressure, quality) - power * energy - opex * days) / days
    fuel is computed from steam tables (deterministic physics, not learned);
    oil, production days and SRP energy come from the surrogate.
 4. Re-simulate the top-k candidates AND the well's historical-practice controls with the
    physics twin from the well's actual end-of-history state (same noise seed) and report
    the verified outcome. The recommendation is the best VERIFIED candidate.
"""
from __future__ import annotations

import copy
import dataclasses
import math

import numpy as np
import pandas as pd

from ..config import FIELD_YAML, FieldConfig, load_pipeline
from ..data.generate_history import simulate_well
from ..models.css_surrogate import CSSSurrogate
from ..models.features import FEATURES, well_static
from ..physics import steam as stm
from ..physics.reservoir import CSSWellSimulator, CycleControls, economics_from_config
from ..units import BBL

DAY = 86400.0
CONTROL_COLS = ["steam_t", "inj_pressure_MPa", "quality_surface", "soak_days", "cutoff_m3d",
                "min_prod_days", "spm_init"]
INTEGER_COLS = {"soak_days", "min_prod_days"}


def fuel_gj(steam_t, p_mpa, quality, cfg):
    p = p_mpa * 1e6
    h = stm.h_liquid_sat(p) + quality * stm.latent_heat(p) - stm.h_liquid(cfg.value("wellbore.surface_temperature"), p)
    return steam_t * 1e3 * h / cfg.value("steam.boiler_efficiency") / 1e9


def next_cycle_row(well_row, cycles_w):
    last = cycles_w.sort_values("cycle_no").iloc[-1]
    row = dict(well_row)
    row.update(cycle_no=last.cycle_no + 1,
               cum_oil=last.cum_oil + last.oil_m3_true, cum_water=last.cum_water + last.water_m3_true,
               cum_steam_t=last.cum_steam_t + last.steam_t, prev_oil_m3=last.oil_m3, prev_sor=last.sor,
               prev_prod_days=last.prod_days, prev_t_bottom_end=last.t_bottom_end,
               prev_peak_oil_m3d=last.peak_oil_m3d, prev_floating_days=last.floating_days)
    return row, last


def sample_candidates(bounds, n, rng):
    c = {}
    for k in CONTROL_COLS:
        lo, hi = bounds[k]
        v = rng.uniform(lo, hi, n)
        c[k] = np.round(v) if k in INTEGER_COLS else v
    return pd.DataFrame(c)


def predicted_value(pred, cand, cfg, econ):
    inj_days = np.ceil(cand["steam_t"] * 1e3 / cfg.value("steam.generator_max_rate") / DAY)
    days = inj_days + cand["soak_days"] + pred["prod_days"]
    fuel = np.array([fuel_gj(s, p, q, cfg) for s, p, q in
                     zip(cand["steam_t"], cand["inj_pressure_MPa"], cand["quality_surface"])])
    value = (pred["oil_m3"] / BBL * econ["oil_price"] - fuel * econ["fuel_cost_per_GJ"]
             - pred["energy_kwh"] * econ["power_cost"] - days * econ["opex_per_day"])
    return value / days, value, days


def controls_from_row(r, cfg, policy):
    return CycleControls(steam_mass=float(r["steam_t"]) * 1e3, inj_pressure=float(r["inj_pressure_MPa"]) * 1e6,
                         quality_surface=float(r["quality_surface"]), soak_days=int(r["soak_days"]),
                         cutoff_rate=float(r["cutoff_m3d"]) / DAY,
                         max_prod_days=int(cfg.value("css_practice.max_production_days") / DAY),
                         spm_policy=policy, spm_init=float(r["spm_init"]), ratio_init=1.0,
                         min_prod_days=int(r["min_prod_days"]))


def simulate_candidate(sim: CSSWellSimulator, st0, controls, noise_seed, econ):
    st = copy.deepcopy(st0)
    sim.rng = np.random.default_rng(noise_seed)
    try:
        _, s = sim.run_cycle(st, controls)
    except ValueError as e:  # e.g. injection pressure below current reservoir pressure
        return dict(feasible=False, reason=str(e))
    v, vd = sim.cycle_value(s, econ)
    return dict(feasible=True, value=v, value_per_day=vd, oil_m3=s["oil_m3"], sor=s["sor"],
                days=s["end_day"] - s["start_day"] + 1, energy_kwh=s["energy_kwh"],
                floating_days=s["floating_days"], overload_days=s["overload_days"])


def optimise_well(well_idx, wells, cycles, surrogate: CSSSurrogate, seed, rng, n_candidates, top_k,
                  policy="manual"):
    cfg = FieldConfig()
    pl = load_pipeline()
    econ = economics_from_config(cfg)
    wid = wells.iloc[well_idx]["well_id"]
    static = well_static(wells).set_index("well_id").loc[wid].to_dict()
    static["cold_oil_m3d"] = float(wells.set_index("well_id").loc[wid, "cold_oil_m3d"])
    cyc_w = cycles[cycles.well_id == wid]
    base_row, last = next_cycle_row(static, cyc_w)
    cand = sample_candidates(surrogate.info["control_bounds"], n_candidates, rng)
    X = pd.DataFrame([base_row] * len(cand))
    for k in CONTROL_COLS:
        X[k] = cand[k].to_numpy()
    pred = surrogate.predict(X[FEATURES])
    vpd, val, days = predicted_value(pred, cand, cfg, econ)
    cand = cand.assign(pred_value_per_day=vpd, pred_oil_m3=pred["oil_m3"].to_numpy(),
                       pred_days=days.to_numpy()).sort_values("pred_value_per_day", ascending=False)
    # physics twin at the well's end-of-history state (deterministic re-simulation)
    _, _, _, sim, st0 = simulate_well(str(FIELD_YAML), pl["simulation"], well_idx, seed, return_state=True)
    assert sim.w.well_id == wid
    noise_seed = [seed, well_idx, 99]
    practice = last[CONTROL_COLS].to_dict()  # what the operator did last cycle
    out = dict(well_id=wid, practice=practice,
               practice_sim=simulate_candidate(sim, st0, controls_from_row(practice, cfg, policy), noise_seed, econ))
    Xp = pd.DataFrame([base_row])
    for k in CONTROL_COLS:
        Xp[k] = practice[k]
    pp = surrogate.predict(Xp[FEATURES])
    out["practice_pred_value_per_day"] = float(predicted_value(pp, pd.DataFrame([practice]), cfg, econ)[0].iloc[0])
    verified = []
    for _, r in cand.head(top_k).iterrows():
        res = simulate_candidate(sim, st0, controls_from_row(r, cfg, policy), noise_seed, econ)
        verified.append(dict(controls={k: float(r[k]) for k in CONTROL_COLS},
                             pred_value_per_day=float(r["pred_value_per_day"]),
                             pred_oil_m3=float(r["pred_oil_m3"]), sim=res))
    feas = [v for v in verified if v["sim"]["feasible"]]
    out["candidates"] = verified
    out["best"] = max(feas, key=lambda v: v["sim"]["value_per_day"]) if feas else None
    return out


def run(n_wells=None, policy="manual"):
    pl = load_pipeline()
    from ..models.features import DATA
    wells = pd.read_csv(DATA / "wells.csv")
    cycles = pd.read_csv(DATA / "cycles.csv")
    sur = CSSSurrogate()
    rng = np.random.default_rng(pl["seed"])
    oc = pl["optimizer"]
    idxs = range(len(wells)) if n_wells is None else range(min(n_wells, len(wells)))
    results = [optimise_well(i, wells, cycles, sur, pl["seed"], rng, oc["n_candidates"], oc["top_k_verify"], policy)
               for i in idxs]
    return results


def summarise(results):
    rows = []
    for r in results:
        p = r["practice_sim"]
        b = r["best"]
        if not (p.get("feasible") and b):
            continue
        rows.append(dict(well_id=r["well_id"], test_well=None,
                         practice_value_day=p["value_per_day"], opt_value_day=b["sim"]["value_per_day"],
                         practice_sor=p["sor"], opt_sor=b["sim"]["sor"],
                         practice_oil=p["oil_m3"], opt_oil=b["sim"]["oil_m3"],
                         practice_days=p["days"], opt_days=b["sim"]["days"],
                         pred_value_day=b["pred_value_per_day"],
                         practice_pred_value_day=r["practice_pred_value_per_day"]))
    return pd.DataFrame(rows)
