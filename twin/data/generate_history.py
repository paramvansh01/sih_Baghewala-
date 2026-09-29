"""Synthesise the field operating history the ML models are trained on.

For every producing well (count from field.yaml): sample reservoir/fluid/equipment
properties, simulate cold production until the well's (sampled) CSS start date, then
consecutive CSS cycles until the history end date. Operating parameters of every cycle
are drawn from the 'historical practice' envelope (css_practice.*) and the SRP is run
with the reactive manual policy - i.e. the data represent CURRENT practice.

Outputs (artifacts/data/): wells.csv, cycles.csv, daily.csv
Only columns an operator can actually measure are used as ML features downstream; the
'*_true' columns and hidden state are kept for validation only.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..config import ROOT, FieldConfig, load_pipeline
from ..physics.reservoir import CSSWellSimulator, CycleControls
from ..physics.steam import WaterProps
from .wells import sample_well

DAY = 86400.0
OUT = ROOT / "artifacts" / "data"


def practice_controls(cfg, rng, policy="manual", spm_init=None):
    s = lambda k: cfg.sample(k, rng)  # noqa: E731
    return CycleControls(
        steam_mass=s("css_practice.steam_mass"),
        inj_pressure=s("css_practice.injection_pressure"),
        quality_surface=s("steam.surface_quality"),
        soak_days=int(round(s("css_practice.soak_days") / DAY)),
        cutoff_rate=s("css_practice.cutoff_oil_rate"),
        min_prod_days=int(round(s("css_practice.min_production_days") / DAY)),
        max_prod_days=int(cfg.value("css_practice.max_production_days") / DAY),
        spm_policy=policy,
        spm_init=spm_init if spm_init is not None else round(s("srp.manual_spm_range") * 2) / 2,
        ratio_init=1.0,
    )


def simulate_well(cfg_path, settings, well_idx, seed, perm_multiplier=1.0, policy="manual",
                  return_state=False):
    cfg = FieldConfig(cfg_path)
    rng = np.random.default_rng([seed, well_idx])
    water = WaterProps()
    well = sample_well(cfg, rng, f"BGW-S{well_idx + 1:02d}")
    well = dataclasses.replace(well, k=well.k * perm_multiplier)
    sim = CSSWellSimulator(cfg, well, settings, water, rng)
    st = sim.initial_state()

    first = date.fromisoformat(cfg.value("field.css_first_date"))
    end = date.fromisoformat(cfg.value("field.history_end_date"))
    spread = cfg.value("field.css_start_spread_years")
    css_start = first + timedelta(days=int(rng.uniform(0, spread * 365.25)))
    pre_days = int(settings["pre_css_cold_days"])
    record_start = css_start - timedelta(days=pre_days)

    daily = []
    cold = CycleControls(0.0, 0.0, 0.0, 0, 0.0, pre_days, policy,
                         spm_init=cfg.value("srp.spm_min"))
    daily += sim.produce(st, cold, phase="cold", min_days=10**9)
    cold_rate = float(np.mean([r["oil_m3d"] for r in daily]))
    cycles = []
    while True:
        c = practice_controls(cfg, rng, policy)
        # guard: injection pressure must exceed current reservoir pressure
        p_needed = max(st.p_e, st.p_in) * 1.05
        if c.inj_pressure <= p_needed:
            c = dataclasses.replace(c, inj_pressure=min(p_needed, cfg.range("css_practice.injection_pressure")[1]))
            if c.inj_pressure <= max(st.p_e, st.p_in):
                break
        prev = cycles[-1] if cycles else None
        cum_before = dict(cum_oil=st.cum_oil, cum_water=st.cum_water, cum_steam_t=st.cum_steam / 1e3)
        recs, summ = sim.run_cycle(st, c)
        summ.update(cum_before)
        summ["prev_oil_m3"] = prev["oil_m3"] if prev else np.nan
        summ["prev_sor"] = prev["sor"] if prev else np.nan
        summ["prev_prod_days"] = prev["prod_days"] if prev else np.nan
        summ["prev_t_bottom_end"] = prev["t_bottom_end"] if prev else np.nan
        summ["prev_peak_oil_m3d"] = prev["peak_oil_m3d"] if prev else np.nan
        summ["prev_floating_days"] = prev["floating_days"] if prev else np.nan
        summ["cold_oil_m3d"] = cold_rate
        summ["start_date"] = (record_start + timedelta(days=summ["start_day"] - 1)).isoformat()
        summ["end_date"] = (record_start + timedelta(days=summ["end_day"] - 1)).isoformat()
        cycles.append(summ)
        daily += recs
        if record_start + timedelta(days=st.day) >= end:
            break
    wrow = dataclasses.asdict(well)
    wrow.update(css_start=css_start.isoformat(), record_start=record_start.isoformat(),
                cold_oil_m3d=cold_rate, pump_depth=sim.srp.pump_depth,
                n_cycles=len(cycles), oil_viscosity_res_cP=float(sim.oil.viscosity(well.t_r)) * 1e3)
    for d in daily:
        d["well_id"] = well.well_id
        d["date"] = (record_start + timedelta(days=d["day"] - 1)).isoformat()
    for c in cycles:
        c["well_id"] = well.well_id
    if return_state:
        return wrow, cycles, daily, sim, st
    return wrow, cycles, daily


def generate(perm_multiplier=1.0, seed=None, n_wells=None, policy="manual", out=OUT, save=True):
    cfg = FieldConfig()
    pl = load_pipeline()
    seed = pl["seed"] if seed is None else seed
    n = n_wells or pl["history"]["n_wells"] or int(cfg.value("field.n_producing_wells"))
    res = Parallel(n_jobs=-1)(delayed(simulate_well)(str(cfg_path()), pl["simulation"], i, seed,
                                                     perm_multiplier, policy) for i in range(n))
    wells = pd.DataFrame([r[0] for r in res])
    cycles = pd.DataFrame([c for r in res for c in r[1]])
    daily = pd.DataFrame([d for r in res for d in r[2]])
    if save:
        out.mkdir(parents=True, exist_ok=True)
        wells.to_csv(out / "wells.csv", index=False)
        cycles.to_csv(out / "cycles.csv", index=False)
        daily.to_csv(out / "daily.csv.gz", index=False)
    return wells, cycles, daily


def cfg_path():
    from ..config import FIELD_YAML
    return FIELD_YAML


def field_rate_bopd(daily, end_date, days=365):
    """Mean field oil rate (bbl/d) over the last `days` of the history (measured rates)."""
    from ..units import BBL
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    end = pd.Timestamp(end_date)
    win = d[(d["date"] > end - pd.Timedelta(days=days)) & (d["date"] <= end)]
    per_day = win.groupby("date")["oil_m3d"].sum()
    return float(per_day.mean() / BBL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm-multiplier", type=float, default=1.0)
    a = ap.parse_args()
    t0 = time.time()
    w, c, d = generate(a.perm_multiplier)
    cfg = FieldConfig()
    print(f"{len(w)} wells, {len(c)} cycles, {len(d)} daily rows in {time.time()-t0:.1f}s")
    print("field rate last 365 d (bopd):", round(field_rate_bopd(d, cfg.value('field.history_end_date')), 1))
    print(c[["oil_m3", "water_m3", "sor", "prod_days", "floating_days", "overload_days"]].describe().round(2))
