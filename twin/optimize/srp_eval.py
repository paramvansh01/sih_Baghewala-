"""SRP policy evaluation: reactive manual practice vs the twin controller.

For every well, starting from its actual end-of-history state, the next `n_cycles` CSS
cycles are simulated twice with IDENTICAL CSS controls and noise: once with the manual
reactive SPM rules, once with the twin controller (daily choice of SPM and downstroke/
upstroke speed ratio that avoids rod floating and Goodman overload and targets fillage).

Twin decisions are then spot-checked with the independent finite-difference rod/pump model:
at the chosen (SPM, ratio) and that day's tubing viscosity profile, the FD model must show
no floating (minimum polished-rod load > 0).

Policies: manual (reactive rules), twin (controller + lift-limited cut-off: ends the cycle
when no feasible SPM/ratio exists), twin_no_cutoff (controller only; keeps pumping at the
least-violating point). The cut-off trades oil for rod protection; with no rod-failure cost
model the twin cannot price that trade-off, so both variants are reported.

A sensitivity case with a higher permeability multiplier (more oil, lower water cut, more
viscous tubing flow) is also run, because rod floating is rare in the base synthetic field.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from ..config import FIELD_YAML, FieldConfig, load_pipeline
from ..data.generate_history import simulate_well
from ..physics.rodpump_fd import PumpCondition, simulate_card
from ..units import G
from .css_optimizer import CONTROL_COLS, controls_from_row

METRIC_KEYS = ["oil_m3", "prod_days", "floating_days", "overload_days", "impact_strokes", "energy_kwh",
               "steam_t", "calendar_days"]
POLICIES = {"manual": ("manual", False), "twin": ("twin", True), "twin_no_cutoff": ("twin", False)}


def _run_policy(sim, st0, controls_list, policy, noise_seed):
    policy, lift_cutoff = POLICIES[policy]
    st = copy.deepcopy(st0)
    sim.rng = np.random.default_rng(noise_seed)
    tot = dict.fromkeys(METRIC_KEYS, 0.0)
    days = []
    for c in controls_list:
        c = copy.copy(c)
        c.spm_policy, c.lift_cutoff = policy, lift_cutoff
        try:
            recs, s = sim.run_cycle(st, c)
        except ValueError:
            break
        s = dict(s, calendar_days=s["end_day"] - s["start_day"] + 1)
        for k in METRIC_KEYS:
            tot[k] += s[k]
        days += [r for r in recs if r["phase"] == "production"]
    _derive(tot)
    return tot, days


def _derive(t):
    t["kwh_per_m3_oil"] = t["energy_kwh"] / max(t["oil_m3"], 1e-9)
    t["oil_per_calendar_day"] = t["oil_m3"] / max(t["calendar_days"], 1e-9)
    t["sor"] = t["steam_t"] / max(t["oil_m3"], 1e-9)


def fd_verify(sim, days, n_samples, rng, cfg, n_el):
    """Run the FD model at sampled production days; returns list of (analytic_floating, fd_min_load)."""
    if not days:
        return []
    pick = rng.choice(len(days), size=min(n_samples, len(days)), replace=False)
    out = []
    k_bulk, m_pl = cfg.value("oil.bulk_modulus"), cfg.value("srp.plunger_mass")
    for i in pick:
        r = days[i]
        mu, rho = sim.tubing_viscosity_nodes(r["t_bottom_true"], r["q_prev_m3d"] / 86400.0, r["wc_srp"],
                                             r["prod_day"], n_el + 1)
        p_d = sim.p_wh + rho * G * sim.srp.pump_depth
        o = simulate_card(sim.srp, r["spm"], r["speed_ratio"], mu, rho, sim.p_wf_min, p_d,
                          PumpCondition(fill=float(min(max(r["fillage"], 0.05), 1.0))), k_bulk, m_pl)
        out.append(dict(analytic_floating=bool(r["floating"]), fd_min_load_kN=float(o["min_load"] / 1e3),
                        fd_floating=bool(o["min_load"] < 0), spm=r["spm"], ratio=r["speed_ratio"],
                        mu_pump_cP=float(mu[-1] * 1e3), mu_top_cP=float(mu[0] * 1e3)))
    return out


def run(perm_multipliers=(1.0, 4.0), n_cycles=3, fd_samples_per_well=6):
    cfg = FieldConfig()
    pl = load_pipeline()
    seed = pl["seed"]
    n_wells = int(pl["history"]["n_wells"] or cfg.value("field.n_producing_wells"))
    cycles_hist = pd.read_csv(FIELD_YAML.parents[1] / "artifacts" / "data" / "cycles.csv")
    rng = np.random.default_rng(seed)
    n_el = int(pl["cards"]["rod_elements"])
    scenarios = {}
    for mult in perm_multipliers:
        per_well, fd_twin, fd_manual = [], [], []
        for i in range(n_wells):
            wrow, cyc, _, sim, st0 = simulate_well(str(FIELD_YAML), pl["simulation"], i, seed,
                                                   perm_multiplier=mult, return_state=True)
            last = (pd.DataFrame(cyc).iloc[-1] if mult != 1.0 else
                    cycles_hist[cycles_hist.well_id == wrow["well_id"]].sort_values("cycle_no").iloc[-1])
            ctrl = controls_from_row(last[CONTROL_COLS].to_dict(), cfg, "manual")
            clist = [ctrl] * n_cycles
            row = dict(well_id=wrow["well_id"])
            for p in POLICIES:
                tot_p, days_p = _run_policy(sim, st0, clist, p, [seed, i, 7])
                row.update({f"{p}_{k}": v for k, v in tot_p.items()})
                if p == "twin":
                    feas = [d for d in days_p if d["twin_feasible"]]
                    fd_twin += fd_verify(sim, feas, fd_samples_per_well, rng, cfg, n_el)
                if p == "manual":
                    fd_manual += fd_verify(sim, [d for d in days_p if d["floating"]], 2, rng, cfg, n_el)
            per_well.append(row)
        df = pd.DataFrame(per_well)
        tot = {p: {k: float(df[f"{p}_{k}"].sum()) for k in METRIC_KEYS} for p in POLICIES}
        for p in tot:
            _derive(tot[p])
        ft = pd.DataFrame(fd_twin)
        fm = pd.DataFrame(fd_manual)
        scenarios[f"perm_x{mult:g}"] = dict(
            perm_multiplier=mult, n_wells=n_wells, n_cycles=n_cycles, totals=tot,
            per_well=df.to_dict(orient="records"),
            fd_check_twin_feasible_days=dict(
                n=len(ft), fd_floating=int(ft["fd_floating"].sum()) if len(ft) else 0,
                min_fd_load_kN=float(ft["fd_min_load_kN"].min()) if len(ft) else None),
            fd_check_manual_floating_days=dict(n=len(fm),
                                               fd_confirms_floating=int(fm["fd_floating"].sum()) if len(fm) else 0),
        )
    return dict(scenarios=scenarios,
                field_totals={k: v["totals"] for k, v in scenarios.items()})
