"""Live (real-time) digital-twin engine.

Data source: a SIMULATED FIELD FEED. Every well's physics simulator is advanced one day per
tick and emits exactly the measurements a SCADA/historian would provide (measured rates with
noise, temperatures, SPM, loads, energy). Everything downstream - the ML predictions, the card
diagnosis, the recommendations and alerts - works only on that feed plus the trained models,
exactly as it would on real data.

Two fleets run in lockstep from the SAME end-of-history state and noise seeds:
  * practice : reactive manual SPM rules + historical-practice CSS settings
  * twin     : twin SPM/speed controller (+ lift-limited cut-off) + CSS settings recommended by
               the surrogate search and verified in the physics twin before being applied
so the dashboard shows the twin's effect against a live counterfactual.
"""
from __future__ import annotations

import copy
import dataclasses
import threading
import time
from collections import deque
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..config import FIELD_YAML, FieldConfig, load_pipeline
from ..data.generate_cards import CLASSES
from ..data.generate_history import practice_controls, simulate_well
from ..models.card_cnn import CardDiagnoser
from ..models.css_surrogate import CSSSurrogate
from ..models.features import DATA, FEATURES, well_static
from ..models.trajectory import TrajectoryModel
from ..optimize.css_optimizer import (CONTROL_COLS, controls_from_row, next_cycle_row,
                                      predicted_value, sample_candidates, simulate_candidate)
from ..physics.reservoir import economics_from_config
from ..physics.rodpump_fd import PumpCondition, simulate_card
from ..units import BBL, G

DAY = 86400.0
FLEETS = ("practice", "twin")


def _controls_dict(c):
    return dict(steam_t=c.steam_mass / 1e3, inj_pressure_MPa=c.inj_pressure / 1e6,
                quality_surface=c.quality_surface, soak_days=c.soak_days, cutoff_m3d=c.cutoff_rate * DAY,
                min_prod_days=c.min_prod_days, spm_init=c.spm_init)


class LiveWell:
    def __init__(self, fleet, idx, sim, st, cycles, static, models, cfg, lcfg, seed, start_date):
        self.fleet, self.idx, self.sim, self.st = fleet, idx, sim, st
        self.id = sim.w.well_id
        self.cycles = [dict(c) for c in cycles]
        self.static, self.models, self.cfg, self.lcfg = static, models, cfg, lcfg
        self.seed = seed
        self.start_date = start_date  # date of simulator day 1
        self.rng = np.random.default_rng([seed, idx, 4242])  # practice-controls sampler (same in both fleets)
        self.queue = deque()
        self.gen = None
        self.controls = None
        self.records = None
        self.ctx = None
        self.cum_before = None
        self.series = deque(maxlen=int(lcfg["history_days_kept"]))
        self.last = None
        self.pred = None
        self.css_plan = None
        self.cycle_start_index = None
        self.shut_in = False
        self.totals = dict(oil=0.0, water=0.0, steam_t=0.0, energy=0.0, floating_days=0, overload_days=0,
                           impact_strokes=0.0, fuel_GJ=0.0)
        self.ticks = 0

    # ------------------------------------------------------------ cycles
    def _feature_row(self):
        row, _ = next_cycle_row(self.static, pd.DataFrame(self.cycles))
        return row

    def _recommend_css(self, row):
        """Surrogate search inside the training envelope, top-k verified in the physics twin."""
        sur = self.models["surrogate"]
        econ = economics_from_config(self.cfg)
        cand = sample_candidates(sur.info["control_bounds"], int(self.lcfg["css_candidates"]),
                                 np.random.default_rng([self.seed, self.idx, len(self.cycles)]))
        X = pd.DataFrame([row] * len(cand))
        for k in CONTROL_COLS:
            X[k] = cand[k].to_numpy()
        pred = sur.predict(X[FEATURES])
        vpd, _, _ = predicted_value(pred, cand, self.cfg, econ)
        cand = cand.assign(pred_value_per_day=vpd.to_numpy(), pred_oil_m3=pred["oil_m3"].to_numpy())
        cand = cand.sort_values("pred_value_per_day", ascending=False).head(int(self.lcfg["css_top_k"]))
        rng_backup = self.sim.rng
        best = None
        for _, r in cand.iterrows():
            ctrl = controls_from_row(r, self.cfg, "twin")
            res = simulate_candidate(self.sim, self.st, ctrl, [self.seed, self.idx, 99], econ)
            if res.get("feasible") and (best is None or res["value_per_day"] > best["verified"]["value_per_day"]):
                best = dict(controls={k: float(r[k]) for k in CONTROL_COLS},
                            pred_value_per_day=float(r["pred_value_per_day"]),
                            pred_oil_m3=float(r["pred_oil_m3"]), verified=res)
        self.sim.rng = rng_backup
        return best

    def begin_cycle(self):
        row = self._feature_row()
        practice = practice_controls(self.cfg, self.rng, "manual")
        plan = None
        if self.fleet == "twin":
            plan = self._recommend_css(row)
            controls = controls_from_row(plan["controls"], self.cfg, "twin") if plan else \
                dataclasses.replace(practice, spm_policy="twin")
        else:
            controls = practice
        p_now = max(self.st.p_e, self.st.p_in)
        if controls.inj_pressure <= p_now * 1.05:
            hi = self.cfg.range("css_practice.injection_pressure")[1]
            controls = dataclasses.replace(controls, inj_pressure=min(p_now * 1.05, hi))
            if controls.inj_pressure <= p_now:
                self.shut_in = True
                return
        controls.lift_cutoff = self.fleet == "twin"
        self.controls = controls
        self.css_plan = plan
        self.cum_before = dict(cum_oil=self.st.cum_oil, cum_water=self.st.cum_water,
                               cum_steam_t=self.st.cum_steam / 1e3)
        self.records = []
        self.ctx = self.sim.start_cycle(self.st, controls, self.records)
        self.queue.extend(self.records)
        self.gen = self.sim.produce_iter(self.st, controls, records=self.records)
        # ML predictions for this cycle, from pre-cycle information only
        x = dict(row)
        x.update(_controls_dict(controls))
        xs = pd.Series(x)
        horizon = int(self.lcfg["prediction_horizon_days"])
        traj = self.models["trajectory"].predict_cycle(xs, horizon)
        tot = self.models["surrogate"].predict(pd.DataFrame([x])[FEATURES]).iloc[0]
        self.pred = dict(prod_day=traj.index.tolist(), oil=traj["oil_m3d"].round(3).tolist(),
                         t_bottom=traj["t_bottom"].round(2).tolist(),
                         cycle_oil_m3=float(tot["oil_m3"]), cycle_prod_days=float(tot["prod_days"]),
                         cycle_energy_kwh=float(tot["energy_kwh"]))
        self.totals["steam_t"] += controls.steam_mass / 1e3
        self.totals["fuel_GJ"] += self.ctx["inj"]["fuel_GJ"]

    def _finish_cycle(self):
        s = self.sim.summarise_cycle(self.st, self.controls, self.records, self.ctx)
        prev = self.cycles[-1] if self.cycles else None
        s.update(self.cum_before)
        for k in ("oil_m3", "sor", "prod_days", "t_bottom_end", "peak_oil_m3d", "floating_days"):
            s[f"prev_{k}"] = prev[k] if prev else np.nan
        s["well_id"] = self.id
        self.cycles.append(s)
        self.gen = None
        return s

    # -------------------------------------------------------------- tick
    def step(self, alerts, date_str):
        """Advance one day. Returns the day's record (or None if shut in)."""
        if self.shut_in:
            return None
        if not self.queue and self.gen is None:
            self.begin_cycle()
            if self.shut_in:
                alerts.append(dict(date=date_str, well=self.id, fleet=self.fleet, level="warn",
                                   msg="Cannot steam: reservoir pressure is above the allowed injection pressure."))
                return None
            c = _controls_dict(self.controls)
            alerts.append(dict(date=date_str, well=self.id, fleet=self.fleet, level="info",
                               msg=(f"Started steaming for cycle {self.st.cycle_no}: {c['steam_t']:.0f} t at "
                                    f"{c['inj_pressure_MPa']:.1f} MPa, then a {c['soak_days']}-day soak"
                                    + (". Settings chosen by the optimiser and checked in the physics twin."
                                       if self.css_plan else "."))))
        if self.queue:
            rec = self.queue.popleft()
        else:
            try:
                rec = next(self.gen)
            except StopIteration:
                s = self._finish_cycle()
                alerts.append(dict(date=date_str, well=self.id, fleet=self.fleet, level="info",
                                   msg=f"Cycle {s['cycle_no']} ended at its maximum length: {s['oil_m3']:.0f} m³ oil, "
                                       f"{s['sor']:.1f} t steam per m³."))
                return self.step(alerts, date_str)
        self.ticks += 1
        if rec["phase"] == "production":
            spm_r, ratio_r, ok = self.sim.choose_twin(rec["t_bottom_true"], rec["q_prev_m3d"] / DAY, rec["wc_srp"],
                                                      rec["prod_day"], rec["potential_m3d"] / DAY)
            rec = dict(rec, rec_spm=spm_r, rec_ratio=ratio_r, rec_feasible=ok)
            t = self.totals
            t["oil"] += rec["oil_m3d"]
            t["water"] += rec["water_m3d"]
            t["energy"] += rec["motor_kwh"]
            t["floating_days"] += int(rec["floating"])
            t["overload_days"] += int(rec["goodman"] > 1.0)
            t["impact_strokes"] += rec["impact_strokes"]
            if rec["floating"]:
                alerts.append(dict(date=date_str, well=self.id, fleet=self.fleet, level="alarm",
                                   msg=(f"Rods floating at {rec['spm']:.2f} strokes/min. "
                                        + (f"The twin suggests {spm_r:.2f} strokes/min with a {ratio_r:.2f}x slower downstroke."
                                           if ok else "No safe pump speed exists; the cycle should end and the well be re-steamed."))))
            elif rec["goodman"] > 1.0:
                alerts.append(dict(date=date_str, well=self.id, fleet=self.fleet, level="alarm",
                                   msg=f"Rod stress at {rec['goodman'] * 100:.0f}% of the safe limit."))
            if rec.get("stop_reason"):
                s = self._finish_cycle()
                why = ("pumping became unsafe for the rods" if rec["stop_reason"] == "lift_cutoff"
                       else "the oil rate fell below the cut-off")
                alerts.append(dict(date=date_str, well=self.id, fleet=self.fleet, level="info",
                                   msg=f"Cycle {s['cycle_no']} ended because {why}: {s['oil_m3']:.0f} m³ oil, "
                                       f"{s['sor']:.1f} t steam per m³."))
        rec = dict(rec, date=date_str)
        self.last = rec
        self.series.append(rec)
        return rec


INCOMPLETE_FILL = {"fluid_pound", "gas_interference"}  # same operational meaning: chamber not filled


def _grouped(c):
    return "incomplete_fillage" if c in INCOMPLETE_FILL else c


def validate_live_cards(n_days=90, every=15, fault_wells=6):
    """CNN vs physics label on cards generated at LIVE operating points (both fleets), plus
    injected-fault diagnosis. Written to artifacts/metrics/live_card_check.json by the CLI."""
    field = LiveField()
    rows = []
    for day in range(n_days):
        field.tick()
        if day % every == every - 1:
            for f in FLEETS:
                for w in field.fleets[f]:
                    c = field._card(w)
                    if c:
                        rows.append(dict(expected=c["expected"], cnn=c["top"]))
    faults = []
    prod = [w for w in field.fleets["twin"] if w.last and w.last["phase"] == "production"][:fault_wells]
    for fault in field.lcfg["fault_presets"]:
        for w in prod:
            field.faults[w.id] = fault
            c = field._card(w)
            faults.append(dict(expected=c["expected"], cnn=c["top"]))
            field.faults[w.id] = "none"
    df, fd = pd.DataFrame(rows), pd.DataFrame(faults)
    return dict(
        n_live_cards=len(df), exact_agreement=float((df.expected == df.cnn).mean()),
        grouped_agreement=float((df.expected.map(_grouped) == df.cnn.map(_grouped)).mean()),
        confusion=pd.crosstab(df.expected, df.cnn).to_dict(),
        n_fault_cards=len(fd), fault_agreement=float((fd.expected == fd.cnn).mean()) if len(fd) else None,
        days=n_days, every=every)


class LiveField:
    def __init__(self):
        self.cfg = FieldConfig()
        self.pl = load_pipeline()
        self.lcfg = self.pl["live"]
        seed = self.pl["seed"]
        self.models = dict(surrogate=CSSSurrogate(), trajectory=TrajectoryModel(), cards=CardDiagnoser())
        wells_df = pd.read_csv(DATA / "wells.csv")
        n = len(wells_df)
        self.fleets = {f: [] for f in FLEETS}
        for i in range(n):
            wrow, cyc, _, sim, st = simulate_well(str(FIELD_YAML), self.pl["simulation"], i, seed,
                                                  return_state=True)
            static = well_static(pd.DataFrame([wrow])).iloc[0].drop("well_id").to_dict()
            static["cold_oil_m3d"] = wrow["cold_oil_m3d"]
            start = date.fromisoformat(wrow["record_start"])
            for f in FLEETS:
                s2, st2 = copy.deepcopy(sim), copy.deepcopy(st)
                s2.rng = np.random.default_rng([seed, i, 777])  # identical noise in both fleets
                self.fleets[f].append(LiveWell(f, i, s2, st2, cyc, static, self.models, self.cfg, self.lcfg,
                                               seed, start))
        self.date = date.fromisoformat(self.cfg.value("field.history_end_date"))
        self.alerts = deque(maxlen=int(self.lcfg["alerts_kept"]))
        self.lock = threading.Lock()
        self.new_snapshot = threading.Condition(self.lock)
        self.running = True
        self.seconds_per_day = float(self.lcfg["seconds_per_day"])
        self.selected = dict(well=self.fleets["twin"][0].id, fleet="twin")
        self.faults = {}  # well_id -> fault preset name (affects the card simulation only)
        self.snapshot = None
        self.version = 0
        self.tick_ms = 0.0
        self._card_cache = {}
        self.card_tally = dict(n=0, agree=0, agree_grouped=0)  # live CNN-vs-physics check, this session
        self.kpi_history = deque(maxlen=int(self.lcfg["history_days_kept"]))
        self.positions = self._layout(n, seed)

    def _layout(self, n, seed):
        """SCHEMATIC map positions (real well coordinates are not public): staggered grid with
        a small deterministic jitter; units = grid spacing."""
        m = self.lcfg["map"]
        cols = int(np.ceil(np.sqrt(n * float(m["aspect"]))))
        rng = np.random.default_rng([seed, 31])
        pos = []
        for i in range(n):
            r, c = divmod(i, cols)
            jx, jy = rng.uniform(-1, 1, 2) * float(m["jitter"])
            pos.append((round(c + 0.5 * (r % 2) + jx, 3), round(r * 0.866 + jy, 3)))
        return pos

    # ------------------------------------------------------------- ticking
    def tick(self):
        t0 = time.time()
        self.date += timedelta(days=1)
        ds = self.date.isoformat()
        new_alerts = []
        for f in FLEETS:
            for w in self.fleets[f]:
                w.step(new_alerts, ds)
        snap = self._build_snapshot(new_alerts)
        self.kpi_history.append(dict(date=snap["date"], **{f: snap["kpis"][f] for f in FLEETS}))
        snap["kpi_history"] = list(self.kpi_history)
        self.tick_ms = (time.time() - t0) * 1e3
        snap["tick_ms"] = round(self.tick_ms, 1)
        with self.new_snapshot:
            self.alerts.extend(new_alerts)
            snap["alerts"] = list(self.alerts)[::-1][: int(self.lcfg["alerts_sent"])]
            self.snapshot = snap
            self.version += 1
            self.new_snapshot.notify_all()

    def run_forever(self):
        while True:
            if self.running:
                start = time.time()
                try:
                    self.tick()
                except Exception as e:  # keep the service alive, surface the error in the UI
                    with self.new_snapshot:
                        self.alerts.append(dict(date=self.date.isoformat(), well="-", fleet="-", level="alarm",
                                                msg=f"engine error: {type(e).__name__}: {e}"))
                time.sleep(max(0.0, self.seconds_per_day - (time.time() - start)))
            else:
                time.sleep(0.1)

    def step_once(self):
        """Advance exactly one day (used by the 'next day' button while paused)."""
        if not self.running:
            self.tick()

    def refresh(self):
        """Rebuild the snapshot without advancing time (after a selection/fault change)."""
        snap = self._build_snapshot([])
        snap["kpi_history"] = list(self.kpi_history)
        with self.new_snapshot:
            snap["alerts"] = list(self.alerts)[::-1][: int(self.lcfg["alerts_sent"])]
            snap["tick_ms"] = round(self.tick_ms, 1)
            self.snapshot = snap
            self.version += 1
            self.new_snapshot.notify_all()

    # ------------------------------------------------------------ snapshot
    def _kpis(self, fleet):
        ws = self.fleets[fleet]
        last = [w.last for w in ws if w.last is not None]
        prod = [r for r in last if r["phase"] == "production"]
        tot = {k: sum(w.totals[k] for w in ws) for k in ws[0].totals}
        return dict(
            oil_bopd=sum(r["oil_m3d"] for r in prod) / BBL,
            water_m3d=sum(r["water_m3d"] for r in prod),
            energy_kwh_day=sum(r["motor_kwh"] for r in prod),
            producing=len(prod), injecting=sum(r["phase"] == "injection" for r in last),
            soaking=sum(r["phase"] == "soak" for r in last),
            floating_now=sum(bool(r["floating"]) for r in prod),
            overload_now=sum(r["goodman"] > 1.0 for r in prod),
            cum_oil_m3=tot["oil"], cum_steam_t=tot["steam_t"],
            sor=tot["steam_t"] / tot["oil"] if tot["oil"] > 0 else None,
            kwh_per_m3=tot["energy"] / tot["oil"] if tot["oil"] > 0 else None,
            floating_days=tot["floating_days"], overload_days=tot["overload_days"],
            impact_strokes=tot["impact_strokes"],
            net_value_usd=self._net_value(tot),
        )

    def _net_value(self, tot):
        e = economics_from_config(self.cfg)
        return (tot["oil"] / BBL * e["oil_price"] - tot["fuel_GJ"] * e["fuel_cost_per_GJ"]
                - tot["energy"] * e["power_cost"])

    def status(self, w):
        """Plain-language status, severity and recommended action for a well (both fleets).
        Thresholds come from config (low-fillage threshold, controller grid step)."""
        r = w.last
        if r is None:
            if w.shut_in:
                return ("shut_in", "Shut in", "warn",
                        "Reservoir pressure is above the allowed injection pressure, so the well cannot be steamed.")
            return ("starting", "Starting", "info", "")
        if r["phase"] == "injection":
            return ("steaming", "Steaming", "info", f"Steam is being injected for cycle {r['cycle_no']}.")
        if r["phase"] == "soak":
            return ("soaking", "Soaking", "info", "The well is shut in so the steam heat soaks into the oil.")
        twin = w.fleet == "twin"
        spm, rs, rr = r["spm"], r["rec_spm"], r["rec_ratio"]
        tol = float(self.pl["simulation"]["spm_grid_step"]) + 1e-6
        if not r["rec_feasible"]:
            return ("no_safe_speed", "No safe pump speed", "alarm",
                    "The oil in the tubing is too thick to pump without the rods floating or overloading at any "
                    "allowed speed. End this cycle and re-steam."
                    + (" The twin does this automatically after a few days." if twin else ""))
        if r["floating"]:
            return ("floating", "Rods floating", "alarm",
                    f"Slow the pump to {rs:.2f} strokes/min with a {rr:.2f}x slower downstroke to stop the rods floating.")
        if r["goodman"] > 1.0:
            return ("overload", "Rods overloaded", "alarm",
                    f"Rod stress is above the safe limit. Change to {rs:.2f} strokes/min, downstroke {rr:.2f}x slower.")
        if r["fillage"] < self.cfg.value("srp.low_fillage_threshold"):
            if spm <= self.cfg.value("srp.spm_min") + 1e-6:
                return ("low_inflow", "Low inflow", "ok",
                        f"The pump fills {r['fillage'] * 100:.0f}% per stroke at its slowest speed. Inflow is low, "
                        "which is normal late in a cycle; no action needed.")
            return ("pumped_off", "Pump running dry", "warn",
                    f"The pump fills only {r['fillage'] * 100:.0f}% per stroke. "
                    f"Slow it to {rs:.2f} strokes/min to match what the well delivers.")
        if abs(spm - rs) > tol or abs(r["speed_ratio"] - rr) > tol:
            return ("off_target", "Speed off target", "warn",
                    f"The twin suggests {rs:.2f} strokes/min (downstroke {rr:.2f}x the upstroke) instead of {spm:.2f}.")
        return ("ok", "Producing normally", "ok", "No action needed.")

    def _row(self, w):
        r = w.last
        code, label, level, action = self.status(w)
        pos = self.positions[w.idx]
        out = dict(status=code, status_label=label, level=level, action=action, x=pos[0], y=pos[1],
                   r_heated=round(w.st.r_h, 2), r_e=round(w.sim.w.r_e, 1))
        if r is None:
            out.update(phase="shut-in" if w.shut_in else "-")
            return out
        out.update(phase=r["phase"], cycle=r["cycle_no"], t_bottom=round(float(r.get("t_bottom", r["t_bottom_true"])), 1),
                   t_zone=round(float(w.sim.heated_temperature(w.st)), 1))
        if r["phase"] == "production":
            out.update(prod_day=r["prod_day"], oil=round(r["oil_m3d"], 2), water=round(r["water_m3d"], 2),
                       spm=r["spm"], ratio=r["speed_ratio"], fill=round(r["fillage"], 2),
                       floating=bool(r["floating"]), overload=bool(r["goodman"] > 1.0),
                       goodman=round(r["goodman"], 2), rec_spm=r["rec_spm"], rec_ratio=r["rec_ratio"],
                       rec_feasible=bool(r["rec_feasible"]), wc=round(r["water_cut"], 2))
        return out

    def _card(self, w):
        """Dynamometer card at the well's current operating point + CNN diagnosis."""
        r = w.last
        if r is None or r["phase"] != "production":
            return None
        fault = self.faults.get(w.id, "none")
        key = (w.fleet, w.id, r["day"], fault)
        if key in self._card_cache:
            return self._card_cache[key]
        sim, cfg = w.sim, self.cfg
        n_el = int(self.pl["cards"]["rod_elements"])
        mu, rho = sim.tubing_viscosity_nodes(r["t_bottom_true"], r["q_prev_m3d"] / DAY, r["wc_srp"],
                                             r["prod_day"], n_el + 1)
        preset = dict(self.lcfg["fault_presets"].get(fault, {}))
        p_intake = float(preset.pop("p_intake_MPa", sim.p_wf_min / 1e6)) * 1e6
        # an injected fault is simulated at full fillage (as in the training data); otherwise
        # the pump sees today's fillage from the twin
        fill = 1.0 if fault != "none" else float(min(max(r["fillage"], 0.02), 1.0))
        cond = PumpCondition(fill=fill, **preset)
        p_d = sim.p_wh + rho * G * sim.srp.pump_depth
        o = simulate_card(sim.srp, r["spm"], r["speed_ratio"], mu, rho, p_intake, p_d, cond,
                          cfg.value("oil.bulk_modulus"), cfg.value("srp.plunger_mass"),
                          n_strokes=int(self.pl["cards"]["strokes_to_simulate"]),
                          n_out=int(self.pl["cards"]["points_per_card"]))
        rng = np.random.default_rng([self.pl["seed"], w.idx, r["day"]])
        pos = o["position"] / sim.srp.stroke + cfg.value("measurement_noise.position_sd") * rng.standard_normal(len(o["position"]))
        load = (np.maximum(o["load"], 0.0) / o["w_air"]
                + cfg.value("measurement_noise.load_relative_sd") * rng.standard_normal(len(o["load"])))
        proba = self.models["cards"].predict_proba(np.stack([pos, load])[None].astype(np.float32))[0]
        # label the physics implies, with the same definitions/priority as the training data
        pound_max = float(self.pl["cards"]["scenario"]["pound_fill"][1])
        expected = ("rod_floating" if o["min_load"] < 0 else fault if fault != "none"
                    else "fluid_pound" if fill < pound_max else "normal")
        card = dict(pos=np.round(pos, 4).tolist(), load=np.round(load, 4).tolist(),
                    downhole_pos=np.round(o["plunger"] / sim.srp.stroke, 4).tolist(),
                    downhole_load=np.round(o["pump_force"] / o["w_air"], 4).tolist(),
                    diagnosis={c: round(float(p), 4) for c, p in zip(CLASSES, proba)},
                    top=CLASSES[int(np.argmax(proba))], expected=expected, fill=round(fill, 3),
                    min_load_kN=round(o["min_load"] / 1e3, 2),
                    fault_injected=fault, spm=r["spm"], ratio=r["speed_ratio"],
                    mu_top_cP=round(float(mu[0]) * 1e3), mu_pump_cP=round(float(mu[-1]) * 1e3, 1))
        self._card_cache = {key: card}
        t = self.card_tally
        t["n"] += 1
        t["agree"] += int(card["top"] == expected)
        t["agree_grouped"] += int(_grouped(card["top"]) == _grouped(expected))
        return card

    def _detail(self):
        sel = self.selected
        w = next(x for x in self.fleets[sel["fleet"]] if x.id == sel["well"])
        keys = ["date", "phase", "cycle_no", "prod_day", "oil_m3d", "water_m3d", "t_bottom_true", "t_bottom",
                "spm", "speed_ratio", "rec_spm", "rec_ratio", "fillage", "prl_min_kN", "prl_max_kN", "goodman",
                "motor_kwh", "floating", "water_cut", "steam_t"]
        series = {k: [(round(float(r[k]), 4) if isinstance(r.get(k), (int, float, np.floating)) and not isinstance(r.get(k), bool)
                       else r.get(k)) for r in w.series] for k in keys}
        # series index of production day 1 of the current cycle (may lie in the future while the
        # well is still injecting/soaking) - the trajectory forecast is drawn from there
        prod_start = None
        if w.records is not None and w.last is not None:
            idx = [i for i, r in enumerate(w.series) if r["cycle_no"] == w.st.cycle_no]
            if idx:
                n_pre = sum(r["phase"] in ("injection", "soak") for r in w.records)
                prod_start = idx[0] + n_pre
        return dict(well=w.id, fleet=w.fleet, series=series, prod_start_index=prod_start,
                    prediction=w.pred, card=self._card(w),
                    controls=_controls_dict(w.controls) if w.controls else None,
                    css_plan=w.css_plan, fault=self.faults.get(w.id, "none"),
                    static={k: (round(v, 4) if isinstance(v, float) else v) for k, v in w.static.items()},
                    well_state=dict(cycle_no=w.st.cycle_no, heated_radius_m=round(w.st.r_h, 2),
                                    shut_in=w.shut_in))

    def _build_snapshot(self, new_alerts):
        wells = []
        for p, t in zip(self.fleets["practice"], self.fleets["twin"]):
            wells.append(dict(id=p.id, practice=self._row(p), twin=self._row(t)))
        days = (self.date - date.fromisoformat(self.cfg.value("field.history_end_date"))).days
        return dict(date=self.date.isoformat(), days_elapsed=days, running=self.running,
                    seconds_per_day=self.seconds_per_day,
                    feed="SIMULATED FIELD FEED (physics twin) - replace with SCADA/historian for field use",
                    kpis={f: self._kpis(f) for f in FLEETS}, wells=wells, selected=self.selected,
                    detail=self._detail(), classes=CLASSES, card_tally=dict(self.card_tally),
                    fault_presets=["none"] + list(self.lcfg["fault_presets"]),
                    sor_chart_start_day=int(self.lcfg["sor_chart_start_day"]))
