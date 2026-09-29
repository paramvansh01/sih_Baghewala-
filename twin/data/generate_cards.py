"""Labelled dynamometer-card dataset from the finite-difference rod/pump model.

Classes: normal, fluid_pound, gas_interference, tv_leak, sv_leak, pump_unseated, rod_floating.
The class label is the injected pump condition; `rod_floating` is additionally VERIFIED by
the simulation (minimum polished-rod load < 0) and every other class must stay clearly
above zero load, so no label contradicts the physics. Label priority when conditions
coincide (as in the live field): rod_floating > injected fault > fluid_pound > normal.

What the classifier sees is what a load cell/position sensor would record: polished-rod
position / stroke and load / rod weight in air, with sensor noise, and the load clipped at
zero (when the rods float the carrier bar separates from the clamp and the cell reads ~0).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..config import ROOT, FieldConfig, load_pipeline
from ..physics.fluids import OilPVT
from ..physics.rodpump_fd import PumpCondition, simulate_card
from ..physics.steam import WaterProps
from ..physics.srp_analytic import SRPDesign
from ..units import G

CLASSES = ["normal", "fluid_pound", "gas_interference", "tv_leak", "sv_leak", "pump_unseated",
           "rod_floating"]
OUT = ROOT / "artifacts" / "data"
WATER = WaterProps()


def _u(rng, lohi):
    return float(rng.uniform(*lohi))


def _logu(rng, lohi):
    return float(np.exp(rng.uniform(np.log(lohi[0]), np.log(lohi[1]))))


def _one(cls, idx, seed, cc, noise_mult=1.0):
    cfg = FieldConfig()
    sc = cc["scenario"]
    rng = np.random.default_rng([seed, CLASSES.index(cls), idx])
    k_bulk, m_pl = cfg.value("oil.bulk_modulus"), cfg.value("srp.plunger_mass")
    n_el = int(cc["rod_elements"])
    for attempt in range(int(cc["max_attempts"])):
        depth = cfg.value("reservoir.depth") + _u(rng, sc["depth_offset_m"])
        stroke = cfg.sample("srp.stroke_length", rng)
        d = SRPDesign.from_config(cfg, depth, stroke)
        rho = _u(rng, sc["rho_fluid"])
        p_wh = _u(rng, sc["wellhead_MPa"]) * 1e6
        p_i = _u(rng, sc["p_intake_MPa"]) * 1e6
        spm, ratio = _u(rng, sc["spm"]), _u(rng, sc["speed_ratio"])
        cond = PumpCondition()
        oil_cont = rng.uniform() < float(sc["oil_continuous_fraction"])
        t_pump = _u(rng, sc["t_pump_C"])
        if cls == "rod_floating":
            oil_cont = True
            t_pump = _u(rng, sc["floating_t_pump_C"])
            spm, ratio = _u(rng, sc["floating_spm"]), _u(rng, sc["floating_speed_ratio"])
            cond.fill = _u(rng, sc["floating_fill"])
        elif cls == "fluid_pound":
            cond.fill = _u(rng, sc["pound_fill"])
            p_i = _u(rng, sc["pound_p_intake_MPa"]) * 1e6
        elif cls == "gas_interference":
            cond.gvf = _u(rng, sc["gas_gvf"])
            p_i = _u(rng, sc["gas_p_intake_MPa"]) * 1e6
        elif cls == "tv_leak":
            cond.tv_leak = _u(rng, sc["leak_frac"])
        elif cls == "sv_leak":
            cond.sv_leak = _u(rng, sc["leak_frac"])
        elif cls == "pump_unseated":
            cond.unseated = True
        t_top = float(rng.uniform(cfg.value("wellbore.surface_temperature"), t_pump))
        t_nodes = np.linspace(t_top, t_pump, n_el + 1)  # surface -> pump
        if oil_cont:
            oil = OilPVT.from_config(cfg, cfg.sample("oil.api_gravity", rng), cfg.sample("oil.viscosity_at_ref", rng),
                                     cfg.sample("oil.walther_slope_B", rng))
            mu_nodes = np.asarray(oil.viscosity(t_nodes), float)
        else:
            mu_nodes = np.asarray(WATER.viscosity(t_nodes), float)
        mu_pump, top = float(mu_nodes[-1]), float(mu_nodes[0])
        p_d = p_wh + rho * G * d.pump_depth
        o = simulate_card(d, spm, ratio, mu_nodes, rho, p_i, p_d, cond, k_bulk, m_pl,
                          n_strokes=int(cc["strokes_to_simulate"]), n_out=int(cc["points_per_card"]),
                          cfl=float(cc["cfl"]))
        if not (np.all(np.isfinite(o["load"])) and np.isfinite(o["min_load"])):
            continue
        if o["load"].max() > cfg.value("srp.rod_tensile_strength") * d.a_rod:
            continue  # rods would part - such a card cannot be recorded
        floating = o["min_load"] < 0.0
        if cls == "rod_floating" and not floating:
            continue
        if cls != "rod_floating" and o["min_load"] < cc["non_floating_margin"] * o["w_buoyant"]:
            continue
        break
    else:
        return None
    ns = cfg.value("measurement_noise.load_relative_sd") * noise_mult
    ps = cfg.value("measurement_noise.position_sd") * noise_mult
    pos = o["position"] / stroke + ps * rng.standard_normal(o["position"].shape)
    load = np.maximum(o["load"], 0.0) / o["w_air"] + ns * rng.standard_normal(o["load"].shape)
    meta = dict(cls=cls, label=CLASSES.index(cls), idx=idx, attempt=attempt, depth=depth,
                stroke=stroke, spm=spm, ratio=ratio, rho=rho, p_intake=p_i, p_wh=p_wh,
                mu_pump=mu_pump, mu_top=top, t_pump=t_pump, t_top=t_top, oil_continuous=bool(oil_cont),
                fill=cond.fill, gvf=cond.gvf, tv_leak=cond.tv_leak,
                sv_leak=cond.sv_leak, unseated=cond.unseated, min_load_kN=o["min_load"] / 1e3,
                floating=bool(floating), w_air_kN=o["w_air"] / 1e3)
    return meta, pos.astype(np.float32), load.astype(np.float32)


def generate(n_per_class=None, seed=None, save=True, noise_mult=1.0, name="cards"):
    pl = load_pipeline()
    cc = pl["cards"]
    seed = pl["seed"] if seed is None else seed
    n = int(n_per_class or cc["n_per_class"])
    jobs = [(c, i) for c in CLASSES for i in range(n)]
    res = Parallel(n_jobs=cc["n_jobs"], batch_size=32)(delayed(_one)(c, i, seed, cc, noise_mult)
                                                       for c, i in jobs)
    failed = sum(r is None for r in res)
    res = [r for r in res if r is not None]
    meta = pd.DataFrame([r[0] for r in res])
    x = np.stack([np.stack([r[1], r[2]]) for r in res])  # (n, 2, points)
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        meta.to_csv(OUT / f"{name}_meta.csv", index=False)
        np.save(OUT / f"{name}_x.npy", x)
    return meta, x, failed


if __name__ == "__main__":
    t0 = time.time()
    meta, x, failed = generate()
    print(f"{len(meta)} cards ({failed} scenarios failed physics checks after max attempts) "
          f"in {time.time() - t0:.1f}s")
    print(meta.groupby("cls").agg(n=("label", "size"), floating=("floating", "mean"),
                                  attempts=("attempt", "mean"), min_load=("min_load_kN", "mean")).round(2))
