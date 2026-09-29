"""Field assistant for the live dashboard.

Grounded, offline question answering: every answer is computed from the live snapshot, the
well simulators and the saved model metrics. There is no language model, so it cannot invent
facts; questions it does not understand get a list of things it can answer.
"""
from __future__ import annotations

import json
import re
from datetime import date

import numpy as np

from ..config import ROOT
from ..units import BBL

DAY = 86400.0
METRICS = ROOT / "artifacts" / "metrics"

GLOSSARY = {
    "rod floating": "Rod floating happens on the downstroke when the fluid in the tubing is so thick that the "
                    "rods cannot fall as fast as the pumping unit lowers them. The carrier bar separates from the "
                    "polished-rod clamp and they hit each other again on the next stroke (impact loading), which "
                    "damages rods and can unseat the pump. Slowing the downstroke prevents it.",
    "sor": "Steam-oil ratio (SOR): tonnes of steam injected per cubic metre of oil produced. Lower is better.",
    "fillage": "Pump fillage: how full the pump barrel is on each stroke. Low fillage means the pump is taking "
               "more than the well can deliver (running dry, 'fluid pound').",
    "goodman": "Goodman loading compares rod stress with the allowable fatigue limit (API RP 11BR). Above 100% "
               "the rods are overloaded and likely to fail early.",
    "css": "Cyclic steam stimulation (CSS): inject steam into a well, let it soak so the heat reaches the oil, "
           "then pump the heated, thinner oil back until the rate falls to a cut-off, and repeat.",
    "soak": "Soak: the few days after steam injection when the well is shut in so heat can spread into the oil.",
    "spm": "SPM: strokes per minute of the sucker-rod pump.",
    "down/up": "Down/up ratio: how much longer the downstroke takes than the upstroke. A variable-speed or "
               "hydraulic unit can slow just the downstroke, which lets rods fall through thick oil without "
               "floating while keeping the upstroke (which lifts the oil) fast.",
    "cut-off": "Production cut-off: the oil rate at which a cycle is ended and the well is steamed again.",
    "digital twin": "The digital twin is a physics model of every well (reservoir heating, flow into the well, "
                    "tubing temperature and viscosity, rod-pump mechanics) plus machine-learning models trained "
                    "on its data. It forecasts each cycle, diagnoses pump cards and recommends pump speeds and "
                    "steam settings, checking each recommendation in the physics model before applying it.",
    "card": "A dynamometer card plots polished-rod load against rod position over one stroke. Its shape shows "
            "the pump's condition (full, running dry, gas, leaking valves, unseated, rods floating).",
}
GLOSS_KEYS = {"float": "rod floating", "sor": "sor", "steam-oil": "sor", "steam oil": "sor", "fillage": "fillage",
              "goodman": "goodman", "css": "css", "cyclic steam": "css", "soak": "soak", "spm": "spm",
              "strokes per": "spm", "ratio": "down/up", "downstroke": "down/up", "cut-off": "cut-off",
              "cutoff": "cut-off", "twin": "digital twin", "card": "card", "dynamometer": "card"}

CARD_NAMES = {"normal": "a normal full pump", "fluid_pound": "the pump running partly empty (fluid pound)",
              "gas_interference": "gas in the pump", "tv_leak": "a leaking travelling valve",
              "sv_leak": "a leaking standing valve", "pump_unseated": "an unseated pump",
              "rod_floating": "rods floating"}
FLEET_NAME = {"twin": "twin-controlled field", "practice": "current-practice field"}
STATE = {"floating": "floating its rods", "no_safe_speed": "unable to pump without floating or overloading its rods",
         "overload": "overloading its rods", "pumped_off": "running its pump partly dry", "low_inflow": "producing at low inflow",
         "off_target": "pumping at an off-target speed", "ok": "producing normally", "steaming": "being steamed",
         "soaking": "soaking", "shut_in": "shut in", "starting": "starting up"}
FOLLOW = {"floating": "Explain rod floating", "no_safe_speed": "Explain rod floating", "overload": "What is Goodman loading?",
          "pumped_off": "What is fillage?", "low_inflow": "What is fillage?", "off_target": "What is the down/up ratio?"}


def _b(x):
    return f"**{x}**"


def _find_well(q, ids):
    m = re.search(r"(?:bgw[\s-]*s?|\bs|well\s*#?\s*)(\d{1,2})\b", q)
    if not m:
        return None
    wid = f"BGW-S{int(m.group(1)):02d}"
    return wid if wid in ids else None


def _snapshot(field):
    with field.lock:
        return field.snapshot


def _well(field, fleet, wid):
    return next(w for w in field.fleets[fleet] if w.id == wid)


def _bbl(m3):
    return m3 / BBL


def _chips(*xs):
    return [x for x in xs if x]


def _nice_date(d):
    return date.fromisoformat(d).strftime("%a %-d %b %Y")


def _why_chip(wid, level):
    return f"Why does {wid} need attention?" if level in ("alarm", "warn") else f"Why is {wid} producing this much?"


def _other_fleet_note(field, fleet, wid):
    other = "practice" if fleet == "twin" else "twin"
    w = _well(field, other, wid)
    code, _, level, _ = field.status(w)
    if level == "alarm":
        return f"\n\nFor comparison, in the {FLEET_NAME[other]} {wid} is {_b(STATE.get(code, code))} right now."
    return ""


# --------------------------------------------------------------------------- intents
def _summary(field, snap, fleet):
    k = snap["kpis"][fleet]
    hist = snap.get("kpi_history") or []
    rows = [w[fleet] for w in snap["wells"]]
    alarms = [(w["id"], w[fleet]) for w in snap["wells"] if w[fleet]["level"] == "alarm"]
    warns = [(w["id"], w[fleet]) for w in snap["wells"] if w[fleet]["level"] == "warn"]
    txt = f"On {_b(_nice_date(snap['date']))} the {FLEET_NAME[fleet]} is producing {_b(f'{k['oil_bopd']:,.0f} bbl/d')} of oil"
    if len(hist) >= 2:
        y = hist[-2][fleet]["oil_bopd"]
        d = k["oil_bopd"] - y
        if abs(d) >= 0.5:
            txt += f", {'up' if d > 0 else 'down'} {abs(d):,.0f} bbl/d from yesterday"
    txt += (f". {k['producing']} wells are pumping, {k['injecting']} are being steamed and {k['soaking']} are soaking. "
            f"Pumps used {k['energy_kwh_day']:,.0f} kWh today.")
    if alarms:
        names = ", ".join(f"{i} ({r['status_label'].lower()})" for i, r in alarms[:4])
        txt += f"\n\n{_b(f'{len(alarms)} wells need attention now')}: {names}{' and more' if len(alarms) > 4 else ''}."
    elif warns:
        txt += f"\n\nNothing urgent. {len(warns)} wells have minor issues worth a look."
    else:
        txt += "\n\nNothing needs attention right now."
    first = alarms[0][0] if alarms else (warns[0][0] if warns else None)
    return dict(text=txt, well=None, chips=_chips("What needs attention?", first and f"Why does {first} need attention?",
                                                  "Twin vs current practice?"))


def _attention(field, snap, fleet):
    order = {"alarm": 0, "warn": 1}
    items = sorted([(order[w[fleet]["level"]], w["id"], w[fleet]) for w in snap["wells"]
                    if w[fleet]["level"] in order], key=lambda t: (t[0], t[1]))
    if not items:
        return dict(text=f"All wells in the {FLEET_NAME[fleet]} are fine right now.", well=None,
                    chips=_chips("How is the field today?"))
    groups = {}  # wells with identical advice are listed together
    for _, i, r in items:
        groups.setdefault((r["status_label"], r["action"]), []).append(i)
    lines = []
    for (label, action), wells in list(groups.items())[:6]:
        names = ", ".join(_b(w) for w in wells[:6]) + (f" and {len(wells) - 6} more" if len(wells) > 6 else "")
        lines.append(f"- {names} — {label}: {action}")
    more = ("\n\n…and more (see the Wells page, filter 'Needs attention')." if len(groups) > 6 else "")
    return dict(text=f"{len(items)} wells need attention in the {FLEET_NAME[fleet]}:\n" + "\n".join(lines) + more,
                well=items[0][1], chips=_chips(_why_chip(items[0][1], items[0][2]["level"]),
                                               "Explain rod floating"))


def _forecast_left(w):
    p, r = w.pred, w.last
    if not p or not r or r["phase"] != "production":
        return None
    d = int(r["prod_day"])
    total_days = int(round(p["cycle_prod_days"]))
    left = [v for day, v in zip(p["prod_day"], p["oil"]) if d < day <= total_days]
    return float(np.sum(left)), max(total_days - d, 0), p["cycle_oil_m3"]


def _well_answer(field, snap, fleet, wid):
    w = _well(field, fleet, wid)
    r = w.last
    code, label, level, action = field.status(w)
    if r is None:
        return dict(text=f"{_b(wid)}: {label}. {action}", well=wid, chips=[])
    if r["phase"] != "production":
        c = w.controls
        txt = (f"{_b(wid)} is {_b(STATE.get(code, label.lower()))} (cycle {r['cycle_no']}). "
               f"This cycle uses {c.steam_mass / 1e3:,.0f} t of steam at {c.inj_pressure / 1e6:.1f} MPa with a "
               f"{c.soak_days}-day soak; production restarts after the soak.")
        if w.pred:
            txt += (f" The models expect about {_b(f'{w.pred['cycle_oil_m3']:,.0f} m³')} of oil over "
                    f"{w.pred['cycle_prod_days']:.0f} production days.")
        if w.css_plan:
            v = w.css_plan["verified"]
            txt += (f" These settings were picked by the optimiser and checked in the physics twin "
                    f"(verified: {v['oil_m3']:,.0f} m³ oil, SOR {v['sor']:.1f}).")
        txt += _other_fleet_note(field, fleet, wid)
        return dict(text=txt, well=wid, chips=_chips("How is the field today?", "What is a soak?"))
    txt = (f"{_b(wid)} — {_b(label)}. Production day {r['prod_day']} of cycle {r['cycle_no']}: "
           f"{r['oil_m3d']:.2f} m³/d oil ({_bbl(r['oil_m3d']):.1f} bbl/d), water cut {r['water_cut'] * 100:.0f}%, "
           f"bottom-hole {r['t_bottom']:.0f} °C.\n\n"
           f"Pump: {r['spm']:.2f} strokes/min, downstroke {r['speed_ratio']:.2f}x the upstroke, "
           f"{r['fillage'] * 100:.0f}% full. The twin recommends {r['rec_spm']:.2f} strokes/min, "
           f"downstroke {r['rec_ratio']:.2f}x.")
    fl = _forecast_left(w)
    if fl:
        left, days_left, total = fl
        txt += (f"\n\nForecast: about {_b(f'{left:,.0f} m³')} more oil over the next ~{days_left} days "
                f"(cycle total ≈ {total:,.0f} m³).")
    if field.selected == dict(well=wid, fleet=fleet):
        c = field._card(w)
        if c:
            txt += f"\n\nToday's pump card looks like {CARD_NAMES.get(c['top'], c['top'])} ({max(c['diagnosis'].values()) * 100:.0f}% confidence)."
    txt += f"\n\n{_b('Recommended:')} {action}" + _other_fleet_note(field, fleet, wid)
    return dict(text=txt, well=wid, chips=_chips(_why_chip(wid, level), "What needs attention?"))


def _why(field, snap, fleet, wid):
    w = _well(field, fleet, wid)
    r = w.last
    code, label, level, action = field.status(w)
    if r is None or r["phase"] != "production":
        return _well_answer(field, snap, fleet, wid)
    sim = w.sim
    mu, _ = sim.tubing_viscosity_nodes(r["t_bottom_true"], r["q_prev_m3d"] / DAY, r["wc_srp"], r["prod_day"], 41)
    inv = sim.wc_inv
    oil_flow = r["wc_srp"] < inv
    parts = [f"{_b(wid)} is {STATE.get(code, label.lower())}."]
    if code in ("floating", "overload", "no_safe_speed"):
        parts.append(
            f"The produced stream is {r['wc_srp'] * 100:.0f}% water, "
            + (f"below the ~{inv * 100:.0f}% point where the tubing flow turns oil-continuous, so the rods move through "
               f"oil: about {mu[-1] * 1e3:,.0f} cP at the pump and {mu[0] * 1e3:,.0f} cP near the surface, where it has "
               f"cooled to about {r.get('t_wellhead', float('nan')):.0f} °C. "
               if oil_flow else f"so the tubing flow is water-continuous (about {mu[0] * 1e3:.1f} cP). ")
            + "Thick fluid slows the rods' fall on the downstroke; if the unit lowers them faster than they can fall, "
              "they float (and on the upstroke the drag adds to the rod stress).")
        if code == "no_safe_speed":
            parts.append("Even the slowest allowed speed with the longest downstroke cannot avoid it, so lifting is "
                         "limited by the cold tubing, not by the reservoir. Re-steaming reheats the well.")
    elif code in ("pumped_off", "low_inflow"):
        parts.append(f"The reservoir can deliver about {r['potential_m3d']:.1f} m³/d of liquid but the pump can move "
                     f"{r['capacity_m3d']:.1f} m³/d at this speed, so the barrel is only partly filled each stroke. "
                     f"The heated zone has cooled to {r['t_bottom_true']:.0f} °C, which makes the oil thicker and "
                     f"inflow lower.")
    else:
        start = w.ctx["t_prod_start"] if w.ctx else None
        parts.append(f"The heated zone is at {r['t_bottom_true']:.0f} °C"
                     + (f" (it started this cycle at {start:.0f} °C)" if start else "")
                     + f"; the oil there is about {sim.oil.viscosity(r['t_bottom_true']) * 1e3:,.1f} cP versus "
                       f"{sim.mu_cold * 1e3:,.0f} cP in the cold reservoir. Rate falls as the zone cools and its "
                       f"mobile oil drains.")
    parts.append(f"{_b('Recommended:')} {action}")
    return dict(text=" ".join(parts) + _other_fleet_note(field, fleet, wid), well=wid,
                chips=_chips(FOLLOW.get(code), "What needs attention?"))


def _compare(field, snap, fleet):
    p, t = snap["kpis"]["practice"], snap["kpis"]["twin"]
    days = snap.get("days_elapsed", 0)
    if not p["cum_oil_m3"] or not t["cum_oil_m3"]:
        return dict(text="Not enough production yet to compare — give it a few more days.", well=None, chips=[])
    lines = [f"Over the {days} days since the live run started (both fields started from the same state):",
             f"- Oil: twin {t['cum_oil_m3']:,.0f} m³ vs practice {p['cum_oil_m3']:,.0f} m³ "
             f"({(t['cum_oil_m3'] / p['cum_oil_m3'] - 1) * 100:+.0f}%)"]
    if t["sor"] and p["sor"]:
        lines.append(f"- Steam per m³ of oil (SOR): {t['sor']:.1f} vs {p['sor']:.1f}")
    if t["kwh_per_m3"] and p["kwh_per_m3"]:
        lines.append(f"- Pump energy: {t['kwh_per_m3']:,.0f} vs {p['kwh_per_m3']:,.0f} kWh per m³")
    lines.append(f"- Rod-floating well-days: {t['floating_days']:,} vs {p['floating_days']:,}; "
                 f"impact strokes {t['impact_strokes']:,.0f} vs {p['impact_strokes']:,.0f}")
    lines.append(f"- Net value (placeholder prices): ${t['net_value_usd']:,.0f} vs ${p['net_value_usd']:,.0f}")
    if days < 60:
        lines.append("\nEarly days are dominated by cycle timing (the twin re-steams sooner), so judge after a few months.")
    return dict(text="\n".join(lines), well=None, chips=_chips("How is the field today?", "What is SOR?"))


def _steam(field, snap, fleet):
    ws = [w for w in field.fleets[fleet] if w.last is not None and w.last["phase"] in ("injection", "soak")]
    if not ws:
        return dict(text=f"No wells are being steamed or soaking in the {FLEET_NAME[fleet]} right now.", well=None,
                    chips=_chips("How is the field today?"))
    lines = []
    for w in ws[:8]:
        c = w.controls
        lines.append(f"- {_b(w.id)} {w.last['phase']}: {c.steam_mass / 1e3:,.0f} t at {c.inj_pressure / 1e6:.1f} MPa, "
                     f"{c.soak_days}-day soak"
                     + (f"; forecast {w.pred['cycle_oil_m3']:,.0f} m³ oil" if w.pred else ""))
    how = ("The twin chooses these settings by scoring thousands of options with the cycle model and "
           "checking the best few in the physics twin." if fleet == "twin"
           else "These follow historical practice.")
    return dict(text=f"{len(ws)} wells are in their steam/soak phase:\n" + "\n".join(lines) + f"\n\n{how}",
                well=ws[0].id, chips=_chips("Twin vs current practice?"))


def _energy(field, snap, fleet):
    p, t = snap["kpis"]["practice"], snap["kpis"]["twin"]
    return dict(text=(f"Pumps are using {t['energy_kwh_day']:,.0f} kWh today in the twin-controlled field and "
                      f"{p['energy_kwh_day']:,.0f} kWh in the current-practice field. Per cubic metre of oil since the "
                      f"start: {t['kwh_per_m3'] or 0:,.0f} vs {p['kwh_per_m3'] or 0:,.0f} kWh. Most pump energy is lost "
                      f"to drag when rods move quickly through thick oil, which the twin avoids by slowing down."),
                well=None, chips=_chips("Twin vs current practice?"))


def _ranking(field, snap, fleet, worst):
    prod = [(w[fleet]["oil"], w["id"]) for w in snap["wells"] if w[fleet].get("phase") == "production"]
    if not prod:
        return dict(text="No wells are pumping right now.", well=None, chips=[])
    prod.sort(reverse=not worst)
    top = prod[:3]
    lines = [f"- {_b(i)}: {o:.2f} m³/d ({_bbl(o):.1f} bbl/d)" for o, i in top]
    return dict(text=f"{'Lowest' if worst else 'Highest'} producers right now:\n" + "\n".join(lines),
                well=top[0][1], chips=_chips(f"Why is {top[0][1]} so low?" if worst else f"Tell me about {top[0][1]}"))


def _data(field, snap, fleet):
    txt = ("The live data here come from a **simulated field feed**: the physics twin of each well steps forward one "
           "day at a time and reports what SCADA would (rates, temperatures, pump speed, loads). Real Baghewala "
           "data are not public, so the models were trained on data generated the same way.")
    try:
        sur = json.loads((METRICS / "css_surrogate.json").read_text())["results"]["oil_m3"]
        traj = json.loads((METRICS / "trajectory.json").read_text())["results"]
        cnn = json.loads((METRICS / "card_cnn.json").read_text())
        txt += (f"\n\nOn wells they never saw during training, the cycle model explains "
                f"{sur['test'][sur['chosen']]['r2'] * 100:.0f}% of the variation in oil per cycle "
                f"(repeating the last cycle explains {sur['test']['previous']['r2'] * 100:.0f}%), the daily forecast "
                f"{traj['oil_m3d']['gbm']['r2'] * 100:.0f}% of daily oil rate and "
                f"{traj['t_bottom']['gbm']['r2'] * 100:.0f}% of temperature, and the card model is "
                f"{cnn['cnn']['accuracy'] * 100:.1f}% accurate on test cards.")
    except (OSError, KeyError, json.JSONDecodeError):
        pass
    t = snap.get("card_tally", {})
    if t.get("n"):
        txt += (f" In this session its diagnosis matched the physics on {t['agree']} of {t['n']} live cards "
                f"({t['agree_grouped']} of {t['n']} counting 'running dry' and 'gas' as one).")
    return dict(text=txt, well=None, chips=_chips("What is a digital twin?"))


def _help():
    return dict(text=("I can tell you how the field is doing today, which wells need attention and what to do, why a "
                      "well is behaving the way it is, how twin control compares with current practice, which wells "
                      "are being steamed, and explain terms like rod floating or SOR."),
                well=None, chips=["How is the field today?", "What needs attention?", "Twin vs current practice?",
                                  "Explain rod floating"])


def answer(field, question, fleet="twin"):
    fleet = fleet if fleet in ("twin", "practice") else "twin"
    q = question.lower().strip()
    snap = _snapshot(field)
    if snap is None or not q:
        return _help()
    ids = [w["id"] for w in snap["wells"]]
    wid = _find_well(q, ids)
    if "practice" in q and "vs" not in q and "compare" not in q and "twin" not in q:
        fleet = "practice"
    if re.search(r"\b(what is|what's a|what does|explain|meaning|define)\b", q) and not wid:
        for key, g in GLOSS_KEYS.items():
            if key in q:
                return dict(text=GLOSSARY[g], well=None, chips=_chips("What needs attention?", "How is the field today?"))
    if wid:
        if re.search(r"\b(why|cause|reason|explain)\b", q):
            return _why(field, snap, fleet, wid)
        return _well_answer(field, snap, fleet, wid)
    if re.search(r"\b(compare|vs|versus|benefit|better|saving|savings|difference|improve)", q):
        return _compare(field, snap, fleet)
    if re.search(r"(attention|problem|issue|alarm|alert|wrong|fix|action|todo|to do|urgent|floating|overload)", q):
        return _attention(field, snap, fleet)
    if re.search(r"(steam|soak|inject|cycle|sor\b)", q):
        return _steam(field, snap, fleet)
    if re.search(r"(energy|power|kwh|electric)", q):
        return _energy(field, snap, fleet)
    if re.search(r"(worst|lowest|poor|weak)", q):
        return _ranking(field, snap, fleet, worst=True)
    if re.search(r"(best|top|highest|most)", q):
        return _ranking(field, snap, fleet, worst=False)
    if re.search(r"(data|accura|model|trust|real|simulat|how do you know)", q):
        return _data(field, snap, fleet)
    if re.search(r"(today|summary|overview|status|how is|how's|field|production|doing)", q):
        return _summary(field, snap, fleet)
    for key, g in GLOSS_KEYS.items():
        if key in q:
            return dict(text=GLOSSARY[g], well=None, chips=_chips("What needs attention?"))
    if re.search(r"^(hi|hello|hey|help|good (morning|afternoon|evening))", q):
        return _help()
    out = _help()
    out["text"] = "I didn't quite get that. " + out["text"]
    return out
