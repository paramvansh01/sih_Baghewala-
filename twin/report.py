"""Builds artifacts/report.md + figures from the saved datasets, models and metrics.
Every number in the report is read from artifacts/ - nothing is typed in by hand."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import ROOT, FieldConfig  # noqa: E402
from .data.generate_cards import CLASSES  # noqa: E402
from .data.generate_history import field_rate_bopd  # noqa: E402

A = ROOT / "artifacts"
D, M, FIG = A / "data", A / "metrics", A / "figures"


def _j(name):
    return json.loads(Path(M / name).read_text())


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=110)
    plt.close(fig)
    return f"figures/{name}"


def fig_well_history(daily, well):
    d = daily[daily.well_id == well].copy()
    d["date"] = pd.to_datetime(d["date"])
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    p = d[d.phase.isin(["production", "cold"])]
    ax[0].plot(p.date, p.oil_m3d, lw=0.7, label="oil (measured)")
    ax[0].plot(p.date, p.water_m3d, lw=0.7, alpha=0.6, label="water (measured)")
    ax[0].set_ylabel("m3/d")
    ax[0].legend(loc="upper right")
    ax[1].plot(d.date, d.t_bottom_true, lw=0.7, color="tab:red")
    ax[1].set_ylabel("heated-zone T (C)")
    ax[2].step(p.date, p.spm, lw=0.7, color="tab:green")
    ax[2].set_ylabel("SPM (manual)")
    inj = d[d.phase == "injection"]
    for a in ax:
        a.vlines(inj.date, *a.get_ylim(), color="orange", alpha=0.08)
    ax[0].set_title(f"{well}: synthetic operating history (orange = steam injection days)")
    return _save(fig, "well_history.png")


def fig_cards(meta, x):
    fig, ax = plt.subplots(1, len(CLASSES), figsize=(18, 2.8))
    for a, c in zip(ax, CLASSES):
        i = meta.index[meta.cls == c][0]
        a.plot(x[i, 0], x[i, 1], lw=1)
        a.set_title(c, fontsize=9)
        a.set_xlabel("position / stroke", fontsize=8)
    ax[0].set_ylabel("load / rod weight")
    return _save(fig, "cards.png")


def fig_parity():
    t = pd.read_csv(M / "css_surrogate_test_predictions.csv")
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    for a, c in zip(ax, ["oil_m3", "water_m3", "prod_days", "energy_kwh"]):
        a.scatter(t[c], t[f"pred_{c}"], s=10, alpha=0.6)
        lim = [min(t[c].min(), t[f"pred_{c}"].min()), max(t[c].max(), t[f"pred_{c}"].max())]
        a.plot(lim, lim, "k--", lw=0.8)
        a.set_xlabel(f"actual {c}")
        a.set_ylabel("predicted")
        a.set_title(c)
    fig.suptitle("CSS surrogate on held-out wells", y=1.02)
    return _save(fig, "surrogate_parity.png")


def fig_trajectory():
    t = pd.read_csv(M / "trajectory_test_predictions.csv.gz")
    key = t.groupby(["well_id", "cycle_no"]).size().sort_values().index[-1]
    s = t[(t.well_id == key[0]) & (t.cycle_no == key[1])].sort_values("prod_day")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].plot(s.prod_day, s.oil_m3d, ".", ms=4, label="measured")
    ax[0].plot(s.prod_day, s.pred_oil_m3d, "-", label="predicted")
    ax[0].set_ylabel("oil m3/d")
    ax[1].plot(s.prod_day, s.t_bottom, ".", ms=4, label="measured")
    ax[1].plot(s.prod_day, s.pred_t_bottom, "-", label="predicted")
    ax[1].set_ylabel("bottom-hole T (C)")
    for a in ax:
        a.set_xlabel("production day")
        a.legend()
    fig.suptitle(f"Trajectory model, held-out well {key[0]} cycle {key[1]}", y=1.02)
    return _save(fig, "trajectory.png")


def fig_confusion(r):
    cm = np.array(r["cnn"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Card CNN, test set")
    return _save(fig, "cnn_confusion.png")


def fig_css(summary):
    s = summary.sort_values("practice_value_day")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    x = np.arange(len(s))
    ax[0].bar(x - 0.2, s.practice_value_day, 0.4, label="historical practice")
    ax[0].bar(x + 0.2, s.opt_value_day, 0.4, label="optimised (verified in twin)")
    ax[0].set_ylabel("net value, USD per cycle-day")
    ax[1].bar(x - 0.2, s.practice_sor, 0.4, label="historical practice")
    ax[1].bar(x + 0.2, s.opt_sor, 0.4, label="optimised (verified in twin)")
    ax[1].set_ylabel("SOR (t steam / m3 oil)")
    for a in ax:
        a.set_xticks(x, s.well_id, rotation=90, fontsize=7)
        a.legend()
    return _save(fig, "css_optimisation.png")


def fig_srp(srp):
    sc = srp["scenarios"]
    keys = list(sc)
    metrics = [("oil_m3", "oil (m3)"), ("oil_per_calendar_day", "oil per calendar day (m3/d)"),
               ("kwh_per_m3_oil", "SRP kWh per m3 oil"), ("floating_days", "rod-floating days"),
               ("impact_strokes", "impact strokes")]
    pols = list(sc[keys[0]]["totals"])
    fig, ax = plt.subplots(1, len(metrics), figsize=(20, 3.8))
    w = 0.8 / len(pols)
    for a, (k, lab) in zip(ax, metrics):
        xs = np.arange(len(keys))
        for j, p in enumerate(pols):
            a.bar(xs + (j - (len(pols) - 1) / 2) * w, [sc[s]["totals"][p][k] for s in keys], w, label=p)
        a.set_xticks(xs, keys)
        a.set_title(lab, fontsize=10)
        if k in ("kwh_per_m3_oil", "impact_strokes", "floating_days"):
            a.set_yscale("symlog")
    ax[0].legend(fontsize=8)
    return _save(fig, "srp_policy.png")


def _tbl(df, floatfmt=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def _pytest_summary():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], capture_output=True,
                       text=True, cwd=ROOT)
    lines = [ln for ln in r.stdout.strip().splitlines() if "passed" in ln or "failed" in ln]
    return lines[-1] if lines else r.stdout[-300:]


def build():
    cfg = FieldConfig()
    prov = pd.DataFrame(cfg.provenance_table())
    wells = pd.read_csv(D / "wells.csv")
    cycles = pd.read_csv(D / "cycles.csv")
    daily = pd.read_csv(D / "daily.csv.gz")
    cmeta = pd.read_csv(D / "cards_meta.csv")
    cx = np.load(D / "cards_x.npy")
    sur, traj, cnn = _j("css_surrogate.json"), _j("trajectory.json"), _j("card_cnn.json")
    css = pd.read_csv(M / "css_optimisation_summary.csv")
    srp = _j("srp_evaluation.json")

    rate = field_rate_bopd(daily, cfg.value("field.history_end_date"))
    c1 = cycles[cycles.cycle_no == 1].merge(wells[["well_id", "cold_oil_m3d"]], on="well_id",
                                            suffixes=("", "_w"))
    stim = (c1.oil_m3 / c1.prod_days) / c1.cold_oil_m3d_w
    figs = dict(hist=fig_well_history(daily, wells.well_id.iloc[0]), cards=fig_cards(cmeta, cx),
                parity=fig_parity(), traj=fig_trajectory(), cm=fig_confusion(cnn), css=fig_css(css),
                srp=fig_srp(srp))

    from .models.css_surrogate import summary_table
    st = summary_table(sur["results"])[["target", "model", "cv_r2", "r2", "mae", "mape"]]
    tr = traj["results"]
    traj_tbl = pd.DataFrame([dict(target=t, model=m, **v) for t, r in tr.items() for m, v in r.items()
                             if isinstance(v, dict)])
    per = pd.DataFrame(cnn["cnn"]["per_class"]).T.reset_index().rename(columns={"index": "class"})
    per = per[["class", "precision", "recall", "f1-score", "support"]]
    srp_rows = []
    for s, v in srp["scenarios"].items():
        for p in v["totals"]:
            srp_rows.append(dict(scenario=s, policy=p, **{k: v["totals"][p][k] for k in
                                                          ("oil_m3", "calendar_days", "oil_per_calendar_day",
                                                           "sor", "kwh_per_m3_oil", "floating_days",
                                                           "overload_days", "impact_strokes")}))
    srp_tbl = pd.DataFrame(srp_rows)
    css_imp = dict(n=len(css), better=int((css.opt_value_day > css.practice_value_day).sum()),
                   med_dv=float((css.opt_value_day - css.practice_value_day).median()),
                   sor_p=float(css.practice_sor.median()), sor_o=float(css.opt_sor.median()),
                   pred_err=float((css.pred_value_day - css.opt_value_day).abs().median()))

    live_txt = ""
    if (M / "live_card_check.json").exists():
        lc = _j("live_card_check.json")
        live_txt = (f"**Live operating points** (cards generated at the live twin's daily conditions, both fleets, "
                    f"n={lc['n_live_cards']}): exact agreement with the physics label {lc['exact_agreement']:.3f}; "
                    f"{lc['grouped_agreement']:.3f} when fluid pound and gas interference are grouped as "
                    f"'incomplete fillage' (in this pump model both mean a chamber part-filled with gas at intake "
                    f"pressure, differing only in that pressure). Injected faults: {lc['fault_agreement']:.3f} "
                    f"(n={lc['n_fault_cards']}).")
    counts = prov.status.value_counts().to_dict()
    pub = prov[prov.status.isin(["published", "problem_statement"])][["parameter", "value", "unit", "source"]]
    md = f"""# Baghewala CSS + SRP Digital Twin — results report

All numbers below are produced by `python -m twin.cli all` and read from `artifacts/`.
**The training data are SYNTHETIC** (generated by the physics twin), because Baghewala
field data are not public. The pipeline is built so OIL's real `wells/cycles/daily` tables
can replace the synthetic ones (see README).

Verification suite: `{_pytest_summary()}`

## 1. Parameter provenance
{len(prov)} parameters: {counts}. Every value has a source in `config/field.yaml`;
`python -m twin.cli params` prints them all. Published / problem-statement values used:

{_tbl(pub)}

## 2. Synthetic field vs published figures (sanity checks — NOT fitted)
| check | synthetic | published |
|---|---|---|
| field oil rate, last 365 days of history | {rate:.0f} bopd | ~1,100–1,202 bopd (OIL website; BusinessToday 2026) |
| mean cycle-1 oil rate / pre-steam cold rate (median over wells) | {stim.median():.1f}× | "five to six fold" (OIL CSS pilot) |
| cycles simulated | {len(cycles)} over {len(wells)} wells | 34 producing wells |

The synthetic field produces **far less than the real field**: the assumed (unpublished)
reservoir properties and tubing-flow assumptions are pessimistic. Absolute numbers from
this twin therefore must not be read as Baghewala forecasts; relative comparisons
(model vs baseline, policy A vs B) are what the synthetic study supports.

![history]({figs['hist']})

## 3. CSS cycle surrogate (trained from scratch, test = {len(sur['info']['test_wells'])} unseen wells)
Train {sur['info']['n_train_cycles']} cycles / test {sur['info']['n_test_cycles']} cycles. Model per target chosen
by grouped CV on training wells only (`*`). `previous` = repeat the well's last cycle
(historical-experience baseline).

{_tbl(st)}

![parity]({figs['parity']})

## 4. Heating/cooling & production trajectory model
{_tbl(traj_tbl)}

![trajectory]({figs['traj']})

## 5. Dynamometer-card CNN (PyTorch, random init)
{cnn['n_parameters']} parameters; initial validation loss {cnn['initial_val_loss']:.3f} (chance ln7 = {cnn['chance_loss']:.3f})
→ best {cnn['best_val_loss']:.4f} after {cnn['epochs_run']} epochs.
Test accuracy **{cnn['cnn']['accuracy']:.4f}** (macro-F1 {cnn['cnn']['macro_f1']:.4f}) vs logistic regression
{cnn['logistic_regression']['accuracy']:.4f}. Independent robustness set (new seed, 3× sensor noise):
CNN {cnn.get('robustness', {}).get('accuracy', float('nan')):.4f} vs logistic regression
{cnn.get('robustness', {}).get('logistic_regression_accuracy', float('nan')):.4f}.
Rod floating is also caught by the simple physics rule (min measured load ≤ {cnn['floating_rule']['threshold_load_over_w_air']:.2f}·W_air,
F1 {cnn['floating_rule']['f1']:.3f}) — detection is easy; *preventing* it is the SRP controller's job (§7).
Caveat: synthetic cards are cleaner than field cards; expect lower accuracy on real data.

{_tbl(per)}

{live_txt}

![cards]({figs['cards']})
![cm]({figs['cm']})

## 6. CSS optimisation (surrogate search → physics-twin verification)
For each of {css_imp['n']} wells, next-cycle controls were searched with the surrogate inside the
training-data envelope; the top candidates and the well's last (historical-practice) controls were
re-simulated in the physics twin from the well's actual state.
Verified improvement in net value/cycle-day on {css_imp['better']}/{css_imp['n']} wells (median
{css_imp['med_dv']:+.0f} USD/day); median SOR {css_imp['sor_p']:.1f} → {css_imp['sor_o']:.1f}.
Median |surrogate-predicted − verified| value: {css_imp['pred_err']:.0f} USD/day. Economics use
placeholder prices from `config/field.yaml` — replace with OIL figures.

![css]({figs['css']})

## 7. SRP control: reactive manual practice vs twin controller
Next {srp['scenarios'][list(srp['scenarios'])[0]]['n_cycles']} cycles per well, identical CSS controls and noise. Policies:
`manual` = reactive operator rules; `twin` = controller + lift-limited cut-off (ends the cycle
when no SPM/ratio within unit limits avoids floating and overload, protecting the rods);
`twin_no_cutoff` = controller only. The cut-off trades oil for rod protection; there is no
rod-failure cost model, so the choice between the two twin variants is left to the operator.
`perm_x4` is a sensitivity case (4× permeability).

{_tbl(srp_tbl, 2)}

FD-model spot checks of twin decisions on days it judged feasible:
{json.dumps({k: v['fd_check_twin_feasible_days'] for k, v in srp['scenarios'].items()})}
FD confirmation of manual floating days: {json.dumps({k: v['fd_check_manual_floating_days'] for k, v in srp['scenarios'].items()})}
(floating = FD minimum polished-rod load < 0).

![srp]({figs['srp']})

## 8. Limitations (read before using any number)
* Training data are synthetic; {counts.get('assumption', 0)} of the {len(prov)} parameters are assumptions (see §1).
* Reservoir model: single-layer, radial, lumped heated zone, two-region pressure, Dupuit gravity
  drainage; no steam flashing/solution-gas drive, no geomechanics, no inter-well interference.
* Tubing flow: oil-/water-continuous switch at an assumed inversion water cut; emulsion
  viscosity increase ignored; rod drag uses annular Couette flow (fluid flow neglected).
* Card model: single-taper rods, harmonic polished-rod motion, no tubing stretch, ideal valves
  (plus laminar leaks); field cards contain effects not modelled here.
* Surrogate is only valid inside the training envelope of controls; the optimiser is constrained to it.
* SRP results are dominated by the assumed emulsion inversion water cut and VIT heat-transfer
  coefficient: below the inversion point the tubing flow uses cold-oil viscosity (10^4–10^5 cP near
  surface), which is what makes floating frequent. Measure both before trusting SRP numbers.
* There is no rod-failure / workover cost model, so the twin cannot price rod protection against
  lost oil; both twin variants are reported and the energy objective uses placeholder prices.
"""
    out = A / "report.md"
    out.write_text(md)
    return out
