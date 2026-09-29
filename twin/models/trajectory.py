"""Within-cycle trajectory model: predicts the daily oil rate and bottom-hole (heated-zone)
temperature on production day d of a planned cycle -> 'reservoir heating, cooling and
production performance'. Trained from scratch (gradient-boosted trees) on measured daily
data; evaluated on the same held-out WELLS as the cycle surrogate.
Baseline: the average training profile by production day.
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from ..config import ROOT, load_pipeline
from .css_surrogate import METRICS, MODELS, metrics, split_wells
from .features import DATA, FEATURES, cycle_table

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
TARGETS = {"oil_m3d": "log", "t_bottom": "raw"}
TRAJ_FEATURES = FEATURES + ["prod_day"]


def daily_table(data_dir=DATA, every=1):
    daily = pd.read_csv(data_dir / "daily.csv.gz")
    prod = daily[daily.phase == "production"][["well_id", "cycle_no", "prod_day", "oil_m3d", "t_bottom"]]
    prod = prod[prod.prod_day % every == 1 % every] if every > 1 else prod
    cyc = cycle_table(data_dir)
    return prod.merge(cyc[["well_id"] + FEATURES], on=["well_id", "cycle_no"], how="inner")  # cycle_no is a feature


def _fwd(t, y):
    return np.log1p(y) if TARGETS[t] == "log" else np.asarray(y, float)


def _inv(t, y):
    return np.expm1(y) if TARGETS[t] == "log" else np.asarray(y, float)


def train(save=True):
    with threadpool_limits(1):
        return _train(save)


def _train(save):
    pl = load_pipeline()
    cfg, seed = pl["trajectory_model"], pl["seed"]
    df = daily_table(every=int(cfg["day_sample_every"]))
    # identical well split to the cycle surrogate (split is made on the cycle table)
    tr_c, te_c = split_wells(cycle_table(), pl["css_surrogate"]["test_fraction_wells"], seed)
    tr = df[df.well_id.isin(set(tr_c.well_id))].reset_index(drop=True)
    te = df[df.well_id.isin(set(te_c.well_id))].reset_index(drop=True)
    models, results = {}, {}
    for t in TARGETS:
        m = HistGradientBoostingRegressor(random_state=seed, **cfg["hgb"]).fit(tr[TRAJ_FEATURES], _fwd(t, tr[t]))
        pred = _inv(t, m.predict(te[TRAJ_FEATURES]))
        prof = tr.groupby("prod_day")[t].mean()
        base = te["prod_day"].map(prof).fillna(tr[t].mean()).to_numpy()
        results[t] = dict(gbm=metrics(te[t], pred), mean_profile=metrics(te[t], base),
                          n_iter=int(m.n_iter_))
        models[t] = m
    info = dict(n_train_rows=len(tr), n_test_rows=len(te), test_wells=sorted(te.well_id.unique()),
                features=TRAJ_FEATURES, hyperparameters=cfg["hgb"], seed=seed)
    if save:
        MODELS.mkdir(parents=True, exist_ok=True)
        METRICS.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(models=models, features=TRAJ_FEATURES, info=info), MODELS / "trajectory.joblib")
        with open(METRICS / "trajectory.json", "w") as f:
            json.dump(dict(info=info, results=results), f, indent=2)
        out = te[["well_id", "cycle_no", "prod_day"] + list(TARGETS)].copy()
        for t in TARGETS:
            out[f"pred_{t}"] = _inv(t, models[t].predict(te[TRAJ_FEATURES]))
        out.to_csv(METRICS / "trajectory_test_predictions.csv.gz", index=False)
    return models, results, info


class TrajectoryModel:
    def __init__(self, path=None):
        b = joblib.load(path or MODELS / "trajectory.joblib")
        self.models, self.features = b["models"], b["features"]

    def predict_cycle(self, cycle_features: pd.Series, days: int) -> pd.DataFrame:
        X = pd.DataFrame([cycle_features[FEATURES].to_dict()] * days)
        X["prod_day"] = np.arange(1, days + 1)
        return pd.DataFrame({t: _inv(t, m.predict(X[self.features])) for t, m in self.models.items()},
                            index=X["prod_day"])


if __name__ == "__main__":
    _, res, info = train()
    print(f"rows train {info['n_train_rows']}, test {info['n_test_rows']} (wells {info['test_wells']})")
    for t, r in res.items():
        print(t, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in r.items() if isinstance(v, dict)},
              "iters", r["n_iter"])
