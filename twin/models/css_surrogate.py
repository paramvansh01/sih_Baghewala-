"""CSS cycle-performance surrogate, trained from scratch on the cycle history.

Predicts, for a planned cycle: oil (m3), water (m3), production days and SRP energy (kWh).
Evaluation is grouped by WELL: the test wells are never seen during training or model
selection, which is the realistic use case (forecasting a well the model was not fitted on).

Model selection, per target, by grouped K-fold CV on the training wells only:
  * gradient-boosted trees (small hyper-parameter grid from pipeline.yaml)
  * ridge regression (standardised features, median imputation + missing indicators)
  * ensemble = mean of the best GBM and the ridge model
Baselines reported on the test wells:
  * mean      - training mean
  * previous  - "historical experience": repeat the well's previous cycle outcome
"""
from __future__ import annotations

import itertools
import json
import warnings

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import ROOT, load_pipeline
from .features import FEATURES, cycle_table, inverse_target, transform_target

MODELS = ROOT / "artifacts" / "models"
METRICS = ROOT / "artifacts" / "metrics"
PREV = {"oil_m3": "prev_oil_m3", "prod_days": "prev_prod_days"}

# numpy 2.x linked to Apple Accelerate emits spurious "divide by zero/overflow encountered in
# matmul" warnings on finite inputs (reproducible on a random 50x50 product). Results were
# verified against a BLAS-free Gram-matrix computation (max coefficient difference 1.6e-13),
# see tests/test_models.py::test_ridge_matches_blas_free_solution.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


def split_wells(df, test_fraction, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    tr, te = next(gss.split(df, groups=df["well_id"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = dict(r2=float(r2_score(y, p)), mae=float(mean_absolute_error(y, p)))
    nz = y > 1e-6
    out["mape"] = float(np.mean(np.abs(p[nz] - y[nz]) / y[nz])) if nz.any() else float("nan")
    return out


def _gbm_factory(params, seed):
    return lambda: HistGradientBoostingRegressor(random_state=seed, **params)


def _ridge_factory(alpha):
    return lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                 StandardScaler(), Ridge(alpha=alpha))


class Ensemble:
    def __init__(self, members):
        self.members = members

    def fit(self, X, y):
        for m in self.members:
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.members], axis=0)


def _cv_r2(factory, X, y_t, y_raw, groups, target, folds):
    scores = []
    for tr, va in GroupKFold(n_splits=folds).split(X, y_t, groups):
        m = factory().fit(X.iloc[tr], y_t[tr])
        scores.append(r2_score(y_raw[va], inverse_target(target, m.predict(X.iloc[va]))))
    return float(np.mean(scores)), float(np.std(scores))


def train(df=None, save=True, verbose=False):
    # single-threaded: ~3x faster on these small tables (avoids OpenMP oversubscription)
    with threadpool_limits(1):
        return _train(df, save)


def _train(df, save):
    pl = load_pipeline()
    cfg = pl["css_surrogate"]
    seed = pl["seed"]
    df = cycle_table() if df is None else df
    train_df, test_df = split_wells(df, cfg["test_fraction_wells"], seed)
    X_tr, X_te = train_df[FEATURES], test_df[FEATURES]
    groups = train_df["well_id"].to_numpy()
    grid = [dict(zip(cfg["hgb_grid"], v)) for v in itertools.product(*cfg["hgb_grid"].values())]
    results, models = {}, {}
    for t in cfg["targets"]:
        y_raw = train_df[t].to_numpy(float)
        y_t = transform_target(t, y_raw)
        cv = {}
        best_gbm, best_gbm_score = None, -np.inf
        for params in grid:
            s = _cv_r2(_gbm_factory(params, seed), X_tr, y_t, y_raw, groups, t, cfg["cv_folds"])
            if s[0] > best_gbm_score:
                best_gbm, best_gbm_score, cv["gbm"] = params, s[0], s
        best_alpha, best_ridge_score = None, -np.inf
        for a in cfg["ridge_alphas"]:
            s = _cv_r2(_ridge_factory(a), X_tr, y_t, y_raw, groups, t, cfg["cv_folds"])
            if s[0] > best_ridge_score:
                best_alpha, best_ridge_score, cv["ridge"] = a, s[0], s
        ens_factory = lambda: Ensemble([_gbm_factory(best_gbm, seed)(), _ridge_factory(best_alpha)()])  # noqa: E731
        cv["ensemble"] = _cv_r2(ens_factory, X_tr, y_t, y_raw, groups, t, cfg["cv_folds"])
        factories = {"gbm": _gbm_factory(best_gbm, seed), "ridge": _ridge_factory(best_alpha),
                     "ensemble": ens_factory}
        chosen = max(cv, key=lambda k: cv[k][0])
        test_scores = {}
        fitted = {}
        for name, fac in factories.items():
            fitted[name] = fac().fit(X_tr, y_t)
            test_scores[name] = metrics(test_df[t], inverse_target(t, fitted[name].predict(X_te)))
        test_scores["mean"] = metrics(test_df[t], np.full(len(test_df), y_raw.mean()))
        if t in PREV:
            first = train_df.loc[train_df.cycle_no == 1, t].median()
            test_scores["previous"] = metrics(test_df[t], test_df[PREV[t]].fillna(first).to_numpy())
        results[t] = dict(chosen=chosen, cv_r2={k: dict(mean=v[0], std=v[1]) for k, v in cv.items()},
                          gbm_params=best_gbm, ridge_alpha=best_alpha, test=test_scores)
        models[t] = fitted[chosen]
    oil_pred = inverse_target("oil_m3", models["oil_m3"].predict(X_te))
    results["sor_derived"] = dict(test=dict(selected=metrics(test_df["steam_t"] / test_df["oil_m3"],
                                                             test_df["steam_t"] / np.maximum(oil_pred, 1e-6))))
    info = dict(n_train_cycles=len(train_df), n_test_cycles=len(test_df),
                train_wells=sorted(train_df.well_id.unique()), test_wells=sorted(test_df.well_id.unique()),
                features=FEATURES, seed=seed,
                control_bounds={c: [float(train_df[c].min()), float(train_df[c].max())]
                                for c in ["steam_t", "inj_pressure_MPa", "quality_surface", "soak_days",
                                          "cutoff_m3d", "min_prod_days", "spm_init"]})
    if save:
        MODELS.mkdir(parents=True, exist_ok=True)
        METRICS.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(models=models, features=FEATURES, info=info), MODELS / "css_surrogate.joblib")
        with open(METRICS / "css_surrogate.json", "w") as f:
            json.dump(dict(info=info, results=results), f, indent=2)
        out = test_df[["well_id", "cycle_no"] + cfg["targets"]].copy()
        for t in cfg["targets"]:
            out[f"pred_{t}"] = inverse_target(t, models[t].predict(X_te))
        out.to_csv(METRICS / "css_surrogate_test_predictions.csv", index=False)
    return models, results, info


class CSSSurrogate:
    def __init__(self, path=None):
        b = joblib.load(path or MODELS / "css_surrogate.joblib")
        self.models, self.features, self.info = b["models"], b["features"], b["info"]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.features]
        return pd.DataFrame({t: inverse_target(t, m.predict(X)) for t, m in self.models.items()},
                            index=X.index)


def summary_table(results):
    rows = []
    for t, r in results.items():
        if t == "sor_derived":
            rows.append(dict(target="SOR (derived)", model="selected", **r["test"]["selected"]))
            continue
        for k, v in r["test"].items():
            rows.append(dict(target=t, model=k + (" *" if k == r["chosen"] else ""),
                             cv_r2=r["cv_r2"].get(k, {}).get("mean", np.nan), **v))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    _, res, info = train()
    print(f"train cycles {info['n_train_cycles']} ({len(info['train_wells'])} wells), "
          f"test cycles {info['n_test_cycles']} (wells {info['test_wells']})")
    print(summary_table(res).round(3).to_string(index=False))
    print("(* = chosen by grouped CV on training wells)")
