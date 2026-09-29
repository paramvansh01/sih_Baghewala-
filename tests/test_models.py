"""ML hygiene tests. Metric tests need trained artifacts (python -m twin.cli train)."""
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from twin.config import ROOT
from twin.models.features import FEATURES, HISTORY, STATIC

METRICS = ROOT / "artifacts" / "metrics"
needs = pytest.mark.skipif(not (METRICS / "css_surrogate.json").exists(), reason="train models first")


def test_no_hidden_state_or_outcomes_in_features():
    hidden = {"skin", "n_rem", "r_h", "r_heated", "t_prod_start_true", "x_bottom", "q_heat_MW",
              "oil_m3", "water_m3", "prod_days", "energy_kwh", "sor", "floating_days", "peak_oil_m3d",
              "t_bottom_start", "t_bottom_end", "end_day", "overload_days"}
    assert not hidden & set(FEATURES)
    assert not [f for f in FEATURES if f.endswith("_true")]
    # every history feature refers to the past (prev_*) or cumulative-before-cycle values
    assert all(f.startswith(("prev_", "cum_")) or f in ("cycle_no", "cold_oil_m3d") for f in HISTORY)


def test_ridge_matches_blas_free_solution():
    """Guards the suppression of Accelerate's spurious matmul warnings: BLAS results must equal a
    pure-python Gram-matrix solve."""
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 12))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = z @ rng.standard_normal(12) + 0.1 * rng.standard_normal(200)
        coef = Ridge(alpha=1.0).fit(z, y).coef_
    zc, yc = z - z.mean(0), y - y.mean()
    g = np.array([[sum(a * b for a, b in zip(zc[:, i], zc[:, j])) for j in range(12)] for i in range(12)])
    h = np.array([sum(a * b for a, b in zip(zc[:, i], yc)) for i in range(12)])
    ref = np.linalg.solve(g + np.eye(12), h)
    assert np.allclose(coef, ref, atol=1e-10)


@needs
def test_surrogate_split_is_by_well_and_beats_baselines():
    m = json.loads(Path(METRICS / "css_surrogate.json").read_text())
    info, res = m["info"], m["results"]
    assert not set(info["train_wells"]) & set(info["test_wells"])
    oil = res["oil_m3"]["test"]
    chosen = res["oil_m3"]["chosen"]
    assert oil[chosen]["r2"] > oil["mean"]["r2"] + 0.3
    assert oil[chosen]["r2"] > oil["previous"]["r2"]
    # control bounds for the optimiser come from the training data
    assert set(info["control_bounds"]) >= {"steam_t", "inj_pressure_MPa", "soak_days"}


@needs
def test_cnn_trained_from_scratch():
    r = json.loads(Path(METRICS / "card_cnn.json").read_text())
    # random initialisation: initial validation loss is at chance level ln(7)
    assert abs(r["initial_val_loss"] - r["chance_loss"]) < 0.1
    assert r["best_val_loss"] < 0.2 * r["chance_loss"]
    assert r["cnn"]["accuracy"] > r["logistic_regression"]["accuracy"]
    assert r["n_train"] + r["n_val"] + r["n_test"] == sum(sum(row) for row in r["cnn"]["confusion_matrix"]) \
        + r["n_train"] + r["n_val"]
