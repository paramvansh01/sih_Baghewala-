"""Feature definitions. ONLY quantities an operator can know BEFORE a cycle starts:
well/lab data, the well's own production history, and the planned operating controls.
Hidden simulator state (skin, saturations, heated radius, *_true columns) is excluded."""
import numpy as np
import pandas as pd

from ..config import ROOT

DATA = ROOT / "artifacts" / "data"

STATIC = ["k_mD", "h", "phi", "s_oi", "r_e", "p_i_MPa", "t_r", "api", "mu_ref_cP", "walther_b",
          "stroke", "pump_depth"]
HISTORY = ["cycle_no", "cum_oil", "cum_water", "cum_steam_t", "prev_oil_m3", "prev_sor",
           "prev_prod_days", "prev_t_bottom_end", "prev_peak_oil_m3d", "prev_floating_days",
           "cold_oil_m3d"]
CONTROLS = ["steam_t", "inj_pressure_MPa", "quality_surface", "soak_days", "cutoff_m3d",
            "min_prod_days", "spm_init"]
FEATURES = STATIC + HISTORY + CONTROLS
TARGETS = ["oil_m3", "water_m3", "prod_days", "energy_kwh", "floating_days"]
LOG_TARGETS = {"oil_m3", "water_m3", "energy_kwh", "floating_days"}


def well_static(wells: pd.DataFrame) -> pd.DataFrame:
    w = wells.copy()
    w["k_mD"] = w["k"] / 9.869233e-16
    w["p_i_MPa"] = w["p_i"] / 1e6
    w["mu_ref_cP"] = w["mu_ref"] * 1e3
    return w[["well_id"] + STATIC]


def cycle_table(data_dir=DATA) -> pd.DataFrame:
    cycles = pd.read_csv(data_dir / "cycles.csv")
    wells = pd.read_csv(data_dir / "wells.csv")
    cyc = cycles.drop(columns=[c for c in ("cold_oil_m3d",) if c in cycles and c in wells], errors="ignore")
    df = cyc.merge(well_static(wells), on="well_id", how="left")
    df = df.merge(wells[["well_id", "cold_oil_m3d"]], on="well_id", how="left")
    return df


def transform_target(name, y):
    return np.log1p(y) if name in LOG_TARGETS else np.asarray(y, float)


def inverse_target(name, y):
    return np.expm1(y) if name in LOG_TARGETS else np.asarray(y, float)
