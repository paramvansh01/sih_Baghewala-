"""Loads config/field.yaml, validates provenance of every parameter, converts to SI."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .units import to_si

ROOT = Path(__file__).resolve().parents[1]
FIELD_YAML = ROOT / "config" / "field.yaml"
PIPELINE_YAML = ROOT / "config" / "pipeline.yaml"
VALID_STATUS = {"published", "problem_statement", "assumption", "physical_constant"}


class ConfigError(ValueError):
    pass


def _walk(node, prefix=""):
    for k, v in node.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and "status" in v:
            yield path, v
        elif isinstance(v, dict):
            yield from _walk(v, path)
        else:
            raise ConfigError(f"{path}: every parameter must be a mapping with status/source/unit")


def _validate(raw):
    for path, p in _walk(raw):
        missing = {"status", "source", "unit"} - p.keys()
        if missing:
            raise ConfigError(f"{path}: missing {sorted(missing)}")
        if p["status"] not in VALID_STATUS:
            raise ConfigError(f"{path}: status '{p['status']}' not in {sorted(VALID_STATUS)}")
        if ("value" in p) == ("range" in p):
            raise ConfigError(f"{path}: needs exactly one of 'value' or 'range'")
        if "range" in p:
            lo, hi = p["range"]
            if not lo <= hi:
                raise ConfigError(f"{path}: range low > high")
            if p.get("dist", "uniform") not in ("uniform", "loguniform"):
                raise ConfigError(f"{path}: unknown dist {p.get('dist')}")
        vals = p["range"] if "range" in p else p["value"]
        vals = vals if isinstance(vals, list) else [vals]
        if p["unit"] != "date" and not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                           for v in vals):
            raise ConfigError(f"{path}: non-numeric value {vals!r} (YAML needs e.g. 2.4e+6)")
        if not str(p["source"]).strip():
            raise ConfigError(f"{path}: empty source")


class FieldConfig:
    """Access parameters by dotted path, always returned in SI (temperatures in degC)."""

    def __init__(self, path: Path | str = FIELD_YAML):
        with open(path) as f:
            self.raw = yaml.safe_load(f)
        _validate(self.raw)
        self.params = dict(_walk(self.raw))

    def _p(self, key):
        if key not in self.params:
            raise KeyError(f"parameter '{key}' not defined in field.yaml")
        return self.params[key]

    def value(self, key):
        p = self._p(key)
        if "value" not in p:
            raise ConfigError(f"{key} is a range; use .range() or .sample()")
        return to_si(p["value"], p["unit"])

    def range(self, key):
        p = self._p(key)
        if "range" not in p:
            raise ConfigError(f"{key} is a single value; use .value()")
        return tuple(to_si(p["range"], p["unit"]))

    def mid(self, key):
        """Central value: the value, or the (geometric for loguniform) mid of the range."""
        p = self._p(key)
        if "value" in p:
            return self.value(key)
        lo, hi = self.range(key)
        return float(np.sqrt(lo * hi)) if p.get("dist") == "loguniform" else 0.5 * (lo + hi)

    def sample(self, key, rng: np.random.Generator):
        p = self._p(key)
        if "value" in p:
            return self.value(key)
        lo, hi = self.range(key)
        if p.get("dist") == "loguniform":
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        return float(rng.uniform(lo, hi))

    def provenance_table(self):
        rows = []
        for k, p in self.params.items():
            rows.append({
                "parameter": k,
                "value": p.get("value", p.get("range")),
                "unit": p["unit"],
                "status": p["status"],
                "source": p["source"],
            })
        return rows


def load_pipeline(path: Path | str = PIPELINE_YAML) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
