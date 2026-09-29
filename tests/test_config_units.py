import copy
from pathlib import Path

import pytest
import yaml

from twin.config import FIELD_YAML, ConfigError, FieldConfig, _validate
from twin.units import to_si


def test_every_parameter_has_provenance():
    cfg = FieldConfig()
    assert len(cfg.params) > 50
    for k, p in cfg.params.items():
        assert p["status"] in {"published", "problem_statement", "assumption", "physical_constant"}, k
        assert str(p["source"]).strip(), k


def test_published_values_match_sources():
    cfg = FieldConfig()
    assert cfg.value("reservoir.depth") == 1150.0            # OIL website
    assert cfg.range("oil.viscosity_at_ref") == (10.0, 13.0)  # 10,000-13,000 cP in Pa.s
    assert cfg.value("oil.viscosity_ref_temperature") == 50.0
    assert cfg.value("field.n_producing_wells") == 34
    assert cfg.range("reservoir.temperature") == (46.0, 48.0)  # problem statement
    assert cfg.range("oil.api_gravity") == (17.0, 19.0)


def test_unknown_unit_raises():
    with pytest.raises(ValueError):
        to_si(1.0, "furlong")


def test_validation_rejects_missing_source_and_string_numbers():
    raw = yaml.safe_load(Path(FIELD_YAML).read_text())
    bad = copy.deepcopy(raw)
    del bad["reservoir"]["depth"]["source"]
    with pytest.raises(ConfigError):
        _validate(bad)
    bad = copy.deepcopy(raw)
    bad["reservoir"]["depth"]["value"] = "2.4e6"
    with pytest.raises(ConfigError):
        _validate(bad)


def test_si_conversions():
    assert to_si(1.0, "mD") == pytest.approx(9.869233e-16)
    assert to_si(150.0, "t/d") == pytest.approx(150e3 / 86400)
    assert to_si(1.0, "cP") == pytest.approx(1e-3)
