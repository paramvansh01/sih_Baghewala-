"""Smoke test of the live engine (needs trained artifacts)."""
import json

import pytest

from twin.config import ROOT

needs = pytest.mark.skipif(not (ROOT / "artifacts" / "models" / "card_cnn.pt").exists(),
                           reason="train models first")


@needs
def test_live_engine_ticks_and_snapshot_is_valid_json():
    from twin.live.engine import FLEETS, LiveField
    from twin.live.server import _dumps
    f = LiveField()
    start = f.date
    for _ in range(25):
        f.tick()
    assert (f.date - start).days == 25
    snap = json.loads(_dumps(f.snapshot))  # strict JSON (no NaN)
    assert set(snap["kpis"]) == set(FLEETS)
    assert len(snap["wells"]) == len(f.fleets["twin"])
    # both fleets started from the identical state
    assert [w.id for w in f.fleets["practice"]] == [w.id for w in f.fleets["twin"]]
    # every twin cycle started so far used physics-verified CSS settings
    assert all(w.css_plan is not None for w in f.fleets["twin"] if w.controls is not None and not w.shut_in)
