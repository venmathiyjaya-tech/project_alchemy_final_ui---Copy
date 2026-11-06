"""tests/test_calibration.py — range conversion and the switchable arousal offset."""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.recommender.calibration import DEFAULT_AROUSAL_OFFSET, to_scent_space  # noqa: E402


def raw(v, a, d):
    return {"valence": v, "arousal": a, "dominance": d}


def test_zero_offset_is_exactly_the_original_range_conversion():
    out = to_scent_space(raw(-0.65, 0.70, -0.50))
    assert out == {"valence": 0.175, "arousal": 0.85, "dominance": 0.25}


@pytest.mark.parametrize("v", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_range_conversion_maps_minus1_1_onto_0_1(v):
    out = to_scent_space(raw(v, v, v))
    assert out["valence"] == pytest.approx((v + 1) / 2) and out["dominance"] == pytest.approx((v + 1) / 2)


def test_offset_changes_only_arousal():
    base, shifted = to_scent_space(raw(0.4, 0.6, -0.2)), to_scent_space(raw(0.4, 0.6, -0.2), 0.25)
    assert shifted["valence"] == base["valence"] and shifted["dominance"] == base["dominance"]
    assert shifted["arousal"] == pytest.approx(base["arousal"] - 0.25)


def test_arousal_is_clipped_at_zero():
    assert to_scent_space(raw(0, -0.9, 0), 0.25)["arousal"] == 0.0


def test_inputs_outside_the_range_are_clipped():
    out = to_scent_space(raw(2.0, -3.0, 9.0))
    assert out == {"valence": 1.0, "arousal": 0.0, "dominance": 1.0}


def test_default_offset_is_the_documented_value():
    assert DEFAULT_AROUSAL_OFFSET == 0.25


def test_missing_key_is_rejected():
    with pytest.raises(KeyError):
        to_scent_space({"valence": 0.1, "arousal": 0.1})
