"""
ml/recommender/calibration.py — express the user's emotional state in the scent library's
coordinate system.

Two different things were mismatched between the emotion side and the scent side:

1. RANGE. The classifier works on [-1, 1]; the scent library is on [0, 1].
   Conversion: (x + 1) / 2. Stored values (journal entries, recommendations) stay on
   [-1, 1]; only the value handed to the recommender is converted, here, in one place.

2. AROUSAL OFFSET (switchable). The two tables were written separately, with different ideas
   of what "energetic" means. For the five emotions that also appear as scent tags (caring,
   excitement, joy, neutral, pride), the tagged scents sit 0.15 to 0.34 lower in arousal than
   the emotion table does (mean 0.24). Left uncorrected, almost every mood looks energetic
   next to the library (emotions average arousal 0.72, scents 0.45) and the recommender keeps
   choosing energising scents; it also makes low-energy negative moods such as sadness look
   high-arousal, so the policy calms them instead of lifting them.
   The fix is one number subtracted from arousal. Valence and dominance are left alone: the
   library is pleasant by nature, so shifting valence would hide how a user really feels.

DEFAULT_AROUSAL_OFFSET = 0.25 was chosen as the 0.24 measured above, rounded. It sits below
~0.27, where a neutral state starts landing on the sleepy scent. It is a judgement from five
tag-matched labels, not a validated value; set it to 0 to switch calibration off.
"""
from __future__ import annotations

from typing import Mapping

DEFAULT_AROUSAL_OFFSET = 0.25
_KEYS = ("valence", "arousal", "dominance")


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def to_scent_space(raw_vad: Mapping[str, float], arousal_offset: float = 0.0) -> dict:
    """
    Convert a VAD point from classifier space ([-1, 1]) to scent-library space ([0, 1]),
    then subtract `arousal_offset` from arousal (clipped to [0, 1]).

    With arousal_offset = 0 this is exactly the original (x + 1) / 2 conversion.
    """
    for k in _KEYS:
        if k not in raw_vad:
            raise KeyError(f"vad is missing required key: '{k}'")
    offset = _clip(arousal_offset, 0.0, 1.0)

    out = {k: (_clip(raw_vad[k], -1.0, 1.0) + 1.0) / 2.0 for k in _KEYS}
    out["arousal"] = _clip(out["arousal"] - offset, 0.0, 1.0)
    return {k: round(v, 6) for k, v in out.items()}
