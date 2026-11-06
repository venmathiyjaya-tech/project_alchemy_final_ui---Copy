"""
ml/recommender/policy.py — what should the scent do for the user's state?

The classifier answers "which scent sits closest to this VAD point?". That is the
right question when the user feels good, but a poor one when they do not:
matching an anxious, high-arousal state returned a stimulating scent
(Basil & Peppermint), and the library holds few scents anywhere near sad or
angry states.

This module decides the VAD point the recommender should aim at:

  match     aim at the user's own state (the original behaviour)
  adaptive  neutral or positive state  -> match
            clearly negative state     -> move the aim toward a calming goal
                                          (if arousal is high) or an uplifting
                                          goal (if arousal is low)

The shift fades in smoothly as valence falls from 0.5 to 0.25, so a small change
in the entry never flips the recommendation abruptly.

All values are on the [0, 1] scale used by the scent library. The two goals are
the centroids of scents the Vasya catalogue itself tags as calming / uplifting
(see GOALS below), not invented numbers. That adaptive regulation helps users is
a DESIGN HYPOTHESIS: it has not been validated, and user ratings are how it
will be tested.
"""
from __future__ import annotations

from typing import Mapping

POLICIES = ("match", "adaptive")

# Goals as (valence, arousal, dominance) on [0, 1], derived from the catalogue:
#   G_CALM : mean of scents tagged calm / relaxation / soothed / peaceful with
#            arousal <= 0.30 and valence >= 0.65  (VASY_01, 04, 07, 10, 12, 13, 18, 24, 26, 28)
#   G_LIFT : mean of scents tagged joy / well-being / happiness / sensory pleasure /
#            caring with arousal 0.28-0.50 and valence >= 0.75  (VASY_05, 06, 13, 21, 23)
G_CALM = (0.75, 0.25, 0.60)
G_LIFT = (0.83, 0.43, 0.61)

NEUTRAL_VALENCE = 0.50     # at or above this the user's state is matched as is
FADE_SPAN = 0.25           # full regulation at valence <= NEUTRAL_VALENCE - FADE_SPAN
AROUSAL_LOW = 0.35         # below this arousal the uplifting goal is used
AROUSAL_SPAN = 0.30        # above LOW + SPAN the calming goal is used (blended between)

_KEYS = ("valence", "arousal", "dominance")


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def regulation_weight(vad: Mapping[str, float], strength: float = 1.0) -> float:
    """How strongly to regulate, 0 (match) to `strength` (full), from valence only."""
    drop = (NEUTRAL_VALENCE - _clip01(vad["valence"])) / FADE_SPAN
    return _clip01(drop) * _clip01(strength)


def apply_policy(vad: Mapping[str, float], mode: str = "adaptive", strength: float = 1.0) -> dict:
    """Return the VAD point to recommend for (dict with valence/arousal/dominance)."""
    if mode not in POLICIES:
        raise ValueError(f"Unknown recommendation policy '{mode}'. Choose from {POLICIES}.")
    for k in _KEYS:
        if k not in vad:
            raise KeyError(f"vad is missing required key: '{k}'")

    user = [_clip01(vad[k]) for k in _KEYS]
    if mode == "match":
        return dict(zip(_KEYS, (round(x, 6) for x in user)))

    w = regulation_weight(vad, strength)
    # arousal decides the goal: aroused -> calm down, low energy -> lift
    hi = _clip01((user[1] - AROUSAL_LOW) / AROUSAL_SPAN)
    goal = [hi * c + (1.0 - hi) * l for c, l in zip(G_CALM, G_LIFT)]
    target = [u + w * (g - u) for u, g in zip(user, goal)]
    return dict(zip(_KEYS, (round(_clip01(x), 6) for x in target)))
