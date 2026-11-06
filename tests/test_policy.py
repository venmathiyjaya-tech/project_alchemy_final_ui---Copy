"""tests/test_policy.py — the recommendation policy (no model or database needed)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.recommender.policy import (  # noqa: E402
    G_CALM, G_LIFT, POLICIES, apply_policy, regulation_weight,
)

SCENTS = json.loads((_ROOT / 'data' / 'scents.json').read_text(encoding='utf-8'))
PROTOS = np.array([[s['valence'], s['arousal'], s['dominance']] for s in SCENTS])


def vad(v, a, d):
    return {'valence': v, 'arousal': a, 'dominance': d}


def vec(t):
    return np.array([t['valence'], t['arousal'], t['dominance']])


def test_match_returns_the_users_own_state():
    s = vad(0.18, 0.85, 0.20)
    assert apply_policy(s, 'match') == s


@pytest.mark.parametrize('state', [vad(0.5, 0.5, 0.5), vad(0.9, 0.88, 0.8), vad(0.7, 0.2, 0.4)])
def test_neutral_and_positive_states_are_matched_in_adaptive_mode(state):
    assert apply_policy(state, 'adaptive') == state


def test_anxious_state_is_moved_toward_calm():
    anxious = vad(0.18, 0.85, 0.20)
    t = apply_policy(anxious, 'adaptive')
    assert t['arousal'] < 0.40 < anxious['arousal']        # calmed down
    assert t['valence'] > anxious['valence']               # and lifted


def test_sad_low_energy_state_is_lifted_not_energised():
    sad = vad(0.15, 0.14, 0.22)
    t = apply_policy(sad, 'adaptive')
    assert t['valence'] > 0.7                              # lifted toward the warm goal
    assert 0.14 < t['arousal'] < 0.5                       # gently, not to peak energy


def test_regulation_weight_fades_in_with_negativity():
    ws = [regulation_weight(vad(v, 0.5, 0.5)) for v in (0.6, 0.5, 0.45, 0.375, 0.3, 0.25, 0.1)]
    assert ws == sorted(ws)                                # never decreases as valence falls
    assert ws[0] == ws[1] == 0.0 and ws[-1] == 1.0


def test_small_input_changes_never_cause_large_target_jumps():
    worst = max(
        np.linalg.norm(vec(apply_policy(vad(v, 0.6, 0.4))) - vec(apply_policy(vad(v + 0.01, 0.6, 0.4))))
        for v in np.linspace(0.05, 0.6, 200))
    assert worst < 0.03


def test_strength_zero_equals_match():
    s = vad(0.1, 0.8, 0.2)
    assert apply_policy(s, 'adaptive', strength=0.0) == apply_policy(s, 'match')


def test_outputs_stay_inside_the_unit_cube():
    for v in (0, 0.2, 0.5, 1):
        for a in (0, 0.5, 1):
            t = apply_policy(vad(v, a, 0.5))
            assert all(0.0 <= x <= 1.0 for x in t.values())


def test_unknown_policy_is_rejected():
    assert POLICIES == ('match', 'adaptive')
    with pytest.raises(ValueError):
        apply_policy(vad(0.5, 0.5, 0.5), 'soothe')


@pytest.mark.parametrize('goal', [G_CALM, G_LIFT])
def test_goals_sit_inside_the_catalogue(goal):
    """If the catalogue changes, re-derive the goals (see policy.py) rather than ignore this."""
    assert np.linalg.norm(PROTOS - np.array(goal), axis=1).min() < 0.08


def test_adaptive_changes_which_scent_an_anxious_state_gets():
    anxious = vad(0.18, 0.85, 0.20)
    near = lambda t: SCENTS[int(np.linalg.norm(PROTOS - vec(t), axis=1).argmin())]
    matched, regulated = near(apply_policy(anxious, 'match')), near(apply_policy(anxious, 'adaptive'))
    assert matched['arousal'] >= 0.6          # matching an anxious state gives an energising scent
    assert regulated['arousal'] <= 0.35       # regulating gives a calming one
