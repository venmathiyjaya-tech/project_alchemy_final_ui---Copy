"""
tests/test_emotion.py — real assertions for the emotion classifier.

Most tests need no model and run instantly. The few that run the network are
skipped automatically when models/roberta_finetuned is missing.

    pytest tests/test_emotion.py -v
"""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.nlp.emotion_classifier import EmotionClassifier  # noqa: E402

MODEL_DIR = _ROOT / 'models' / 'roberta_finetuned'


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def bare():
    """An instance without a model: enough for the pure-logic methods."""
    return EmotionClassifier.__new__(EmotionClassifier)


@pytest.fixture(scope='module')
def clf():
    if not (MODEL_DIR / 'config.json').is_file():
        pytest.skip(f'No model at {MODEL_DIR}')
    return EmotionClassifier(str(MODEL_DIR))


# ── label tables (no model needed) ────────────────────────────────────────

def test_canonical_and_legacy_label_sets_match():
    assert len(EmotionClassifier.LABELS) == 28
    assert set(EmotionClassifier.LABELS) == set(EmotionClassifier.LEGACY_LABELS)


def test_label_orders_are_the_two_known_layouts():
    assert EmotionClassifier.LABELS[27] == 'neutral'          # GoEmotions / SamLowe order
    assert EmotionClassifier.LEGACY_LABELS[20] == 'neutral'   # old notebook order


def test_vad_table_covers_every_label_within_range():
    assert set(EmotionClassifier.VAD_TABLE) == set(EmotionClassifier.LABELS)
    for label, vad in EmotionClassifier.VAD_TABLE.items():
        assert len(vad) == 3 and all(-1.0 <= x <= 1.0 for x in vad), label


def test_emotion_groups_partition_all_labels():
    members = [m for group in EmotionClassifier.EMOTION_GROUPS.values() for m in group]
    assert sorted(members) == sorted(EmotionClassifier.LABELS)   # every label in exactly one group


# ── pure logic (no model needed) ──────────────────────────────────────────

def test_predict_vad_of_a_single_emotion_is_that_emotions_vad(bare):
    v, a, d = EmotionClassifier.VAD_TABLE['joy']
    out = bare.predict_vad({'joy': 0.9})
    assert (out['valence'], out['arousal'], out['dominance']) == (round(v, 4), round(a, 4), round(d, 4))


def test_predict_vad_of_empty_distribution_is_neutral(bare):
    assert bare.predict_vad({l: 0.0 for l in EmotionClassifier.LABELS}) == \
        {'valence': 0.0, 'arousal': 0.0, 'dominance': 0.0}


def test_group_emotions_takes_the_max_member(bare):
    probs = {l: 0.0 for l in EmotionClassifier.LABELS}
    probs['gratitude'], probs['joy'] = 0.7, 0.2
    grouped = bare.group_emotions(probs)
    assert set(grouped) == set(EmotionClassifier.EMOTION_GROUPS)
    assert grouped['joy'] == 0.7


@pytest.mark.parametrize('text', ['', '   ', 'ab', '12345 67890 11111', 'a a a'])
def test_unusable_entries_are_rejected(bare, text):
    with pytest.raises(EmotionClassifier.InvalidEntryError):
        bare._validate_text(text.strip())


def test_a_normal_sentence_passes_validation(bare):
    bare._validate_text('I felt calm and happy today.')


# ── model-dependent (skipped without the model folder) ────────────────────

def test_model_uses_labels_json_with_neutral_last(clf):
    assert (MODEL_DIR / 'labels.json').is_file(), 'Copy the v5 folder (it contains labels.json).'
    assert len(clf.labels) == 28 and clf.labels.index('neutral') == 27


def test_predict_returns_28_probabilities_in_unit_range(clf):
    probs = clf.predict('I felt overwhelmed today but somehow hopeful.')
    assert set(probs) == set(clf.labels)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_positive_text_has_higher_valence_than_negative_text(clf):
    pos = clf.predict_vad(clf.predict('Today was absolutely amazing. I got a promotion!'))
    neg = clf.predict_vad(clf.predict('I feel so lost and alone. Nothing is going right.'))
    assert pos['valence'] > neg['valence']
    assert pos['valence'] > 0 > neg['valence']
