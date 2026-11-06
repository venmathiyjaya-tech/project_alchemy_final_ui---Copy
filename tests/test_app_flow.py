"""
tests/test_app_flow.py — end-to-end checks of the recommendation flow.

Covers issue 9 (Observatory shows VAD on the right scale) and issue 10
(no duplicate recommendations; /decoding only opens when there is something new).
Uses an in-memory SQLite database and the real routes, templates and recommender.

    pytest tests/test_app_flow.py -v
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app import create_app, db, _seed_scents          # noqa: E402
from app.models import User, JournalEntry, Recommendation  # noqa: E402


class TestConfig:
    __test__ = False                                   # not a test class
    TESTING = True
    SECRET_KEY = 'test-secret'
    SQLALCHEMY_DATABASE_URI = 'sqlite://'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    USE_LEXICON_BLEND = False


@pytest.fixture()
def app():
    application = create_app(TestConfig)
    with application.app_context():
        db.create_all()
        _seed_scents(application)
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    c = app.test_client()
    r = c.post('/auth/register', data={
        'username': 'tester', 'password': 'Test#1234', 'confirm_password': 'Test#1234',
        'recovery_phrase': 'blue lantern'})
    assert r.status_code == 302, 'registration should log the user in and redirect'
    return c


def _uid():
    return User.query.filter_by(username='tester').first().id


def _add_entry(seconds_ahead=0, vad=(0.4, 0.2, 0.3), emotions=None):
    """Add a journal entry; seconds_ahead>0 makes it newer than any recommendation made so far."""
    e = JournalEntry(
        user_id=_uid(), entry_text='A calm and happy day with friends.',
        emotion_vector=json.dumps(emotions or {'joy': 0.8, 'love': 0.3, 'neutral': 0.1, 'sadness': 0.05}),
        vad_valence=vad[0], vad_arousal=vad[1], vad_dominance=vad[2],
        submitted_at=datetime.utcnow() + timedelta(seconds=seconds_ahead))
    db.session.add(e)
    db.session.commit()
    return e


def _rec_count():
    return Recommendation.query.filter_by(user_id=_uid()).count()


# ── issue 10: generate is idempotent ──────────────────────────────────────

def test_generate_without_entries_is_rejected(client):
    r = client.post('/recommend/generate', json={})
    assert r.status_code == 400
    assert _rec_count() == 0


def test_first_generate_creates_one_recommendation(client):
    _add_entry()
    r = client.post('/recommend/generate', json={})
    assert r.status_code == 201
    assert _rec_count() == 1


def test_second_generate_without_a_new_entry_reuses_the_recommendation(client):
    _add_entry()
    first = client.post('/recommend/generate', json={}).get_json()
    second = client.post('/recommend/generate', json={})
    assert second.status_code == 200
    body = second.get_json()
    assert body['reused'] is True
    assert body['rec_id'] == first['rec_id'] and body['scent_id'] == first['scent_id']
    assert _rec_count() == 1


def test_a_new_entry_produces_a_new_recommendation(client):
    _add_entry()
    first = client.post('/recommend/generate', json={}).get_json()
    _add_entry(seconds_ahead=5)
    second = client.post('/recommend/generate', json={})
    assert second.status_code == 201
    assert second.get_json()['rec_id'] != first['rec_id']
    assert _rec_count() == 2


# ── issue 10: /decoding only opens when there is something to decode ─────

def test_decoding_without_entries_goes_to_the_confessional(client):
    r = client.get('/decoding')
    assert r.status_code == 302 and '/journal/write' in r.headers['Location']


def test_decoding_opens_for_a_fresh_entry_then_redirects_once_decoded(client):
    _add_entry()
    assert client.get('/decoding').status_code == 200            # fresh entry: show the chamber
    client.post('/recommend/generate', json={})
    r = client.get('/decoding')                                  # reload after generating
    assert r.status_code == 302 and '/recommend/dashboard' in r.headers['Location']
    _add_entry(seconds_ahead=5)
    assert client.get('/decoding').status_code == 200            # a newer entry reopens it


# ── issue 9 + 10: the Observatory ─────────────────────────────────────────

def test_observatory_maps_vad_from_minus1_1_to_percent(client):
    db.session.add(Recommendation(
        user_id=_uid(), scent_id='VASY_01',
        aggregated_vad=json.dumps({'valence': -0.5, 'arousal': 0.2, 'dominance': 0.0}),
        dominant_emotions=json.dumps(['sadness', 'fear', 'joy'])))
    db.session.commit()
    html = client.get('/recommend/dashboard').get_data(as_text=True)
    assert 'width: 25%' in html      # valence -0.5 -> 25%  (used to collapse to 0%)
    assert 'width: 60%' in html      # arousal   0.2 -> 60%
    assert 'width: 50%' in html      # dominance 0.0 -> 50%


def test_observatory_no_longer_offers_regenerate(client):
    db.session.add(Recommendation(
        user_id=_uid(), scent_id='VASY_01',
        aggregated_vad=json.dumps({'valence': 0.3, 'arousal': 0.3, 'dominance': 0.3}),
        dominant_emotions=json.dumps(['joy'])))
    db.session.commit()
    html = client.get('/recommend/dashboard').get_data(as_text=True)
    assert 'Regenerate Ritual' not in html and 'id="regen-btn"' not in html


# ── voice journaling is present on the Confessional page ──────────────────

def test_confessional_page_has_voice_controls(client):
    html = client.get('/journal/write').get_data(as_text=True)
    for needle in ('id="mic-btn"', 'id="voice-lang"', 'id="readback-btn"',
                   'id="speak-prompt-btn"', 'js/voice.js'):
        assert needle in html, needle


# ── recommendation policy (issues 6 and 7) ────────────────────────────────

def _scent_arousal(scent_id):
    from app.models import Scent
    return db.session.get(Scent, scent_id).vad_arousal


def test_anxious_entry_gets_a_calming_scent_under_the_adaptive_policy(app, client):
    app.config['RECOMMENDATION_POLICY'] = 'adaptive'
    _add_entry(vad=(-0.65, 0.70, -0.50), emotions={'fear': 0.9, 'nervousness': 0.6})
    body = client.post('/recommend/generate', json={}).get_json()
    assert body['policy'] == 'adaptive'
    assert _scent_arousal(body['scent_id']) <= 0.35
    assert 'policy:adaptive' in db.session.get(Recommendation, body['rec_id']).model_version


def test_the_match_policy_keeps_the_original_behaviour(app, client):
    app.config['RECOMMENDATION_POLICY'] = 'match'
    _add_entry(vad=(-0.65, 0.70, -0.50), emotions={'fear': 0.9, 'nervousness': 0.6})
    body = client.post('/recommend/generate', json={}).get_json()
    assert body['policy'] == 'match'
    assert _scent_arousal(body['scent_id']) >= 0.55          # matching an anxious state energises


def test_positive_entries_are_unaffected_by_the_policy(app, client):
    _add_entry(vad=(0.6, 0.2, 0.3))
    app.config['RECOMMENDATION_POLICY'] = 'adaptive'
    adaptive = client.post('/recommend/generate', json={}).get_json()['scent_id']
    Recommendation.query.delete(); db.session.commit()
    app.config['RECOMMENDATION_POLICY'] = 'match'
    matched = client.post('/recommend/generate', json={}).get_json()['scent_id']
    assert adaptive == matched


# ── arousal calibration (switchable) ──────────────────────────────────────

def test_a_low_energy_sad_entry_is_lifted_when_calibrated_and_calmed_when_not(app, client):
    """Sadness is low-energy. Without calibration its arousal looks high, so it is calmed;
    with the arousal offset it is recognised as low-energy and lifted (a livelier scent)."""
    _add_entry(vad=(-0.76, 0.21, -0.42), emotions={'sadness': 0.9, 'grief': 0.4})
    app.config['RECOMMENDATION_POLICY'] = 'adaptive'

    app.config['AROUSAL_OFFSET'] = 0.0
    off = client.post('/recommend/generate', json={}).get_json()
    Recommendation.query.delete(); db.session.commit()

    app.config['AROUSAL_OFFSET'] = 0.25
    on = client.post('/recommend/generate', json={}).get_json()

    assert off['arousal_offset'] == 0.0 and on['arousal_offset'] == 0.25
    assert _scent_arousal(on['scent_id']) > _scent_arousal(off['scent_id'])
    assert db.session.get(Recommendation, on['rec_id']).model_version.endswith('cal:0.25')


def test_a_high_energy_anxious_entry_is_still_calmed_when_calibrated(app, client):
    app.config.update(RECOMMENDATION_POLICY='adaptive', AROUSAL_OFFSET=0.25)
    _add_entry(vad=(-0.70, 0.72, -0.57), emotions={'fear': 0.9, 'nervousness': 0.6})
    body = client.post('/recommend/generate', json={}).get_json()
    assert _scent_arousal(body['scent_id']) <= 0.35
