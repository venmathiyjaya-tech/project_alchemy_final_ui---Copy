import os
import json
from datetime import datetime, timezone, timedelta

from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user

from app import db
from app.models import JournalEntry
from ml.nlp.emotion_classifier import EmotionClassifier
from ml.nlp.vad_lexicon import VADLexicon, blend_vad

journal = Blueprint('journal', __name__, url_prefix='/journal')

# ---------------------------------------------------------------------------
# Module-level classifier cache — loaded once per process on first request.
# ---------------------------------------------------------------------------
_classifier: EmotionClassifier | None = None


def _get_classifier() -> EmotionClassifier:
    """Return a cached EmotionClassifier, loading it on first call."""
    global _classifier
    if _classifier is None:
        local_model_path = os.path.normpath(
            os.path.join(current_app.root_path, '..', 'models', 'roberta_finetuned')
        )
        model_name = local_model_path if os.path.isdir(local_model_path) else 'roberta-base'
        current_app.logger.info('Loading EmotionClassifier from: %s', model_name)
        _classifier = EmotionClassifier(model_name)
    return _classifier


_lexicon: VADLexicon | None = None


def _get_lexicon() -> VADLexicon | None:
    """
    Return the cached NRC VAD lexicon, or None when blending is switched off
    (config USE_LEXICON_BLEND = False). Raises FileNotFoundError with download
    instructions if blending is on but the lexicon file is missing.
    """
    global _lexicon
    if not current_app.config.get('USE_LEXICON_BLEND', True):
        return None
    if _lexicon is None:
        path = current_app.config.get('NRC_VAD_PATH', 'data/NRC-VAD-Lexicon.txt')
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(current_app.root_path, '..', path))
        current_app.logger.info('Loading NRC VAD lexicon from: %s', path)
        _lexicon = VADLexicon(path)
    return _lexicon


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@journal.route('/', methods=['GET'])
@login_required
def index():
    """Render the dashboard page with the user's last 7 entries."""
    entries = (
        JournalEntry.query
        .filter_by(user_id=current_user.id)
        .order_by(JournalEntry.submitted_at.desc())
        .limit(7)
        .all()
    )
    recommendation = None
    return render_template(
        'dashboard/dashboard.html',
        entries=entries,
        recommendation=recommendation,
    )


@journal.route('/write', methods=['GET'])
@login_required
def write_page():
    """Render the journal entry page."""
    # FIX (Bug 5): pass entries to the template so {% if not entries %} and
    # {% for entry in entries %} don't raise a Jinja UndefinedError.
    entries = (
        JournalEntry.query
        .filter_by(user_id=current_user.id)
        .order_by(JournalEntry.submitted_at.desc())
        .limit(7)
        .all()
    )
    return render_template('journal/entry.html', entries=entries)


@journal.route('/entry', methods=['POST'])
@login_required
def create_entry():
    """
    Accept a JSON body {entry_text: str}, run emotion + VAD prediction,
    persist a JournalEntry, and return the saved data as JSON.
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': 'Request body must be JSON.'}), 400

    entry_text = body.get('entry_text', '')

    if not entry_text or not entry_text.strip():
        return jsonify({'error': 'entry_text must not be empty.'}), 422

    if len(entry_text) > 2000:
        return jsonify({
            'error': f'entry_text must not exceed 2000 characters (got {len(entry_text)}).'
        }), 422

    try:
        clf = _get_classifier()
        # predict() now validates text first — raises InvalidEntryError on gibberish
        emotions         = clf.predict(entry_text)
        # group_emotions() collapses 27 labels → 8 parent categories for display
        grouped_emotions = clf.group_emotions(emotions)
        vad              = clf.predict_vad(emotions)   # VAD still uses all 27 labels

        # Lexicon signal (Mohammad, 2018): blended 65/35 with the model VAD, with
        # the lexicon's weight shrinking automatically when few words match.
        lexicon = _get_lexicon()
        if lexicon is not None:
            vad = blend_vad(
                vad,
                lexicon.score(entry_text),
                model_weight=current_app.config.get('LEXICON_MODEL_WEIGHT', 0.65),
            )
    except FileNotFoundError as exc:
        # A missing model/lexicon file is a deployment problem, not a user problem.
        current_app.logger.error('Required file missing: %s', exc)
        return jsonify({'error': 'Server configuration error. Please contact the administrator.'}), 500
    except EmotionClassifier.InvalidEntryError as exc:
        # Text failed quality checks — return readable 422, not a 500
        return jsonify({'error': str(exc)}), 422
    except Exception as exc:
        current_app.logger.exception('EmotionClassifier failed: %s', exc)
        return jsonify({'error': 'Emotion analysis failed. Please try again later.'}), 500

    entry = JournalEntry(
        user_id=current_user.id,
        entry_text=entry_text,
        emotion_vector=json.dumps(grouped_emotions),   # store 8 groups for display
        vad_valence=vad['valence'],
        vad_arousal=vad['arousal'],
        vad_dominance=vad['dominance'],
        submitted_at=datetime.now(timezone.utc),
    )
    db.session.add(entry)
    db.session.commit()

    return jsonify({
        'entry_id':       entry.id,
        'emotion_vector': entry.emotion_vector,   # 8-group JSON string
        'vad_valence':    entry.vad_valence,
        'vad_arousal':    entry.vad_arousal,
        'vad_dominance':  entry.vad_dominance,
        'submitted_at':   entry.submitted_at.isoformat(),
        'message':        'Entry saved',
    }), 201


@journal.route('/entry/<int:entry_id>', methods=['DELETE'])
@login_required
def delete_entry(entry_id: int):
    """
    Delete a journal entry belonging to the current user,
    provided it was submitted within the last 24 hours.
    """
    entry = db.session.get(JournalEntry, entry_id)

    if entry is None or entry.user_id != current_user.id:
        return jsonify({'error': 'Entry not found.'}), 404

    cutoff    = datetime.now(timezone.utc) - timedelta(hours=24)
    submitted = entry.submitted_at
    if submitted.tzinfo is None:
        submitted = submitted.replace(tzinfo=timezone.utc)

    if submitted < cutoff:
        return jsonify({'error': 'Entries can only be deleted within 24 hours of submission.'}), 403

    db.session.delete(entry)
    db.session.commit()
    return jsonify({'success': True, 'deleted_entry_id': entry_id}), 200


@journal.route('/entries', methods=['GET'])
@login_required
def list_entries():
    """Return the current user's last 7 journal entries as a JSON list."""
    entries = (
        JournalEntry.query
        .filter_by(user_id=current_user.id)
        .order_by(JournalEntry.submitted_at.desc())
        .limit(7)
        .all()
    )

    return jsonify([
        {
            'entry_id':       e.id,
            'entry_text':     e.entry_text,
            'emotion_vector': e.emotion_vector,
            'vad_valence':    e.vad_valence,
            'vad_arousal':    e.vad_arousal,
            'vad_dominance':  e.vad_dominance,
            'submitted_at':   e.submitted_at.isoformat(),
        }
        for e in entries
    ]), 200