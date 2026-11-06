import json
import os
from datetime import datetime, timezone, timedelta

from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user

from app import db
from app.models import JournalEntry, Recommendation, Rating
from ml.temporal.aggregator import TemporalAggregator
from ml.recommender.recommender import MLRecommender
from ml.recommender.policy import apply_policy
from ml.recommender.calibration import to_scent_space

recommend = Blueprint('recommend', __name__, url_prefix='/recommend')

# ---------------------------------------------------------------------------
# Module-level recommender cache — loaded once per process on first use.
# ---------------------------------------------------------------------------
_recommender: MLRecommender | None = None


def _get_recommender() -> MLRecommender:
    """Return a cached MLRecommender, loading it from disk on first call."""
    global _recommender
    if _recommender is None:
        # 1. Point to the top-level data folder where scents.json lives
        scents_path = os.path.normpath(
            os.path.join(current_app.root_path, '..', 'data', 'scents.json')
        )
        
        # 2. Point to the top-level models folder where saved_model.joblib lives
        model_path = os.path.normpath(
            os.path.join(current_app.root_path, '..', 'models', 'saved_model.joblib')
        )
        
        current_app.logger.info('Loading MLRecommender from: %s', model_path)
        
        # Instantiate the engine
        _recommender = MLRecommender(scents_path=scents_path, model_path=model_path)
        
    return _recommender


def _as_utc(dt):
    """SQLite hands back naive datetimes (stored as UTC); make them comparable."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _generate_payload(rec: Recommendation) -> dict:
    """The JSON the frontend gets back from /recommend/generate for an existing rec."""
    return {
        'rec_id':                     rec.id,
        'scent_id':                   rec.scent_id,
        'scent_name':                 rec.scent.name if rec.scent else None,
        'scent_description':          rec.scent.description if rec.scent else '',
        'aggregated_vad':             json.loads(rec.aggregated_vad) if rec.aggregated_vad else {},
        'dominant_emotions':          json.loads(rec.dominant_emotions) if rec.dominant_emotions else [],
        'scientific_prove_reference': rec.scent.scientific_prove_reference if rec.scent else '',
    }


def _recommendation_to_dict(rec: Recommendation, rating: Rating | None = None,
                             next_rec_generated_at=None) -> dict:
    """Serialise a Recommendation (+ optional Rating) to a JSON-safe dict.

    next_rec_generated_at: the generated_at of the chronologically NEXT
    recommendation for this user (None if rec is the latest). Used to
    compute whether the rating window is still open.
    """
    fallback_deadline = rec.generated_at + timedelta(hours=48)
    deadline = min(next_rec_generated_at, fallback_deadline) if next_rec_generated_at else fallback_deadline

    now = datetime.now(timezone.utc)
    gen_at = rec.generated_at
    if gen_at.tzinfo is None:
        gen_at = gen_at.replace(tzinfo=timezone.utc)
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)

    rating_open = now <= deadline

    data = {
        'rec_id':                     rec.id,
        'scent_id':                   rec.scent_id,
        'scent_name':                 rec.scent.name if rec.scent else None,
        'scent_description':          rec.scent.description if rec.scent else None,
        'aggregated_vad':             rec.aggregated_vad,
        'dominant_emotions':          json.loads(rec.dominant_emotions) if rec.dominant_emotions else [],
        'scientific_prove_reference': rec.scent.scientific_prove_reference if rec.scent else None,
        'created_at':                 rec.generated_at.isoformat(),
        'rating_open':                rating_open,
        'rating_deadline':            deadline.isoformat(),
    }
    if rating is not None:
        data['rating'] = {
            'rating_id': rating.id,
            'score':     rating.score,
            'rated_at':  rating.rated_at.isoformat(),
        }
    else:
        data['rating'] = None
    return data


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@recommend.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    """Render the dashboard, passing the user's most recent recommendation."""
    latest = (
        Recommendation.query
        .filter_by(user_id=current_user.id)
        .order_by(Recommendation.generated_at.desc())
        .first()
    )

    # Fetch today's most recent journal entry's emotion_vector for the badge strip
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_entry = (
        JournalEntry.query
        .filter_by(user_id=current_user.id)
        .filter(JournalEntry.submitted_at >= today_start)
        .order_by(JournalEntry.submitted_at.desc())
        .first()
    )

    today_emotions = None
    if today_entry and today_entry.emotion_vector:
        try:
            ev = json.loads(today_entry.emotion_vector)
            # Sort by score desc, keep top 5 with score > 0.05
            today_emotions = sorted(
                [(k, v) for k, v in ev.items() if v > 0.05],
                key=lambda x: x[1], reverse=True
            )[:3]
        except (json.JSONDecodeError, AttributeError):
            today_emotions = None

    return render_template(
        'recommend/dashboard.html',
        recommendation=latest,
        today_emotions=today_emotions,
    )

@recommend.route('/generate', methods=['POST'])
@login_required
def generate():
    """
    Run the full recommendation pipeline:
        entries → TemporalAggregator → true_vad_coords → MLRecommender → Recommendation
    """
    # 1. Fetch last 7 journal entries
    raw_entries = (
        JournalEntry.query
        .filter_by(user_id=current_user.id)
        .order_by(JournalEntry.submitted_at.desc())
        .limit(7)
        .all()
    )

    # 2. Guard Clause: Handle the case where there is no data to aggregate yet
    if not raw_entries:
        current_app.logger.warning("User attempted to generate recommendation with 0 entries.")
        return jsonify({
            "error": "Please write at least one journal entry before generating a scent recommendation!"
        }), 400

    # Idempotency (issue 10): the same entries always give the same scent, so if the
    # latest recommendation is newer than the latest entry there is nothing new to
    # distil. Reloading /decoding or double-clicking must not create duplicate
    # rituals (a duplicate would also close the previous ritual's rating window).
    latest_rec = (
        Recommendation.query
        .filter_by(user_id=current_user.id)
        .order_by(Recommendation.generated_at.desc())
        .first()
    )
    if latest_rec is not None and (
        _as_utc(latest_rec.generated_at) >= _as_utc(raw_entries[0].submitted_at)
    ):
        payload = _generate_payload(latest_rec)
        payload['reused'] = True
        return jsonify(payload), 200

    # Flip entries chronologically so index 0 is oldest and index -1 is the newest entry (today)
    entries = list(reversed(raw_entries))

    # 3 & 4. Temporal aggregation
    try:
        aggregator = TemporalAggregator()
        summary = aggregator.summarise(entries)
    except Exception as exc:
        current_app.logger.exception('TemporalAggregator failed: %s', exc)
        return jsonify({'error': 'Failed to aggregate journal data. Please try again later.'}), 500

    # 5 & 6. ML recommendation
    try:
        recommender = _get_recommender()
        
        # Extract the nested raw flat dictionary out of the aggregator's summary object
        raw_vad = summary["aggregated_vad"]

        # Range + arousal calibration in one place (ml/recommender/calibration.py):
        # classifier space [-1, 1] -> scent space [0, 1], then the switchable arousal offset
        # (config AROUSAL_OFFSET; 0 = off). Stored values stay on [-1, 1].
        arousal_offset = current_app.config.get('AROUSAL_OFFSET', 0.0)
        true_vad_coords = to_scent_space(raw_vad, arousal_offset=arousal_offset)

        # Policy (issues 6/7): decide which VAD point to aim at. Neutral and positive
        # states are matched as is; clearly negative states are nudged toward a
        # calming or uplifting goal taken from the catalogue (see ml/recommender/policy.py).
        policy_mode = current_app.config.get('RECOMMENDATION_POLICY', 'adaptive')
        target_vad = apply_policy(
            true_vad_coords,
            mode=policy_mode,
            strength=current_app.config.get('POLICY_STRENGTH', 1.0),
        )
        scent = recommender.recommend(target_vad)
        
    except Exception as exc:
        current_app.logger.exception('MLRecommender failed: %s', exc)
        return jsonify({'error': 'Recommendation engine failed. Please try again later.'}), 500

    # 7. Persist to Database using confirmed table columns
    dominant_emotions = summary.get('dominant_emotions', [])
    rec = Recommendation(
        user_id=current_user.id,
        scent_id=scent['id'],
        aggregated_vad=json.dumps(summary["aggregated_vad"]),
        dominant_emotions=json.dumps(dominant_emotions),
        model_version=f"ml_recommender_v1+policy:{policy_mode}+cal:{float(arousal_offset):.2f}"
    )
    db.session.add(rec)
    db.session.commit()

    # 8. Return JSON Response back to your frontend handler
    return jsonify({
        'rec_id':                     rec.id,
        'scent_id':                   rec.scent_id,
        'scent_name':                 scent['name'],
        'scent_description':          scent.get('description', ''),
        'aggregated_vad':             summary["aggregated_vad"],
        'dominant_emotions':          dominant_emotions,
        'scientific_prove_reference': scent.get('scientific_prove_reference', ''),
        'policy':                     policy_mode,
        'arousal_offset':             float(arousal_offset),
    }), 201


@recommend.route('/rate', methods=['POST'])
@login_required
def rate():
    """
    Accept {rec_id: int, score: int 1-5}.
    Creates or updates a Rating for the given Recommendation.

    Rating window: open until the user submits their next journal entry
    (i.e. generates their next recommendation), or 48 hours after the
    recommendation was generated — whichever comes first.
    This enforces the ritual loop: wear → confess → rate → repeat.
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': 'Request body must be JSON.'}), 400

    rec_id = body.get('rec_id')
    score  = body.get('score')

    # --- Validate inputs ---
    if rec_id is None or not isinstance(rec_id, int):
        return jsonify({'error': 'rec_id must be an integer.'}), 422

    if score is None or not isinstance(score, int) or not (1 <= score <= 5):
        return jsonify({'error': 'score must be an integer between 1 and 5.'}), 422

    # --- Verify ownership ---
    rec = db.session.get(Recommendation, rec_id)
    if rec is None or rec.user_id != current_user.id:
        return jsonify({'error': 'Recommendation not found.'}), 404

    # --- Rating window check ---
    # The window closes when the user generates their NEXT recommendation,
    # or after 48 hours — whichever comes first.
    next_rec = (
        Recommendation.query
        .filter_by(user_id=current_user.id)
        .filter(Recommendation.generated_at > rec.generated_at)
        .order_by(Recommendation.generated_at.asc())
        .first()
    )

    fallback_deadline = rec.generated_at + timedelta(hours=48)
    deadline = min(
        next_rec.generated_at if next_rec else fallback_deadline,
        fallback_deadline,
    )

    # Make deadline timezone-aware for comparison
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)

    if now > deadline:
        # Determine a user-friendly reason for the closed window
        if next_rec:
            reason = (
                "This ritual has been sealed — a new confession was submitted "
                "before this scent could be marked. The archive holds it as it was."
            )
        else:
            reason = (
                "The rating window for this ritual has closed (48 hours have passed). "
                "Submit a new confession to begin the next ritual."
            )
        return jsonify({'error': reason}), 403

    # --- Upsert rating ---
    existing = Rating.query.filter_by(
        user_id=current_user.id,
        rec_id=rec_id,
    ).first()

    if existing:
        existing.score = score
        db.session.commit()
        rating_id = existing.id
        created   = False
    else:
        rating = Rating(
            user_id=current_user.id,
            rec_id=rec_id,
            score=score,
        )
        db.session.add(rating)
        db.session.commit()
        rating_id = rating.id
        created   = True

    return jsonify({
        'success':   True,
        'rating_id': rating_id,
        'rec_id':    rec_id,
        'score':     score,
        'created':   created,
    }), 201 if created else 200


@recommend.route('/by-entry/<int:entry_id>', methods=['DELETE'])
@login_required
def delete_by_entry(entry_id: int):
    """
    Delete all Recommendations that were generated within a time window
    of the given journal entry being submitted.

    Called by the frontend when a journal entry is deleted, so the archive
    stays consistent — no orphaned recommendation cards appear in history.
    """
    # Verify the entry belongs to this user
    entry = db.session.get(JournalEntry, entry_id)
    if entry is None or entry.user_id != current_user.id:
        return jsonify({'error': 'Entry not found.'}), 404

    submitted = entry.submitted_at
    if submitted.tzinfo is None:
        submitted = submitted.replace(tzinfo=timezone.utc)

    # Find recommendations generated within 24h after this entry was submitted
    window_start = submitted
    window_end   = submitted + timedelta(hours=24)

    recs_to_delete = (
        Recommendation.query
        .filter_by(user_id=current_user.id)
        .filter(
            Recommendation.generated_at >= window_start,
            Recommendation.generated_at <= window_end,
        )
        .all()
    )

    deleted_count = len(recs_to_delete)
    for rec in recs_to_delete:
        db.session.delete(rec)
    db.session.commit()

    return jsonify({
        'success':       True,
        'deleted_recs':  deleted_count,
        'entry_id':      entry_id,
    }), 200


@recommend.route('/history', methods=['GET'])
@login_required
def history_page():
    """Render the history HTML template context (resolves base.html BuildError)."""
    return render_template('recommend/history.html')


@recommend.route('/history/data', methods=['GET'])
@login_required
def history_data():
    """Fetch user historical data arrays as asynchronous JSON objects."""
    recs = (
        Recommendation.query
        .filter_by(user_id=current_user.id)
        .order_by(Recommendation.generated_at.desc())
        .all()
    )

    rec_ids = [r.id for r in recs]
    ratings = (
        Rating.query
        .filter(
            Rating.user_id == current_user.id,
            Rating.rec_id.in_(rec_ids),
        )
        .all()
    ) if rec_ids else []

    rating_map: dict[int, Rating] = {r.rec_id: r for r in ratings}

    # recs is ordered NEWEST first. For each rec, the "next" one
    # chronologically is the one immediately BEFORE it in this list
    # (i.e. recs[i-1], since recs[0] is the newest with no "next").
    result = []
    for i, rec in enumerate(recs):
        next_rec = recs[i - 1] if i > 0 else None
        next_generated_at = next_rec.generated_at if next_rec else None
        result.append(
            _recommendation_to_dict(rec, rating_map.get(rec.id), next_generated_at)
        )

    return jsonify(result), 200