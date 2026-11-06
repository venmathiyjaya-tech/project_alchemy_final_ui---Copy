# app/routes/landing.py
from flask import Blueprint, render_template
from flask_login import current_user, login_required
from flask import redirect, url_for

from app.models import JournalEntry, Recommendation
from app.routes.recommend import _as_utc

landing = Blueprint('landing', __name__)


@landing.route('/')
def index():
    """The Invitation — landing page for unauthenticated visitors.
    Authenticated users are sent straight to the Observatory.
    """
    if current_user.is_authenticated:
        return redirect(url_for('recommend.dashboard'))
    return render_template('landing.html')

@landing.route('/decoding')
@login_required
def decoding():
    """The Decoding Chamber — ceremonial transition page.
 
    Shown after a journal entry is recorded. The page itself triggers
    /recommend/generate via JS and redirects to /recommend/dashboard
    once the pipeline completes (or on failure).
    """
    # Gate (issue 10): only show the chamber when there is something new to decode.
    latest_entry = (
        JournalEntry.query
        .filter_by(user_id=current_user.id)
        .order_by(JournalEntry.submitted_at.desc())
        .first()
    )
    if latest_entry is None:
        return redirect(url_for('journal.write_page'))

    latest_rec = (
        Recommendation.query
        .filter_by(user_id=current_user.id)
        .order_by(Recommendation.generated_at.desc())
        .first()
    )
    if latest_rec is not None and (
        _as_utc(latest_rec.generated_at) >= _as_utc(latest_entry.submitted_at)
    ):
        return redirect(url_for('recommend.dashboard'))

    return render_template('decoding.html')
