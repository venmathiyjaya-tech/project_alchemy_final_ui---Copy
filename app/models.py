from datetime import datetime
from flask_login import UserMixin
import json
from werkzeug.security import generate_password_hash, check_password_hash
from app.extensions import db


# ─────────────────────────────────────────────
# USER
# ─────────────────────────────────────────────
class User(UserMixin, db.Model):
    __tablename__ = "user"

    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    
    # Secure hash storage for the recovery phrase (prevents plain-text data leaks)
    recovery_hash = db.Column(db.String(256), nullable=False)
    
    created_at    = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    journal_entries = db.relationship("JournalEntry",  back_populates="user", cascade="all, delete-orphan")
    recommendations = db.relationship("Recommendation", back_populates="user", cascade="all, delete-orphan")
    ratings         = db.relationship("Rating",         back_populates="user", cascade="all, delete-orphan")

    # Passphrase Cryptography
    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    # Recovery Phrase Cryptography
    def set_recovery_phrase(self, phrase: str) -> None:
        """Normalizes and securely hashes the recovery string."""
        # Lowercase and strip whitespace so matching isn't broken by minor typos or spacing
        normalized = phrase.strip().lower()
        self.recovery_hash = generate_password_hash(normalized)

    def check_recovery_phrase(self, phrase: str) -> bool:
        """Verifies an incoming string against the stored hash."""
        normalized = phrase.strip().lower()
        return check_password_hash(self.recovery_hash, normalized)

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r}>"


# ─────────────────────────────────────────────
# JOURNAL ENTRY
# ─────────────────────────────────────────────
class JournalEntry(db.Model):
    __tablename__ = "journal_entry"

    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    entry_text     = db.Column(db.Text, nullable=False)
    submitted_at   = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Emotion classification output — stored as a JSON string
    # e.g. '{"joy": 0.82, "sadness": 0.12, "fear": 0.06}'
    emotion_vector = db.Column(db.Text, nullable=True)

    # VAD (Valence-Arousal-Dominance) coordinates derived from emotion_vector
    vad_valence    = db.Column(db.Float, nullable=True)
    vad_arousal    = db.Column(db.Float, nullable=True)
    vad_dominance  = db.Column(db.Float, nullable=True)

    # Relationships
    user = db.relationship("User", back_populates="journal_entries")

    def __repr__(self) -> str:
        return (
            f"<JournalEntry id={self.id} user_id={self.user_id} "
            f"submitted_at={self.submitted_at:%Y-%m-%d}>"
        )


# ─────────────────────────────────────────────
# SCENT
# ─────────────────────────────────────────────
class Scent(db.Model):
    __tablename__ = "scent"

    id            = db.Column(db.String(20), primary_key=True)
    name          = db.Column(db.String(120), nullable=False)
    description   = db.Column(db.Text, nullable=True)

    # VAD coordinates that describe the emotional quality of the scent
    vad_valence   = db.Column(db.Float, nullable=False)
    vad_arousal   = db.Column(db.Float, nullable=False)
    vad_dominance = db.Column(db.Float, nullable=False)

    # Emotion tags as a JSON string, e.g. '["calm", "relief", "contentment"]'
    emotion_tags  = db.Column(db.Text, nullable=True)

    @property
    def tags_list(self):
        if self.emotion_tags:
            return json.loads(self.emotion_tags)
        return []

    @tags_list.setter
    def tags_list(self, value):
        self.emotion_tags = json.dumps(value) if value is not None else None

    brand         = db.Column(db.String(80), nullable=False, default="Vasya")

    # Scientific references supporting the scent-emotion associations
    scientific_prove_reference = db.Column(db.Text, nullable=True)

    # Relationships
    recommendations = db.relationship("Recommendation", back_populates="scent")

    def __repr__(self) -> str:
        return f"<Scent id={self.id} name={self.name!r} brand={self.brand!r}>"


# ─────────────────────────────────────────────
# RECOMMENDATION
# ─────────────────────────────────────────────
class Recommendation(db.Model):
    __tablename__ = "recommendation"

    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("user.id"),     nullable=False)
    scent_id       = db.Column(db.String(20), db.ForeignKey("scent.id"), nullable=False)
    generated_at   = db.Column(db.DateTime, default=datetime.utcnow,    nullable=False)

    # Weighted-average VAD profile that drove this recommendation
    # e.g. '{"valence": 0.45, "arousal": 0.68, "dominance": 0.52}'
    aggregated_vad = db.Column(db.Text, nullable=True)

    # FIX (Bug 9): store dominant_emotions from the aggregator as a JSON string
    # e.g. '["joy", "optimism", "gratitude"]'
    dominant_emotions = db.Column(db.Text, nullable=True)

    # Which model version produced this recommendation
    model_version  = db.Column(db.String(50), nullable=True)

    # Relationships
    user   = db.relationship("User",  back_populates="recommendations")
    scent  = db.relationship("Scent", back_populates="recommendations")
    rating = db.relationship("Rating", back_populates="recommendation",
                             uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return (
            f"<Recommendation id={self.id} user_id={self.user_id} "
            f"scent_id={self.scent_id} generated_at={self.generated_at:%Y-%m-%d}>"
        )


# ─────────────────────────────────────────────
# RATING
# ─────────────────────────────────────────────
class Rating(db.Model):
    __tablename__ = "rating"

    id       = db.Column(db.Integer, primary_key=True)
    # FIX (Bug 3 — models side): column is rec_id, matching the FK name used everywhere
    rec_id   = db.Column(db.Integer, db.ForeignKey("recommendation.id"), nullable=False)
    user_id  = db.Column(db.Integer, db.ForeignKey("user.id"),           nullable=False)
    score    = db.Column(db.Integer, nullable=False)   # constrained to 1–5 in app logic
    rated_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    recommendation = db.relationship("Recommendation", back_populates="rating")
    user           = db.relationship("User",           back_populates="ratings")

    def __repr__(self) -> str:
        return f"<Rating id={self.id} rec_id={self.rec_id} score={self.score}>"