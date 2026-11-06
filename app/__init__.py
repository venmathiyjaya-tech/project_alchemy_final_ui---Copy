# app/__init__.py
import json
import os
import click
from flask import Flask, redirect, url_for
from flask_login import LoginManager

from app.extensions import db, login_manager, migrate


def create_app(config_object="config.Config"):
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object(config_object)

    # ── Extensions ──────────────────────────────────────────────────────
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    
    login_manager.login_view = "auth.login"
    login_manager.login_message_category = "error"

    # ── Jinja2 custom filter ─────────────────────────────────────────────
    @app.template_filter('from_json')
    def from_json_filter(value):
        """Deserialise a JSON string stored in a DB column to a Python object."""
        if not value:
            return {}
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return {}

    # ── Blueprints ───────────────────────────────────────────────────────
    with app.app_context():
        from app.routes.auth     import auth     as auth_bp
        from app.routes.journal  import journal  as journal_bp
        from app.routes.recommend import recommend as recommend_bp
        from app.routes.landing  import landing  as landing_bp

        app.register_blueprint(landing_bp)   # handles '/'
        app.register_blueprint(auth_bp)
        app.register_blueprint(journal_bp)
        app.register_blueprint(recommend_bp)

    # ── User loader ──────────────────────────────────────────────────────
    @login_manager.user_loader
    def load_user(user_id):
        from app.models import User
        return db.session.get(User, int(user_id))

    # ── CLI — init-db ────────────────────────────────────────────────────
    @app.cli.command("init-db")
    def init_db_command():
        db.create_all()
        click.echo("✓ Tables created.")
        _seed_scents(app)
        click.echo("✓ Done.")

    return app


def _seed_scents(app):
    from app.models import Scent
    json_path = os.path.normpath(
        os.path.join(app.root_path, "..", "data", "scents.json")
    )
    if not os.path.exists(json_path):
        click.echo(f"  ⚠ scents.json not found at {json_path} — skipping seed.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        scents_data = json.load(f)

    for item in scents_data:
        scent = Scent(
            id=item["id"],
            name=item["name"],
            description=item.get("description"),
            
            # Map your flat JSON keys directly to your database column names
            vad_valence=item["valence"],
            vad_arousal=item["arousal"],
            vad_dominance=item["dominance"],
            
            brand=item.get("brand", "Vasya"),
            scientific_prove_reference=item.get("scientific_prove_reference"),
        )
        scent.tags_list = item.get("emotion_tags", [])
        db.session.merge(scent)

    db.session.commit()
    click.echo(f"  ✓ Seeded {len(scents_data)} scents.")