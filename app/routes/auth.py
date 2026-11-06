import re
from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user

from app.models import User
from app.extensions import db

auth = Blueprint('auth', __name__, url_prefix='/auth')


def is_password_strong(password: str) -> tuple[bool, str]:
    """Validates password strength rules matching the frontend design tokens."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one number."
    if not re.search(r"[!@#$%^&*(),.?':{}|<>_+-]", password):
        return False, "Password must contain at least one special character."
    return True, ""


@auth.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('recommend.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        recovery_phrase = request.form.get('recovery_phrase', '').strip()

        # 1. Missing fields validation
        if not username or not password or not confirm_password or not recovery_phrase:
            flash('All fields are required to open a record.', 'error')
            return render_template('auth/login.html')

        if len(recovery_phrase) < 6:
            flash('Recovery phrase must be at least 6 characters.', 'error')
            return render_template('auth/login.html')

        # 2. Backend password matching check
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('auth/login.html')

        # 3. Strength verification
        is_strong, error_msg = is_password_strong(password)
        if not is_strong:
            flash(error_msg, 'error')
            return render_template('auth/login.html')  

        # 4. Duplicate identity validation
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already taken.', 'error')
            return render_template('auth/login.html')  

        # 5. Model initialization & secure hashing
        new_user = User(username=username)
        new_user.set_password(password)
        new_user.set_recovery_phrase(recovery_phrase) # Uses the secure hashing logic from your model
        
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        return redirect(url_for('recommend.dashboard'))

    return render_template('auth/login.html')


@auth.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('recommend.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('auth/login.html')

        user = User.query.filter_by(username=username).first()

        if not user:
            flash('User not found.', 'error')
            return render_template('auth/login.html')

        if not user.check_password(password):
            flash('Invalid password.', 'error')
            return render_template('auth/login.html')

        login_user(user)
        next_page = request.args.get('next')
        return redirect(next_page or url_for('recommend.dashboard'))

    return render_template('auth/login.html')


@auth.route('/reset', methods=['GET', 'POST'])
def reset():
    """Handles archive access restoration via the registered identity and recovery token."""
    if current_user.is_authenticated:
        return redirect(url_for('recommend.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        recovery_phrase = request.form.get('recovery_phrase', '').strip()
        new_password = request.form.get('password', '').strip()

        if not username or not recovery_phrase or not new_password:
            flash('All validation tokens are required to perform restoration.', 'error')
            return render_template('auth/reset.html')

        user = User.query.filter_by(username=username).first()
        
        # Combined secure hash match check (replaces old plain text fallback checks)
        if not user or not user.check_recovery_phrase(recovery_phrase):
            flash('Invalid identity or recovery validation token failed.', 'error')
            return render_template('auth/reset.html')

        # Password strength validation for the new password
        is_strong, error_msg = is_password_strong(new_password)
        if not is_strong:
            flash(error_msg, 'error')
            return render_template('auth/reset.html')

        # Persist the new passphrase securely
        user.set_password(new_password)
        db.session.commit()

        flash('Your passphrase has been restored to the archive. You may now enter.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/reset.html')


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))