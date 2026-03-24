"""
Authentication module for AI-Fibrosis.

- SQLite user store with bcrypt-hashed passwords
- JWT access tokens (short-lived) + refresh tokens (longer-lived)
- Flask blueprint with /login, /refresh, /me endpoints
- CLI helper to add users: python auth.py add <username> <password>
"""

import os
import sys
import sqlite3
import secrets
from datetime import datetime, timedelta, timezone
from functools import wraps

import bcrypt
import jwt as pyjwt
from flask import Blueprint, request, jsonify, g, current_app

# ── Config ──────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

# Generate a persistent secret key file so it survives restarts
_SECRET_KEY_PATH = os.path.join(os.path.dirname(__file__), '.jwt_secret')

def _load_or_create_secret():
    if os.path.exists(_SECRET_KEY_PATH):
        with open(_SECRET_KEY_PATH, 'r') as f:
            return f.read().strip()
    key = secrets.token_hex(64)
    with open(_SECRET_KEY_PATH, 'w') as f:
        f.write(key)
    return key

JWT_SECRET = _load_or_create_secret()
JWT_ALGORITHM = 'HS256'
ACCESS_TOKEN_MINUTES = 30
REFRESH_TOKEN_DAYS = 7

# ── Database helpers ────────────────────────────────────────────────────

def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create users table if it doesn't exist."""
    conn = _get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL UNIQUE COLLATE NOCASE,
            password    TEXT    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    ''')
    conn.commit()
    conn.close()


def add_user(username: str, password: str):
    """Hash the password with bcrypt and insert a new user."""
    if len(password) < 8:
        raise ValueError('Password must be at least 8 characters')
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12))
    conn = _get_db()
    try:
        conn.execute(
            'INSERT INTO users (username, password) VALUES (?, ?)',
            (username, hashed.decode('utf-8')),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError(f'User "{username}" already exists')
    conn.close()


def verify_user(username: str, password: str):
    """Return the user row if credentials are valid, else None."""
    conn = _get_db()
    row = conn.execute(
        'SELECT * FROM users WHERE username = ?', (username,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    if bcrypt.checkpw(password.encode('utf-8'), row['password'].encode('utf-8')):
        return dict(row)
    return None


# ── Token helpers ───────────────────────────────────────────────────────

def _create_token(user_id: int, username: str, token_type: str, lifetime: timedelta):
    now = datetime.now(timezone.utc)
    payload = {
        'sub': str(user_id),
        'username': username,
        'type': token_type,
        'iat': now,
        'exp': now + lifetime,
    }
    return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_access_token(user_id: int, username: str):
    return _create_token(user_id, username, 'access', timedelta(minutes=ACCESS_TOKEN_MINUTES))


def create_refresh_token(user_id: int, username: str):
    return _create_token(user_id, username, 'refresh', timedelta(days=REFRESH_TOKEN_DAYS))


def decode_token(token: str, expected_type: str = 'access'):
    """Decode and validate a JWT. Returns the payload dict or None."""
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get('type') != expected_type:
            return None
        return payload
    except (pyjwt.ExpiredSignatureError, pyjwt.InvalidTokenError):
        return None


# ── Flask decorator ─────────────────────────────────────────────────────

def login_required(f):
    """Decorator that protects a route with JWT access-token auth.
    Accepts token from Authorization header OR ?token= query param (for EventSource/SSE)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
        else:
            token = request.args.get('token')
        if not token:
            return jsonify({'error': 'Missing or malformed Authorization header'}), 401
        payload = decode_token(token, expected_type='access')
        if payload is None:
            return jsonify({'error': 'Invalid or expired token'}), 401
        g.user_id = int(payload['sub'])
        g.username = payload['username']
        return f(*args, **kwargs)
    return decorated


# ── Blueprint ───────────────────────────────────────────────────────────

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    user = verify_user(username, password)
    if user is None:
        return jsonify({'error': 'Invalid credentials'}), 401

    return jsonify({
        'access_token': create_access_token(user['id'], user['username']),
        'refresh_token': create_refresh_token(user['id'], user['username']),
        'username': user['username'],
    }), 200


@auth_bp.route('/refresh', methods=['POST'])
def refresh():
    data = request.get_json(silent=True)
    if not data or not data.get('refresh_token'):
        return jsonify({'error': 'refresh_token is required'}), 400

    payload = decode_token(data['refresh_token'], expected_type='refresh')
    if payload is None:
        return jsonify({'error': 'Invalid or expired refresh token'}), 401

    return jsonify({
        'access_token': create_access_token(payload['sub'], payload['username']),
        'username': payload['username'],
    }), 200


@auth_bp.route('/me', methods=['GET'])
@login_required
def me():
    return jsonify({'user_id': g.user_id, 'username': g.username}), 200


# ── CLI: python auth.py add <username> <password> ──────────────────────

if __name__ == '__main__':
    init_db()
    if len(sys.argv) >= 4 and sys.argv[1] == 'add':
        username = sys.argv[2]
        password = sys.argv[3]
        try:
            add_user(username, password)
            print(f'User "{username}" created successfully.')
        except ValueError as e:
            print(f'Error: {e}', file=sys.stderr)
            sys.exit(1)
    else:
        print('Usage: python auth.py add <username> <password>')
        sys.exit(1)
