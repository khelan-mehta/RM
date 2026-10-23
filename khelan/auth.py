"""Authentication for Khelan — standard-library only.

- Passwords hashed with PBKDF2-HMAC-SHA256 (salted), stored in the settings
  table.
- Sessions are stateless signed cookies: base64(payload) + "." + HMAC-SHA256,
  signed with a per-install secret key. No external dependency.

On first run an 'admin' account is created. The password is taken from the
KHELAN_PASSWORD env var if set, otherwise a random one is generated and written
to khelan/ADMIN_LOGIN.txt (and logged once) so it is never guessable.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COOKIE = "khelan_session"
SESSION_TTL = 60 * 60 * 24 * 7  # 7 days
_PBKDF2_ROUNDS = 200_000


# ---------------- password hashing ----------------
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, dk_hex = stored.split("$")
        if scheme != "pbkdf2":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), _PBKDF2_ROUNDS)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except (ValueError, AttributeError):
        return False


# ---------------- bootstrap ----------------
def ensure_admin(db):
    """Create the admin account + signing secret on first run. Returns the
    plaintext password only when it was newly generated (else None)."""
    if not db.get_setting("secret_key"):
        db.set_setting("secret_key", secrets.token_hex(32))
    db.set_setting("auth_user", db.get_setting("auth_user", "admin"))
    new_password = None
    if not db.get_setting("auth_pw"):
        pw = os.environ.get("KHELAN_PASSWORD")
        if not pw:
            pw = secrets.token_urlsafe(9)
            new_password = pw
            try:
                with open(os.path.join(HERE, "ADMIN_LOGIN.txt"), "w") as f:
                    f.write(f"Khelan login\nusername: admin\npassword: {pw}\n")
            except OSError:
                pass
        db.set_setting("auth_pw", hash_password(pw))
    return new_password


def check_login(db, username: str, password: str) -> bool:
    if username != db.get_setting("auth_user", "admin"):
        return False
    stored = db.get_setting("auth_pw", "")
    return bool(stored) and verify_password(password, stored)


def set_password(db, new_password: str):
    db.set_setting("auth_pw", hash_password(new_password))


# ---------------- signed session tokens ----------------
def _secret(db) -> bytes:
    return db.get_setting("secret_key", "khelan-dev-secret").encode()


def make_token(db, username: str) -> str:
    payload = {"u": username, "exp": int(time.time()) + SESSION_TTL}
    raw = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    sig = hmac.new(_secret(db), raw.encode(), hashlib.sha256).hexdigest()
    return f"{raw}.{sig}"


def verify_token(db, token: str):
    if not token or "." not in token:
        return None
    raw, sig = token.rsplit(".", 1)
    expected = hmac.new(_secret(db), raw.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        pad = "=" * (-len(raw) % 4)
        payload = json.loads(base64.urlsafe_b64decode(raw + pad))
    except (ValueError, json.JSONDecodeError):
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload.get("u")
