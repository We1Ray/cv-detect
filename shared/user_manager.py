"""User permission management for industrial inspection systems.

Provides role-based access control with three levels:
- Operator: Can run inspections, view results
- Engineer: Can modify parameters, train models, configure system
- Administrator: Full access including user management

Uses SQLite for persistence with pbkdf2 password hashing.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import sqlite3
import string
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums & data models
# ---------------------------------------------------------------------------


class UserRole(IntEnum):
    """Role hierarchy: higher numeric value = higher privilege."""

    OPERATOR = 1
    ENGINEER = 2
    ADMIN = 3


@dataclass
class UserRecord:
    """Immutable snapshot of a user row."""

    id: int
    username: str
    role: UserRole
    password_hash: str
    created_at: str
    last_login: Optional[str]
    is_active: bool
    force_password_change: bool = False


# ---------------------------------------------------------------------------
# Password utilities
# ---------------------------------------------------------------------------

_HASH_ITERATIONS = 260_000
_SALT_LENGTH = 32


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """Return ``salt_hex:hash_hex`` using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = os.urandom(_SALT_LENGTH)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _HASH_ITERATIONS)
    return f"{salt.hex()}:{dk.hex()}"


def _generate_secure_password() -> str:
    """Generate a random 16-char password meeting the policy."""
    alphabet = string.ascii_letters + string.digits + "!@#$%"
    while True:
        pw = ''.join(secrets.choice(alphabet) for _ in range(16))
        if (any(c.isupper() for c in pw) and any(c.isdigit() for c in pw)
                and any(c in "!@#$%" for c in pw)):
            return pw


def _verify_password(password: str, stored: str) -> bool:
    """Verify *password* against a ``salt_hex:hash_hex`` string."""
    try:
        salt_hex, _ = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(_hash_password(password, salt), stored)


# ---------------------------------------------------------------------------
# UserManager
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT    NOT NULL UNIQUE,
    role        INTEGER NOT NULL DEFAULT 1,
    password_hash TEXT  NOT NULL,
    created_at  TEXT    NOT NULL,
    last_login  TEXT,
    is_active   INTEGER NOT NULL DEFAULT 1,
    force_password_change INTEGER DEFAULT 0
);
"""


class UserManager:
    """SQLite-backed user & permission manager.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Defaults to ``users.db`` in the
        current working directory.
    """

    def __init__(self, db_path: str = "users.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._current_user: Optional[UserRecord] = None
        self._init_db()

    # -- internal helpers ---------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(_CREATE_TABLE_SQL)
                conn.commit()
                # Auto-create default admin if the table is empty
                row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
                if row[0] == 0:
                    now = datetime.now(timezone.utc).isoformat()
                    default_pw = _generate_secure_password()
                    pw_hash = _hash_password(default_pw)
                    conn.execute(
                        "INSERT INTO users (username, role, password_hash, created_at, force_password_change) "
                        "VALUES (?, ?, ?, ?, 1)",
                        ("admin", UserRole.ADMIN, pw_hash, now),
                    )
                    conn.commit()
                    logger.warning(
                        "Default admin created. Initial password: %s — CHANGE IMMEDIATELY.",
                        default_pw,
                    )
            finally:
                conn.close()

    @staticmethod
    def _row_to_record(row: tuple) -> UserRecord:
        return UserRecord(
            id=row[0],
            username=row[1],
            role=UserRole(row[2]),
            password_hash=row[3],
            created_at=row[4],
            last_login=row[5],
            is_active=bool(row[6]),
            force_password_change=bool(row[7]) if len(row) > 7 else False,
        )

    # -- public API ---------------------------------------------------------

    @staticmethod
    def check_password_policy(password: str) -> Tuple[bool, str]:
        """Validate a password against the security policy.

        Rules
        -----
        - Minimum 8 characters.
        - At least 1 uppercase letter.
        - At least 1 digit.

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` if the password meets all requirements, otherwise
            ``(False, "<reason>")`` describing the first violated rule.
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters long."
        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least 1 uppercase letter."
        if not re.search(r"[0-9]", password):
            return False, "Password must contain at least 1 digit."
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return False, "Password must contain at least 1 special character."
        return True, ""

    def create_user(self, username: str, password: str, role: UserRole = UserRole.OPERATOR) -> int:
        """Create a new user and return the user id.

        Raises
        ------
        ValueError
            If the username already exists or the password does not meet
            the security policy.
        """
        valid, reason = self.check_password_policy(password)
        if not valid:
            raise ValueError(reason)
        now = datetime.now(timezone.utc).isoformat()
        pw_hash = _hash_password(password)
        with self._lock:
            conn = self._connect()
            try:
                try:
                    cur = conn.execute(
                        "INSERT INTO users (username, role, password_hash, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (username, int(role), pw_hash, now),
                    )
                    conn.commit()
                    user_id: int = cur.lastrowid  # type: ignore[assignment]
                    logger.info("User '%s' created with role %s.", username, role.name)
                    return user_id
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"Username '{username}' already exists.") from exc
            finally:
                conn.close()

    def authenticate(self, username: str, password: str) -> Optional[UserRecord]:
        """Authenticate a user.  Returns the ``UserRecord`` on success, else ``None``."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM users WHERE username = ? AND is_active = 1",
                    (username,),
                ).fetchone()
                if row is None:
                    # Prevent timing oracle on username existence
                    _hash_password("dummy_password_to_prevent_timing_oracle")
                    logger.warning("Failed login attempt for unknown user '%s'.", username)
                    return None
                record = self._row_to_record(row)
                if not _verify_password(password, record.password_hash):
                    logger.warning("Failed login attempt for user '%s'.", username)
                    return None
                # Update last_login
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "UPDATE users SET last_login = ? WHERE id = ?",
                    (now, record.id),
                )
                conn.commit()
                return UserRecord(
                    id=record.id,
                    username=record.username,
                    role=record.role,
                    password_hash=record.password_hash,
                    created_at=record.created_at,
                    last_login=now,
                    is_active=record.is_active,
                    force_password_change=record.force_password_change,
                )
            finally:
                conn.close()

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change a user's password.  Returns ``True`` on success.

        Raises
        ------
        ValueError
            If *new_password* does not meet the security policy.
        """
        valid, reason = self.check_password_policy(new_password)
        if not valid:
            raise ValueError(reason)
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM users WHERE username = ?", (username,)
                ).fetchone()
                if row is None:
                    return False
                record = self._row_to_record(row)
                if not _verify_password(old_password, record.password_hash):
                    return False
                new_hash = _hash_password(new_password)
                conn.execute(
                    "UPDATE users SET password_hash = ?, force_password_change = 0 WHERE id = ?",
                    (new_hash, record.id),
                )
                conn.commit()
                logger.info("Password changed for user '%s'.", username)
                return True
            finally:
                conn.close()

    def needs_password_change(self, username: str) -> bool:
        """Return ``True`` if *username* is flagged for a forced password change."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT force_password_change FROM users WHERE username = ?",
                    (username,),
                ).fetchone()
                if row is None:
                    return False
                return bool(row[0])
            finally:
                conn.close()

    def set_role(self, username: str, role: UserRole) -> None:
        """Set a user's role.

        Raises
        ------
        ValueError
            If the user does not exist.
        """
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "UPDATE users SET role = ? WHERE username = ?",
                    (int(role), username),
                )
                conn.commit()
                if cur.rowcount == 0:
                    raise ValueError(f"User '{username}' not found.")
                logger.info("Role for '%s' set to %s.", username, role.name)
            finally:
                conn.close()

    def list_users(self) -> List[UserRecord]:
        """Return all users (active and inactive)."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
                return [self._row_to_record(r) for r in rows]
            finally:
                conn.close()

    def deactivate_user(self, username: str) -> None:
        """Soft-delete a user by setting ``is_active = 0``."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE users SET is_active = 0 WHERE username = ?", (username,)
                )
                conn.commit()
                logger.info("User '%s' deactivated.", username)
            finally:
                conn.close()

    def activate_user(self, username: str) -> None:
        """Re-activate a deactivated user."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE users SET is_active = 1 WHERE username = ?", (username,)
                )
                conn.commit()
                logger.info("User '%s' activated.", username)
            finally:
                conn.close()

    @staticmethod
    def check_permission(user: UserRecord, required_role: UserRole) -> bool:
        """Return ``True`` if *user*'s role meets or exceeds *required_role*.

        Role hierarchy: ADMIN (3) >= ENGINEER (2) >= OPERATOR (1).
        """
        return user.role >= required_role

    def get_current_user(self) -> Optional[UserRecord]:
        """Return the currently logged-in user (session-level, in-memory)."""
        return self._current_user

    def set_current_user(self, user: Optional[UserRecord]) -> None:
        """Set the currently logged-in user."""
        self._current_user = user
