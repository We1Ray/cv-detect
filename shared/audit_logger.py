"""Audit trail logger for regulatory compliance.

Records all significant operations with timestamps, user identity,
and operation details. Designed for FDA 21 CFR Part 11 style
requirements (though full compliance requires additional measures).

Features:
- Immutable audit log (append-only SQLite table)
- Operation categories: AUTH, CONFIG, MODEL, INSPECT, EXPORT, SYSTEM
- Searchable by date range, user, category
- Export to CSV/JSON for external review
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums & data models
# ---------------------------------------------------------------------------


class AuditCategory(str, Enum):
    """Category tags for audit log entries."""

    AUTH = "AUTH"
    CONFIG = "CONFIG"
    MODEL = "MODEL"
    INSPECT = "INSPECT"
    EXPORT = "EXPORT"
    SYSTEM = "SYSTEM"


@dataclass
class AuditEntry:
    """A single audit log record."""

    id: int
    timestamp: str
    username: str
    category: str
    action: str
    detail: str
    ip_address: str


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS audit_trail (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    username    TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    detail      TEXT    NOT NULL DEFAULT '',
    ip_address  TEXT    NOT NULL DEFAULT ''
);
"""


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------


class AuditLogger:
    """Append-only audit trail backed by SQLite.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Defaults to ``audit_trail.db``.
    """

    def __init__(self, db_path: str = "audit_trail.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
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
            finally:
                conn.close()

    @staticmethod
    def _row_to_entry(row: tuple) -> AuditEntry:
        return AuditEntry(
            id=row[0],
            timestamp=row[1],
            username=row[2],
            category=row[3],
            action=row[4],
            detail=row[5],
            ip_address=row[6],
        )

    # -- public API ---------------------------------------------------------

    def log(
        self,
        category: AuditCategory,
        action: str,
        detail: str = "",
        username: str = "system",
        ip_address: str = "",
    ) -> int:
        """Append a new audit entry and return its id.

        Parameters
        ----------
        category:
            One of the :class:`AuditCategory` values.
        action:
            Short description of the operation (e.g. ``"login_success"``).
        detail:
            Optional extended information.
        username:
            Identity of the acting user; defaults to ``"system"``.
        ip_address:
            Optional IP address of the client.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT INTO audit_trail (timestamp, username, category, action, detail, ip_address) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (now, username, category.value, action, detail, ip_address),
                )
                conn.commit()
                entry_id: int = cur.lastrowid  # type: ignore[assignment]
                logger.debug(
                    "Audit [%s] %s: %s (%s)", category.value, username, action, detail
                )
                return entry_id
            finally:
                conn.close()

    def query(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        username: Optional[str] = None,
        category: Optional[AuditCategory] = None,
        limit: int = 1000,
    ) -> List[AuditEntry]:
        """Search the audit trail with optional filters.

        Parameters
        ----------
        start_date / end_date:
            ISO-8601 date or datetime strings for range filtering.
        username:
            Filter by acting user.
        category:
            Filter by :class:`AuditCategory`.
        limit:
            Maximum number of rows to return (default 1000).
        """
        clauses: list[str] = []
        params: list[object] = []

        if start_date is not None:
            clauses.append("timestamp >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append("timestamp <= ?")
            params.append(end_date)
        if username is not None:
            clauses.append("username = ?")
            params.append(username)
        if category is not None:
            clauses.append("category = ?")
            params.append(category.value)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM audit_trail{where} ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [self._row_to_entry(r) for r in rows]
            finally:
                conn.close()

    def get_recent(self, limit: int = 50) -> List[AuditEntry]:
        """Return the most recent *limit* entries."""
        return self.query(limit=limit)

    # -- export -------------------------------------------------------------

    def _fetch_for_export(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[AuditEntry]:
        """Fetch entries within a date range for export (ascending order)."""
        clauses: list[str] = []
        params: list[object] = []
        if start_date is not None:
            clauses.append("timestamp >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append("timestamp <= ?")
            params.append(end_date)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM audit_trail{where} ORDER BY id ASC"

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [self._row_to_entry(r) for r in rows]
            finally:
                conn.close()

    def export_csv(
        self,
        path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """Export audit entries to a CSV file.  Returns the number of rows written."""
        entries = self._fetch_for_export(start_date, end_date)
        fieldnames = ["id", "timestamp", "username", "category", "action", "detail", "ip_address"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for entry in entries:
                writer.writerow(asdict(entry))
        logger.info("Exported %d audit entries to %s", len(entries), path)
        return len(entries)

    def export_json(
        self,
        path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """Export audit entries to a JSON file.  Returns the number of rows written."""
        entries = self._fetch_for_export(start_date, end_date)
        data = [asdict(e) for e in entries]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        logger.info("Exported %d audit entries to %s", len(entries), path)
        return len(entries)
