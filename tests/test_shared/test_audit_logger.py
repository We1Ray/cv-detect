"""Tests for shared.audit_logger -- append-only audit trail."""

from __future__ import annotations

import csv
import json
import sqlite3

import pytest

from shared.audit_logger import AuditCategory, AuditEntry, AuditLogger


@pytest.fixture
def audit(tmp_path) -> AuditLogger:
    """Create an AuditLogger backed by a temporary database."""
    db = str(tmp_path / "test_audit.db")
    return AuditLogger(db_path=db)


# ------------------------------------------------------------------
# Basic logging
# ------------------------------------------------------------------


class TestLogEntry:
    """Tests for the log() method."""

    def test_log_entry(self, audit: AuditLogger) -> None:
        """A logged entry should be retrievable."""
        entry_id = audit.log(AuditCategory.AUTH, "login_success", username="alice")
        assert isinstance(entry_id, int)
        assert entry_id > 0
        entries = audit.get_recent(limit=1)
        assert len(entries) == 1
        assert entries[0].action == "login_success"
        assert entries[0].username == "alice"
        assert entries[0].category == "AUTH"

    def test_log_with_detail_and_ip(self, audit: AuditLogger) -> None:
        """Detail and ip_address fields should be stored."""
        audit.log(
            AuditCategory.CONFIG,
            "param_change",
            detail="threshold: 0.5 -> 0.7",
            username="bob",
            ip_address="192.168.1.10",
        )
        entries = audit.get_recent(limit=1)
        assert entries[0].detail == "threshold: 0.5 -> 0.7"
        assert entries[0].ip_address == "192.168.1.10"

    def test_log_default_username(self, audit: AuditLogger) -> None:
        """Default username should be 'system'."""
        audit.log(AuditCategory.SYSTEM, "startup")
        entries = audit.get_recent(limit=1)
        assert entries[0].username == "system"


# ------------------------------------------------------------------
# Queries
# ------------------------------------------------------------------


class TestQuery:
    """Tests for the query() method."""

    def _seed(self, audit: AuditLogger) -> None:
        """Insert a variety of entries for query tests."""
        audit.log(AuditCategory.AUTH, "login", username="alice")
        audit.log(AuditCategory.CONFIG, "param_update", username="bob")
        audit.log(AuditCategory.INSPECT, "run_inspection", username="alice")
        audit.log(AuditCategory.MODEL, "train_start", username="bob")
        audit.log(AuditCategory.EXPORT, "export_report", username="alice")

    def test_query_by_category(self, audit: AuditLogger) -> None:
        """Filtering by category should return only matching entries."""
        self._seed(audit)
        results = audit.query(category=AuditCategory.AUTH)
        assert all(e.category == "AUTH" for e in results)
        assert len(results) == 1

    def test_query_by_username(self, audit: AuditLogger) -> None:
        """Filtering by username should return only that user's entries."""
        self._seed(audit)
        results = audit.query(username="bob")
        assert all(e.username == "bob" for e in results)
        assert len(results) == 2

    def test_query_by_date_range(self, audit: AuditLogger) -> None:
        """Date range filtering should work with ISO-8601 boundaries."""
        self._seed(audit)
        # All entries were just created, so a wide range returns everything
        results = audit.query(start_date="2000-01-01", end_date="2099-12-31")
        assert len(results) == 5

        # A range in the far past returns nothing
        results = audit.query(start_date="1990-01-01", end_date="1990-12-31")
        assert len(results) == 0

    def test_query_limit(self, audit: AuditLogger) -> None:
        """The limit parameter should cap the result count."""
        self._seed(audit)
        results = audit.query(limit=2)
        assert len(results) == 2

    def test_get_recent(self, audit: AuditLogger) -> None:
        """get_recent should return entries in reverse chronological order."""
        self._seed(audit)
        recent = audit.get_recent(limit=3)
        assert len(recent) == 3
        # IDs should be descending (most recent first)
        assert recent[0].id > recent[1].id > recent[2].id


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------


class TestExport:
    """Tests for export_csv and export_json."""

    def _seed(self, audit: AuditLogger) -> None:
        audit.log(AuditCategory.AUTH, "login", username="alice")
        audit.log(AuditCategory.CONFIG, "change", username="bob")
        audit.log(AuditCategory.INSPECT, "run", username="alice")

    def test_export_csv(self, audit: AuditLogger, tmp_path) -> None:
        """CSV export should produce a valid CSV with correct row count."""
        self._seed(audit)
        csv_path = str(tmp_path / "audit.csv")
        count = audit.export_csv(csv_path)
        assert count == 3

        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["username"] == "alice"
        assert "timestamp" in rows[0]

    def test_export_json(self, audit: AuditLogger, tmp_path) -> None:
        """JSON export should produce a valid JSON array."""
        self._seed(audit)
        json_path = str(tmp_path / "audit.json")
        count = audit.export_json(json_path)
        assert count == 3

        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)

        assert isinstance(data, list)
        assert len(data) == 3
        assert data[0]["category"] == "AUTH"

    def test_export_csv_with_date_filter(self, audit: AuditLogger, tmp_path) -> None:
        """Export with a far-future start_date should yield zero rows."""
        self._seed(audit)
        csv_path = str(tmp_path / "empty.csv")
        count = audit.export_csv(csv_path, start_date="2099-01-01")
        assert count == 0


# ------------------------------------------------------------------
# Immutability
# ------------------------------------------------------------------


class TestImmutability:
    """Verify the audit trail is append-only."""

    def test_immutable_no_delete(self, audit: AuditLogger, tmp_path) -> None:
        """The AuditLogger class provides no delete or update methods.

        Direct SQL DELETE/UPDATE on the database is outside the class contract.
        We verify that the public API cannot remove entries.
        """
        entry_id = audit.log(AuditCategory.SYSTEM, "test_action")
        # No delete or update method exists on AuditLogger
        assert not hasattr(audit, "delete")
        assert not hasattr(audit, "update")
        assert not hasattr(audit, "remove")
        # The entry still exists
        entries = audit.get_recent(limit=100)
        ids = [e.id for e in entries]
        assert entry_id in ids

    def test_entries_persist_across_instances(self, tmp_path) -> None:
        """Entries written by one instance should be visible to another."""
        db = str(tmp_path / "persist.db")
        a1 = AuditLogger(db_path=db)
        a1.log(AuditCategory.AUTH, "login", username="carol")

        a2 = AuditLogger(db_path=db)
        entries = a2.get_recent(limit=10)
        assert len(entries) == 1
        assert entries[0].username == "carol"
