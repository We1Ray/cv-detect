"""Tests for shared.core.results_db -- SQLite inspection results database."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pytest

from shared.core.results_db import (
    InspectionRecord,
    ResultsDatabase,
    SPCMetrics,
    TrendData,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_record(
    *,
    timestamp: str | None = None,
    model_type: str = "autoencoder",
    anomaly_score: float = 0.5,
    is_defective: bool = False,
    batch_id: str | None = None,
    notes: str = "",
) -> InspectionRecord:
    """Create an InspectionRecord with sensible defaults."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    return InspectionRecord(
        timestamp=timestamp,
        image_path="/fake/image.png",
        model_type=model_type,
        anomaly_score=anomaly_score,
        threshold=0.7,
        is_defective=is_defective,
        defect_count=3 if is_defective else 0,
        total_defect_area=150 if is_defective else 0,
        max_defect_area=80 if is_defective else 0,
        batch_id=batch_id,
        notes=notes,
    )


def _make_sequential_records(
    n: int,
    base_time: datetime | None = None,
    model_type: str = "autoencoder",
    score_fn=None,
) -> List[InspectionRecord]:
    """Create n records with sequential timestamps."""
    if base_time is None:
        base_time = datetime(2025, 1, 1, 8, 0, 0)
    if score_fn is None:
        score_fn = lambda i: 0.3 + 0.02 * i
    records = []
    for i in range(n):
        ts = (base_time + timedelta(minutes=i)).isoformat()
        score = score_fn(i)
        records.append(
            _make_record(
                timestamp=ts,
                model_type=model_type,
                anomaly_score=score,
                is_defective=score > 0.7,
            )
        )
    return records


# ------------------------------------------------------------------
# TestDatabaseInit
# ------------------------------------------------------------------

class TestDatabaseInit:
    """Tests for database initialisation."""

    def test_creates_db_file(self, tmp_path: Path) -> None:
        """The database file should be created on construction."""
        db_path = str(tmp_path / "test.db")
        db = ResultsDatabase(db_path)

        assert os.path.exists(db_path)

    def test_tables_exist(self, tmp_path: Path) -> None:
        """The inspections table should exist after initialisation."""
        db_path = str(tmp_path / "test.db")
        db = ResultsDatabase(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='inspections'"
        )
        tables = cursor.fetchall()
        conn.close()

        assert len(tables) == 1
        assert tables[0][0] == "inspections"


# ------------------------------------------------------------------
# TestCRUD
# ------------------------------------------------------------------

class TestCRUD:
    """Tests for insert, get, update, and delete operations."""

    def test_insert_and_get(self, tmp_path: Path) -> None:
        """Insert a record and retrieve it by id; fields should match."""
        db = ResultsDatabase(str(tmp_path / "crud.db"))
        record = _make_record(
            model_type="patchcore",
            anomaly_score=0.85,
            is_defective=True,
            notes="test note",
        )

        row_id = db.insert_record(record)

        assert isinstance(row_id, int)
        assert row_id >= 1

        fetched = db.get_record(row_id)
        assert fetched is not None
        assert fetched.id == row_id
        assert fetched.model_type == "patchcore"
        assert fetched.anomaly_score == pytest.approx(0.85)
        assert fetched.is_defective is True
        assert fetched.notes == "test note"

    def test_insert_batch(self, tmp_path: Path) -> None:
        """Batch insert should return the correct number of ids."""
        db = ResultsDatabase(str(tmp_path / "batch.db"))
        records = _make_sequential_records(5)

        ids = db.insert_batch(records)

        assert len(ids) == 5
        assert len(set(ids)) == 5  # all unique

        for rid in ids:
            assert db.get_record(rid) is not None

    def test_update_record(self, tmp_path: Path) -> None:
        """Updating the notes field should persist the change."""
        db = ResultsDatabase(str(tmp_path / "update.db"))
        row_id = db.insert_record(_make_record(notes="original"))

        db.update_record(row_id, notes="updated note")

        fetched = db.get_record(row_id)
        assert fetched is not None
        assert fetched.notes == "updated note"

    def test_update_is_defective(self, tmp_path: Path) -> None:
        """Updating is_defective should work with boolean values."""
        db = ResultsDatabase(str(tmp_path / "update_def.db"))
        row_id = db.insert_record(_make_record(is_defective=False))

        db.update_record(row_id, is_defective=True)

        fetched = db.get_record(row_id)
        assert fetched is not None
        assert fetched.is_defective is True

    def test_delete_record(self, tmp_path: Path) -> None:
        """Deleting a record should make get_record return None."""
        db = ResultsDatabase(str(tmp_path / "delete.db"))
        row_id = db.insert_record(_make_record())

        db.delete_record(row_id)

        assert db.get_record(row_id) is None

    def test_update_invalid_column(self, tmp_path: Path) -> None:
        """Updating a non-existent column should raise ValueError."""
        db = ResultsDatabase(str(tmp_path / "invalid_col.db"))
        row_id = db.insert_record(_make_record())

        with pytest.raises(ValueError, match="Unknown column"):
            db.update_record(row_id, nonexistent_column="value")

    def test_get_nonexistent_record(self, tmp_path: Path) -> None:
        """Getting a record that does not exist should return None."""
        db = ResultsDatabase(str(tmp_path / "norecord.db"))

        assert db.get_record(9999) is None


# ------------------------------------------------------------------
# TestQuery
# ------------------------------------------------------------------

class TestQuery:
    """Tests for query_records with filters."""

    def test_query_all(self, tmp_path: Path) -> None:
        """Querying without filters should return all inserted records."""
        db = ResultsDatabase(str(tmp_path / "query.db"))
        db.insert_batch(_make_sequential_records(5))

        results = db.query_records()

        assert len(results) == 5

    def test_query_by_model_type(self, tmp_path: Path) -> None:
        """Filtering by model_type should return only matching records."""
        db = ResultsDatabase(str(tmp_path / "model.db"))
        db.insert_batch(_make_sequential_records(3, model_type="autoencoder"))
        db.insert_batch(_make_sequential_records(2, model_type="patchcore"))

        ae_results = db.query_records(model_type="autoencoder")
        pc_results = db.query_records(model_type="patchcore")

        assert len(ae_results) == 3
        assert all(r.model_type == "autoencoder" for r in ae_results)
        assert len(pc_results) == 2
        assert all(r.model_type == "patchcore" for r in pc_results)

    def test_query_defective_only(self, tmp_path: Path) -> None:
        """Filtering with defective_only should return only defective records."""
        db = ResultsDatabase(str(tmp_path / "defective.db"))
        records = [
            _make_record(anomaly_score=0.9, is_defective=True),
            _make_record(anomaly_score=0.3, is_defective=False),
            _make_record(anomaly_score=0.8, is_defective=True),
        ]
        db.insert_batch(records)

        results = db.query_records(defective_only=True)

        assert len(results) == 2
        assert all(r.is_defective for r in results)

    def test_query_by_date_range(self, tmp_path: Path) -> None:
        """Filtering by date range should return records within bounds."""
        db = ResultsDatabase(str(tmp_path / "dates.db"))
        base = datetime(2025, 6, 1, 10, 0, 0)
        records = _make_sequential_records(10, base_time=base)
        db.insert_batch(records)

        start = (base + timedelta(minutes=3)).isoformat()
        end = (base + timedelta(minutes=6)).isoformat()
        results = db.query_records(start_date=start, end_date=end)

        # Should include records with timestamps in [start, end]
        assert len(results) >= 1
        for r in results:
            assert r.timestamp >= start
            assert r.timestamp <= end

    def test_query_limit(self, tmp_path: Path) -> None:
        """The limit parameter should cap the number of returned records."""
        db = ResultsDatabase(str(tmp_path / "limit.db"))
        db.insert_batch(_make_sequential_records(10))

        results = db.query_records(limit=2)

        assert len(results) == 2


# ------------------------------------------------------------------
# TestSPCMetrics
# ------------------------------------------------------------------

class TestSPCMetrics:
    """Tests for compute_spc_metrics."""

    def test_basic_metrics(self, tmp_path: Path) -> None:
        """SPC metrics from known scores should have correct mean/std/ucl/lcl."""
        db = ResultsDatabase(str(tmp_path / "spc.db"))
        scores = [0.5] * 20
        records = []
        base = datetime(2025, 1, 1)
        for i, s in enumerate(scores):
            records.append(
                _make_record(
                    timestamp=(base + timedelta(minutes=i)).isoformat(),
                    anomaly_score=s,
                )
            )
        db.insert_batch(records)

        metrics = db.compute_spc_metrics(field="anomaly_score")

        assert metrics.mean == pytest.approx(0.5, abs=1e-6)
        assert metrics.std == pytest.approx(0.0, abs=1e-6)
        assert metrics.n_samples == 20

    def test_ucl_lcl_calculation(self, tmp_path: Path) -> None:
        """UCL and LCL should be mean +/- 3*std."""
        db = ResultsDatabase(str(tmp_path / "ucl.db"))
        rng = np.random.RandomState(42)
        scores = rng.normal(5.0, 1.0, 30).tolist()
        base = datetime(2025, 1, 1)
        records = [
            _make_record(
                timestamp=(base + timedelta(minutes=i)).isoformat(),
                anomaly_score=s,
            )
            for i, s in enumerate(scores)
        ]
        db.insert_batch(records)

        metrics = db.compute_spc_metrics(field="anomaly_score")

        arr = np.array(scores)
        expected_mean = float(np.mean(arr))
        expected_std = float(np.std(arr, ddof=1))

        assert metrics.mean == pytest.approx(expected_mean, abs=1e-4)
        assert metrics.std == pytest.approx(expected_std, abs=1e-4)
        assert metrics.ucl == pytest.approx(expected_mean + 3 * expected_std, abs=1e-4)
        assert metrics.lcl == pytest.approx(expected_mean - 3 * expected_std, abs=1e-4)

    def test_capability_indices(self, tmp_path: Path) -> None:
        """With USL/LSL, Cp and Cpk should be computed correctly."""
        db = ResultsDatabase(str(tmp_path / "cp.db"))
        rng = np.random.RandomState(42)
        scores = rng.normal(5.0, 0.5, 50).tolist()
        base = datetime(2025, 1, 1)
        records = [
            _make_record(
                timestamp=(base + timedelta(minutes=i)).isoformat(),
                anomaly_score=s,
            )
            for i, s in enumerate(scores)
        ]
        db.insert_batch(records)

        usl, lsl = 7.0, 3.0
        metrics = db.compute_spc_metrics(field="anomaly_score", usl=usl, lsl=lsl)

        assert metrics.cp is not None
        assert metrics.cpk is not None
        assert metrics.usl == usl
        assert metrics.lsl == lsl

        arr = np.array(scores)
        expected_std = float(np.std(arr, ddof=1))
        expected_mean = float(np.mean(arr))
        expected_cp = (usl - lsl) / (6 * expected_std)
        expected_cpk = min(
            (usl - expected_mean) / (3 * expected_std),
            (expected_mean - lsl) / (3 * expected_std),
        )
        assert metrics.cp == pytest.approx(expected_cp, abs=1e-4)
        assert metrics.cpk == pytest.approx(expected_cpk, abs=1e-4)

    def test_insufficient_data(self, tmp_path: Path) -> None:
        """Fewer than 2 records should return zero-valued metrics."""
        db = ResultsDatabase(str(tmp_path / "insuff.db"))
        db.insert_record(_make_record(anomaly_score=0.5))

        metrics = db.compute_spc_metrics(field="anomaly_score")

        assert metrics.mean == 0.0
        assert metrics.std == 0.0
        assert metrics.ucl == 0.0
        assert metrics.lcl == 0.0
        assert metrics.n_samples == 1

    def test_invalid_field(self, tmp_path: Path) -> None:
        """An invalid field name should raise ValueError."""
        db = ResultsDatabase(str(tmp_path / "invalid.db"))
        db.insert_record(_make_record())

        with pytest.raises(ValueError, match="Invalid field"):
            db.compute_spc_metrics(field="nonexistent_field")


# ------------------------------------------------------------------
# TestWesternElectric
# ------------------------------------------------------------------

class TestWesternElectric:
    """Tests for detect_out_of_control (Western Electric rules)."""

    def test_rule1_beyond_3sigma(self, tmp_path: Path) -> None:
        """A single extreme point should trigger rule 1."""
        db = ResultsDatabase(str(tmp_path / "rule1.db"))
        base = datetime(2025, 1, 1)
        # 19 normal points + 1 extreme outlier
        scores = [5.0] * 19 + [100.0]
        records = [
            _make_record(
                timestamp=(base + timedelta(minutes=i)).isoformat(),
                anomaly_score=s,
            )
            for i, s in enumerate(scores)
        ]
        db.insert_batch(records)

        violations = db.detect_out_of_control(field="anomaly_score", rules=[1])

        assert len(violations) >= 1
        rule_1_indices = [v["index"] for v in violations if v["rule"] == 1]
        assert 19 in rule_1_indices

    def test_rule4_same_side(self, tmp_path: Path) -> None:
        """8+ consecutive points above the mean should trigger rule 4."""
        db = ResultsDatabase(str(tmp_path / "rule4.db"))
        base = datetime(2025, 1, 1)
        # Mean will be around 5.5; first 10 points below, next 10 above
        scores = [1.0] * 10 + [10.0] * 10
        records = [
            _make_record(
                timestamp=(base + timedelta(minutes=i)).isoformat(),
                anomaly_score=s,
            )
            for i, s in enumerate(scores)
        ]
        db.insert_batch(records)

        violations = db.detect_out_of_control(field="anomaly_score", rules=[4])

        assert len(violations) > 0
        assert any(v["rule"] == 4 for v in violations)

    def test_no_violations(self, tmp_path: Path) -> None:
        """Data centred around the mean with small variance should have
        no rule violations."""
        db = ResultsDatabase(str(tmp_path / "norule.db"))
        base = datetime(2025, 1, 1)
        rng = np.random.RandomState(42)
        # Alternating slightly above/below mean to avoid run violations
        scores = []
        for i in range(20):
            if i % 2 == 0:
                scores.append(5.0 + rng.uniform(0.01, 0.1))
            else:
                scores.append(5.0 - rng.uniform(0.01, 0.1))

        records = [
            _make_record(
                timestamp=(base + timedelta(minutes=i)).isoformat(),
                anomaly_score=s,
            )
            for i, s in enumerate(scores)
        ]
        db.insert_batch(records)

        violations = db.detect_out_of_control(field="anomaly_score")

        assert violations == []


# ------------------------------------------------------------------
# TestExport
# ------------------------------------------------------------------

class TestExport:
    """Tests for export_to_csv and export_to_json."""

    def test_export_csv(self, tmp_path: Path) -> None:
        """Exported CSV should have correct header and row count."""
        db = ResultsDatabase(str(tmp_path / "export.db"))
        db.insert_batch(_make_sequential_records(5))

        csv_path = str(tmp_path / "output.csv")
        db.export_to_csv(csv_path)

        assert os.path.exists(csv_path)
        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 5
        assert "id" in reader.fieldnames
        assert "anomaly_score" in reader.fieldnames
        assert "model_type" in reader.fieldnames
        assert "timestamp" in reader.fieldnames

    def test_export_json(self, tmp_path: Path) -> None:
        """Exported JSON should be valid and contain the right number of records."""
        db = ResultsDatabase(str(tmp_path / "export.db"))
        db.insert_batch(_make_sequential_records(3))

        json_path = str(tmp_path / "output.json")
        db.export_to_json(json_path)

        assert os.path.exists(json_path)
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)

        assert isinstance(data, list)
        assert len(data) == 3
        assert "anomaly_score" in data[0]
        assert "model_type" in data[0]


# ------------------------------------------------------------------
# TestTrendData
# ------------------------------------------------------------------

class TestTrendData:
    """Tests for get_trend_data."""

    def test_basic_trend(self, tmp_path: Path) -> None:
        """Trend data should have matching lengths for timestamps, values,
        and moving_avg."""
        db = ResultsDatabase(str(tmp_path / "trend.db"))
        n = 25
        db.insert_batch(_make_sequential_records(n))

        trend = db.get_trend_data(field="anomaly_score", window=5)

        assert len(trend.timestamps) == n
        assert len(trend.values) == n
        assert len(trend.moving_avg) == n
        assert len(trend.ucl) == n
        assert len(trend.lcl) == n
        assert len(trend.mean_line) == n

    def test_empty_trend(self, tmp_path: Path) -> None:
        """An empty database should return empty TrendData."""
        db = ResultsDatabase(str(tmp_path / "empty_trend.db"))

        trend = db.get_trend_data(field="anomaly_score")

        assert trend.timestamps == []
        assert trend.values == []
        assert trend.moving_avg == []

    def test_moving_average_values(self, tmp_path: Path) -> None:
        """The moving average should be NaN-padded at the start and
        have correct values after the window."""
        db = ResultsDatabase(str(tmp_path / "ma.db"))
        base = datetime(2025, 1, 1)
        # Constant score of 1.0 -- moving average should also be 1.0
        records = [
            _make_record(
                timestamp=(base + timedelta(minutes=i)).isoformat(),
                anomaly_score=1.0,
            )
            for i in range(10)
        ]
        db.insert_batch(records)

        trend = db.get_trend_data(field="anomaly_score", window=3)

        # First (window-1) entries should be None (NaN padded)
        assert trend.moving_avg[0] is None
        assert trend.moving_avg[1] is None
        # After the window, values should be 1.0
        assert trend.moving_avg[2] == pytest.approx(1.0)


# ------------------------------------------------------------------
# TestSummary
# ------------------------------------------------------------------

class TestSummary:
    """Tests for get_summary."""

    def test_group_by_day(self, tmp_path: Path) -> None:
        """Grouping by day should aggregate records per day."""
        db = ResultsDatabase(str(tmp_path / "summary.db"))
        day1 = datetime(2025, 3, 1, 10, 0, 0)
        day2 = datetime(2025, 3, 2, 10, 0, 0)

        records_day1 = _make_sequential_records(3, base_time=day1)
        records_day2 = _make_sequential_records(2, base_time=day2)
        db.insert_batch(records_day1 + records_day2)

        summary = db.get_summary(group_by="day")

        assert len(summary) == 2
        # Each entry should have the expected keys
        for entry in summary:
            assert "period" in entry
            assert "total" in entry
            assert "defect_count" in entry
            assert "defect_rate" in entry
            assert "avg_score" in entry

        totals = [s["total"] for s in summary]
        assert sorted(totals) == [2, 3]

    def test_invalid_group_by(self, tmp_path: Path) -> None:
        """An invalid group_by value should raise ValueError."""
        db = ResultsDatabase(str(tmp_path / "bad_group.db"))

        with pytest.raises(ValueError, match="group_by"):
            db.get_summary(group_by="year")

    @pytest.mark.parametrize("group_by", ["hour", "day", "week", "month"])
    def test_valid_group_by_options(self, tmp_path: Path, group_by: str) -> None:
        """All valid group_by options should execute without error."""
        db = ResultsDatabase(str(tmp_path / f"group_{group_by}.db"))
        db.insert_batch(_make_sequential_records(5))

        summary = db.get_summary(group_by=group_by)

        assert isinstance(summary, list)
