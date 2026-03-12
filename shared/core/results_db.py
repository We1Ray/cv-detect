"""SQLite-based inspection results database with SPC analytics.

Provides structured storage for inspection results, replacing CSV export.
Includes Statistical Process Control (SPC) metrics, Western Electric rule
detection, and matplotlib-based chart generation.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from shared.op_logger import log_operation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InspectionRecord:
    """Single inspection result row."""

    id: Optional[int] = None
    timestamp: str = ""                # ISO format
    image_path: str = ""
    model_type: str = ""               # "autoencoder", "patchcore", "variation_model"
    anomaly_score: float = 0.0
    threshold: float = 0.0
    is_defective: bool = False
    defect_count: int = 0
    total_defect_area: int = 0
    max_defect_area: int = 0
    batch_id: Optional[str] = None     # production batch identifier
    line_id: Optional[str] = None      # production line
    notes: str = ""
    metadata: str = ""                 # JSON string for extra data


@dataclass
class SPCMetrics:
    """SPC control chart statistics."""

    mean: float                        # X-bar
    std: float
    ucl: float                         # Upper Control Limit (mean + 3*std)
    lcl: float                         # Lower Control Limit (mean - 3*std)
    usl: Optional[float] = None       # Upper Spec Limit (user-defined)
    lsl: Optional[float] = None       # Lower Spec Limit (user-defined)
    cp: Optional[float] = None        # Process Capability
    cpk: Optional[float] = None       # Process Capability Index
    pp: Optional[float] = None        # Process Performance
    ppk: Optional[float] = None       # Process Performance Index
    n_samples: int = 0
    n_out_of_control: int = 0


@dataclass
class TrendData:
    """Time-series trend data for control charts."""

    timestamps: List[str] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    moving_avg: List[float] = field(default_factory=list)      # moving average
    ucl: List[float] = field(default_factory=list)              # UCL line
    lcl: List[float] = field(default_factory=list)              # LCL line
    mean_line: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    image_path TEXT,
    model_type TEXT,
    anomaly_score REAL,
    threshold REAL,
    is_defective INTEGER,
    defect_count INTEGER,
    total_defect_area INTEGER,
    max_defect_area INTEGER,
    batch_id TEXT,
    line_id TEXT,
    notes TEXT,
    metadata TEXT
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_timestamp ON inspections(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_batch ON inspections(batch_id);",
    "CREATE INDEX IF NOT EXISTS idx_defective ON inspections(is_defective);",
]

_INSERT_SQL = """
INSERT INTO inspections (
    timestamp, image_path, model_type, anomaly_score, threshold,
    is_defective, defect_count, total_defect_area, max_defect_area,
    batch_id, line_id, notes, metadata
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_COLUMNS = [
    "id", "timestamp", "image_path", "model_type", "anomaly_score",
    "threshold", "is_defective", "defect_count", "total_defect_area",
    "max_defect_area", "batch_id", "line_id", "notes", "metadata",
]

# Date grouping expressions for SQLite
_GROUP_BY_EXPR = {
    "hour": "strftime('%Y-%m-%d %H:00', timestamp)",
    "day": "strftime('%Y-%m-%d', timestamp)",
    "week": "strftime('%Y-W%W', timestamp)",
    "month": "strftime('%Y-%m', timestamp)",
}


# ---------------------------------------------------------------------------
# Helper: row -> InspectionRecord
# ---------------------------------------------------------------------------

def _row_to_record(row: sqlite3.Row) -> InspectionRecord:
    """Convert a sqlite3.Row to an InspectionRecord."""
    d = dict(row)
    d["is_defective"] = bool(d.get("is_defective", 0))
    return InspectionRecord(**d)


def _record_to_values(record: InspectionRecord) -> Tuple:
    """Extract insertion values from record (excludes id)."""
    return (
        record.timestamp,
        record.image_path,
        record.model_type,
        record.anomaly_score,
        record.threshold,
        int(record.is_defective),
        record.defect_count,
        record.total_defect_area,
        record.max_defect_area,
        record.batch_id,
        record.line_id,
        record.notes,
        record.metadata,
    )


# ---------------------------------------------------------------------------
# ResultsDatabase
# ---------------------------------------------------------------------------

class ResultsDatabase:
    """SQLite-backed inspection results store with SPC analytics.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.  Created automatically if it
        does not exist.
    """

    def __init__(self, db_path: str = "inspection_results.db") -> None:
        self.db_path = db_path
        self._init_db()

    # -- connection management ----------------------------------------------

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            for idx_sql in _CREATE_INDEXES_SQL:
                conn.execute(idx_sql)
            conn.commit()
        logger.info("Database initialised at %s", self.db_path)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that yields a connection with row_factory set."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # CRUD
    # -----------------------------------------------------------------------

    @log_operation(logger)
    def insert_record(self, record: InspectionRecord) -> int:
        """Insert a single inspection record and return its row id."""
        with self._connect() as conn:
            cur = conn.execute(_INSERT_SQL, _record_to_values(record))
            conn.commit()
            row_id: int = cur.lastrowid  # type: ignore[assignment]
            logger.debug("Inserted record id=%d", row_id)
            return row_id

    @log_operation(logger)
    def insert_batch(self, records: Sequence[InspectionRecord]) -> List[int]:
        """Bulk-insert multiple records and return their row ids."""
        ids: List[int] = []
        with self._connect() as conn:
            for rec in records:
                cur = conn.execute(_INSERT_SQL, _record_to_values(rec))
                ids.append(cur.lastrowid)  # type: ignore[arg-type]
            conn.commit()
        logger.info("Batch-inserted %d records", len(ids))
        return ids

    def get_record(self, record_id: int) -> Optional[InspectionRecord]:
        """Fetch a single record by id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM inspections WHERE id = ?", (record_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    @log_operation(logger)
    def query_records(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model_type: Optional[str] = None,
        batch_id: Optional[str] = None,
        defective_only: bool = False,
        limit: int = 1000,
    ) -> List[InspectionRecord]:
        """Flexible query with optional filters.

        Parameters
        ----------
        start_date, end_date : str, optional
            ISO-format date/datetime bounds (inclusive).
        model_type : str, optional
            Filter by model type.
        batch_id : str, optional
            Filter by production batch.
        defective_only : bool
            If *True*, return only defective records.
        limit : int
            Maximum number of records to return.
        """
        clauses: List[str] = []
        params: List[Any] = []

        if start_date is not None:
            clauses.append("timestamp >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append("timestamp <= ?")
            params.append(end_date)
        if model_type is not None:
            clauses.append("model_type = ?")
            params.append(model_type)
        if batch_id is not None:
            clauses.append("batch_id = ?")
            params.append(batch_id)
        if defective_only:
            clauses.append("is_defective = 1")

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM inspections{where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_record(r) for r in rows]

    def update_record(self, record_id: int, **kwargs: Any) -> None:
        """Update specific fields on an existing record.

        Parameters
        ----------
        record_id : int
            Primary key of the record to update.
        **kwargs
            Column-name / value pairs to set.  ``is_defective`` values
            are automatically cast to ``int`` for SQLite.
        """
        if not kwargs:
            return
        valid_cols = {f.name for f in fields(InspectionRecord)} - {"id"}
        for k in kwargs:
            if k not in valid_cols:
                raise ValueError(f"Unknown column: {k}")
        if "is_defective" in kwargs:
            kwargs["is_defective"] = int(kwargs["is_defective"])
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        params = list(kwargs.values()) + [record_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE inspections SET {set_clause} WHERE id = ?", params,
            )
            conn.commit()
        logger.debug("Updated record id=%d fields=%s", record_id, list(kwargs))

    def delete_record(self, record_id: int) -> None:
        """Delete a record by id."""
        with self._connect() as conn:
            conn.execute("DELETE FROM inspections WHERE id = ?", (record_id,))
            conn.commit()
        logger.debug("Deleted record id=%d", record_id)

    # -----------------------------------------------------------------------
    # Aggregation / summary
    # -----------------------------------------------------------------------

    @log_operation(logger)
    def get_summary(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        group_by: str = "day",
    ) -> List[Dict[str, Any]]:
        """Return aggregate statistics grouped by time period.

        Parameters
        ----------
        group_by : str
            One of ``"hour"``, ``"day"``, ``"week"``, ``"month"``.

        Returns
        -------
        list of dict
            Each dict contains: ``period``, ``total``, ``defect_count``,
            ``defect_rate``, ``avg_score``.
        """
        if group_by not in _GROUP_BY_EXPR:
            raise ValueError(
                f"group_by must be one of {list(_GROUP_BY_EXPR)}, got {group_by!r}"
            )
        expr = _GROUP_BY_EXPR[group_by]

        clauses: List[str] = []
        params: List[Any] = []
        if start_date is not None:
            clauses.append("timestamp >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append("timestamp <= ?")
            params.append(end_date)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT
                {expr} AS period,
                COUNT(*) AS total,
                SUM(is_defective) AS defect_count,
                ROUND(CAST(SUM(is_defective) AS REAL) / COUNT(*), 4) AS defect_rate,
                ROUND(AVG(anomaly_score), 4) AS avg_score
            FROM inspections
            {where}
            GROUP BY period
            ORDER BY period;
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # SPC analytics
    # -----------------------------------------------------------------------

    _ALLOWED_FIELDS = frozenset({
        "anomaly_score", "threshold", "defect_count",
        "total_defect_area", "max_defect_area",
    })

    def _fetch_field_values(
        self,
        field_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[List[str], List[float]]:
        """Fetch (timestamps, values) for a numeric field."""
        if field_name not in self._ALLOWED_FIELDS:
            raise ValueError(f"Invalid field: {field_name!r}")
        clauses: List[str] = []
        params: List[Any] = []
        if start_date is not None:
            clauses.append("timestamp >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append("timestamp <= ?")
            params.append(end_date)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT timestamp, {field_name} FROM inspections{where} ORDER BY timestamp"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        timestamps = [r["timestamp"] for r in rows]
        values = [float(r[field_name]) for r in rows]
        return timestamps, values

    @log_operation(logger)
    def compute_spc_metrics(
        self,
        field: str = "anomaly_score",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        usl: Optional[float] = None,
        lsl: Optional[float] = None,
    ) -> SPCMetrics:
        """Compute SPC control chart statistics for a numeric field.

        Parameters
        ----------
        field : str
            Database column to analyse.
        usl, lsl : float, optional
            Upper / Lower specification limits for capability indices.

        Returns
        -------
        SPCMetrics
        """
        _, values = self._fetch_field_values(field, start_date, end_date)
        if len(values) < 2:
            return SPCMetrics(
                mean=0.0, std=0.0, ucl=0.0, lcl=0.0,
                n_samples=len(values),
            )

        arr = np.array(values, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))

        ucl_val = mean + 3 * std
        lcl_val = mean - 3 * std

        n_ooc = int(np.sum((arr > ucl_val) | (arr < lcl_val)))

        cp: Optional[float] = None
        cpk: Optional[float] = None
        pp: Optional[float] = None
        ppk: Optional[float] = None

        if usl is not None and lsl is not None and std > 0:
            cp = (usl - lsl) / (6 * std)
            cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))

            # Pp / Ppk use population std (ddof=0)
            std_pop = float(np.std(arr, ddof=0))
            if std_pop > 0:
                pp = (usl - lsl) / (6 * std_pop)
                ppk = min(
                    (usl - mean) / (3 * std_pop),
                    (mean - lsl) / (3 * std_pop),
                )

        return SPCMetrics(
            mean=mean,
            std=std,
            ucl=ucl_val,
            lcl=lcl_val,
            usl=usl,
            lsl=lsl,
            cp=cp,
            cpk=cpk,
            pp=pp,
            ppk=ppk,
            n_samples=len(values),
            n_out_of_control=n_ooc,
        )

    @log_operation(logger)
    def get_trend_data(
        self,
        field: str = "anomaly_score",
        window: int = 20,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> TrendData:
        """Return time-series trend data with moving average and control lines.

        Parameters
        ----------
        field : str
            Numeric database column.
        window : int
            Moving-average window size.
        """
        timestamps, values = self._fetch_field_values(field, start_date, end_date)
        if not values:
            return TrendData()

        arr = np.array(values, dtype=np.float64)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        ucl_val = mean_val + 3 * std_val
        lcl_val = mean_val - 3 * std_val

        # Moving average (pad shorter sequences with NaN)
        if len(arr) >= window:
            kernel = np.ones(window) / window
            ma = np.convolve(arr, kernel, mode="valid")
            # Pad the beginning so lengths match
            pad = np.full(window - 1, np.nan)
            ma = np.concatenate([pad, ma])
        else:
            ma = np.full(len(arr), np.nan)

        return TrendData(
            timestamps=timestamps,
            values=values,
            moving_avg=[float(v) if not np.isnan(v) else None for v in ma],  # type: ignore[misc]
            ucl=[ucl_val] * len(values),
            lcl=[lcl_val] * len(values),
            mean_line=[mean_val] * len(values),
        )

    @log_operation(logger)
    def detect_out_of_control(
        self,
        field: str = "anomaly_score",
        rules: Optional[List[int]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Detect out-of-control points using Western Electric rules.

        Rules
        -----
        1. 1 point beyond 3 sigma.
        2. 2 of 3 consecutive points beyond 2 sigma (same side).
        3. 4 of 5 consecutive points beyond 1 sigma (same side).
        4. 8 consecutive points on the same side of the centre line.

        Parameters
        ----------
        rules : list of int, optional
            Which rules to apply (1-4).  Defaults to all four.

        Returns
        -------
        list of dict
            Each with keys ``rule``, ``index``, ``timestamp``, ``value``.
        """
        if rules is None:
            rules = [1, 2, 3, 4]

        timestamps, values = self._fetch_field_values(field, start_date, end_date)
        if len(values) < 2:
            return []

        arr = np.array(values, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        if std == 0:
            return []

        violations: List[Dict[str, Any]] = []
        seen: set[Tuple[int, int]] = set()  # (rule, index)

        def _add(rule: int, idx: int) -> None:
            key = (rule, idx)
            if key not in seen:
                seen.add(key)
                violations.append({
                    "rule": rule,
                    "index": idx,
                    "timestamp": timestamps[idx],
                    "value": values[idx],
                })

        # Rule 1: single point beyond 3 sigma
        if 1 in rules:
            for i, v in enumerate(arr):
                if abs(v - mean) > 3 * std:
                    _add(1, i)

        # Rule 2: 2 of 3 consecutive points beyond 2 sigma (same side)
        if 2 in rules:
            for i in range(2, len(arr)):
                window_vals = arr[i - 2: i + 1]
                above = np.sum(window_vals > mean + 2 * std)
                below = np.sum(window_vals < mean - 2 * std)
                if above >= 2:
                    for j in range(i - 2, i + 1):
                        if arr[j] > mean + 2 * std:
                            _add(2, j)
                if below >= 2:
                    for j in range(i - 2, i + 1):
                        if arr[j] < mean - 2 * std:
                            _add(2, j)

        # Rule 3: 4 of 5 consecutive points beyond 1 sigma (same side)
        if 3 in rules:
            for i in range(4, len(arr)):
                window_vals = arr[i - 4: i + 1]
                above = np.sum(window_vals > mean + std)
                below = np.sum(window_vals < mean - std)
                if above >= 4:
                    for j in range(i - 4, i + 1):
                        if arr[j] > mean + std:
                            _add(3, j)
                if below >= 4:
                    for j in range(i - 4, i + 1):
                        if arr[j] < mean - std:
                            _add(3, j)

        # Rule 4: 8 consecutive points on the same side of the centre
        if 4 in rules:
            run_above = 0
            run_below = 0
            for i, v in enumerate(arr):
                if v > mean:
                    run_above += 1
                    run_below = 0
                elif v < mean:
                    run_below += 1
                    run_above = 0
                else:
                    run_above = 0
                    run_below = 0

                if run_above >= 8:
                    for j in range(i - 7, i + 1):
                        _add(4, j)
                if run_below >= 8:
                    for j in range(i - 7, i + 1):
                        _add(4, j)

        violations.sort(key=lambda d: d["index"])
        return violations

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    @log_operation(logger)
    def export_to_csv(
        self,
        output_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        """Export records to a CSV file."""
        records = self.query_records(
            start_date=start_date, end_date=end_date, limit=999_999_999,
        )
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
            writer.writeheader()
            for rec in records:
                d = asdict(rec)
                d["is_defective"] = int(d["is_defective"])
                writer.writerow(d)
        logger.info("Exported %d records to %s", len(records), output_path)

    @log_operation(logger)
    def export_to_json(
        self,
        output_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        """Export records to a JSON file."""
        records = self.query_records(
            start_date=start_date, end_date=end_date, limit=999_999_999,
        )
        data = [asdict(r) for r in records]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        logger.info("Exported %d records to %s", len(records), output_path)

    # -----------------------------------------------------------------------
    # Visualization helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _fig_to_array(fig: plt.Figure) -> np.ndarray:
        """Render a matplotlib figure to an RGB numpy array and close it."""
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img = buf[:, :, :3].copy()  # RGBA -> RGB
        plt.close(fig)
        return img

    @log_operation(logger)
    def plot_control_chart(
        self,
        field: str = "anomaly_score",
        width: int = 800,
        height: int = 400,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> np.ndarray:
        """Generate an X-bar control chart as a numpy RGB image.

        Out-of-control points are highlighted in red.
        """
        trend = self.get_trend_data(field, start_date=start_date, end_date=end_date)
        if not trend.values:
            fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
            ax.set_title(f"Control Chart - {field}")
            return self._fig_to_array(fig)

        ooc = self.detect_out_of_control(field, start_date=start_date, end_date=end_date)
        ooc_indices = {v["index"] for v in ooc}

        x = np.arange(len(trend.values))
        vals = np.array(trend.values)
        normal_mask = np.array([i not in ooc_indices for i in range(len(vals))])

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Normal points
        ax.scatter(
            x[normal_mask], vals[normal_mask],
            c="steelblue", s=12, zorder=3, label="In control",
        )
        # Out-of-control points
        if not np.all(normal_mask):
            ax.scatter(
                x[~normal_mask], vals[~normal_mask],
                c="red", s=20, zorder=4, label="Out of control",
            )

        # Control lines
        if trend.mean_line:
            ax.axhline(trend.mean_line[0], color="green", linestyle="-", linewidth=1, label="Mean")
        if trend.ucl:
            ax.axhline(trend.ucl[0], color="red", linestyle="--", linewidth=1, label="UCL")
        if trend.lcl:
            ax.axhline(trend.lcl[0], color="red", linestyle="--", linewidth=1, label="LCL")

        ax.set_title(f"Control Chart - {field}")
        ax.set_xlabel("Sample")
        ax.set_ylabel(field)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        return self._fig_to_array(fig)

    @log_operation(logger)
    def plot_histogram(
        self,
        field: str = "anomaly_score",
        bins: int = 50,
        width: int = 600,
        height: int = 400,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        usl: Optional[float] = None,
        lsl: Optional[float] = None,
    ) -> np.ndarray:
        """Distribution histogram with normal curve overlay and spec lines."""
        _, values = self._fetch_field_values(field, start_date, end_date)

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        if not values:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
            ax.set_title(f"Histogram - {field}")
            return self._fig_to_array(fig)

        arr = np.array(values, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))

        ax.hist(arr, bins=bins, density=True, alpha=0.7, color="steelblue", edgecolor="white")

        # Normal curve overlay
        if std > 0:
            x_curve = np.linspace(float(arr.min()), float(arr.max()), 200)
            y_curve = (
                1.0 / (std * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((x_curve - mean) / std) ** 2)
            )
            ax.plot(x_curve, y_curve, "r-", linewidth=1.5, label="Normal fit")

        # Control limits
        ucl_val = mean + 3 * std
        lcl_val = mean - 3 * std
        ax.axvline(ucl_val, color="red", linestyle="--", linewidth=1, label="UCL")
        ax.axvline(lcl_val, color="red", linestyle="--", linewidth=1, label="LCL")

        # Spec limits
        if usl is not None:
            ax.axvline(usl, color="orange", linestyle="-.", linewidth=1, label="USL")
        if lsl is not None:
            ax.axvline(lsl, color="orange", linestyle="-.", linewidth=1, label="LSL")

        ax.set_title(f"Histogram - {field}")
        ax.set_xlabel(field)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        return self._fig_to_array(fig)

    @log_operation(logger)
    def plot_trend(
        self,
        field: str = "anomaly_score",
        width: int = 800,
        height: int = 400,
        window: int = 20,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> np.ndarray:
        """Time-series trend plot with moving average."""
        trend = self.get_trend_data(
            field, window=window, start_date=start_date, end_date=end_date,
        )

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        if not trend.values:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
            ax.set_title(f"Trend - {field}")
            return self._fig_to_array(fig)

        x = np.arange(len(trend.values))
        ax.plot(x, trend.values, ".-", color="steelblue", markersize=3, linewidth=0.8, label=field)

        # Moving average
        ma = np.array([v if v is not None else np.nan for v in trend.moving_avg])
        valid = ~np.isnan(ma)
        if np.any(valid):
            ax.plot(x[valid], ma[valid], "-", color="orange", linewidth=1.5, label=f"MA({window})")

        # Control lines
        if trend.mean_line:
            ax.axhline(trend.mean_line[0], color="green", linestyle="-", linewidth=1, label="Mean")
        if trend.ucl:
            ax.axhline(trend.ucl[0], color="red", linestyle="--", linewidth=1, label="UCL")
        if trend.lcl:
            ax.axhline(trend.lcl[0], color="red", linestyle="--", linewidth=1, label="LCL")

        ax.set_title(f"Trend - {field}")
        ax.set_xlabel("Sample")
        ax.set_ylabel(field)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        return self._fig_to_array(fig)

    @log_operation(logger)
    def plot_pareto(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        width: int = 600,
        height: int = 400,
    ) -> np.ndarray:
        """Pareto chart of defect frequency by model type."""
        clauses: List[str] = ["is_defective = 1"]
        params: List[Any] = []
        if start_date is not None:
            clauses.append("timestamp >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append("timestamp <= ?")
            params.append(end_date)

        where = " WHERE " + " AND ".join(clauses)
        sql = f"""
            SELECT model_type, COUNT(*) AS cnt
            FROM inspections
            {where}
            GROUP BY model_type
            ORDER BY cnt DESC;
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        fig, ax1 = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        if not rows:
            ax1.text(0.5, 0.5, "No defects found", ha="center", va="center", fontsize=14)
            ax1.set_title("Pareto Chart - Defects")
            return self._fig_to_array(fig)

        labels = [r["model_type"] or "unknown" for r in rows]
        counts = [r["cnt"] for r in rows]
        total = sum(counts)
        cumulative = np.cumsum(counts) / total * 100

        x_pos = np.arange(len(labels))
        ax1.bar(x_pos, counts, color="steelblue", edgecolor="white")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax1.set_ylabel("Count")
        ax1.set_title("Pareto Chart - Defects by Model Type")

        ax2 = ax1.twinx()
        ax2.plot(x_pos, cumulative, "ro-", markersize=5, linewidth=1.5)
        ax2.set_ylabel("Cumulative %")
        ax2.set_ylim(0, 110)

        fig.tight_layout()
        return self._fig_to_array(fig)
