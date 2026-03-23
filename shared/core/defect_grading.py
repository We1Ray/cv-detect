"""Defect severity grading and multi-class classification system.

Provides configurable rule-based grading for defect inspection results.
Supports weighted multi-criteria evaluation, batch grading, and Pareto
analysis to identify dominant defect types.

Typical usage::

    config = GradingConfig.from_json("grading_rules.json")
    grader = DefectGrader(config)

    result = grader.grade_from_score(score=0.85, defect_count=3, area_ratio=0.02)
    print(result.grade, result.reasons)

    batch = grader.grade_batch(detection_results)
    pareto = grader.pareto_analysis(detection_results)
"""
from __future__ import annotations

import json
import operator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Grade levels
# ---------------------------------------------------------------------------


class GradeLevel(Enum):
    """Defect severity grade from best to worst.

    Two naming conventions are provided.  The numeric ``value`` allows
    natural ordering (lower is better).
    """

    PASS = 0
    MINOR = 1
    MAJOR = 2
    CRITICAL = 3
    REJECT = 4

    # Convenience letter aliases ------------------------------------------------
    A = 0
    B = 1
    C = 2
    D = 3
    F = 4

    @classmethod
    def from_string(cls, name: str) -> "GradeLevel":
        """Resolve a grade from its name (case-insensitive)."""
        key = name.strip().upper()
        try:
            return cls[key]
        except KeyError:
            raise ValueError(
                f"Unknown grade level '{name}'. "
                f"Valid names: {[m.name for m in cls]}"
            )

    def __lt__(self, other: "GradeLevel") -> bool:
        if not isinstance(other, GradeLevel):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: "GradeLevel") -> bool:
        if not isinstance(other, GradeLevel):
            return NotImplemented
        return self.value <= other.value


# ---------------------------------------------------------------------------
# Operator look-up
# ---------------------------------------------------------------------------

_OPERATOR_MAP: Dict[str, Callable[[Any, Any], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GradeRule:
    """A single grading rule that maps a feature comparison to a grade.

    Attributes:
        field_name: Name of the feature to inspect (e.g. ``"area_ratio"``).
        op: Comparison operator as a string (``">"``, ``">="``, etc.).
        threshold: Numeric threshold for the comparison.
        grade: Grade to assign when the rule fires.
        weight: Optional importance weight used during weighted grading.
        description: Human-readable explanation of the rule.
    """

    field_name: str
    op: str
    threshold: float
    grade: GradeLevel
    weight: float = 1.0
    description: str = ""

    def evaluate(self, value: float) -> bool:
        """Return *True* if *value* satisfies this rule's condition."""
        cmp_fn = _OPERATOR_MAP.get(self.op)
        if cmp_fn is None:
            raise ValueError(
                f"Unsupported operator '{self.op}'. "
                f"Supported: {list(_OPERATOR_MAP.keys())}"
            )
        return cmp_fn(value, self.threshold)

    # Serialisation helpers ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "op": self.op,
            "threshold": self.threshold,
            "grade": self.grade.name,
            "weight": self.weight,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GradeRule":
        return cls(
            field_name=data["field_name"],
            op=data["op"],
            threshold=data["threshold"],
            grade=GradeLevel.from_string(data["grade"]),
            weight=data.get("weight", 1.0),
            description=data.get("description", ""),
        )


@dataclass
class GradingConfig:
    """Configuration container holding all grading rules and thresholds.

    Attributes:
        rules: Ordered list of :class:`GradeRule` instances.
        default_grade: Grade assigned when no rule fires.
        score_thresholds: Optional mapping of ``{grade_name: min_score}``
            used by :meth:`DefectGrader.grade_from_score`.
        use_worst_grade: If *True*, batch / multi-rule evaluation returns the
            worst (highest severity) grade among all fired rules.
    """

    rules: List[GradeRule] = field(default_factory=list)
    default_grade: GradeLevel = GradeLevel.PASS
    score_thresholds: Dict[str, float] = field(default_factory=dict)
    use_worst_grade: bool = True

    # --- JSON persistence ----------------------------------------------------

    def to_json(self, path: Union[str, Path]) -> None:
        """Serialise the configuration to a JSON file at *path*."""
        payload = {
            "rules": [r.to_dict() for r in self.rules],
            "default_grade": self.default_grade.name,
            "score_thresholds": self.score_thresholds,
            "use_worst_grade": self.use_worst_grade,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "GradingConfig":
        """Load configuration from a JSON file at *path*."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        rules = [GradeRule.from_dict(r) for r in raw.get("rules", [])]
        return cls(
            rules=rules,
            default_grade=GradeLevel.from_string(
                raw.get("default_grade", "PASS")
            ),
            score_thresholds=raw.get("score_thresholds", {}),
            use_worst_grade=raw.get("use_worst_grade", True),
        )

    @classmethod
    def default_config(cls) -> "GradingConfig":
        """Return a sensible factory-default configuration."""
        return cls(
            rules=[],
            default_grade=GradeLevel.PASS,
            score_thresholds={
                "PASS": 0.90,
                "MINOR": 0.70,
                "MAJOR": 0.40,
                "CRITICAL": 0.20,
                "REJECT": 0.0,
            },
            use_worst_grade=True,
        )


@dataclass
class GradingResult:
    """Outcome of a single grading evaluation.

    Attributes:
        grade: Final assigned grade.
        reasons: Human-readable reasons explaining the grade.
        scores: Per-rule or per-criterion numeric scores.
        fired_rules: Indices of the rules that evaluated to *True*.
        weighted_score: Combined weighted score (0.0 -- 1.0) if applicable.
    """

    grade: GradeLevel
    reasons: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    fired_rules: List[int] = field(default_factory=list)
    weighted_score: float = 0.0


# ---------------------------------------------------------------------------
# Pareto result
# ---------------------------------------------------------------------------


@dataclass
class ParetoEntry:
    """One row in a Pareto analysis table."""

    defect_type: str
    count: int
    percentage: float
    cumulative_percentage: float


# ---------------------------------------------------------------------------
# DefectGrader
# ---------------------------------------------------------------------------


class DefectGrader:
    """Rule-based defect grading engine.

    Parameters:
        config: A :class:`GradingConfig` instance.  If *None* the default
            configuration is used.
    """

    def __init__(self, config: Optional[GradingConfig] = None) -> None:
        self.config = config or GradingConfig.default_config()

    # ------------------------------------------------------------------
    # Score-based grading
    # ------------------------------------------------------------------

    def grade_from_score(
        self,
        score: float,
        defect_count: int = 0,
        area_ratio: float = 0.0,
    ) -> GradingResult:
        """Assign a grade based on a numeric *score* (0.0 -- 1.0).

        The grade is determined by comparing *score* against the ordered
        ``score_thresholds`` in the configuration.  The first threshold whose
        minimum value is <= *score* wins.

        Parameters:
            score: Overall quality / anomaly score in [0, 1].
            defect_count: Number of detected defects (informational).
            area_ratio: Ratio of defect area to total inspection area.

        Returns:
            A :class:`GradingResult` with the determined grade.
        """
        thresholds = self.config.score_thresholds
        if not thresholds:
            return GradingResult(
                grade=self.config.default_grade,
                reasons=["No score thresholds configured."],
                scores={"score": score, "defect_count": defect_count, "area_ratio": area_ratio},
            )

        # Sort thresholds descending by value so we match the highest first.
        ordered: List[Tuple[str, float]] = sorted(
            thresholds.items(), key=lambda kv: kv[1], reverse=True
        )

        reasons: List[str] = []
        assigned = self.config.default_grade
        for grade_name, min_score in ordered:
            if score >= min_score:
                assigned = GradeLevel.from_string(grade_name)
                reasons.append(
                    f"Score {score:.4f} >= threshold {min_score:.4f} -> {grade_name}"
                )
                break

        if not reasons:
            reasons.append(
                f"Score {score:.4f} below all thresholds -> {assigned.name}"
            )

        if defect_count > 0:
            reasons.append(f"Defect count: {defect_count}")
        if area_ratio > 0.0:
            reasons.append(f"Area ratio: {area_ratio:.6f}")

        return GradingResult(
            grade=assigned,
            reasons=reasons,
            scores={
                "score": score,
                "defect_count": float(defect_count),
                "area_ratio": area_ratio,
            },
        )

    # ------------------------------------------------------------------
    # Rule-based grading
    # ------------------------------------------------------------------

    def grade_from_rules(
        self, features: Dict[str, float]
    ) -> GradingResult:
        """Evaluate all configured rules against *features* and return a grade.

        If ``config.use_worst_grade`` is *True* the worst (highest severity)
        grade among all fired rules is returned.  Otherwise the first matching
        rule wins.

        When rules carry weights a ``weighted_score`` is computed as the
        weight-normalised sum of fired-rule severity values.

        Parameters:
            features: Mapping of feature names to their numeric values.

        Returns:
            A :class:`GradingResult`.
        """
        fired_indices: List[int] = []
        fired_grades: List[GradeLevel] = []
        reasons: List[str] = []
        total_weight = 0.0
        weighted_sum = 0.0

        for idx, rule in enumerate(self.config.rules):
            value = features.get(rule.field_name)
            if value is None:
                continue
            if rule.evaluate(value):
                fired_indices.append(idx)
                fired_grades.append(rule.grade)
                desc = rule.description or (
                    f"{rule.field_name} ({value}) {rule.op} {rule.threshold}"
                )
                reasons.append(f"Rule #{idx}: {desc} -> {rule.grade.name}")
                weighted_sum += rule.weight * rule.grade.value
                total_weight += rule.weight

        if not fired_grades:
            return GradingResult(
                grade=self.config.default_grade,
                reasons=["No rule matched; default grade assigned."],
                scores=features,
            )

        if self.config.use_worst_grade:
            final_grade = max(fired_grades)
        else:
            final_grade = fired_grades[0]

        w_score = (
            weighted_sum / (total_weight * GradeLevel.REJECT.value)
            if total_weight > 0 and GradeLevel.REJECT.value > 0
            else 0.0
        )

        return GradingResult(
            grade=final_grade,
            reasons=reasons,
            scores=features,
            fired_rules=fired_indices,
            weighted_score=round(w_score, 6),
        )

    # ------------------------------------------------------------------
    # Batch grading
    # ------------------------------------------------------------------

    def grade_batch(
        self,
        results_list: Sequence[Dict[str, float]],
    ) -> List[GradingResult]:
        """Grade a batch of inspection results.

        Each element of *results_list* must be a dictionary of feature
        values suitable for :meth:`grade_from_rules`.

        Returns:
            A list of :class:`GradingResult`, one per input element.
        """
        return [self.grade_from_rules(feat) for feat in results_list]

    # ------------------------------------------------------------------
    # Pareto analysis
    # ------------------------------------------------------------------

    @staticmethod
    def pareto_analysis(
        results_list: Sequence[Dict[str, Any]],
        type_key: str = "defect_type",
    ) -> List[ParetoEntry]:
        """Perform Pareto analysis on a list of defect records.

        Each record in *results_list* should contain a key named *type_key*
        whose value is the defect category string.  The output is sorted by
        descending frequency with cumulative percentages.

        Parameters:
            results_list: Sequence of defect record dicts.
            type_key: Key name that holds the defect-type label.

        Returns:
            A list of :class:`ParetoEntry` sorted from most to least frequent.
        """
        counts: Dict[str, int] = {}
        for rec in results_list:
            dtype = str(rec.get(type_key, "unknown"))
            counts[dtype] = counts.get(dtype, 0) + 1

        total = sum(counts.values()) or 1
        sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

        entries: List[ParetoEntry] = []
        cumulative = 0.0
        for dtype, cnt in sorted_items:
            pct = cnt / total * 100.0
            cumulative += pct
            entries.append(
                ParetoEntry(
                    defect_type=dtype,
                    count=cnt,
                    percentage=round(pct, 2),
                    cumulative_percentage=round(cumulative, 2),
                )
            )
        return entries

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def summarize_batch(
        grading_results: Sequence[GradingResult],
    ) -> Dict[str, Any]:
        """Produce a summary of a batch of grading results.

        Returns a dictionary with per-grade counts and an overall pass rate.
        """
        counts: Dict[str, int] = {g.name: 0 for g in GradeLevel}
        for r in grading_results:
            counts[r.grade.name] = counts.get(r.grade.name, 0) + 1

        total = len(grading_results) or 1
        pass_count = counts.get(GradeLevel.PASS.name, 0)

        return {
            "total": len(grading_results),
            "counts": counts,
            "pass_rate": round(pass_count / total, 4),
            "worst_grade": (
                max((r.grade for r in grading_results), default=GradeLevel.PASS).name
            ),
        }
