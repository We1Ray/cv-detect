"""SPC real-time alerting module.

Monitors inspection scores and fires alerts based on Western Electric rules
and capability index thresholds.
"""

import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertRule(Enum):
    """Western Electric / Nelson rules for control chart interpretation."""

    BEYOND_3SIGMA = "beyond_3sigma"  # 1 point beyond 3 sigma
    TWO_OF_THREE_2SIGMA = "2of3_2sigma"  # 2 of 3 consecutive beyond 2 sigma
    FOUR_OF_FIVE_1SIGMA = "4of5_1sigma"  # 4 of 5 consecutive beyond 1 sigma
    EIGHT_SAME_SIDE = "8_same_side"  # 8 consecutive on same side of mean
    SIX_TRENDING = "6_trending"  # 6 consecutive increasing or decreasing
    CPK_LOW = "cpk_low"  # Cpk drops below threshold
    YIELD_LOW = "yield_low"  # Yield rate drops below threshold
    CONSECUTIVE_NG = "consecutive_ng"  # N consecutive NG results


@dataclass
class SPCAlert:
    """A single SPC alert event."""

    timestamp: str
    level: AlertLevel
    rule: AlertRule
    message: str
    value: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class SPCMonitor:
    """Real-time SPC monitoring with configurable alert rules.

    Maintains a rolling window of inspection scores and checks each new
    result against the Western Electric rules, Cpk thresholds, yield
    thresholds, and consecutive NG limits.

    Parameters
    ----------
    window_size : int
        Number of recent scores to keep for statistical calculations.
    cpk_threshold : float
        Minimum acceptable process capability index (Cpk).
    yield_threshold : float
        Minimum acceptable yield rate in percent.
    consecutive_ng_limit : int
        Number of consecutive NG results that triggers an alert.
    usl : float or None
        Upper specification limit for Cpk calculation.
    lsl : float or None
        Lower specification limit for Cpk calculation.
    """

    def __init__(
        self,
        window_size: int = 50,
        cpk_threshold: float = 1.33,
        yield_threshold: float = 95.0,
        consecutive_ng_limit: int = 3,
        usl: Optional[float] = None,
        lsl: Optional[float] = None,
    ) -> None:
        self._window_size = window_size
        self._cpk_threshold = cpk_threshold
        self._yield_threshold = yield_threshold
        self._consecutive_ng_limit = consecutive_ng_limit
        self._usl = usl
        self._lsl = lsl

        # Rolling window of scores
        self._scores: Deque[float] = deque(maxlen=window_size)

        # Pass/fail tracking for yield calculation
        self._total_count: int = 0
        self._pass_count: int = 0

        # Consecutive NG counter
        self._consecutive_ng_count: int = 0

        # Alert callbacks
        self._callbacks: List[Callable[[SPCAlert], None]] = []

    def add_result(self, score: float, is_pass: bool) -> List[SPCAlert]:
        """Process a new inspection result and return any triggered alerts.

        Parameters
        ----------
        score : float
            The inspection score (e.g. anomaly score, similarity score).
        is_pass : bool
            Whether the inspection result is a pass (OK) or fail (NG).

        Returns
        -------
        list of SPCAlert
            All alerts triggered by this result.
        """
        self._scores.append(score)
        self._total_count += 1
        if is_pass:
            self._pass_count += 1

        alerts: List[SPCAlert] = []

        # Check consecutive NG first (does not require statistical window)
        alert = self._check_consecutive_ng(is_pass)
        if alert is not None:
            alerts.append(alert)

        # Statistical checks require at least 2 data points for std
        if len(self._scores) >= 2:
            alert = self._check_beyond_3sigma(score)
            if alert is not None:
                alerts.append(alert)

            alert = self._check_2of3_2sigma()
            if alert is not None:
                alerts.append(alert)

            alert = self._check_4of5_1sigma()
            if alert is not None:
                alerts.append(alert)

            alert = self._check_8_same_side()
            if alert is not None:
                alerts.append(alert)

        # Trending check requires at least 6 data points
        if len(self._scores) >= 6:
            alert = self._check_6_trending()
            if alert is not None:
                alerts.append(alert)

        # Cpk check requires spec limits and enough data
        if len(self._scores) >= 2:
            alert = self._check_cpk()
            if alert is not None:
                alerts.append(alert)

        # Yield check
        alert = self._check_yield()
        if alert is not None:
            alerts.append(alert)

        # Fire callbacks for each alert
        for a in alerts:
            logger.warning("SPC Alert: [%s] %s - %s", a.level.value, a.rule.value, a.message)
            for cb in self._callbacks:
                try:
                    cb(a)
                except Exception:
                    logger.exception("Error in SPC alert callback")

        return alerts

    def register_callback(self, callback: Callable[[SPCAlert], None]) -> None:
        """Register a callback for alert notifications.

        The callback receives an ``SPCAlert`` instance each time an alert
        fires.
        """
        self._callbacks.append(callback)

    def get_current_metrics(self) -> Dict[str, Any]:
        """Return current SPC metrics.

        Returns a dictionary containing:
        - ``count``: total inspections processed
        - ``window_count``: number of scores in the rolling window
        - ``mean``: mean of the rolling window
        - ``std``: standard deviation of the rolling window
        - ``ucl``: upper control limit (mean + 3*std)
        - ``lcl``: lower control limit (mean - 3*std)
        - ``cpk``: current process capability index (None if spec limits not set)
        - ``yield_rate``: current yield rate in percent
        - ``consecutive_ng``: current consecutive NG count
        """
        n = len(self._scores)
        if n == 0:
            return {
                "count": self._total_count,
                "window_count": 0,
                "mean": None,
                "std": None,
                "ucl": None,
                "lcl": None,
                "cpk": None,
                "yield_rate": 100.0 if self._total_count == 0 else 0.0,
                "consecutive_ng": self._consecutive_ng_count,
            }

        mean = statistics.mean(self._scores)
        std = statistics.pstdev(self._scores) if n >= 2 else 0.0
        ucl = mean + 3 * std
        lcl = mean - 3 * std

        cpk = self._compute_cpk(mean, std)

        yield_rate = (self._pass_count / self._total_count * 100.0) if self._total_count > 0 else 100.0

        return {
            "count": self._total_count,
            "window_count": n,
            "mean": mean,
            "std": std,
            "ucl": ucl,
            "lcl": lcl,
            "cpk": cpk,
            "yield_rate": yield_rate,
            "consecutive_ng": self._consecutive_ng_count,
        }

    def reset(self) -> None:
        """Reset all counters and history."""
        self._scores.clear()
        self._total_count = 0
        self._pass_count = 0
        self._consecutive_ng_count = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mean_std(self) -> tuple:
        """Return (mean, std) of the current window using population stdev."""
        mean = statistics.mean(self._scores)
        std = statistics.pstdev(self._scores)
        return mean, std

    def _compute_cpk(self, mean: float, std: float) -> Optional[float]:
        """Compute the process capability index Cpk.

        Cpk = min((USL - mean) / (3 * std), (mean - LSL) / (3 * std))

        Returns None if either spec limit is not set or std is zero.
        """
        if self._usl is None or self._lsl is None:
            return None
        if std <= 0:
            return None
        cpu = (self._usl - mean) / (3 * std)
        cpl = (mean - self._lsl) / (3 * std)
        return min(cpu, cpl)

    # ------------------------------------------------------------------
    # Rule checks
    # ------------------------------------------------------------------

    def _check_beyond_3sigma(self, score: float) -> Optional[SPCAlert]:
        """Rule: 1 point beyond 3 sigma from the mean."""
        mean, std = self._mean_std()
        if std <= 0:
            return None
        ucl = mean + 3 * std
        lcl = mean - 3 * std
        if score > ucl or score < lcl:
            return SPCAlert(
                timestamp=datetime.now().isoformat(),
                level=AlertLevel.CRITICAL,
                rule=AlertRule.BEYOND_3SIGMA,
                message=f"Score {score:.4f} exceeds 3-sigma limits [{lcl:.4f}, {ucl:.4f}]",
                value=score,
                details={"mean": mean, "std": std, "ucl": ucl, "lcl": lcl},
            )
        return None

    def _check_2of3_2sigma(self) -> Optional[SPCAlert]:
        """Rule: 2 of 3 consecutive points beyond 2 sigma on the same side."""
        if len(self._scores) < 3:
            return None
        mean, std = self._mean_std()
        if std <= 0:
            return None
        last3 = list(self._scores)[-3:]
        upper_2s = mean + 2 * std
        lower_2s = mean - 2 * std

        above = sum(1 for v in last3 if v > upper_2s)
        below = sum(1 for v in last3 if v < lower_2s)

        if above >= 2:
            return SPCAlert(
                timestamp=datetime.now().isoformat(),
                level=AlertLevel.WARNING,
                rule=AlertRule.TWO_OF_THREE_2SIGMA,
                message=f"2 of last 3 points above 2-sigma ({upper_2s:.4f})",
                value=last3[-1],
                details={"last3": last3, "upper_2sigma": upper_2s, "mean": mean, "std": std},
            )
        if below >= 2:
            return SPCAlert(
                timestamp=datetime.now().isoformat(),
                level=AlertLevel.WARNING,
                rule=AlertRule.TWO_OF_THREE_2SIGMA,
                message=f"2 of last 3 points below 2-sigma ({lower_2s:.4f})",
                value=last3[-1],
                details={"last3": last3, "lower_2sigma": lower_2s, "mean": mean, "std": std},
            )
        return None

    def _check_4of5_1sigma(self) -> Optional[SPCAlert]:
        """Rule: 4 of 5 consecutive points beyond 1 sigma on the same side."""
        if len(self._scores) < 5:
            return None
        mean, std = self._mean_std()
        if std <= 0:
            return None
        last5 = list(self._scores)[-5:]
        upper_1s = mean + std
        lower_1s = mean - std

        above = sum(1 for v in last5 if v > upper_1s)
        below = sum(1 for v in last5 if v < lower_1s)

        if above >= 4:
            return SPCAlert(
                timestamp=datetime.now().isoformat(),
                level=AlertLevel.WARNING,
                rule=AlertRule.FOUR_OF_FIVE_1SIGMA,
                message=f"4 of last 5 points above 1-sigma ({upper_1s:.4f})",
                value=last5[-1],
                details={"last5": last5, "upper_1sigma": upper_1s, "mean": mean, "std": std},
            )
        if below >= 4:
            return SPCAlert(
                timestamp=datetime.now().isoformat(),
                level=AlertLevel.WARNING,
                rule=AlertRule.FOUR_OF_FIVE_1SIGMA,
                message=f"4 of last 5 points below 1-sigma ({lower_1s:.4f})",
                value=last5[-1],
                details={"last5": last5, "lower_1sigma": lower_1s, "mean": mean, "std": std},
            )
        return None

    def _check_8_same_side(self) -> Optional[SPCAlert]:
        """Rule: 8 consecutive points on the same side of the mean."""
        if len(self._scores) < 8:
            return None
        mean, std = self._mean_std()
        last8 = list(self._scores)[-8:]

        all_above = all(v > mean for v in last8)
        all_below = all(v < mean for v in last8)

        if all_above:
            return SPCAlert(
                timestamp=datetime.now().isoformat(),
                level=AlertLevel.WARNING,
                rule=AlertRule.EIGHT_SAME_SIDE,
                message=f"8 consecutive points above mean ({mean:.4f})",
                value=last8[-1],
                details={"last8": last8, "mean": mean, "side": "above"},
            )
        if all_below:
            return SPCAlert(
                timestamp=datetime.now().isoformat(),
                level=AlertLevel.WARNING,
                rule=AlertRule.EIGHT_SAME_SIDE,
                message=f"8 consecutive points below mean ({mean:.4f})",
                value=last8[-1],
                details={"last8": last8, "mean": mean, "side": "below"},
            )
        return None

    def _check_6_trending(self) -> Optional[SPCAlert]:
        """Rule: 6 consecutive points steadily increasing or decreasing."""
        if len(self._scores) < 6:
            return None
        last6 = list(self._scores)[-6:]

        increasing = all(last6[i] < last6[i + 1] for i in range(5))
        decreasing = all(last6[i] > last6[i + 1] for i in range(5))

        if increasing:
            return SPCAlert(
                timestamp=datetime.now().isoformat(),
                level=AlertLevel.WARNING,
                rule=AlertRule.SIX_TRENDING,
                message=f"6 consecutive increasing points: {last6[0]:.4f} -> {last6[-1]:.4f}",
                value=last6[-1],
                details={"last6": last6, "direction": "increasing"},
            )
        if decreasing:
            return SPCAlert(
                timestamp=datetime.now().isoformat(),
                level=AlertLevel.WARNING,
                rule=AlertRule.SIX_TRENDING,
                message=f"6 consecutive decreasing points: {last6[0]:.4f} -> {last6[-1]:.4f}",
                value=last6[-1],
                details={"last6": last6, "direction": "decreasing"},
            )
        return None

    def _check_cpk(self) -> Optional[SPCAlert]:
        """Rule: Cpk drops below the configured threshold."""
        if self._usl is None or self._lsl is None:
            return None
        if len(self._scores) < 2:
            return None
        mean, std = self._mean_std()
        cpk = self._compute_cpk(mean, std)
        if cpk is None:
            return None
        if cpk < self._cpk_threshold:
            level = AlertLevel.CRITICAL if cpk < 1.0 else AlertLevel.WARNING
            return SPCAlert(
                timestamp=datetime.now().isoformat(),
                level=level,
                rule=AlertRule.CPK_LOW,
                message=f"Cpk={cpk:.3f} below threshold {self._cpk_threshold:.2f}",
                value=cpk,
                details={"cpk": cpk, "threshold": self._cpk_threshold, "mean": mean, "std": std},
            )
        return None

    def _check_yield(self) -> Optional[SPCAlert]:
        """Rule: yield rate drops below the configured threshold."""
        if self._total_count < 1:
            return None
        yield_rate = self._pass_count / self._total_count * 100.0
        if yield_rate < self._yield_threshold:
            level = AlertLevel.CRITICAL if yield_rate < self._yield_threshold - 5.0 else AlertLevel.WARNING
            return SPCAlert(
                timestamp=datetime.now().isoformat(),
                level=level,
                rule=AlertRule.YIELD_LOW,
                message=f"Yield {yield_rate:.1f}% below threshold {self._yield_threshold:.1f}%",
                value=yield_rate,
                details={
                    "yield_rate": yield_rate,
                    "threshold": self._yield_threshold,
                    "pass_count": self._pass_count,
                    "total_count": self._total_count,
                },
            )
        return None

    def _check_consecutive_ng(self, is_pass: bool) -> Optional[SPCAlert]:
        """Rule: N consecutive NG results."""
        if is_pass:
            self._consecutive_ng_count = 0
            return None

        self._consecutive_ng_count += 1

        if self._consecutive_ng_count >= self._consecutive_ng_limit:
            return SPCAlert(
                timestamp=datetime.now().isoformat(),
                level=AlertLevel.CRITICAL,
                rule=AlertRule.CONSECUTIVE_NG,
                message=f"{self._consecutive_ng_count} consecutive NG results (limit: {self._consecutive_ng_limit})",
                value=float(self._consecutive_ng_count),
                details={
                    "consecutive_count": self._consecutive_ng_count,
                    "limit": self._consecutive_ng_limit,
                },
            )
        return None
