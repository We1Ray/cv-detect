"""Tests for the SPC real-time alerting module."""
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.spc_alert import (
    AlertLevel,
    AlertRule,
    SPCAlert,
    SPCMonitor,
)


# ======================================================================
# Helpers
# ======================================================================

def _add_many(monitor: SPCMonitor, scores: list, is_pass: bool = True):
    """Feed a list of scores into the monitor, all with the same pass/fail."""
    all_alerts = []
    for s in scores:
        all_alerts.extend(monitor.add_result(s, is_pass=is_pass))
    return all_alerts


# ======================================================================
# Basic behaviour
# ======================================================================

class TestSPCMonitorBasic:
    def test_init_defaults(self):
        mon = SPCMonitor()
        metrics = mon.get_current_metrics()
        assert metrics["count"] == 0
        assert metrics["window_count"] == 0
        assert metrics["mean"] is None

    def test_single_result_no_alert(self):
        mon = SPCMonitor()
        alerts = mon.add_result(0.5, is_pass=True)
        # With only 1 data point, statistical checks are skipped
        assert len(alerts) == 0

    def test_normal_data_no_alerts(self):
        """A stable process centred on 1.0 with small noise should not alert."""
        mon = SPCMonitor(window_size=50)
        rng_values = [1.0 + 0.01 * (i % 5 - 2) for i in range(30)]
        alerts = _add_many(mon, rng_values, is_pass=True)
        # Filter out yield/cpk alerts that are unrelated to WE rules
        we_alerts = [a for a in alerts if a.rule not in (AlertRule.CPK_LOW, AlertRule.YIELD_LOW)]
        assert len(we_alerts) == 0

    def test_reset_clears_state(self):
        mon = SPCMonitor()
        mon.add_result(1.0, is_pass=True)
        mon.add_result(2.0, is_pass=False)
        mon.reset()
        metrics = mon.get_current_metrics()
        assert metrics["count"] == 0
        assert metrics["window_count"] == 0
        assert metrics["consecutive_ng"] == 0

    def test_metrics_after_results(self):
        mon = SPCMonitor(window_size=10)
        for v in [1.0, 2.0, 3.0]:
            mon.add_result(v, is_pass=True)
        metrics = mon.get_current_metrics()
        assert metrics["count"] == 3
        assert metrics["window_count"] == 3
        assert metrics["mean"] == pytest.approx(2.0)
        assert metrics["yield_rate"] == pytest.approx(100.0)


# ======================================================================
# Western Electric rule: beyond 3 sigma
# ======================================================================

class TestBeyond3Sigma:
    def test_point_beyond_3sigma_triggers_alert(self):
        mon = SPCMonitor(window_size=20)
        # Build a stable baseline
        baseline = [10.0] * 10
        _add_many(mon, baseline)
        # Inject a small spread so std > 0
        mon.add_result(10.1, is_pass=True)
        mon.add_result(9.9, is_pass=True)
        # Now inject an extreme outlier
        alerts = mon.add_result(100.0, is_pass=True)
        rules = [a.rule for a in alerts]
        assert AlertRule.BEYOND_3SIGMA in rules

    def test_point_within_3sigma_no_alert(self):
        mon = SPCMonitor(window_size=20)
        # Spread data to have a moderate std
        for v in [8.0, 9.0, 10.0, 11.0, 12.0]:
            mon.add_result(v, is_pass=True)
        alerts = mon.add_result(10.5, is_pass=True)
        rules = [a.rule for a in alerts]
        assert AlertRule.BEYOND_3SIGMA not in rules


# ======================================================================
# Western Electric rule: 2 of 3 beyond 2 sigma
# ======================================================================

class TestTwoOfThree2Sigma:
    def test_2of3_above_2sigma_triggers(self):
        mon = SPCMonitor(window_size=50)
        # Build baseline around 0 with std ~1
        baseline = [0.0, 0.5, -0.5, 0.2, -0.2, 0.1, -0.1, 0.3, -0.3, 0.0]
        _add_many(mon, baseline)
        # Push 3 high values; at least 2 should be beyond 2 sigma
        mon.add_result(5.0, is_pass=True)
        mon.add_result(0.0, is_pass=True)
        alerts = mon.add_result(5.0, is_pass=True)
        rules = [a.rule for a in alerts]
        assert AlertRule.TWO_OF_THREE_2SIGMA in rules


# ======================================================================
# Western Electric rule: 4 of 5 beyond 1 sigma
# ======================================================================

class TestFourOfFive1Sigma:
    def test_4of5_above_1sigma_triggers(self):
        mon = SPCMonitor(window_size=50)
        # Baseline around 0
        baseline = [0.0, 0.1, -0.1, 0.2, -0.2, 0.0, 0.1, -0.1, 0.0, 0.05]
        _add_many(mon, baseline)
        # Push 5 high values
        for _ in range(5):
            alerts = mon.add_result(3.0, is_pass=True)
        rules = [a.rule for a in alerts]
        assert AlertRule.FOUR_OF_FIVE_1SIGMA in rules


# ======================================================================
# Western Electric rule: 8 consecutive same side
# ======================================================================

class TestEightSameSide:
    def test_8_above_mean_triggers(self):
        mon = SPCMonitor(window_size=50)
        # Mix values around 0
        baseline = [0.0, 1.0, -1.0, 0.5, -0.5, 0.0]
        _add_many(mon, baseline)
        # Now add 8 values well above the current mean
        alerts_all = []
        for _ in range(8):
            alerts_all.extend(mon.add_result(10.0, is_pass=True))
        rules = [a.rule for a in alerts_all]
        assert AlertRule.EIGHT_SAME_SIDE in rules

    def test_mixed_sides_no_alert(self):
        mon = SPCMonitor(window_size=50)
        baseline = [0.0, 1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5]
        alerts = _add_many(mon, baseline)
        rules = [a.rule for a in alerts]
        assert AlertRule.EIGHT_SAME_SIDE not in rules


# ======================================================================
# Western Electric rule: 6 consecutive trending
# ======================================================================

class TestSixTrending:
    def test_6_increasing_triggers(self):
        mon = SPCMonitor(window_size=50)
        # Build some baseline
        baseline = [5.0, 5.0, 5.0]
        _add_many(mon, baseline)
        # Now add 6 strictly increasing values
        alerts_all = []
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            alerts_all.extend(mon.add_result(v, is_pass=True))
        rules = [a.rule for a in alerts_all]
        assert AlertRule.SIX_TRENDING in rules

    def test_6_decreasing_triggers(self):
        mon = SPCMonitor(window_size=50)
        baseline = [5.0, 5.0, 5.0]
        _add_many(mon, baseline)
        alerts_all = []
        for v in [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]:
            alerts_all.extend(mon.add_result(v, is_pass=True))
        rules = [a.rule for a in alerts_all]
        assert AlertRule.SIX_TRENDING in rules

    def test_non_monotonic_no_alert(self):
        mon = SPCMonitor(window_size=50)
        values = [1.0, 3.0, 2.0, 4.0, 3.5, 5.0, 4.5, 6.0]
        alerts = _add_many(mon, values)
        rules = [a.rule for a in alerts]
        assert AlertRule.SIX_TRENDING not in rules


# ======================================================================
# Cpk rule
# ======================================================================

class TestCpkLow:
    def test_cpk_low_triggers_alert(self):
        mon = SPCMonitor(
            window_size=50,
            cpk_threshold=1.33,
            usl=12.0,
            lsl=8.0,
        )
        # Data with high variance relative to spec limits -> low Cpk
        alerts_all = []
        for v in [8.5, 11.5, 8.5, 11.5, 9.0, 11.0]:
            alerts_all.extend(mon.add_result(v, is_pass=True))
        rules = [a.rule for a in alerts_all]
        assert AlertRule.CPK_LOW in rules

    def test_cpk_good_no_alert(self):
        mon = SPCMonitor(
            window_size=50,
            cpk_threshold=1.0,
            usl=15.0,
            lsl=5.0,
        )
        # Tight data within wide spec -> high Cpk
        alerts = _add_many(mon, [10.0, 10.01, 9.99, 10.02, 9.98])
        rules = [a.rule for a in alerts]
        assert AlertRule.CPK_LOW not in rules

    def test_no_spec_limits_no_cpk_alert(self):
        mon = SPCMonitor(window_size=50, cpk_threshold=1.33)
        alerts = _add_many(mon, [10.0, 10.1, 9.9, 10.2, 9.8])
        rules = [a.rule for a in alerts]
        assert AlertRule.CPK_LOW not in rules


# ======================================================================
# Yield rule
# ======================================================================

class TestYieldLow:
    def test_low_yield_triggers_alert(self):
        mon = SPCMonitor(yield_threshold=95.0)
        # 10 results: 8 pass, 2 fail = 80% yield
        for _ in range(8):
            mon.add_result(1.0, is_pass=True)
        alerts_all = []
        for _ in range(2):
            alerts_all.extend(mon.add_result(1.0, is_pass=False))
        rules = [a.rule for a in alerts_all]
        assert AlertRule.YIELD_LOW in rules

    def test_high_yield_no_alert(self):
        mon = SPCMonitor(yield_threshold=90.0)
        alerts = _add_many(mon, [1.0] * 20, is_pass=True)
        rules = [a.rule for a in alerts]
        assert AlertRule.YIELD_LOW not in rules


# ======================================================================
# Consecutive NG rule
# ======================================================================

class TestConsecutiveNG:
    def test_consecutive_ng_triggers(self):
        mon = SPCMonitor(consecutive_ng_limit=3)
        alerts_all = []
        for _ in range(3):
            alerts_all.extend(mon.add_result(1.0, is_pass=False))
        rules = [a.rule for a in alerts_all]
        assert AlertRule.CONSECUTIVE_NG in rules

    def test_ng_reset_on_pass(self):
        mon = SPCMonitor(consecutive_ng_limit=3)
        mon.add_result(1.0, is_pass=False)
        mon.add_result(1.0, is_pass=False)
        mon.add_result(1.0, is_pass=True)  # resets counter
        alerts = mon.add_result(1.0, is_pass=False)
        rules = [a.rule for a in alerts]
        assert AlertRule.CONSECUTIVE_NG not in rules

    def test_consecutive_ng_keeps_firing(self):
        mon = SPCMonitor(consecutive_ng_limit=2)
        mon.add_result(1.0, is_pass=False)
        alerts2 = mon.add_result(1.0, is_pass=False)
        alerts3 = mon.add_result(1.0, is_pass=False)
        # Both the 2nd and 3rd should trigger (count >= limit)
        assert any(a.rule == AlertRule.CONSECUTIVE_NG for a in alerts2)
        assert any(a.rule == AlertRule.CONSECUTIVE_NG for a in alerts3)


# ======================================================================
# Edge cases
# ======================================================================

class TestEdgeCases:
    def test_all_same_values_no_statistical_alerts(self):
        """When all values are identical, std=0 and sigma rules should not fire."""
        mon = SPCMonitor(window_size=20)
        alerts = _add_many(mon, [5.0] * 15)
        # With std=0, beyond_3sigma / 2of3 / 4of5 checks return None
        sigma_rules = {
            AlertRule.BEYOND_3SIGMA,
            AlertRule.TWO_OF_THREE_2SIGMA,
            AlertRule.FOUR_OF_FIVE_1SIGMA,
        }
        triggered = {a.rule for a in alerts}
        assert triggered.isdisjoint(sigma_rules)

    def test_empty_monitor_metrics(self):
        mon = SPCMonitor()
        metrics = mon.get_current_metrics()
        assert metrics["count"] == 0
        assert metrics["mean"] is None
        assert metrics["std"] is None
        assert metrics["cpk"] is None

    def test_single_data_point_metrics(self):
        mon = SPCMonitor()
        mon.add_result(42.0, is_pass=True)
        metrics = mon.get_current_metrics()
        assert metrics["count"] == 1
        assert metrics["mean"] == pytest.approx(42.0)
        assert metrics["std"] == pytest.approx(0.0)


# ======================================================================
# Callback mechanism
# ======================================================================

class TestCallbacks:
    def test_callback_fires_on_alert(self):
        received = []
        mon = SPCMonitor(consecutive_ng_limit=2)
        mon.register_callback(lambda alert: received.append(alert))
        mon.add_result(1.0, is_pass=False)
        mon.add_result(1.0, is_pass=False)
        assert len(received) >= 1
        assert isinstance(received[0], SPCAlert)

    def test_callback_exception_does_not_crash(self):
        def bad_callback(alert):
            raise RuntimeError("callback error")

        mon = SPCMonitor(consecutive_ng_limit=1)
        mon.register_callback(bad_callback)
        # Should not raise
        alerts = mon.add_result(1.0, is_pass=False)
        assert len(alerts) >= 1

    def test_multiple_callbacks(self):
        counters = {"a": 0, "b": 0}
        mon = SPCMonitor(consecutive_ng_limit=1)
        mon.register_callback(lambda alert: counters.__setitem__("a", counters["a"] + 1))
        mon.register_callback(lambda alert: counters.__setitem__("b", counters["b"] + 1))
        mon.add_result(1.0, is_pass=False)
        assert counters["a"] >= 1
        assert counters["b"] >= 1


# ======================================================================
# Alert data class
# ======================================================================

class TestSPCAlertDataClass:
    def test_alert_fields(self):
        alert = SPCAlert(
            timestamp="2024-01-01T00:00:00",
            level=AlertLevel.WARNING,
            rule=AlertRule.BEYOND_3SIGMA,
            message="test message",
            value=1.5,
            details={"key": "value"},
        )
        assert alert.level == AlertLevel.WARNING
        assert alert.rule == AlertRule.BEYOND_3SIGMA
        assert alert.value == 1.5
        assert alert.details["key"] == "value"

    def test_alert_default_values(self):
        alert = SPCAlert(
            timestamp="2024-01-01T00:00:00",
            level=AlertLevel.INFO,
            rule=AlertRule.CPK_LOW,
            message="test",
        )
        assert alert.value == 0.0
        assert alert.details == {}
