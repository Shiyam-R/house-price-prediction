# ============================================================
# TEST_DRIFT_MONITOR.PY — DRIFT DETECTION TESTS
# ============================================================
# Tests app/drift_monitor.py directly (unit-level) using the real
# baseline_stats.json shipped with v1.0.0, rather than a fabricated
# baseline — this is what the API actually loads at startup, so
# testing against anything else would risk passing tests against
# behavior the real app doesn't exhibit.

import pytest

from app.drift_monitor import DriftMonitor


@pytest.fixture
def monitor():
    return DriftMonitor(
        baseline_path="artifacts/v1.0.0/baseline_stats.json",
        window_size=500,
        z_threshold=2.0,
    )


def test_baseline_loads_expected_features(monitor):
    assert "OverallQual" in monitor.baseline
    assert "TotalBsmtSF" in monitor.baseline
    assert monitor.baseline["OverallQual"]["mean"] == pytest.approx(6.0993, abs=0.01)


def test_report_before_any_data_shows_insufficient_data(monitor):
    report = monitor.get_report()
    assert report["min_samples_collected"] == 0
    assert report["features"]["OverallQual"]["status"] == "insufficient_data"


def test_recording_near_baseline_values_does_not_flag_drift(monitor):
    """
    Recording values close to the actual training mean should NOT
    trigger a drift flag — this guards against a monitor that's
    miscalibrated to be overly sensitive and cries wolf on normal,
    expected traffic.
    """
    baseline_mean = monitor.baseline["OverallQual"]["mean"]
    for _ in range(50):
        monitor.record({"OverallQual": round(baseline_mean)})

    report = monitor.get_report()
    assert report["features"]["OverallQual"]["drifting"] is False
    assert report["features_drifting"] == 0


def test_recording_far_shifted_values_triggers_drift_flag(monitor):
    """
    Deliberately records a value several standard deviations away
    from the training baseline, repeatedly — this MUST be flagged,
    or the monitor isn't doing its one job.
    """
    baseline_mean = monitor.baseline["OverallQual"]["mean"]
    baseline_std = monitor.baseline["OverallQual"]["std"]
    shifted_value = baseline_mean + (5 * baseline_std)  # 5 std devs away — unambiguous drift

    for _ in range(50):
        monitor.record({"OverallQual": shifted_value})

    report = monitor.get_report()
    assert report["features"]["OverallQual"]["drifting"] is True
    assert report["features"]["OverallQual"]["z_score"] > monitor.z_threshold
    assert report["features_drifting"] >= 1


def test_window_respects_max_size(monitor):
    """
    Recording more values than window_size should not grow the
    window unboundedly — older values must be dropped, keeping
    the signal representative of RECENT traffic only.
    """
    small_monitor = DriftMonitor(
        baseline_path="artifacts/v1.0.0/baseline_stats.json",
        window_size=10,
        z_threshold=2.0,
    )
    for i in range(50):
        small_monitor.record({"OverallQual": 5})

    assert len(small_monitor.windows["OverallQual"]) == 10


def test_record_ignores_untracked_fields_without_error(monitor):
    """
    Recording a payload that includes categorical fields (not
    tracked for drift) must not raise — the drift monitor is a
    side-channel and must never be able to break a real request.
    """
    monitor.record({
        "OverallQual": 6,
        "ExterQual": "Gd",  # categorical, not tracked — should be silently ignored
        "SomeUnknownField": 123,  # not in baseline at all — should also be ignored
    })
    report = monitor.get_report()
    assert report["features"]["OverallQual"]["sample_size"] == 1
