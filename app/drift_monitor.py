# ============================================================
# DRIFT_MONITOR.PY — LIGHTWEIGHT DRIFT AWARENESS
# ============================================================
# Tracks a rolling window of recent numeric input feature values
# and compares their live mean against the model's training-time
# baseline (baseline_stats.json) using a z-score-style signal.
#
# DELIBERATE SCOPE, stated honestly rather than hidden:
#   - In-memory only. Resets on restart, and in a multi-worker or
#     multi-replica deployment, each process/instance keeps its
#     OWN independent window rather than a shared view of traffic.
#     Fine for a single-maintainer project's visibility into "is
#     something obviously off"; NOT a substitute for a real
#     drift-tracking backend (e.g. a shared store, or a dedicated
#     tool like Evidently/whylogs) in a multi-instance production
#     deployment.
#   - Numeric features only — categorical drift (frequency shift
#     in ExterQual/MSZoning/etc.) is a different technique and is
#     intentionally out of scope here.
#   - A simple z-score threshold, not a statistical hypothesis
#     test (e.g. KS-test) — appropriate for a lightweight signal,
#     not a rigorous drift detector.

import json
from collections import deque
from statistics import mean, stdev


class DriftMonitor:
    def __init__(self, baseline_path: str, window_size: int = 500, z_threshold: float = 2.0):
        with open(baseline_path) as f:
            self.baseline = json.load(f)

        self.window_size = window_size
        self.z_threshold = z_threshold

        # One bounded rolling window (deque) per tracked feature.
        # maxlen enforces the window automatically — once full,
        # adding a new value silently drops the oldest one, so
        # this always reflects RECENT traffic, not all-time history.
        self.windows = {
            feature: deque(maxlen=window_size) for feature in self.baseline
        }

    def record(self, input_data: dict) -> None:
        """
        Records one request's numeric feature values into the
        rolling windows. Called once per prediction request.
        Silently ignores any field not in the baseline (e.g.
        categorical fields), rather than raising — this is a
        monitoring side-channel and must never be able to break
        an actual prediction request.
        """
        for feature in self.baseline:
            if feature in input_data:
                self.windows[feature].append(input_data[feature])

    def get_report(self) -> dict:
        """
        Compares each feature's current rolling mean against its
        training baseline mean, expressed as a z-score:
            z = (current_mean - baseline_mean) / baseline_std
        A feature is flagged as "drifting" if |z| exceeds the
        configured threshold (default 2.0 — roughly two baseline
        standard deviations away from the training-time mean).
        """
        features = {}
        drifting_count = 0
        min_samples = min(len(w) for w in self.windows.values()) if self.windows else 0

        for feature, baseline_stats in self.baseline.items():
            window = self.windows[feature]
            sample_size = len(window)

            if sample_size < 2:
                # Not enough data yet for a meaningful comparison —
                # report this honestly rather than a misleading
                # z-score computed from 0-1 samples.
                features[feature] = {
                    "sample_size": sample_size,
                    "status": "insufficient_data"
                }
                continue

            current_mean = round(mean(window), 4)
            baseline_mean = baseline_stats["mean"]
            baseline_std = baseline_stats["std"]

            z_score = round((current_mean - baseline_mean) / baseline_std, 3) if baseline_std > 0 else 0.0
            is_drifting = abs(z_score) > self.z_threshold

            if is_drifting:
                drifting_count += 1

            features[feature] = {
                "sample_size": sample_size,
                "baseline_mean": baseline_mean,
                "current_mean": current_mean,
                "z_score": z_score,
                "drifting": is_drifting,
                "status": "ok"
            }

        return {
            "window_size_configured": self.window_size,
            "z_threshold": self.z_threshold,
            "min_samples_collected": min_samples,
            "features_drifting": drifting_count,
            "total_features_tracked": len(self.baseline),
            "features": features
        }
