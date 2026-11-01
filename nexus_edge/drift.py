"""Lightweight rolling-window drift warning based on standardized mean shift."""

from collections import deque

import numpy as np

from .features import FEATURES, validate_reading


class DriftMonitor:
    def __init__(self, reference, window_size=40, threshold=2.5, min_samples=15):
        self.reference = {}

        for key in FEATURES:
            if key not in reference:
                raise ValueError(f"Missing reference feature: {key}")

            values = np.asarray(reference[key], dtype=float)

            if values.ndim != 1 or len(values) < min_samples:
                raise ValueError(
                    f"Reference data for {key} must contain at least "
                    f"{min_samples} samples."
                )

            if not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid reference values for {key}.")

            self.reference[key] = values

        self.window = deque(maxlen=window_size)
        self.threshold = float(threshold)
        self.min_samples = int(min_samples)

        self.scales = {
            key: max(float(np.std(self.reference[key])), 1e-3)
            for key in FEATURES
        }

    def add(self, reading):
        validate_reading(reading)
        self.window.append({k: float(reading[k]) for k in FEATURES})
        return self.status()

    def status(self):
        if len(self.window) < self.min_samples:
            return {
                "drift": False,
                "ready": False,
                "max_shift": 0.0,
                "feature_shift": {},
            }

        shifts = {}

        for key in FEATURES:
            recent = np.array([r[key] for r in self.window])
            ref_mean = float(np.mean(self.reference[key]))

            shifts[key] = (
                abs(float(np.mean(recent)) - ref_mean) / self.scales[key]
            )

        maximum = max(shifts.values())

        return {
            "drift": maximum >= self.threshold,
            "ready": True,
            "max_shift": float(maximum),
            "feature_shift": shifts,
        }