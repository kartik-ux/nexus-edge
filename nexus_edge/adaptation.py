from dataclasses import dataclass, field
from .features import FEATURES, validate_reading

@dataclass
class AdaptiveBaseline:
    alpha: float = 0.02
    values: dict = field(default_factory=lambda: {
        "vibration_rms": .16, "temperature_c": 26.0, "humidity_pct": 50.0})
    updates: int = 0

    def update(self, reading, confirmed_normal=False):
        validate_reading(reading)
        if not confirmed_normal:
            return False
        for key in FEATURES:
            x = float(reading[key])
            self.values[key] = (1 - self.alpha) * self.values[key] + self.alpha * x
        self.updates += 1
        return True

    def deviation(self, reading):
        validate_reading(reading)
        scales = {"vibration_rms": .05, "temperature_c": 3.0, "humidity_pct": 8.0}
        return {k: abs(float(reading[k]) - self.values[k]) / scales[k] for k in FEATURES}
