import numpy as np
import pandas as pd

def make_dataset(n_normal=1800, n_anomaly=300, seed=7, drift=False):
    """Generate illustrative data only; not a substitute for physical sensor data."""
    rng = np.random.default_rng(seed)
    temp_mean = 29.0 if drift else 26.0
    temp = rng.normal(temp_mean, 2.5, n_normal)
    humidity = np.clip(50 - .45 * (temp - temp_mean) + rng.normal(0, 5, n_normal), 15, 85)
    vibration = np.clip(rng.normal(.16, .035, n_normal) + .0015 * (temp - temp_mean), .02, None)
    normal = pd.DataFrame({"vibration_rms": vibration, "temperature_c": temp,
                           "humidity_pct": humidity, "label": 0})
    n1, n2 = n_anomaly // 3, n_anomaly // 3
    n3 = n_anomaly - n1 - n2
    abnormal = pd.DataFrame({
        "vibration_rms": np.r_[rng.normal(.48, .07, n1), rng.normal(.17, .04, n2), rng.normal(.55, .08, n3)],
        "temperature_c": np.r_[rng.normal(29, 3, n1), rng.normal(49, 4, n2), rng.normal(45, 5, n3)],
        "humidity_pct": np.r_[rng.normal(48, 8, n1), rng.normal(35, 10, n2), rng.normal(75, 8, n3)],
        "label": 1})
    return pd.concat([normal, abnormal], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
