FEATURES = ["vibration_rms", "temperature_c", "humidity_pct"]

def validate_reading(reading):
    missing = [k for k in FEATURES if k not in reading]
    if missing:
        raise ValueError(f"Missing sensor fields: {missing}")
    for key in FEATURES:
        value = float(reading[key])
        if not (-1e6 < value < 1e6):
            raise ValueError(f"Invalid sensor value for {key}: {value}")
