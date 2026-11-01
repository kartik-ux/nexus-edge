from nexus_edge.drift import DriftMonitor

def test_drift_monitor_detects_shift():
    ref = {"vibration_rms": [.15] * 30, "temperature_c": [25.] * 30, "humidity_pct": [50.] * 30}
    monitor = DriftMonitor(ref, window_size=20, threshold=2.0, min_samples=10)
    status = None
    for _ in range(12):
        status = monitor.add({"vibration_rms": .15, "temperature_c": 40., "humidity_pct": 50.})
    assert status["ready"]
    assert status["drift"]
