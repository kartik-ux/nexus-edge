from nexus_edge.adaptation import AdaptiveBaseline

def test_baseline_requires_confirmation():
    baseline = AdaptiveBaseline()
    original = baseline.values["temperature_c"]
    reading = {"vibration_rms": .16, "temperature_c": 30., "humidity_pct": 50.}
    assert not baseline.update(reading, confirmed_normal=False)
    assert baseline.values["temperature_c"] == original
    assert baseline.update(reading, confirmed_normal=True)
    assert baseline.values["temperature_c"] != original
