from sklearn.model_selection import train_test_split
from .data import make_dataset
from .model import fit_model, evaluate_model
from .adaptation import AdaptiveBaseline
from .drift import DriftMonitor
from .engine import EdgeEngine

def main():
    data = make_dataset()
    train, test = train_test_split(data, test_size=.25, random_state=21, stratify=data.label)
    model = fit_model(train)
    print("Synthetic holdout metrics:", evaluate_model(model, test))

    normal_reference = train.loc[train.label == 0]
    drift = DriftMonitor(normal_reference, window_size=40, threshold=2.5)
    engine = EdgeEngine(model, AdaptiveBaseline(), drift)
    print("\\nStable operating readings:")
    for _, row in normal_reference.head(20).iterrows():
        reading = {k: float(row[k]) for k in ("vibration_rms", "temperature_c", "humidity_pct")}
        result = engine.observe(reading, operator_confirmed_normal=True)
    print("Last stable result:", result)

    print("\\nSimulated shifted readings:")
    shifted = make_dataset(n_normal=60, n_anomaly=0, seed=99, drift=True)
    for _, row in shifted.iterrows():
        reading = {k: float(row[k]) for k in ("vibration_rms", "temperature_c", "humidity_pct")}
        result = engine.observe(reading, operator_confirmed_normal=False)
    print("Last shifted result:", result)
    print("Note: synthetic demonstration only; drift is a warning, not an anomaly label.")
if __name__ == "__main__":
    main()
