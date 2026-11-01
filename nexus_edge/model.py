from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from .features import FEATURES

@dataclass
class CalibratedAnomalyModel:
    estimator: object
    threshold: float

    def decision_function(self, X):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=FEATURES)
        return self.estimator.decision_function(frame[FEATURES])

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores < self.threshold, -1, 1)

def fit_model(frame, contamination=0.05, seed=17):
    normal = frame.loc[frame["label"] == 0, FEATURES]
    if len(normal) < 50:
        raise ValueError("Need at least 50 confirmed-normal training rows.")
    estimator = make_pipeline(StandardScaler(), IsolationForest(n_estimators=300, contamination=contamination, random_state=seed))
    estimator.fit(normal)
    return estimator

def calibrate_model(estimator, calibration, max_calibration_fpr=0.08):
    y = calibration["label"].astype(int).to_numpy()
    scores = estimator.decision_function(calibration[FEATURES])
    normal_scores = scores[y == 0]
    if len(normal_scores) < 20 or np.sum(y == 1) < 5:
        raise ValueError("Calibration requires at least 20 normal and 5 anomaly examples.")
    # Candidate cut-offs are selected on calibration data only. Lower scores are more anomalous.
    thresholds = np.unique(np.quantile(normal_scores, np.linspace(0.01, max_calibration_fpr, 40)))
    best = None
    for threshold in thresholds:
        pred = (scores < threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        fpr = fp / max(1, fp + tn)
        if fpr <= max_calibration_fpr:
            f1 = f1_score(y, pred, zero_division=0)
            recall = recall_score(y, pred, zero_division=0)
            candidate = (f1, recall, float(threshold))
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise RuntimeError("Could not find a threshold within the calibration false-positive limit.")
    return CalibratedAnomalyModel(estimator=estimator, threshold=best[2])

def evaluate_model(model, frame):
    y_true = frame["label"].astype(int).to_numpy()
    y_pred = (model.predict(frame[FEATURES]) == -1).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "false_positive_rate": float(fp / max(1, fp + tn)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}

def save_model(model, path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(model,path)
def load_model(path): return joblib.load(path)
