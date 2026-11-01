from .features import FEATURES, validate_reading

class EdgeEngine:
    def __init__(self, model, baseline, drift_monitor=None):
        self.model=model; self.baseline=baseline; self.drift_monitor=drift_monitor
    def score(self, reading):
        validate_reading(reading)
        row={k:[float(reading[k])] for k in FEATURES}
        prediction=int(self.model.predict(__import__("pandas").DataFrame(row))[0])
        normality=float(self.model.decision_function(__import__("pandas").DataFrame(row))[0])
        deviations=self.baseline.deviation(reading); anomaly=prediction==-1
        drift=self.drift_monitor.add(reading) if self.drift_monitor else None
        return {"status":"anomaly" if anomaly else "normal","is_anomaly":anomaly,"normality_score":normality,"baseline_deviation":deviations,"baseline_update_eligible":(not anomaly and max(deviations.values())<3.0),"drift":drift}
    def observe(self,reading,operator_confirmed_normal=False):
        result=self.score(reading); accepted=result["baseline_update_eligible"] and operator_confirmed_normal
        result["baseline_updated"]=self.baseline.update(reading,confirmed_normal=True) if accepted else False
        return result
