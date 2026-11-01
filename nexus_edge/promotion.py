"""Candidate-model quality gates on a fixed labelled validation set."""
from .model import evaluate_model

def assess_candidate(current_model, candidate_model, validation,
                     max_f1_drop=0.02, max_recall_drop=0.03, max_false_positive_rate=0.10):
    candidate = evaluate_model(candidate_model, validation)
    current = evaluate_model(current_model, validation) if current_model is not None else None
    reasons = []
    if candidate["false_positive_rate"] > max_false_positive_rate:
        reasons.append("candidate false-positive rate exceeds limit")
    if current is not None:
        if candidate["f1"] < current["f1"] - max_f1_drop:
            reasons.append("candidate F1 regresses beyond tolerance")
        if candidate["recall"] < current["recall"] - max_recall_drop:
            reasons.append("candidate recall regresses beyond tolerance")
    return {"approved": len(reasons) == 0, "reasons": reasons,
            "candidate_metrics": candidate, "current_metrics": current}
