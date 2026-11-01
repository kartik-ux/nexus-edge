from sklearn.model_selection import train_test_split
from nexus_edge.data import make_dataset
from nexus_edge.model import fit_model, calibrate_model
from nexus_edge.promotion import assess_candidate

def test_promotion_decision_has_required_fields():
    data=make_dataset(500,100)
    train, cal=train_test_split(data,test_size=.4,random_state=4,stratify=data.label)
    estimator=fit_model(train,seed=1)
    candidate=calibrate_model(estimator,cal)
    result=assess_candidate(None,candidate,cal)
    assert "approved" in result and "candidate_metrics" in result and "reasons" in result

def test_calibrated_model_predicts_expected_shape():
    data=make_dataset(500,100)
    train, cal=train_test_split(data,test_size=.4,random_state=4,stratify=data.label)
    model=calibrate_model(fit_model(train),cal)
    assert len(model.predict(cal[["vibration_rms","temperature_c","humidity_pct"]]))==len(cal)
