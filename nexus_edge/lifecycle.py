import argparse
from sklearn.model_selection import train_test_split
from .data import make_dataset
from .model import fit_model, calibrate_model, evaluate_model
from .registry import ModelRegistry
from .promotion import assess_candidate

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--registry",default="models/registry")
    args=parser.parse_args()
    data=make_dataset(seed=11)
    train, holdout=train_test_split(data,test_size=.50,random_state=42,stratify=data.label)
    calibration, holdout2=train_test_split(holdout,test_size=.50,random_state=43,stratify=holdout.label)
    validation, test=train_test_split(holdout2,test_size=.50,random_state=44,stratify=holdout2.label)
    estimator=fit_model(train,seed=17)
    candidate=calibrate_model(estimator,calibration,max_calibration_fpr=.08)
    registry=ModelRegistry(args.registry); current=registry.load_active()
    decision=assess_candidate(current,candidate,validation)
    print("Calibration threshold:",candidate.threshold)
    print("Promotion decision:",decision)
    print("Independent test metrics (candidate):",evaluate_model(candidate,test))
    if decision["approved"]:
        version=registry.promote(candidate,decision["candidate_metrics"],notes="Calibrated candidate passed labelled validation gates.")
        print("PROMOTED:",version)
    else: print("NOT PROMOTED. Active model remains unchanged.")
if __name__=="__main__": main()
