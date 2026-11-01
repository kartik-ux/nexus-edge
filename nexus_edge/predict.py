import argparse
from .registry import ModelRegistry
from .adaptation import AdaptiveBaseline
from .engine import EdgeEngine

def main():
    p=argparse.ArgumentParser(); p.add_argument("--registry",default="models/registry")
    p.add_argument("--vibration-rms",type=float,required=True); p.add_argument("--temperature-c",type=float,required=True); p.add_argument("--humidity-pct",type=float,required=True)
    a=p.parse_args(); model=ModelRegistry(a.registry).load_active()
    if model is None: raise SystemExit("No active model. Run: python -m nexus_edge.lifecycle")
    engine=EdgeEngine(model,AdaptiveBaseline()); reading={"vibration_rms":a.vibration_rms,"temperature_c":a.temperature_c,"humidity_pct":a.humidity_pct}
    print(engine.observe(reading,operator_confirmed_normal=False))
if __name__=="__main__": main()
