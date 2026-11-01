# NEXUS EDGE v0.2 — Drift-aware hybrid Edge AI

Version 0.2 adds:
- rolling-window drift monitoring using a transparent population-stability-style z-score;
- guarded local baseline adaptation;
- candidate model training;
- fixed labelled validation and promotion gates;
- model registry metadata and rollback-friendly versioned artifacts;
- tests for drift, adaptation, and promotion.

**Scope:** this remains a Python simulation/software prototype. Synthetic data is used by the demo. It is not yet running on Heltec hardware and does not yet implement federated learning, OTA, or LoRaWAN.

## Requirements
Python 3.10–3.12 recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run everything
```bash
python -m nexus_edge.demo
pytest -q
```

## Train a candidate and evaluate promotion
```bash
python -m nexus_edge.lifecycle --registry models/registry
```

The lifecycle command:
1. creates a synthetic labelled dataset;
2. splits training, validation and test data;
3. trains a candidate using confirmed-normal training rows only;
4. evaluates current and candidate on the same labelled validation set;
5. promotes only if candidate meets configured quality gates;
6. writes versioned model and metadata into the registry.

The test set is reported separately and is not used for promotion decisions.

## Run a prediction using the active model
```bash
python -m nexus_edge.predict --registry models/registry \
  --vibration-rms 0.18 --temperature-c 25 --humidity-pct 48
```

## Real sensor CSV
Use columns:
`vibration_rms,temperature_c,humidity_pct,label`

`label` is 0 for confirmed normal and 1 for confirmed anomaly. Do not use model-generated predictions as ground-truth labels. Replace the synthetic generator with measured sensor data before claiming field performance.

## Drift monitor
The drift monitor compares a recent window with a reference window using standardized mean shifts. It is a lightweight warning signal, not a formal statistical guarantee. Calibrate thresholds and minimum window sizes on real operating data.

## Promotion safety
A candidate is promoted only when:
- validation F1 is not worse than the current model by more than the configured tolerance;
- candidate false-positive rate is below the configured maximum;
- candidate recall is not worse than the current model beyond the configured tolerance.

Promotion is not automatic proof of safety. Review validation data, class balance, operating conditions and failure cases. Keep the prior model for rollback.

## Heltec deployment roadmap
The desktop scikit-learn model (`joblib`) cannot be copied directly to ESP32-S3. Next we will implement a compact feature pipeline and choose a microcontroller-compatible inference model (e.g. TFLite Micro or a small hand-coded model), then measure RAM, flash, latency and power on the exact Heltec board revision.
