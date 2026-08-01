# Model Card — ServiceRobot PdM LightGBM

## Overview

| Item | Value |
|---|---|
| Model ID | `robot-pdm-lightgbm-enhanced` |
| Task | 9-class real-time service-robot fault diagnosis |
| Artifact | `data/processed/robot_pdm_enhanced.txt` |
| Format | LightGBM native text model |
| Size | 4,505,940 bytes (4.30 MiB) |
| SHA256 | `fc858fe72e8ca241e05f0178d46ebcaf777c449407f19e2141c0cfc328c41171` |

This model diagnoses one normal state plus eight fault states from a 30-step robot
sensor window and static/context features. The design target is CPU-only,
low-latency inference for an operations dashboard and 3D digital twin.

## Validation

| Metric | Value | Notes |
|---|---:|---|
| Official validation accuracy | 0.9329 | AI-Hub official validation split; robots unseen during training |
| Official validation macro-F1 | 0.5838 | Penalized by rare classes with very low support |
| Majority baseline accuracy | 0.83 | Always predicting `정상` |
| Best iteration | 73 | LightGBM early-stopping checkpoint |

The official validation number is the primary deployment-like metric because it
tests generalization to robots not seen during training.

## Input Contract

- Window: 30 timesteps.
- Raw dynamic sensors: `batteryLevel`, `speed`, `x`, `y`, `degree`, `collision`, `obstacle`.
- Model dynamic sensors: `batteryLevel`, `speed`, `degree`, `collision`, `obstacle`.
- Excluded dynamic sensors: `x`, `y`, removed to reduce site-coordinate memorization.
- Static/context features: `isOffline`, `nowCharging`, `emergencyStop`, `batteryUse`, `batteryCycleCount`, `distance`, `crowd`, `deviceType`, `mainState`.
- Engineered feature count: 249.

Feature engineering flattens the retained sequence, appends mean/std/drift
features, appends the first 15 rFFT magnitudes per retained dynamic sensor, and
then appends static/context features.

## Known Limitations

- `E-RBT-N` and `E-RBT-S` have very small validation support, so per-class
  reliability is weaker than headline accuracy.
- Some faults overlap normal sensor ranges, which can produce missed detections.
- The live demo uses replay trajectories and deterministic synthetic sensor
  windows; `/api/snapshot` runs the LightGBM Booster live on each uncached AGV
  frame, but a physical robot/MQTT feed should still be validated before
  claiming field performance.

## Operations

- Serving endpoints: `/predict`, `/health`, `/model-card`, `/api/model-card`.
- Observability endpoints: `/metrics`, `/api/data-quality`, `/api/drift`, `/api/reliability`, `/api/work-orders`.
- Explainability: `/predict` returns LightGBM contribution-based top physical
  signal groups.
- Live twin inference: `/api/snapshot` exposes `inference.mode=live_booster`,
  per-AGV model latency, and replay-vs-live audit fields.
- Maintenance actions: `/api/work-orders` converts predicted faults and low
  health-index assets into P1/P2/P3 work orders with operator status tracking.
- Drift monitoring: `/api/drift` compares live `vib/batt/temp/health/conf` and
  fault rate against the reference operating profile.

Recommended retraining triggers:

- `/api/drift` remains `drift` for multiple monitoring windows.
- Data QA pass rate falls below the release threshold.
- A new robot type, site layout, or sensor calibration profile is introduced.
- Post-maintenance labels show rising missed-detection incidents.

The machine-readable version is available from both servers:

```bash
curl http://127.0.0.1:8000/model-card      # inference API
curl http://127.0.0.1:8000/api/model-card  # realtime twin API
```
