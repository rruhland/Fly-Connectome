# Matching stock actuator timing from contact-free encoder probes

The temporal model predicted encoder increments better than an instantaneous model. Its proposed input filter increased the actual mismatch with stock motion on the alternate arm. The test stopped at the registered bench gate, before playing any compound wrist releases. [Frozen fit and gate](drive-response-lock.json) · [Actual encoder data](drive-response-bench.json.gz)

The bench uses the installed stock servo and the alternate arm with its longer link, wrist payload and changed drive settings. At each starting pose both plants receive the same generic, signed five-axis rate pulses. The time series contains only issued rates and the corresponding public encoder readings. Ball contact is disabled. Fitting uses eight start and pulse pairs, with eight further pairs reserved for qualification. The positive and negative runs at a reset are differenced to suppress passive drift. This uses 112 physical, unscored probe trials of 72 frames each. [Plan and fixed base controller](drive-response-plan.json) · [Portable wrist policy](carried-wrist.md)

A separate 14-tap causal model estimates each joint's encoder increment from its own command history. The input filter fits three finite-difference corrections to make the predicted alternate response resemble the measured stock response, including its lag. The correction has unity steady-state command gain, bounded lead coefficients and clipped actuator commands. The stock route is the identity. The shuffle moves the same filters to different axes. The tracking comparison uses a fixed proportional correction toward the integrated desired joint angle. [Implementation](../../scripts/experimental/tt24_drive_response.py)

| Held normalized encoder-rate prediction error | Temporal model | Instantaneous model |
|---|---:|---:|
| Stock | 0.0494 | 0.0698 |
| Alternate | 0.0616 | 0.0886 |

| Actual alternate trajectories against the same unmodified stock commands | RMS angle discrepancy, milliradians |
|---|---:|
| Unmodified alternate | 12.75 |
| Matched temporal filter | 14.11 |
| Shuffled temporal filter | 14.29 |
| Tracking the integrated desired angle | 80.98 |

The temporal predictor passed the required 10% prediction improvement on both bodies. The physical gate required at least a 10% reduction in held angle discrepancy relative to the unmodified and tracking controls. The matched filter improved original wrist pitch from 5.14 to 4.41 milliradians, while discrepancy rose on the other four joints. This small per-axis prediction model captured useful temporal structure and did not yield a useful physical timing correction at the registered operating point. The held gameplay roster remains unused. [Full per-axis results](drive-response-bench.json.gz)

## Reproduction

Use the repository's table-tennis Python environment from the repository root.

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt24_drive_response
```
