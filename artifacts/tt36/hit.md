# Learning how a joint adjustment changes the outgoing ball

A short-range joint adjustment changed the direction and speed of the observed outgoing ball. The left-side adjustment selected by optical feedback raised outgoing vertical velocity on all three paired first hits with usable stereo observations. It did not make every return better. [Practice commands and observations](hit-practice.json.gz) · [Frozen hit checkpoint](hit-checkpoint.json) · [Controller and measurement](../../scripts/experimental/tt36_hit.py)

The raw pupil tried five bounded joint-angle adjustments on eight paired practice cards, with nominal control included. The adjustment blended in as the currently observed inbound ball moved near the arm. The learner detected a direction reversal in consecutive stereo positions, fitted the following short free-flight segment, and estimated velocity, net height and projected landing. It used these optical estimates to select the left- and right-side adjustments. The actual referee events were kept for scoring and verification. Practice took 40 physical games and 113.17 process CPU seconds.

The selected left adjustment increased vertical velocity by 0.54, 0.16 and 0.54 m/s across three matched observed first hits. The one comparable right-side first hit increased vertical velocity by 0.36 m/s. On release 3855 the left change took an observed missed return to a legal one, while on 13621 the opening legal return was lost. All 47 image-inferred reversals were within 90 ms of a physical near-paddle contact, with 27 ms median absolute difference. First-hit measurements were sparse, especially on the right, where each candidate supplied one usable optical first hit. [Per-action optical response estimates](hit-checkpoint.json)

The exported hook `HitAdjustment.bias(optical, measured_q)` returns five additive joint-angle corrections in radians. It uses the present ball observation, interpolates the two learned corrections across lateral position, and returns zero without a current observation. The caller can pass the sum of independent corrections to `JointPolicy.command(..., offset=...)`.

## Can the outgoing velocity be forecast from motor measurements?

Two fixed ridge regressions were fitted on 47 optically observed practice hits. Both used incoming ball position and velocity. The second also used measured joint angles, their change over the previous control tick, and the most recent issued joint command. The held test replayed 15 previously recorded stock-policy releases outside hit practice and omitted the releases reserved for the changed-continuation comparison. Six held impacts supplied readable optical velocity labels. The independent event audit placed all held input samples before physical contact. [Held and practice signal measurements](hit-prediction.json.gz) · [Forecast test](../../scripts/experimental/tt36_hit_predict.py)

| Input | Outgoing RMS by x, y, z (m/s) | Speed mean absolute error (m/s) | Median direction error | Correct lateral sign |
| --- | ---: | ---: | ---: | ---: |
| Incoming RGB | 0.741, 0.343, 0.560 | 0.278 | 8.97° | 3/5 |
| Incoming RGB and measured motor state | 0.687, 0.276, 0.403 | 0.245 | 5.74° | 4/5 |

Vector RMS fell from 0.990 to 0.843 m/s on those six held hits. Leaving each practice release out in turn also favored the motor inputs, reducing vector RMS from 1.193 to 0.991 m/s. Lateral sign counted only shots with an observed lateral velocity magnitude above 0.20 m/s. These forecasts evaluate the sensory-motor signal and are separate from the two biases used by the deployed hit adjustment.

## Legal returns with the frozen hit adjustment

On 32 new openings the nominal raw policy landed 11 legal replies and the hit adjustment landed 10. Both made three later replies. On the separate 16-game test with changed opponent continuations, the hit adjustment landed 17 versus 12 for nominal and added one later reply. The continuation test used four previously studied openings, so its result does not establish improved coverage of unfamiliar openings. [New opening control](paired-new-fixed.json.gz) · [New opening hit adjustment](paired-new-hit.json.gz) · [Continuation control](paired-replies-fixed.json.gz) · [Continuation hit adjustment](paired-replies-hit.json.gz)

## Reproduce

Use the pinned measured launches from the [endurance reproduction instructions](../tt29/endurance.md). The camera prediction measurement replays the saved commands physically to reconstruct encoders and held RGB observations.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt36_hit practice /tmp/tt25-rallies.json artifacts/tt36/hit-practice.json.gz
python -m scripts.experimental.tt36_hit fit artifacts/tt36/hit-practice.json.gz artifacts/tt36/hit-checkpoint.json
python -m scripts.experimental.tt36_hit_predict
```
