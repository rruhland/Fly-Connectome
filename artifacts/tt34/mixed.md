# Combining learned camera timing and actuator response

With both shifts active, the raw-input pupil landed **five legal returns with both corrections**, four with the camera correction alone, and three with its frozen interface. The full correction matched the earlier camera-delay-only result on the same five releases. All five successful points ended before a second legal pupil reply. [Paired results](mixed-results.json) · [Experiment driver](../../scripts/experimental/tt34_mixed.py)

All three arms played the 16 releases from the actuator study. Each configuration introduced a three-exposure, 100 ms stereo delay and the same asymmetric change in the command increments on the five playing joints. All first deliveries matched down to the far paddle contact, near bounce and receiving-line measurement. The roster was held out from the direct policy, observation-model training, timing fit and actuator probes. It had already been used in the separate timing and actuator assessments. [Held roster](body-novel-plan.json.gz) · [Actuator study](body.md)

| Correction under both shifts | Near legal returns | Alternating legal returns | Near blade contacts |
| --- | ---: | ---: | ---: |
| Frozen interface | 3 | 22 | 9 |
| Carried camera timing and learned observation dynamics | 4 | 24 | 9 |
| Carried camera correction and learned actuator inverse | 5 | 26 | 11 |

The camera correction gained releases 5594 and 10505 and lost 10848. Adding the actuator inverse gained 11139. The independent actuator study had gained a second reply on 3025. Under the combined shifts, every arm landed only its first reply there. The undisturbed pupil scored six near returns and 28 alternating returns on this roster. [Frozen tapes](mixed-frozen.json.gz) · [Camera-only correction tapes](mixed-clock.json.gz) · [Combined correction tapes](mixed-both.json.gz) · [Undisturbed tapes](body-novel-stock.json.gz)

The pupil carried its learned camera age and reused the learned observation forward model. The actuator inverse came from the separate encoder-only fit, and a fresh encoder pulse selected that memory before the combined games. This evaluates composition of independently acquired corrections. Joint identification under simultaneous shifts was outside the experiment. With both corrections enabled, the pupil rescaled 1,046 command vectors and reached 4.65 rad/s peak measured joint speed. [Camera age fit](lag-fit.json.gz) · [Observation model](lag-bank.npz) · [Actuator fit](body-calibration.json.gz)

Command-only replay reproduced referee events for all 12 positive games. Full-CAD replay found zero penetrations across 17,663 physics steps, with 18.6 mm minimum clearance. [Frozen positive-game audit](mixed-frozen-audit.json) · [Camera-only positive-game audit](mixed-clock-audit.json) · [Combined positive-game audit](mixed-both-audit.json)

## Reproduce

Use the pinned measured launches from the [endurance reproduction instructions](../tt29/endurance.md).

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt34_mixed trial /tmp/tt25-rallies.json
python -m scripts.experimental.tt34_mixed summarize
for mode in frozen clock both; do
    python -m scripts.experimental.tt34_mixed audit "artifacts/tt34/mixed-$mode.json.gz" "artifacts/tt34/mixed-$mode-audit.json"
done
```
