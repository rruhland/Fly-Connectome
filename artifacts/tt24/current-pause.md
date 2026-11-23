# Inspecting a changed motor interface before releasing the same held ball

This comparison reruns all three original 12-release single-arm courses. The five-joint wrist policy, encoder surprise monitor, motor-bank recognition thresholds and physical fixtures are unchanged. The controller can request inspection after it observes a public encoder response. The launcher accepts that request while still holding the ball. The actor continues to receive wrist RGB and joint observations, with no body identity, interface identity, fixture phase, or release timer as a policy input. The stock and altered bodies still use their earlier optical calibrations. [Runner](../../scripts/experimental/tt24_current_pause.py) · [Original controller](../../scripts/experimental/tt24_single_arm_course.py)

The fixture suspends its ordinary setup countdown during the additional measured-settling probes. The existing probe callback sends voluntary joint rates to the same physical arm, receives encoders, and returns toward the setup setpoint when needed. When inspection ends, the wrist controller initializes its observation history from the actual posture and resumes the remaining ordinary setup. The same arm state continues through pickup, inspection and discharge. The ordered action record includes every normal setup, probe, settling and live-ball tick. After discharge the fixture declines further inspection requests and the monitor can carry surprise to the next test. Each release admits at most one new inline inspection. [Per-stream actions and referee evidence](current-pause-stream-0.json.gz) · [Stream 1](current-pause-stream-1.json.gz) · [Stream 2](current-pause-stream-2.json.gz)

## Matched results

| Scope | Releases | Same-release inspection | Fixed-release adaptive | Uncompensated | Known map, original schedule |
|---|---:|---:|---:|---:|---:|
| Full original courses | 36 | 529 | 456 | 278 | 537 |
| First invisible switch on the altered arm | 3 | 35 | 2 | 1 | 75 |
| Restored stock body | 9 | 141 | 162 | 165 | 165 |

Counts are physical verified rebounds. The archived controls retain their fixed release and their original, separately measured opening inspections. The known-map benchmark therefore has the original release schedule. The equal-timing control replays the new condition's entire physical held prefix, copies its public controller history and uses the stale map at discharge. Its actual release time, joint positions, joint rates and wrist image are identical to those of the same-release treatment. [Aggregate and checks](current-pause-summary.json.gz)

| Original stream | Ordinary setup ticks | Surprise at tick | Added probe and settling ticks | Rebounds with current inspection | Equal-setup stale map | Fixed-release adaptive |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 19 | 6 | 151 | 1 | 1 | 1 |
| 1 | 26 | 11 | 136 | 9 | 1 | 1 |
| 2 | 15 | 6 | 148 | 25 | 2 | 0 |

The first invisible switch triggered in time in all three streams. Relative matrix error at release ranged from 0.20% to 1.06%. The 35 versus 4 rebound contrast at identical discharge pose and timing came from two streams. The other stream remained at one rebound despite a closely measured map. The three invisible switches consumed 7.25 s of added held time, or 2.27–2.52 s per release.

Eight inline inspections across the course used 914 ticks, or 15.23 s. Including every opening inspection and ordinary preparation, held time rose from 41.07 s in the fixed-release adaptive condition to 47.18 s with current inspection. Some inspections shifted from the next opening to the current release. The restored body made four inline inspections totaling 6.22 s and scored 21 fewer rebounds than fixed-release adaptation. The three courses show a benefit at the silent wiring change with a cost when revisiting the stock fixture. [Original paired release schedules](single-arm-course-0.json.gz) · [Stream 1](single-arm-course-1.json.gz) · [Stream 2](single-arm-course-2.json.gz)

All 108 archived-baseline tapes, 36 current-release tapes and seven equal-setup controls replayed in order to the same complete physical metrics. After each course, reloading the unchanged original wrist policy reproduced an earlier stock trajectory and score exactly, with 24, 28 and 25 rebounds. The final learned map banks also reloaded exactly. This comparison adds no optical refit or reward-training episodes. [Checkpoint and score audit](current-pause-summary.json.gz)

## Reproduce

```sh
export PYTHONPATH=src LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
python -m scripts.experimental.tt24_current_pause
```
