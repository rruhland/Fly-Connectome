# Table motor memory on an inertial puck task

The carried map reached the first puck goal in an average of 100 control frames versus 109.25 for cold exploration. It reached that goal sooner in 18 of 32 paired courses, tied in 11 and was later in 3. Both earned 128 of 128 goals. Starting with the archived map for the earlier wiring earned 114 of 128. [Full summary and evidence index](payload-summary.json.gz)

The task retains the simulated TT6 arm, joint servos and previously calibrated wiring. The puck is a new planar simulation. Every eight control frames its velocity retains a fraction of its previous velocity and receives an impulse from the arm tool's displacement from home. Each course independently draws a two-axis damping matrix, a signed mixing of the tool's three axes into the plane and four visible goals. The tool can point away from the goal while the puck moves toward it, and the controller must arrest inertia to score. A goal earns one point when the puck stays within 3.9 cm and moves no more than 1.9 cm per block for three consecutive blocks. Each goal has 144 control frames. The arm and puck continue across goal changes. [Simulation and controller](../../scripts/experimental/tt16_payload.py)

All three conditions share the same learner, three initial random motor pulses, ongoing exploration rule, projected RGB views and point feedback. The visual color roles, axes and scale are public. The learner is given the model structure and a controller that aims for a velocity toward the visible goal. Ridge regressions estimate the command-to-tool and tool-to-object responses online from quantized RGB positions and requested motor pulses, with no backpropagation. The reward suppresses exploration after a goal is earned. The carried condition imports only the active local motor Jacobian from the coded table checkpoint. The wrong-map condition uses the earlier wiring's Jacobian stored in the same checkpoint. Cold starts its Jacobian at zero. The object dynamics start from the same prior in all three. The learner receives no fixture, wiring, goal list, coupling, damping or exact simulator coordinates. [Coded table checkpoint provenance](transfer.md)

| Starting map | Goals earned | First goal by frame 96 | Mean first-goal frames, capped | Early mean error | Last-goal mean error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Carried | 128/128 | 15/32 | 100.00 | 13.73 cm | 6.67 cm |
| Wrong wiring | 114/128 | 4/32 | 127.75 | 21.59 cm | 6.78 cm |
| Cold | 128/128 | 13/32 | 109.25 | 14.61 cm | 6.77 cm |

Early error averages the first 96 control frames. Thirteen courses with the wrong map missed the first goal and enter the capped mean at 144 frames. The per-course early error was lower with carry than with cold in 22 of 32 courses. Total requested motor effort, measured as absolute normalized commands summed across all port-frames, was 2,678 for carry and 3,707 for cold. The controller learned the puck coupling from scratch in all conditions. Mean relative matrix error at the final decision was 17.5% for carry, 10.9% for wrong wiring and 13.8% for cold. The seed-specific final matrices are recorded in the receipts. [Aggregation and scoring audit](../../scripts/experimental/tt16_payload_report.py)

| Fixture | Mean first-goal frames, carried / wrong / cold | Goals, carried / wrong / cold | Evidence |
| --- | ---: | ---: | --- |
| Gantry 0 | 103 / 125 / 111 | 32 / 30 / 32 | [Tape](payload-gantry-0.json.gz) |
| Gantry 1 | 95 / 131 / 97 | 32 / 27 / 32 | [Tape](payload-gantry-1.json.gz) |
| Five-axis 0 | 106 / 126 / 121 | 32 / 28 / 32 | [Tape](payload-five-0.json.gz) |
| Five-axis 1 | 96 / 129 / 108 | 32 / 29 / 32 | [Tape](payload-five-1.json.gz) |

The held-out comparison uses eight independent courses per fixture with matched learner seeds across conditions. A separate four-course pilot favored cold by four frames on average. The controller was held fixed for the held-out run. All 96 held-out condition tapes reproduced the same tool positions, RGB marker readings and rewards across 6,912 action blocks. No markers reached the view edge. The modest early benefit is conditional on already holding a map on this same arm and wiring. The clock starts on entering the new task. It excludes the earlier optical and table practice used to establish that map. Later scores reach the same ceiling in the carried and cold conditions, and the puck system was only tested in simulation. [Gantry pilot](payload-pilot-gantry-0.json.gz) · [Five-axis pilot](payload-pilot-five-0.json.gz)

## Reproduce

Use the [TT6 simulation environment](../tt6/README.md).

```sh
export PYTHONPATH=.:src:scripts/experimental OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for fixture in gantry-0 gantry-1 five-0 five-1; do
  python -m scripts.experimental.tt16_payload artifacts/tt16 "$fixture"
done
python -m scripts.experimental.tt16_payload_report artifacts/tt16
```
