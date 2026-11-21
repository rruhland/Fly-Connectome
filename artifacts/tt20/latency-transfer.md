# Single-eye transfer of actuator timing

Each camera independently identified an introduced four-frame command delay (133 ms) on the unmarked SO-101 clamp. The delay held requested rates in a queue before they reached the motor servos. The learner had one native RGB stream at 480 × 360, its own probe commands, the control clock and an independently rendered zero-delay factory model. The outside camera measured movement of the red blade. The wrist camera measured movement of the hall background around the blade. Comparing each motion trace with shifted factory traces selected the delay from zero to six control frames. [Trial and sensor boundary](../../scripts/experimental/tt20_latency_trial.py)

| Probe view | Estimated delay | Best-fit loss | Margin to runner-up | Target frames |
| --- | ---: | ---: | ---: | ---: |
| Outside | 4 frames | 0.0095 | 0.4861 | 28 |
| Wrist | 4 frames | 0.0252 | 0.5189 | 28 |

Each receiving branch used only its destination camera. It estimated the ball’s trajectory from natural court lines and the visible ball. A fixed factory catalogue supplied 2,048 motions and the receiver retained the same gantry flight and rebound priors across conditions. The clock correction forecast the ball to the time the first action would reach the servos. The unadapted branch assumed zero delay. The transfer branch received one scalar from the other eye and the destination-adapted branch received the scalar from its own probe. The two four-frame estimates produce identical controllers and share a physical execution on each held start, with both branches checked on replay. [Catalogue](latency-clamp-catalogue.npz) · [Receipt](direct-receive-latency.json.gz)

## First held schedule

Eight rival serves per eye used the same held start seeds, actual rigid-body contacts and the strict end-line referee. All 48 recorded branches replayed exactly.

| Destination eye | First legal returns, unadapted → transferred | Paired gains / losses | Point wins, unadapted → transferred |
| --- | ---: | ---: | ---: |
| Outside | 1/8 → 1/8 | 1 / 1 | 0 → 0 |
| Wrist | 0/8 → 4/8 | 4 / 0 | 0 → 1 |

The outside eye gained start `98275816` and lost `1674636427`. The wrist eye gained `2113866629`, `98275816`, `781867477` and `2012518044`. Three outside first-return plans that the dry model predicted would land legally ended on the learner’s half. Wrist timing also left three starts without a swing, and three legal first returns were followed by a lost exchange. [Point and command records](direct-receive-latency.json.gz)

An evaluator-only physical replay measured the cameras’ range error before contact. In the physical planning window the transfer branch saw 59/71 frames through the outside lens and 48/71 through the wrist. Median 3D error was 28.5 mm outside and 6.3 mm wrist. The 90th percentile was 65.8 mm and 15.3 mm. This difference indicates a remaining error in the outside eye’s contact forecasts, independent of its successful timing probe. [Replay diagnostic](../../scripts/experimental/tt20_latency_trial.py)

## Fresh 20-start replication

The prespecified schedule used RNG seed `3011576803` and reused the original probes, inherited priors and dry catalogue without additional target calibration. The referee replayed all 120 recorded branches exactly. The destination-adapted outcomes and commands matched their cross-eye counterparts on every start.

| Destination eye | First legal returns, unadapted → transferred | Paired gains / losses | Point wins, unadapted → transferred |
| --- | ---: | ---: | ---: |
| Outside | 0/20 → 7/20 | 7 / 0 | 0 → 3 |
| Wrist | 0/20 → 11/20 | 11 / 0 | 0 → 2 |

The outside eye gained starts `576350144`, `1648818577`, `1113927866`, `355435073`, `1193129084`, `1744604808` and `336112787`. The wrist eye gained `2021452621`, `2079566115`, `576350144`, `1648818577`, `1817187318`, `1113927866`, `1132392880`, `355435073`, `1737695571`, `1193129084` and `1744604808`. The replayed range error remained larger for the outside eye, at a median of 34.1 mm across 151 of 171 visible-window frames versus 7.2 mm across 117 of 171 wrist frames. The experiment measures receiving performance against a scripted opponent for this one fixed delay and fixture. [Replication receipt](direct-receive-latency-replication.json.gz)

## Reproduction

Generate the eight-start receipt, then run the separately prespecified 20-start schedule with the frozen estimate from each probe. `--verify` checks the FIFO from the original commands and replays MuJoCo scores and event traces. `--audit-ranging` computes the evaluator-only optical comparison.

```bash
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m scripts.experimental.tt20_latency_trial
python -m scripts.experimental.tt20_latency_trial --fresh --workers 4
python -m scripts.experimental.tt20_latency_trial --fresh --verify
```
