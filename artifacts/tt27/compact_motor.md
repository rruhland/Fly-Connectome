# Regulation motor probes transfer to compact-court rallies

The frozen TT24 stereo policy played live points on the solid compact court with the changed-rebound setting. It used the same five active SO-101 axes and held paddle as the regulation receiver. The two simulators matched on axis order, actuator rates, home angles, joint limits, gains, force limits, and control clock. The far arm independently replanned each reply. [Compact game setup](compact.md) · [Transfer implementation](../../scripts/experimental/tt27_compact_motor.py)

We took the actual empirical decoder estimates from the regulation encoder-probe receipt, stream `53017817`, settings `2` and `5`. Each setting physically permuted and changed signs on the five motor request channels. Setting `5` additionally delivered the prior tick’s request through a one-tick FIFO at 30 Hz. The estimated controller read its wiring from the probe’s `active` record. The stale controller retained the previous accepted map, the known-map reference received the evaluator’s map, and the intact branch used the original interface without a delay. The decoder accounted for its inferred pending command when computing its next target. Each point began from the ordinary simulator home state. [Regulation probe receipt](motor-active-serial.json.gz)

The source probe spent 30 control ticks, or 1.00 second of actuator time per new setting, across five simulator homings. The two estimates were first accepted by ticks 26 and 27. This timing excludes physical homing time and the prior nominal response templates. The target compact task used the saved estimates without new encoder probes or updates from point outcomes.

| Fresh releases, three per map | Estimated near / far | Stale near / far | Known near / far | Intact near / far |
| --- | ---: | ---: | ---: | ---: |
| Signs and permutation, zero added delay | 6 / 6 | 0 / 0 | 6 / 6 | 6 / 6 |
| Signs and permutation, one-frame delay | 0 / 0 | 0 / 0 | 0 / 0 | 2 / 2 |

On fresh zero-delay release `1321462031`, the regulation estimate restored a **12-return rally**, six per arm. The known and intact branches scored the same rally. The estimated and known branches issued identical encoded requests, and the zero-delay inverse reproduced the intact physical motor commands, opposing commands and referee events. The stale map made zero returns on that release. The two other zero-delay starts had zero returns in all branches. A separately labeled retry of the earlier 7+7 release also recovered all 14 alternating returns using the estimate, while stale scored zero. [Fresh cases and executed command tapes](compact_motor.json.gz) · [Original 7+7 point](compact-live.mp4)

The matched video plays both branches at 30 frames per second from the same fresh serve. The stale branch ends after 1.37 seconds and holds its terminal frame while the learned branch continues for 8.13 seconds. The export reconstructs the executed motor rates from the encoded requests and matches both full referee event sequences. [Watch the matched replay](compact_motor-live.mp4) · [Video verification](compact_motor-live.json)

The estimated and known branches executed identical commands through the one-frame delayed interface. They made zero returns on three fresh starts, including one that elicited two returns per side with an intact interface. Knowing the correct delay leaves a physical wait before a newly issued command can reach the arm. In separately labeled diagnostic retries, half-frame and full-frame extrapolation of visual movement still gave zero returns on that fresh start. On the earlier verified 7+7 trajectory, the full-frame lookahead recovered one near return and one far reply with the delayed interface. [Delayed-interface tracking diagnostic](compact_motor-delay.json.gz)

## Reproduce

```sh
export PYTHONPATH=src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt27_compact_motor
python -m scripts.experimental.tt27_compact_motor --delay-diagnostic
python -m scripts.experimental.tt27_compact_motor_movie
```
