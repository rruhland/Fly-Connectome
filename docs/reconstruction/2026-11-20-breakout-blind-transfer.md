# Blind Pong-to-Breakout transfer

The complete Pong checkpoint returned the first incoming ball in every ordinary Breakout course. The same online learner with its learned state reset returned four of eight. Full transfer returned 40 of the first 40 opportunities across these courses, compared with 36 for the matched fresh learner. It missed the preregistered rapid-transfer criterion because only four courses improved and it took longer on average to reach a sustained rally. The motor-and-arrival arm matched its opening returns and saved all 415 opportunities across 128 ordinary lives. [Sealed protocol](../plans/2026-11-20-arcade-transfer-qualification.md)

## Ordinary controls

Each arm played the same eight courses of 16 lives, starting each course with its specified initial state. Reaching three consecutive returns in one life counted as sustained. The elapsed camera frames include previous lives. The averages in the table use courses where the respective event occurred. The paired sustained analysis below also includes failures through the fixed horizon. [Raw course records and summaries](../../artifacts/m2/2026-11-20-breakout-blind-results.zip)

| Arm | First return on opening opportunity | First five across lives | Mean frames to first return | Sustained courses | Mean frames to sustained | All returns | Lost lives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full transfer | 8/8 | 40/40 | 47.9 | 8/8 | 483.0 | 324/341 | 15/128 |
| Motor and arrival | 8/8 | 40/40 | 47.9 | 8/8 | 391.5 | 415/415 | 0/128 |
| Frozen transfer with visual alignment | 8/8 | 39/40 | 47.9 | 8/8 | 483.1 | 322/339 | 17/128 |
| Frozen original Pong orientation | 0/8 | 0/40 | — | 0/8 | — | 0/128 | 128/128 |
| Matched fresh learner | 4/8 | 36/40 | 74.9 | 8/8 | 433.8 | 404/408 | 4/128 |
| Numerically uncalibrated learner | 5/8 | 28/40 | 87.5 | 7/8 | 664.3 | 308/351 | 43/128 |
| Passive centered paddle | 0/8 | 0/40 | — | 0/8 | — | 0/128 | 128/128 |

Against the matched fresh learner, the full and motor-and-arrival arms each gained 10 percentage points over the first five opportunities. Each won four paired courses and tied four. The paired bootstrap 95% interval was +2.5 to +17.5 points and the exact two-sided course sign-flip test gave `p = 0.125`. Full transfer reached sustained returns 119 native ticks later on average, with a paired 95% interval from 330 ticks sooner to 545 later. The motor-and-arrival arm reached sustained returns 100 ticks sooner on average, with a paired interval from 126 ticks later to 406 sooner. It was faster in three courses. [Paired course differences](../../artifacts/m2/2026-11-20-breakout-blind-results.zip)

The frozen arm aligns the visible paddle to the canonical rail within each life and holds the inherited settings fixed. Its ordinary result was close to full transfer. The fresh control uses the same alignment and online controller with uniform motor-order probabilities, random initial relation direction and zero score visits. The numerically uncalibrated diagnostic uses a separate online motor and relation estimator, initialized with zero fitted samples. The frozen-orientation control holds the original Pong rail even when Breakout presents a different axis. [Controller source](../../scripts/experimental/arcade_transfer_session.py) · [Fresh visual learner](../../scripts/experimental/arcade_cold_visual.py)

## Changed button and response conditions

The four separate stress courses used 16 lives per perturbation. First-five and sustained results favored the matched fresh learner over both carried updating arms in this cohort. The arena supplied each learner only the camera stream, elapsed time, and ordinary reward and terminal feedback. [Arena source](../../scripts/experimental/breakout_arena.py)

| Arm | First five across lives | Sustained courses | First incoming under remap | First incoming under delay | First incoming under both |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full transfer | 4/20 | 2/4 | 5/16 | 5/16 | 7/16 |
| Motor and arrival | 6/20 | 2/4 | 5/16 | 5/16 | 7/16 |
| Frozen transfer with visual alignment | 4/20 | 1/4 | 0/16 | 7/16 | 6/16 |
| Frozen original Pong orientation | 0/20 | 0/4 | 0/16 | 0/16 | 2/16 |
| Matched fresh learner | 10/20 | 3/4 | 7/16 | 4/16 | 6/16 |
| Numerically uncalibrated learner | 6/20 | 2/4 | 5/16 | 3/16 | 8/16 |
| Passive centered paddle | 0/20 | 0/4 | 0/16 | 0/16 | 0/16 |

## Receipts

Candidate `54903c9f18c161024fb1dfa568a479eb3185a69f` was frozen before release of the Breakout schedule. An independent check matched all 1,232 life records to 176 paired starts, confirmed identical pre-contact ball trajectories, verified the checkpoint and controller state hashes, and recomputed misses, returns, lost lives and cumulative event times. The archive includes every course, all per-life first-five slots, first-return and sustained times or failure, per-life-ordinal learning curves, and the source, checkpoint and schedule hashes. [Revealed schedule](../../artifacts/m2/2026-11-20-breakout-sealed-schedule.json) · [Committed digests](../../artifacts/m2/2026-11-20-qualification-seals.json) · [Pong checkpoint](../../artifacts/m1b/2026-11-19-pong-after-mixed-course.json) · [Qualification runner](../../scripts/experimental/run_arcade_transfer_qualification.py) · [Raw archive](../../artifacts/m2/2026-11-20-breakout-blind-results.zip)
