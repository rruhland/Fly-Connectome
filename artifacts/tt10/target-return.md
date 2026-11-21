# From precision reaching back to table tennis

Two short cursor sessions were enough to change how the five-axis arm timed its strokes. On installation A the update turned two losses into wins. On installation B it lost points on both fixtures. The current evidence supports retaining the fitted timing as an option specific to A. [Timing fit A](target-time-a-fit.json.gz) · [Timing fit B](target-time-b-fit.json.gz)

| First-person complete game | Inherited model | After cursor timing practice | Qualification detail |
|---|---:|---:|---|
| Five-axis A, first fixture | 8–11 | 11–9 | Normal commissioning for each player |
| Five-axis A, fresh fixture | 3–11 | 12–10 | Identical commissioned state |
| Five-axis B, first fixture | 7–11 | 6–11 | Normal commissioning for each player |
| Five-axis B, fresh fixture | 6–11 | 3–11 | Identical commissioned state |

Timing practice fit the lag and time course of each actuator's response from commands and 30 Hz optical tip displacements on 16 targets per installation. Ordinary undelayed cursor play allowed actuator timing to be estimated separately from transport delay. Geometry, pitch and the strike controller were preserved. Six unseen cursor sessions per installation tested the estimates. Mean error in two-frame displacement fell from 1.555 to 1.510 cm on A and 1.477 to 1.457 cm on B. The small optical gain at cursor speeds on both installations did not predict the direction of the table-tennis effect. [Estimator and source data](../../scripts/experimental/tt10_target_dynamics.py)

In the first game the updated player took one additional commissioning probe on A. The fresh comparison held the entire commissioned player, tracker reference and probe history fixed while substituting only the timing arrays. On A the change raised legal returns from 9 to 13 and successful first follow-ups from 1/4 to 3/5. On B returns fell from 14 to 11 and follow-ups from 3/3 to 2/5. These are four fixtures across two installations. Winning A twice establishes feasibility; broader qualification needs fresh installations. [A first baseline](target-back-a-baseline.json.gz) · [A first update](target-time-a-game.json.gz) · [A held baseline](target-time-a-held-base.json.gz) · [A held update](target-time-a-held-update.json.gz) · [B first baseline](target-back-b-baseline.json.gz) · [B first update](target-time-b-game.json.gz) · [B held baseline](target-time-b-held-base.json.gz) · [B held update](target-time-b-held-update.json.gz)

## What changed when only spatial geometry transferred

A separate intervention used two cursor sessions to update the static translational arm map, leaving actuator timing intact. Across 144 held poses per installation, average position error went from 7.19 to 7.13 cm on A and 6.10 to 6.01 cm on B. The first table games went from 8–11 to 9–11 on A and stayed at 7–11 on B. On 12 identical playable replies to the player's own serve, activating that map only after camera-detected rival contact improved legal first follow-ups from 5 to 7 and later legal returns from 10 to 15. The prospective full games brought a smaller change. A stayed at 5–11 and B went from 5–11 to 6–11. The physically replayed matched-reply effect was larger than the effect in ordinary play. The checkpoint after the B geometry game preserved all 180 original Pong command tapes and 41 points. [Matched replies and independent games](cursor-reply.md) · [Spatial fit A](target-back-a-fit.json) · [Spatial fit B](target-back-b-fit.json) · [Pong retention](target-back-pong.json.gz)

## Reproduction

Run from the repository root in the MuJoCo environment with the same single-threaded renderer as the series experiment. Replacing `offset_a` with `offset_b` repeats the fit on the second arm. The A comparison uses the same seed on both starts. For B use seed 777348143, [the original start](series-b-start.json.gz) and [the updated start](target-time-b-held-start.json.gz). For continuing the winning A player, use the [postgame checkpoint](target-time-a-qualified-player.json.gz).

```bash
python -m scripts.experimental.tt10_target_dynamics offset_a /tmp/a-player.json.gz /tmp/a-fit.json.gz
python -m scripts.experimental.tt10_series_trial /tmp/a-base.json.gz --resume artifacts/tt10/target-time-a-base-start.json.gz --seed 34910671 --games 1
python -m scripts.experimental.tt10_series_trial /tmp/a-update.json.gz --resume artifacts/tt10/target-time-a-held-start.json.gz --seed 34910671 --games 1
```
