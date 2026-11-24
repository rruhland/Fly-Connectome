# Commit a top-ranked joint action at the first causal opportunity

Waiting for a causal visual cue and then keeping the first-ranked short joint action raised legal near returns from 14 to 21 on six paired starts, with fewer than half as many search refreshes. Only three of the six starts delivered an opening, so the result is promising but narrow. [Complete paired results](hindsight-prospective-results.json)

The [missed-shot comparison](hindsight.md) revealed that ordinary camera feedback was already enough to propose five real repairs. Each was the first-ranked action at the first recorded checkpoint. Their choice did not need a physical oracle or retries after seeing how a proposed action turned out. Re-executing the six private decisions reproduced every normalized command exactly. Successful decisions used 55–59 nominal trials and 0.305–0.512 CPU seconds of search, excluding rendering. [Timing receipt](hindsight-causal-timing.json)

The earlier experiment chose the checkpoints retrospectively. A prospective rule recovers all six initial timestamps from the saved causal inputs. It waits until the incoming estimate is supported by a recent stereo exposure, has moved into the near half, and indicates upward motion. That is a timing gate, not a target strike location. The live controller leaves the raw motor law in charge until that cue. It tries the same privately simulated action proposals, with at most one later retry if the first search produces no admissible plan. It then issues the top-ranked short joint motion for at most 21 ticks, unless observed departure ends it earlier. No outcome is required to choose or start an action. [Frozen cue check and assessment cards](hindsight-prospective-plan.json)

Six completely new measured releases were selected for this test. The strict compatibility pool had no unallocated recordings after the other courses and the frozen hindsight assessment. Before either controller ran, a prespecified wider geometric filter ranked unused motion recordings by the same measured compatibility metric. Six were drawn from its closest 24. No actual delivery or return outcome screened the new roster. The opponent’s random placement distribution did not change. Both actors used the same optical estimator, joint archive, nominal model and 34-proposal budget. Neither learned during the assessment. [Controller and replay driver](../../scripts/experimental/tt39_hindsight_prospective.py)

| Six assigned starts | Legal first | Legal later | Legal near | Search refreshes | Search CPU seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard elastic | 2 | 12 | 14 | 62 | 22.99 |
| Causal commitment | 3 | 18 | 21 | 25 | 9.36 |

The three no-offer episodes reproduced identical referee events. The other three first deliveries matched exactly. Their paired near-return counts were 2→4, 0→5, and 12→12. The last pair hit the 24-stroke cap in both arms, so the winner beyond that cap is not known. Total CPU fell from 46.82 to 41.85 seconds, despite playing 37.50 rather than 27.53 simulated seconds. Search CPU per simulated second fell from 0.83 to 0.25. This is still simulated computation, not a proof of real-time hardware performance. [Point outcomes](hindsight-prospective-results.json)

Independent physical replay re-scored the five-own, eleven-alternating rescue on release 3965. Full arm CAD had no penetration samples in 4,173 solver steps and found minimum clearance of 15.51 mm. [Action and CAD audit](hindsight-prospective-audit.json)

This experiment changes the timing of the first search and the commitment after selection together. It does not isolate which change matters. It also does not measure rallies beyond the 24-stroke cap.

## Reproduce

Generate the prerequisite [missed-shot practice data](hindsight.md), then run these stages.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt39_hindsight_prospective plan
python -m scripts.experimental.tt39_hindsight_prospective timing
python -m scripts.experimental.tt39_hindsight_prospective stock
python -m scripts.experimental.tt39_hindsight_prospective commit
python -m scripts.experimental.tt39_hindsight_prospective report
python -m scripts.experimental.tt39_hindsight_prospective audit
```
