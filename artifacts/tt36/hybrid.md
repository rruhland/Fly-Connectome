# Routing each approach between learned reach and staged joint corrections

One frozen direct controller preserved the broad reach scores on both 32-release courses and the staged score on 12 familiar openings with new opponent continuations. It did not preserve the full staged advantage on nearby displaced openings. [Results and exact command comparisons](hybrid-results.json) · [Implementation](../../scripts/experimental/tt36_hybrid.py)

## Recognition before the first correction

The router used the staged controller's existing acquisition rule. On the first incoming optical state with current ball depth below −0.22 m, it compared the six measured ball position and velocity values and five measured joint angles against six saved practice signatures. A normalized distance below 0.4101 selected the existing early-plus-late joint corrections. Other observations selected live learned reach. The router held that choice until the staged observer's reset and used the same raw pupil in both branches.

The cutoff was frozen before held play. It was half the distance to the closest signature among 18 separate, unrelated calibration openings, capped at 0.55. Calibration replayed only the physical prefix required to collect the first signature. The actor never received release IDs, opponent seeds or referee state. [Recognition calibration and component hashes](hybrid-recognition.json)

The routed controller inherits the two previously learned components. The staged branch had six rehearsed openings, with 11 attempts on the dense early-learning branch and eight late probes per opening. The reach branch used the scoreless encoder-to-blade model, 453 rewarded relative observations and its own practice-tuned 0.38 rad cap. This test trained neither branch again. [Staged learning](learning.md) · [Live body feedback](reach.md)

## Held physical play

| Cohort | Raw near legal returns | Standalone component being preserved | Hybrid near legal returns | Hybrid later near returns | Staged / reach selections |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 new releases | 11 | 14 reach | 14 | 3 | 0 / 45 |
| 32 independent confirmation releases | 9 | 12 reach | 12 | 4 | 1 / 43 |
| 6 rehearsed openings × 2 continuations | 0 | 6 staged | 6 | 0 | 12 / 6 |
| 20 displaced openings | 2 | 7 staged | 4 | 1 | 1 / 23 |
| 8 earlier retention continuations | 16 | 14 reach | 14 | 6 | 0 / 22 |

A selection counts one acquisition by the approach observer. Short occlusions can reset the observer, and a point can acquire another approach after a legal exchange. The two 32-release results preserve the standalone reach opening-contact counts of 22 and 19. Their complete tapes and referee events match reach on all 32 new releases and 31 of 32 confirmation releases. Confirmation release 13760 selected the staged branch on its first approach and kept the same legal score. [New games](hybrid-new.json.gz) · [Confirmation games](hybrid-confirm.json.gz)

The hybrid recognized all 12 rehearsed first approaches. On each, its commands matched the staged controller exactly through the first approach. Six complete tapes were also identical. For the other six it subsequently used reach on the later incoming ball, and both policies recorded the same legal score. The displaced case 6861 was the only nearby first ball recognized by the tighter router. This left most displaced shots with live reach and gave up three returns against the existing staged policy. All eight retention tapes match standalone reach exactly, including its loss of two returns relative to the earlier raw controller. [Rehearsed games](hybrid-rehearsed.json.gz) · [Displaced games](hybrid-nearby.json.gz) · [Retention games](hybrid-retained.json.gz)

The five cohorts used 202.90 process CPU seconds for play. Command-only replay reproduced every referee event on 10 selected points, including the recognized unfamiliar point and all positive rehearsal and displaced games. Full-CAD replay found 0 penetrations across 15,517 physics steps, with 18.6 mm minimum clearance. [Independent replays](hybrid-audit.json)

## Reproduce

Use the pinned measured releases from the [regulation reproduction instructions](../tt29/endurance.md) and the existing frozen staged and reach checkpoints.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt36_hybrid calibrate
for cohort in new confirm rehearsed nearby retained; do
    python -m scripts.experimental.tt36_hybrid run /tmp/tt25-rallies.json "$cohort"
done
python -m scripts.experimental.tt36_hybrid summary
python -m scripts.experimental.tt36_hybrid audit
```
