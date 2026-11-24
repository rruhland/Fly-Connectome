# Few-shot correction on six failed openings

We took the nominal raw visual-and-encoder pupil and selected six earlier misses. On three openings the paddle never touched the ball. On three it touched the ball and failed to return it legally. Each new practice point began from the same recorded opening and original opponent seed. A different predetermined seed governed the opponent's subsequent replies on each attempt. The first far contact and near bounce matched across all attempts for each opening. [Matched practice plan](local-plan.json.gz)

The first seven attempts were shared. They tested the original pupil and positive and negative offsets for shoulder pan, a coupled shoulder-lift and elbow adjustment, and a coupled wrist adjustment. Each learner then received four more attempts, using two pairs of opposite adjustments centered on its best observed result. The dense learner could break equal-score ties using the camera-observed ball-to-blade separation and the direction, speed and progress of a visible outgoing arc. The score-only learner ranked the same probes and its own subsequent attempts by legal returns. Both used the same adjustment bounds and reply seeds. Across six openings this took 90 actual practice games, with 11 scored attempts available to each learner on each opening. [Experiment code](../../scripts/experimental/tt36_local.py)

| Opening | Original failure | First legal practice attempt, dense | First legal practice attempt, score only |
| --- | --- | ---: | ---: |
| 6253 | Untouched | None in 11 | None in 11 |
| 6861 | Untouched | None in 11 | None in 11 |
| 12373 | Untouched | None in 11 | None in 11 |
| 3928 | Contacted | None in 11 | None in 11 |
| 4363 | Contacted | None in 11 | None in 11 |
| 12403 | Contacted | 5 | 5 |

On 12403, the fifth attempt landed two arm returns using −0.145 rad shoulder lift and +0.08 rad elbow offset. Both learners also landed two on attempt eight and one on attempt ten. Both kept the original fifth-attempt adjustment. Dense feedback moved the other five policies despite their zero rewards. For example, its selected adjustment narrowed the visible gap on 6861 from 155 to 107 mm. On 12373 the paddle contacted its previously untouched ball and sent it to within 22 mm of the net in the stereo images. Neither adjustment landed a legal return. Practice consumed 156.5 process CPU seconds. [Attempt histories and retained offsets](local-summary.json) · [Committed lookup state](local-checkpoint.json)

## New replies and changed first offers

For held play the policy chose its closest stored context from the current optically estimated ball position and velocity and the measured joint angles. It attenuated the stored adjustment as the observation differed from that context. The actor received no opening ID, opponent placement or future interception forecast. Each held condition was played with both frozen learned policies and the unmodified pupil. [Frozen commands](local-held-fixed.json.gz) · [Dense commands](local-held-dense.json.gz) · [Score-only commands](local-held-score.json.gz)

| Holdout | Unmodified pupil | Dense feedback | Score only |
| --- | ---: | ---: | ---: |
| Same first ball, two new reply schedules per opening | 0 | 2 | 2 |
| One changed first offer per opening | 4 | 3 | 3 |

Only 12403 changed on the new-reply tests. Both learned policies returned the same first ball on both unseen continuations. The opponent then delivered real right-side replies from distinct target requests. Both learned policies missed both replies. The changed-first test screened for at least a 60 mm change in the actual lateral arrival. On the changed 12403 offer, all three policies returned the first ball and the unmodified pupil also landed the next one. The scored information selected a reproducible correction for one first offer. The optical tie-break produced no additional legal conversions or held returns. [Paired outcome and offer checks](local-summary.json)

Independent command replay reproduced the original 12403 miss, its first conversion and all four held gains. Full CAD replay found zero penetrations across 9,048 solver states and measured 18.6 mm minimum clearance. [Contact and clearance audit](local-audit.json)

## Reproduction

Use the pinned DeepMind feeds and regulation dependencies. [Base pupil reproduction](../tt33/joint.md)

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt36_local /tmp/tt25-rallies.json prepare
for index in 0 1 2 3 4 5; do
  python -m scripts.experimental.tt36_local /tmp/tt25-rallies.json practice --index "$index"
done
python -m scripts.experimental.tt36_local /tmp/tt25-rallies.json fit
for mode in fixed dense score; do
  python -m scripts.experimental.tt36_local /tmp/tt25-rallies.json "$mode"
done
python -m scripts.experimental.tt36_local /tmp/tt25-rallies.json summary
python -m scripts.experimental.tt36_local /tmp/tt25-rallies.json audit
```
