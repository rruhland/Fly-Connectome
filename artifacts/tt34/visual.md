# Using misses to select direct joint corrections

The SO-101 started from the frozen raw visual controller in TT33. Its local regression read the current RGB-derived ball position and velocity, joint encoders and measured joint motion. The pupil issued joint targets directly. When the current estimate showed a ball moving toward the arm inside a fixed three-dimensional neighborhood of the fixture, a small correction could be added to the current joint target. The controller had no projected interception plane, future ball waypoint, movement search or true ball state. The correction faded with the ball's current distance from the neighborhood and used the same joint rate limits as the original pupil. [Frozen visual model](../tt33/local-model.npz) · [Controller and training](../../scripts/experimental/tt34_visual.py)

Fourteen matched practice cards were played with seven choices. The choices were no correction and positive or negative offsets on shoulder pan, shoulder lift or wrist flex. A single recorded near-to-far release started each game. The live opponent had to strike the ball and land it on the near half to supply one credited opportunity. On a matched card a new legal return received +1 relative to the frozen controller, while an offset that turned its legal return into a miss received −1. The policy chose a correction from the similarity-weighted mean of those paired outcomes for its current camera and encoder observation. A wins-only control kept the same sample and similarity weights and set the −1 regression labels to zero. Both learned tables were frozen before play. [Frozen experiment plan](visual-plan.json.gz)

The two assessments shared identical held cards across the frozen controller, the signed-outcome learner and the wins-only control. The first cohort uses the eight fresh releases previously played by the frozen controller. The second comprises eight release-disjoint full-interval games selected using only neutral delivery and a three-left, two-middle, three-right first-ball quota. Actual physical returns beyond the opening remained live. [Replay controls and previous fresh games](../tt33/local-fresh.json.gz)

## Practice

| Offset | Arm returns in 14 practice points | New returns | Previous returns lost |
| --- | ---: | ---: | ---: |
| None | 3 | 0 | 0 |
| Shoulder pan +0.12 rad | 3 | 0 | 0 |
| Shoulder pan −0.12 rad | 3 | 0 | 0 |
| Shoulder lift +0.12 rad | 3 | 0 | 0 |
| Shoulder lift −0.12 rad | 2 | 1 | 2 |
| Wrist flex +0.13 rad | 3 | 0 | 0 |
| Wrist flex −0.13 rad | 4 | 1 | 0 |

The 98 attempts all had matched far contact, near bounce and initial sensor inputs when measured against the other choices on the same card. Negative shoulder lift added one return on 13621 and lost two previously successful ones. Negative wrist flex added one return during an alternating point on 12330. [Fitted checkpoint, matched training contexts and signed outcomes](visual-checkpoint.json)

## Held games

| Held games | Frozen | Signed outcomes | Wins only |
| --- | ---: | ---: | ---: |
| Earlier fresh cohort, eight points | 2 | 1 | 1 |
| New release-disjoint cohort, eight points | 4 | 4 | 4 |

Scores count the SO-101's own legal returns. The frozen controller reproduced the earlier eight points exactly. All sixteen first offers matched across all three evaluated policies through far contact, near bounce and the receiving line. The two learned policies chose the same corrections and produced identical motor commands, contacts and scores on every point. Each landed its first shot on four points, equal to the frozen controller. [Paired scores and CPU time](visual-summary.json) · [Frozen games](visual-held-fixed.json.gz) · [Signed-outcome games](visual-held-signed.json.gz) · [Wins-only games](visual-held-positive.json.gz)

The −0.12 rad shoulder-lift correction turned a contacted miss on 9532 into a legal return. It also turned the first legal return on 8239 into an own-half fault and ended that two-return game on its first stroke. Command-only replay of the improved 9532 point reproduced the legal return and all referee events. Full-CAD replay found zero penetrations across 1,370 solver states, with 18.6 mm minimum clearance. [Physical replay audit](visual-9532-audit.json)

## Reproduction

Use the pinned DeepMind feeds and dependencies from the [regulation endurance instructions](../tt29/endurance.md).

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt34_visual /tmp/tt25-rallies.json prepare
for arm in 0 1 2 3 4 5 6; do
  python -m scripts.experimental.tt34_visual /tmp/tt25-rallies.json practice --arm "$arm"
done
python -m scripts.experimental.tt34_visual /tmp/tt25-rallies.json fit
for policy in fixed signed positive; do
  python -m scripts.experimental.tt34_visual /tmp/tt25-rallies.json "$policy"
done
python -m scripts.experimental.tt34_visual /tmp/tt25-rallies.json summary
```
