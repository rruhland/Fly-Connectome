# Practice order on a regulation table

The near player was a simulated table-clamped SO-101 with the same coupled motor limits, force-driven far paddle and stereo RGB tracker used in the earlier regulation games. Both training orders started with the same rewarded motion archive and signed optical shot errors. The learner estimated the ball from RGB, read its encoders, and consulted its own nominal model. A legal landing could add a movement to its archive. Outgoing RGB could add a signed error or a bound on a short stroke. The far paddle's target and the evaluator's exact ball trajectory were unavailable to the learner. The updates used no backpropagation. [Starting movement archive](../tt29/envelope-carry.json.gz) · [Starting optical errors](../tt30/wide-shot-errors.json.gz) · [Policy and prior results](../tt30/shot-correction.md)

Each practice card began with the same pinned measured near-to-far release. The far paddle had to play that ball and land its reply on the near half. Each arm then got one opportunity to return it. The staged course began with eight fixed right-side placements, followed by ten alternating right and left placements and twelve continuous placements. The last stage sampled lateral position, depth, height and pace for six deliveries on each side. The mixed course shuffled the same thirty cards without changing their targets or opening. We measured the incoming path at 1.10 m from the net, using physical contact position if the paddle intercepted earlier. A ball at least 7.5 cm left or right of the shoulder center counted on that side. [Frozen course plan and neutral delivery measurements](course-plan.json.gz) · [Practice driver](../../scripts/experimental/tt31_curriculum.py) · [Physical opponent and measurement](../../scripts/experimental/tt31_curriculum_opponent.py)

## Matched practice

All 30 cards matched across both orders through the far paddle's physical contact, the near bounce and the receiving-line sample. Each order produced 19 actual right-side deliveries and 11 actual left-side deliveries. The actual lateral range was −23.4 to +23.2 cm, and the mean absolute difference between requested x and measured x was 1.6 cm at the reference line. Both arms landed 19/19 from the right and 0/11 from the left. There was no near-paddle contact on those left deliveries. Each archive grew from 223 motor examples and 24 optical errors to 242 and 43. The pupil's commands differed on three of the continuous practice cards. [Staged practice tape](course-practice-staged.json.gz) · [Mixed practice tape](course-practice-mixed.json.gz) · [Staged checkpoint](course-staged-checkpoint.json.gz) · [Mixed checkpoint](course-mixed-checkpoint.json.gz)

## Held games

Held games used ten different, previously unused measured releases with paired opponent seeds. An independent neutral-paddle screen verified five first deliveries to each side. The opponent sampled a fresh continuous placement on each subsequent reply. Requested lateral coordinates were in two bands, between 13 and 26.5 cm from the center on either side. Both players kept playing live after the first reply, and their different strokes could change the ensuing ball trajectories. We froze the motor and optical error archives during the held games, restarted RGB tracking at each opening and capped games at sixteen alternating legal returns. The recorded opening launch is excluded from the scores. [Paired held roster](course-plan.json.gz)

| Checkpoint | SO-101 legal returns | Alternating legal returns | Left deliveries landed | Right deliveries landed |
| --- | ---: | ---: | ---: | ---: |
| Original | 7 | 24 | 0/9 | 7/8 |
| Staged practice | 9 | 28 | 0/9 | 9/10 |
| Mixed practice | 10 | 30 | 0/9 | 10/11 |

All three checkpoints received the same first shot on every held card, down to the far paddle's contact, the near bounce and the measured trajectory at the receiving line. Actual arrivals covered −27.3 to +29.6 cm laterally. Both practice gains came from one opening, 4568. The arm landed two returns with the original checkpoint, four after staged practice and five after mixed practice. The scores on the other nine games tied. Across all three checkpoints, the arm made zero contacts on left-side deliveries. [Per-game scores, placement checks and paired differences](course-summary.json) · [Original tape](course-held-original.json.gz) · [Staged tape](course-held-staged.json.gz) · [Mixed tape](course-held-mixed.json.gz)

Independent command replay reproduced the full referee event streams for the staged and mixed 4568 games. All nine SO-101 landings used the same physical blade face on the right of the shoulder. The full-CAD audit found zero penetrations across 7,822 combined solver states and measured a minimum clearance of 18.6 mm. [Staged audit](course-4568-staged-audit.json) · [Mixed audit](course-4568-mixed-audit.json)

## Reproduction

Use the pinned measured feeds and fixed-stereo dependencies in the [endurance instructions](../tt29/endurance.md). The assessment takes the checkpoint produced by its corresponding practice order. The original can be assessed independently.

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt31_curriculum /tmp/tt25-rallies.json plan
python -m scripts.experimental.tt31_curriculum /tmp/tt25-rallies.json practice --order staged
python -m scripts.experimental.tt31_curriculum /tmp/tt25-rallies.json practice --order mixed
python -m scripts.experimental.tt31_curriculum /tmp/tt25-rallies.json held --order original
python -m scripts.experimental.tt31_curriculum /tmp/tt25-rallies.json held --order staged
python -m scripts.experimental.tt31_curriculum /tmp/tt25-rallies.json held --order mixed
python -m scripts.experimental.tt31_curriculum_summary
```
