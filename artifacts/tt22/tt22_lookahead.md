# A teacher that simulates the pupil's next stroke

The pupil is the frozen recurrent SO-101 servo. It finds and follows its ball in two native fixed RGB streams, predicts a crossing, and makes one rate-limited pulse per inbound flight. The comparison keeps those camera inputs, motor settings, opening family, five-axis far robot, and strict referee fixed. The starts use the narrow stochastic release from the original servo experiment. [Pupil and source experiment](servo.md)

The lookahead coach copies the current motor controller, including its past optical state, and runs it in a separate physical arena. For up to twelve feasible far rate tapes, the coach runs both actuators until the far shot lands legally and the next pupil stroke lands legally, or until a fault. It replaces future images in this scratch copy with projected physical positions plus an archived camera error model. The existing bounce that the preceding pupil shot might still be owed cannot count toward the simulated reply. The exact scorer must first credit the candidate far return, and then credit a new near return. The final choices favor a legal next reply with a reachable far interception and survive two small position perturbations. The live pupil retains its two native RGB streams and its unmodified motor policy. [Coach and replay code](../../scripts/experimental/tt22_lookahead.py)

The camera model uses 286 paired image estimates and evaluator positions from archived stereo servo points. Median position bias was −0.08 mm lateral, −2.39 mm longitudinal, and +4.20 mm vertical. These values and the prior motor pulse are frozen before the physical comparison. The assessment reserves one previously successful development seed and four separate fresh starts. Each sampled point keeps its only opening release and the referee's normal terminal outcome. [Control tapes and shadow events](tt22_lookahead.json.gz)

| Fresh opening seed | Fixed far | Two-stroke coach |
| --- | ---: | ---: |
| `397545534` | 2 legal alternating returns | 4 |
| `235133130` | 0 | 0 |
| `481355939` | 2 | 4 |
| `504601541` | 0 | 0 |

Both conditions made the same initial learner contact with identical preceding motor commands on all four starts. The coach enabled a second live pupil return on both points with a legal first pupil return. Its selected scratch landings predicted pupil credit at steps 74 and 71. The live RGB policy earned those credits at steps 72 and 71. A subsequent far response landed legally in both points. The pupil launched a third pulse in both and missed the ball. The teacher had found no predicted third pupil return among its twelve next-far candidates. Each arm's total went from two to four legal returns. Both conditions lost all four points. The development opening finished with four legal returns in each case, using the ordinary far plans.

The six far planning calls across the fresh fixed points took 6.10 seconds in total. The corresponding coach calls took 13.37 seconds and evaluated 48 nominal two-stroke candidates. All ten physical points, including the development pair, reproduced their exact referee events from both joint-rate tapes.

On a follow-up diagnostic using the two already observed stalls, the coach examined 48 and 14 available late candidates. It found zero legal further pupil returns and both points again ended after four legal exchanges. Both diagnostic points also replayed exactly. A targeted next experiment would learn a separate late-pulse timing and amplitude from public legal-return credit on these physical feeds. [Late candidate coverage diagnostic](tt22_lookahead-deeper.json.gz)

```bash
python -m scripts.experimental.tt22_lookahead --output /tmp/tt22-two-stroke.json.gz
```
