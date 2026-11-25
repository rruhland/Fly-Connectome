# Learn commitment duration from real returns during the rally

The ordinary TT48 player begins from 48 exploratory games of earned joint experience and continues its regular legal-return motor credit during play. This comparison leaves its optical history, outgoing ranker, plant, opponent, candidate sources and physical limits intact. All 16 assignments are the original first half of the first predeclared official stream, including the certified 63-return stock point on 12246. The starting checkpoint, prerecorded launch and opponent random draws are identical for both arms. The treatment does not train on the archived comparison. [Official stream and original runner](../../scripts/experimental/tt48_practice_transfer.py) · [Starting comparison protocol](protocol.json)

A new decision is possible once per actual incoming ball, on a due search right before predicted impact. An already accepted robust motion must still predict a legal return in a fresh nominal check. Normalized position, velocity and encoder disagreement distinguishes two observation bands. The actor then either runs the ordinary search on this tick or keeps the checked direct motion for two more ticks. The first eligible choice is stock. Later bounded probes and confidence-weighted success estimates choose between the same two options. No new motor candidate family is added and no further duration choice is made before the outcome is settled.

Success comes from the physical referee's actual legal landing, immediately available to the actor before the next incoming ball. A real terminal unreturned offer supplies failure, while a capped or otherwise unresolved choice is censored. The actor retains the small observed-outcome table between games and checkpoints it with the usual motor memory. A successful next decision in the same rally can use an observation earned just one return earlier. These factual success rates are not counterfactual estimates of which unissued motion would have scored. [Policy and offline settlement audit](../../scripts/experimental/tt51_learned_commitment.py)

## Held comparison on the official stream

The live duration learner scored **105 legal near returns versus 136 stock** across all 16 games. Alternating legal returns fell from **287 to 225** and the best rally from **63 to 23** near returns. Five games lengthened, five shortened and six tied. Opening success stayed at 13 of 16. The ordinary 63-return 12246 assignment scored nine with the duration learner, while 5335 rose from two to 20. Neither game issued a duration hold, and both already started from different earned motor memories. Their differences are outcomes of the carried learning history, not direct effects of holding a motion on those balls.

| Release | Stock | Learner | Release | Stock | Learner |
| --- | ---: | ---: | --- | ---: | ---: |
| 4317 | 0 | 0 | 14429 | 3 | 3 |
| 12144 | 10 | 23 | 12246 | 63 | 9 |
| 6487 | 14 | 7 | 10339 | 5 | 5 |
| 13344 | 7 | 4 | 3482 | 0 | 0 |
| 10544 | 2 | 5 | 11133 | 5 | 10 |
| 9060 | 8 | 11 | 3527 | 5 | 3 |
| 12866 | 3 | 3 | 14442 | 0 | 0 |
| 12192 | 9 | 2 | 5335 | 2 | 20 |

On 12144, the only game with a duration hold, the two players started with the same earned motor memory. The ordinary duration choice at tick 270 legally scored. The first hold probe at tick 478 responded to that within-rally evidence. The issued commands first changed at tick 480, after 49 exact native events. Both players legally landed the identical incoming ball at the fork. The treatment made a second successful probe later in the rally. All later games used ordinary cadence, although some used a changed motor library. [Exact action and reward trajectories](learning.json.gz) · [Per-game audit](comparison.json)

All 49 eligible choices scored, 47 ordinary and two holds. Thirty-four outcome-weighted choices used evidence from an earlier settled choice in the same rally, but none selected a hold. The robust, low-disagreement eligibility filter supplied no failure labels, so this run did not distinguish the value of the two durations. It showed the fragility of the live motor-learning path to two early schedule changes, not a sustained policy gain. The committed checkpoint restores all 768 motor fragments at serial 112 and the 49 verified duration labels. This variant was not advanced. [Motor checkpoint](learning.npz) · [Duration checkpoint and score audit](comparison.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt51_learned_commitment run
python -m scripts.experimental.tt51_learned_commitment report
python -m pytest -q tests/test_tt51_learned_commitment.py
```
