# Physical reply search on the SO-101 table

On 12 matched legal learner shots, the physical search opponent landed eight legal replies, up from four. It replied on seven points and reached four legal rally shots on one point. The original replied on four points and reached three legal rally shots at most. In the fresh clamp games, the frozen learner lost both 1–11 and created two reply opportunities per game. Each rival converted one. [Qualification receipts](opponent-comparison.json.gz)

| Test | Opponent | Points with a reply / opportunities | Legal rival returns | Legal learner returns | Mean rally length | Longest rally |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 12 matched shots | Original | 4/12 | 4 | 7 | 0.92 | 3 |
| 12 matched shots | Physical search | 7/12 | 8 | 7 | 1.25 | 4 |
| Full clamp game | Original | 1/2 | 1 | 2 | 0.21 | 2 |
| Full clamp game | Physical search | 1/2 | 1 | 2 | 0.21 | 2 |

Three matched points gained a first legal reply and none lost one. The number of matched points with at least two legally landed rally shots rose from three to five, and the longest point grew from 74 to 93 control frames. These were the same 12 initial learner contacts in both branches. The search prefers central placements, but its extra successful replies included wide shots. Median lateral landing distance from the centerline was 12.6 cm for the original opponent and 22.8 cm for the search across their respective landed replies. [Point and landing traces](opponent-comparison.json.gz)

The opponent anticipates an incoming shot from its private ball and arm state. It positions the rival blade, then tests actuator-limited swings in a scratch MuJoCo data instance. The search varies interception depth, timing, loft and forward or withdrawing motion. It tries lateral centering for wide balls and ranks physical trajectories that bounce on the rival's side, reverse at the far end, clear the net and land on the learner's half. The ranking favors a central landing to give the learner time to prepare another shot. The rival attempts every eligible reply and can fall back to the inherited swing. The selected motor rates drive the real rival arm through its existing joint limits and contact solver. The arena supplies the original serve and referee. [Opponent installer](../../scripts/experimental/tt20_opponent.py)

The matched test selects six legal own serves and six legal receiving returns from the archived TT19 rollout games in chronological order. Both opponents see the same physically executed commands through first learner contact on the pedestal and legacy-optic fixture. From that moment the frozen learner again uses its native RGB controller. The two full games use a fresh shared seed and the clamp with outboard camera. The marked views give the frozen TT19 tracker its usual calibration references in both tests. In the clamp games the frozen tracker still assumes a 93° wrist field of view for the 84° outboard lens. All reported returns come from the real TT20 scorer. [Scored trial](../../scripts/experimental/tt20_opponent_trial.py) · [Qualification receipts](opponent-comparison.json.gz)

A reply opportunity is a point with a legal learner serve or legal learner return. A completed reply is a later legal rival return in that point. Rally length counts the legally landed shots by both players after service.

The scorer reproduced all 52 point tapes, including both complete games. An event audit traced each of the 14 scored rival returns to a rival racket contact followed by its matching learner-half table bounce. The scratch search never updates the live ball, controls or events. The trial advances a simulated control clock, and the search adds processing time between frames. [Replay code](../../scripts/experimental/tt20_opponent_trial.py) · [Solver isolation test](../../tests/test_tt20_opponent.py)

## Reproduce

```sh
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt20_opponent_trial artifacts/tt20/opponent-comparison.json.gz \
  --per-role 6 --workers 2
python -m scripts.experimental.tt20_opponent_trial artifacts/tt20/opponent-comparison.json.gz --verify
```
