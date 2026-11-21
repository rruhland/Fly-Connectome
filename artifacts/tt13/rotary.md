# Six-joint geometry recall and first returns

Three RGB probes recovered a rotary motor map that landed 9 of 24 held opponent serves, compared with 8 for an equal-probe cold fit. It landed 5 of 14 first returns across two complete games compared with 1 of 14 for the cold fit. Both players lost 2–11 and 4–11. On the same recalled map, accepting larger, coherent blade sighting corrections reduced the optical position residual on misses and landed 8 of 24 held serves and 5 of 15 in games. All four branches lost both games. [Paired evidence](rotary-comparison.json.gz)

## Calibration and controls

The test used the six-joint installation from TT12 and two older six-joint camera maps from TT9. The cold fit and signed recall shared the first three 34-command probes and 105 visible blade poses. The standard six-channel fitter requires at least 72 observations, which exceeds the maximum from two 34-command episodes. A fourth through seventh probe supplied 139 held blade observations. The recall algorithm chose the donor, permutation and signs by stereo prediction error on the first three probes. A separate evaluator audit recovered the underlying correspondence and showed that the chosen map identified all six ports and five of six signs. [Probe and alignment receipt](rotary-probes.json.gz) · [Port audit](rotary-comparison.json.gz)

The ordinary cold fitter estimated its own timing and geometry jointly from the support probes. The three geometries then ran with the same TT12 cursor-fitted receiving kernels and predictive servo. All branches used the original eight-probe optical tracker, commissioned serving geometry and settling curves, and the selected classic toss stroke. The result tests short calibration of the receiving geometry on this installation with existing timing, optical registration and service calibration. Players received stereo RGB, prior commands and public rewards. Referee events enter the post-game analysis. [TT12 timing and serving evidence](../tt12/body.md) · [Common starting checkpoint](rotary-start.json.gz)

| Receiving geometry | Native held pose error | Held pose error with shared timing |
| --- | ---: | ---: |
| Cold, three probes | 68.2 mm | 78.0 mm |
| Signed recall, three probes | 61.0 mm | 70.9 mm |
| Commissioned, eight probes | 67.0 mm | 74.5 mm |

Pose error measures the mean clipped stereo position and weighted pitch error. Signed alignment reduced the selected donor's support error from 200.3 to 67.1 mm before applying the shared receiving timing.

## Held opponent serves and complete games

Each of the 24 held starts reset the player to the same initial checkpoint. The two full games used independent seed streams for each server, alternating service under the first-to-11, win-by-two rule. Full games carried ordinary response, delay and reward updates across points. Ordinary, soft-racket, delay and mixed conditions appeared in each four-start block. All matched incoming paths agreed through the second service bounce to a numerical tolerance of 1e-8 in logged values. The three paired game comparisons preserved the requested and applied own-service commands through the strike across all 41 matched toss pairs. [Schedules and paired audit](rotary-comparison.json.gz)

| Receiver | Held contacts | Held first returns | Held wins | Games | Game first returns on rival serve | Game rival-serve wins | Legal own serves | First returns after own serve |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Cold geometry | 20/24 | 8/24 | 4/24 | 2–11, 4–11 | 1/14 | 1/14 | 10/14 | 1/6 |
| Signed recall | 18/24 | 9/24 | 4/24 | 2–11, 4–11 | 5/14 | 2/14 | 10/14 | 2/6 |
| Eight-probe reference | 19/24 | 11/24 | 6/24 | 3–11, 2–11 | 4/14 | 1/14 | 9/13 | 2/6 |
| Signed recall with coherent feedback | 19/24 | 8/24 | 3/24 | 3–11, 4–11 | 5/15 | 2/15 | 10/14 | 3/6 |

The recalled map landed four held shots the cold fit failed to land, while the cold fit landed three the recalled map failed to land. Recall gained four first returns across the two matched full games and one rival-serve point win. No branch made a second return on a rival-served point. [Cold held starts](rotary-cold-held.json.gz) · [Recall held starts](rotary-recalled-held.json.gz) · [Eight-probe held starts](rotary-commissioned-held.json.gz) · [Feedback held starts](rotary-coherent-held.json.gz)

## Optical evidence at contact

All 24 held paths produced an optical intercept plan under each branch. With signed recall, six serves missed the blade and nine contacted without a legal return. Eight of the latter bounced on the learner's half. Where stereo provided paired blade and ball sightings around the interception, the median discrepancy between the servo estimate and camera blade position was 105 mm on five misses and 11 mm on seven landed returns. A four-frame optical departure fit was available on four landed shots and five failed contacts. Their median forward speeds were 3.81 and 2.93 m/s, respectively. Those measurements cover just the shots with sufficient sightings. [Optical traces and outcomes](rotary-recalled-held.json.gz)

The follow-up tested the existing coherent predictive feedback rule after baseline misses showed large visual residuals. The rule accepts larger pose innovations when successive blade sightings are spatially consistent. Only the receiving estimator changed. The median optical blade discrepancy was 25 mm on the five observable held misses under this rule. One miss turned into a failed contact and one legal return turned into a failed contact. The game result gained one score point with the same five first returns. The feedback rule remains an experimental option. [Coherent feedback implementation](../../scripts/experimental/tt13_rotary_player.py) · [Paired outcomes](rotary-comparison.json.gz)

## Replay and continuation

The run replayed every recorded applied action and referee event for all 96 held shots and 112 game starts. For the signed-recall checkpoint after the second game, a separate run reconstructed all 15 point action tapes from RGB and reproduced four fresh point tapes and rewards identically from live and restored states. [Signed-recall game 1](rotary-recalled-game-1.json.gz) · [Signed-recall game 2](rotary-recalled-game-2.json.gz) · [Continuation checkpoint](rotary-recalled-game-2-player.json.gz) · [Resume audit](rotary-recalled-resume-audit.json.gz)

A second preparation produced byte-identical probe and starting checkpoint files. The post-game checkpoint preserved all 180 reference native Pong action tapes, 41 earned points and 195 of 206 later returns. [Native Pong audit](rotary-recalled-pong.json.gz)

From the repository root, use the [hall environment](../tt6/README.md). Fixture 0 is the held receiving cohort.

```sh
export PYTHONPATH=.:src:scripts/experimental
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TT3_COMPONENT_LIBRARY=/tmp/tt3-components-final.so
python=/workspace/Fly-Connectome/.venv/bin/python
output=/tmp/tt13-rotary
"$python" -m scripts.experimental.tt13_rotary_trial "$output" prepare
for branch in cold recalled commissioned coherent; do
  for fixture in 0 1 2; do
    "$python" -m scripts.experimental.tt13_rotary_trial "$output" play "$branch" "$fixture"
  done
done
"$python" -m scripts.experimental.tt13_rotary_report "$output"
"$python" -m scripts.experimental.tt13_rotary_trial "$output" audit-resume recalled 2
"$python" -m scripts.experimental.tt8_pong_retention \
  "$output/rotary-recalled-game-2-player.json.gz" "$output/rotary-recalled-pong.json.gz"
```

For a fresh game with the carried player, supply a new seed and a separate output directory. The example below completed a 6–11 game and replayed all 17 point tapes.

```sh
"$python" -m scripts.experimental.tt13_rotary_trial /tmp/tt13-rotary-resume \
  play recalled 1 --resume "$output/rotary-recalled-game-2-player.json.gz" --seed 21757293
```
