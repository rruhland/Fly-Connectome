# Continuous motor calibration from one optical movement

On four fresh gain and coupling fixtures, a bounded fit using the shared 24-command movement produced 32 legal returns and eight points with multiple returns. The frozen signed-map fit made five legal returns and one multiple-return point. TT15's in-game confirmation made 30 returns and eight multiple-return points. TT15 won one full game. The other branches lost all four. A more accurate motor map improved returns here while winning remained harder. [Game and motion audit](motor-summary.json.gz)

## Fixtures and preparation

The gantry and five-axis arm each faced fresh gain, coupling and unchanged fixtures. Two further games per method used TT14's gantry coupling and five-axis attenuation matrices with fresh camera settings and ball trajectories. Every fixture resumed the same archived table player for all branches and installed its motor response before the first scored point. The new responses act on the donor's existing signed port ordering. The study uses the physical MuJoCo arm and shaded first-person stereo. Camera positions, baseline, focal angle and appearance changed independently across the fixtures. The public calibration keeps a common table coordinate system. The new controllers received motor commands, images and ordinary game rewards. The exact mixer and joint state entered only the physical environment and evaluator. [Fixture seeds and camera setup](../../scripts/experimental/tt17_motor_protocol.py)

The three branches shared two 34-frame optical reference movements on the original controls. One began with the same 24-frame command sequence used by the active branches on the new controls. These branches received identical new-wiring commands and images. The signed branch selected a port order and polarity using TT16's frozen nonlinear RGB decoder over the 24-frame tape. The bounded branch also used the paired original-control observation from the common setup to estimate a real-valued gain or mixing correction. The TT15 branch started with its old motor map and a separately charged 28- or 34-frame dry reference. Surprising scored motion triggered another such movement and the established TT15 model selection. [TT16 signed decoding](../../scripts/experimental/tt16_coded_probe.py) · [TT15 dry confirmation](../../scripts/experimental/tt15_motor_probe.py)

## Fitting and validation

The correction minimizes the difference between observed changes in blade motion and the changes predicted by the archived quadratic arm model. Subtracting the measured reference displacement reduces errors that the two optical trajectories share. As in the signed decoder, each output coordinate is weighted by the old map's sensitivity. The fit retains the inherited per-axis settling curves. It tests a gain-only fit and a full mixer around the original and decoded port assignments. It uses alternating three-frame blocks to test the proposed improvement before refitting the selected model on all paired sightings. Diagonal entries stay between 0.42 and 1.35, off-diagonal magnitude stays at most 0.42, and the final singular values stay in the interval from 0.35 to 1.50. The exact stable map is retained whenever the paired response differs by less than 12 mm in the weighted optical measure. The stroke servo uses the fitted dynamics with the inherited public command limits. [Bounded fitter](../../scripts/experimental/tt17_motor_fit.py) · [Predictive servo](../../scripts/experimental/tt14_gain_servo.py)

During scored play an earlier point can propose a limited refinement after a large optical error. To adopt it on the next point, a later point must support it and the new fit must still explain the original active measurement. Both TT15 and bounded branches make these decisions from their own requested rates and optical blade poses. Scored points interleave ordinary, soft, mixed and delayed conditions. The two latter schedules change command delay and noise during play. Ball and disturbance seeds are matched within each fixture. Service roles can diverge as game scores separate. [Game and replay protocol](../../scripts/experimental/tt17_motor_trial.py) · [Physical point conditions](../../scripts/experimental/tt6_continuous_trial.py)

## Held-out physical motion

The evaluator replayed 12 fresh command sequences on both original and altered controls for each of the eight installations. The frozen startup fits predicted the actual change in blade displacement. The pulse course used an independent command schedule with delay and noise. Across the four fresh altered fixtures, the bounded map cut mean error in the change from 53.3 to 18.0 mm and mean absolute displacement error from 76.2 to 44.8 mm. A diagnostic using the planted mixer with the same old camera body yielded 12.3 and 39.2 mm. All motion measurements came after fitting. [Independent motion code](../../scripts/experimental/tt17_motor_audit.py)

| Fixture | Signed fit, change error (mm) | Bounded fit (mm) | Known mixer (mm) |
| --- | ---: | ---: | ---: |
| [Fresh gantry gain](motor-gantry-gain-motion-audit.json.gz) | 41.9 | 11.3 | 13.1 |
| [Fresh gantry coupling](motor-gantry-coupling-motion-audit.json.gz) | 61.1 | 24.5 | 12.6 |
| [Fresh five-axis gain](motor-five-gain-motion-audit.json.gz) | 38.6 | 12.7 | 10.7 |
| [Fresh five-axis coupling](motor-five-coupling-motion-audit.json.gz) | 71.7 | 23.7 | 12.8 |
| [TT14 gantry coupling](motor-gantry-coupling-tt14-motion-audit.json.gz) | 46.9 | 22.9 | 13.2 |
| [TT14 five-axis gain](motor-five-gain-tt14-motion-audit.json.gz) | 70.1 | 15.8 | 10.6 |

The optical rule left both stable maps at their exact original controls. The signed decoder also kept the original maps on those fixtures. Under continuous changes it selected a spurious lateral axis exchange on the fresh five-axis coupling fixture and on the repeated TT14 five-axis gain. The other four changed fixtures kept the correct axis order but lacked the changed gains or cross-axis terms. [Gantry stable pulse audit](motor-gantry-stable-motion-audit.json.gz) · [Five-axis stable pulse audit](motor-five-stable-motion-audit.json.gz)

## Complete games on fresh motor changes

Each cell gives the final score, legal returns and distinct points with at least two legal returns. Complete games can have different numbers of opportunities. [All branches, timing and stage checkpoints](motor-summary.json.gz)

| Fixture | Bounded correction | Frozen signed fit | TT15 confirmation |
| --- | ---: | ---: | ---: |
| [Gantry gain](motor-gantry-gain-bounded.json.gz) | 3–11 / 8 / 2 | 5–11 / 5 / 1 | **11–9** / 15 / 5 |
| [Gantry coupling](motor-gantry-coupling-bounded.json.gz) | 3–11 / 14 / 3 | 1–11 / 0 / 0 | 5–11 / 13 / 3 |
| [Five-axis gain](motor-five-gain-bounded.json.gz) | 2–11 / 3 / 1 | 0–11 / 0 / 0 | 1–11 / 1 / 0 |
| [Five-axis coupling](motor-five-coupling-bounded.json.gz) | 3–11 / 7 / 2 | 0–11 / 0 / 0 | 1–11 / 1 / 0 |
| Total points / legal returns / multiple-return points | 11 / 32 / 8 | 6 / 5 / 1 | 18 / 30 / 8 |

Seven of the bounded branch's multiple-return points came on rival service, compared with zero for the signed fit and six for TT15. None of the bounded maps changed after the initial active fit in the four games. The startup calibration accounts for the improvements in this table. TT15 used 418 additional unscored control frames after its reference movement. Counting shared setup, active moves and scoring, the held-out groups took 4,266 frames for bounded, 2,781 for signed and 5,126 for TT15. They earned 2.58, 2.16 and 3.51 points per 1,000 frames respectively.

Bounded calibration reached its first return after 142, 272, 184 and 141 total frames in the listed order. TT15 took 145, 421, 383 and 412. The signed branch reached a return after 141 frames on gantry gain. Its other three games ended without a return. The elapsed count includes preparation, preceding points, earlier dry confirmation and the control frame of the legal bounce itself. Each signed or bounded game starts with 92 preparation frames. TT15 starts with 96 on gantry or 102 on five-axis before any further confirmation.

On the unchanged installations, all three methods won 11–6 on gantry and 11–9 on five-axis, reproducing all 38 scored action tapes. TT15 triggered 356 extra dry frames and kept the original map. The bounded player kept its original map using the shared startup measurement. [Gantry unchanged](motor-gantry-stable-bounded.json.gz) · [Five-axis unchanged](motor-five-stable-bounded.json.gz)

## Games with the earlier TT14 matrices

| Fixture | Bounded correction | Frozen signed fit | TT15 confirmation |
| --- | ---: | ---: | ---: |
| [Gantry coupling](motor-gantry-coupling-tt14-bounded.json.gz) | 7–11 / 7 / 1 | 2–11 / 3 / 1 | 3–11 / 4 / 0 |
| [Five-axis gain](motor-five-gain-tt14-bounded.json.gz) | 1–11 / 6 / 1 | 0–11 / 1 / 0 | 2–11 / 10 / 2 |

Across all six altered games, bounded calibration made 45 legal returns and landed at least two on ten points. TT15 made 44 returns and also landed at least two on ten points. The signed fit made nine returns and landed at least two on two points. TT15 won the only altered-control game won by any branch. The bounded player accepted no late refinement in any of the games. [All receipts and replay totals](motor-summary.json.gz)

The audit replayed 363 scored point tapes from 24 complete games and reconstructed each camera-only motor choice. The TT15 audit also repeated its 37 between-point dry movements. A tuple-to-list comparison error blocked the earliest TT15 replays. The affected games were restarted from their original checkpoints after correcting the comparison, and each reported clock counts its one completed qualification run.

The fresh coupled gantry recorded 14 legal returns in its complete game. Its post-game bounded checkpoint reproduced all 180 native Pong action tapes and the same 41 earned points. The startup checkpoints separately preserve only the active calibration on the altered controls, before any scored play. [Native Pong audit](motor-native-pong.json.gz) · [Post-game gantry](motor-gantry-coupling-bounded-player.json.gz) · [Calibrated gantry start](motor-gantry-coupling-bounded-start-player.json.gz) · [Calibrated five-axis start](motor-five-coupling-bounded-start-player.json.gz)

## Reproduction

Use the [hall simulation environment](../tt6/README.md) with the experimental scripts on `PYTHONPATH`. Each game receipt includes its clock, referee results, applied commands, action digests and camera-only motor choices. The fixture setup contains the new rig, nominal optical response and physical hardware parameters for exact replay. The saved motor checkpoint uses the TT14 portable format. `tt14_gain_trial.load_map(checkpoint)` returns the body model and the learned requested-rate transform.

```sh
for case in gantry-gain gantry-coupling gantry-stable five-gain five-coupling five-stable \
            gantry-coupling-tt14 five-gain-tt14; do
  python -m scripts.experimental.tt17_motor_protocol artifacts/tt17 "$case"
  python -m scripts.experimental.tt17_motor_audit artifacts/tt17 "$case"
  for branch in signed tt15 bounded; do
    python -m scripts.experimental.tt17_motor_trial artifacts/tt17 "$case" "$branch"
  done
done
python -m scripts.experimental.tt17_motor_report artifacts/tt17
```
