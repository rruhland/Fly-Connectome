# Learning continuous motor effects within a Pong rally

The continuous-gain controller returned the next ball on 20 of 24 fresh starts with an unannounced change in actuator strengths. The shorter held-request calibration returned 9 of 24 on the same starts. The continuous controller alone returned 13 first balls, the shorter controller alone returned two, and both returned seven. The arms issued identical requests and saw identical public camera histories through all 6,000 paired ordinary-control decisions. All 24 first opportunity clocks matched.

## Learned response

The motor observer starts with a motion model fitted from 12 six-second ordinary runs. Four further ordinary runs supplied 1,167 held-out public paddle positions and gave a root-mean-square prediction error of 0.00353 in normalized camera coordinates. Replaying those four runs produced zero change detections or changed requests. The fit used the controlled member and axis of the inherited visual relation, the action IDs requested by the agent, and actual camera intervals. The two-stage temporal response and bounded travel were supplied as modeling assumptions. The response time constants, ordinary gains, speed and travel limits came from fitting public observations.

During play, a filter predicts the next observed position using the current estimate of velocity and filtered drive. It compares that prediction with short-lived alternatives that give each action an independent continuous gain. Each alternative represents a different possible onset time and estimates gains from its own chronological observation errors. Sufficient cumulative likelihood advantage lets the alternative take over. The controller continues to follow the inherited desired direction. It can choose an uncertain request when its estimated upper movement effect could make useful progress in that direction. Known gains later guide ordinary pursuit. The fit and online updates use numerical least squares and conditional Gaussian updates. The learner receives no change signal, true force, native state or detection label.

The shared motion model assumes that changes act through the three gains. Its treatment of clipped travel and raster measurement noise is approximate. These tests varied the gains and the time of the change. The method carries the latest gain distribution and its covariance and uses no catalog of permitted strength layouts or action orderings.

## Matched native comparison

Each layout used eight new starts with four launches in each direction. Each arm began each game from its ordinary checkpoint. The evaluator substituted the three values at the scheduled time. The values are listed in request-ID order.

| Layout | Switch | Forces | First ball, continuous / shorter | Later balls, continuous / shorter |
|---|---:|---|---:|---:|
| A | 3.2 s | +0.91, −0.72, +0.12 | 7/8 · 2/8 | 27/27 · 0/2 |
| B | 5.1 s | −0.34, +0.96, −0.85 | 6/8 · 5/8 | 16/17 · 3/7 |
| C | 6.7 s | +0.76, +0.06, −0.58 | 7/8 · 2/8 | 16/17 · 0/2 |
| All | | | **20/24 · 9/24** | **59/61 · 3/11** |

The median detection latency was 0.208 simulated seconds for the continuous learner. Eight of its first-ball returns preceded a complete supported gain order. In five of those eight paired games, only the continuous learner returned the ball, and it issued requests consistent with the changed direction while at least one gain comparison remained unresolved. A supported order requires each adjacent gain difference to exceed two estimated standard deviations. Those estimates describe the filter's confidence under its assumed model.

The later-opportunity counts reflect how long each arm sustained play. The continuous arm completed six games naturally and reached the 20-second cap in 18. The shorter arm completed 23 naturally and reached the cap in one. The paired comparison used 31,543 transition cameras, 48 initial cameras and 141 seconds of wall time on one worker. The specification and source hashes verified at completion.

## Where it still missed

Two immediate misses in layout B left 0.567 and 0.817 seconds between the change and contact. The learner detected the changes before contact and tested uncertain requests before settling on the useful action. The layout A miss left 1.058 seconds, of which 0.458 elapsed before detection. In the layout C miss, the learner had 3.725 seconds and settled on the direction-consistent request before contact.

A separate replay used the evaluator's true effect ordering immediately at the switch for those four selected misses. The first three recovered with the same inherited visual pursuit policy. Layout C still missed. The replay used privileged ordering solely to diagnose those four starts.

Heald, Lengyel and Wolpert model sensorimotor learning through feedback-conditioned context inference, with memories acquired and expressed across trials. Their implementation predicts context-conditioned outcomes and updates context responsibilities after feedback. This offers a direct precedent for a future test of whether storing several acquired motor responses accelerates recovery on a recurring change. [Primary paper](https://doi.org/10.1038/s41586-021-04129-3) [Authors’ implementation](https://github.com/jamesheald/COIN/blob/796a8dad3861cbc7c5702ae2d0925f2421e9b9eb/COIN.m#L608-L845)

## Reproduction

The archive contains the ordinary camera recordings, fitted model, starting checkpoints, fixed conditions, paired receipts and separate oracle receipts. Its prospective specification SHA-256 is `97d03906ee6498c73d1f89a2c5b3518f9a236fe8c829bd8a020e63d259a29077`. [Experiment archive](../../artifacts/m1b/2026-11-19-continuous-motor-effects.zip)

Run from the repository using its Python environment and an unused output directory.

```bash
export PYTHONPATH=src:scripts/experimental OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unzip artifacts/m1b/2026-11-19-continuous-motor-effects.zip -d /tmp/pong-continuous-input
python scripts/experimental/run_pong_continuous_motor_scout.py /tmp/pong-continuous-replay \
  /tmp/pong-continuous-input/receipts/conditions.json \
  /tmp/pong-continuous-input/acquisition/plant-fit.json \
  /tmp/pong-continuous-input/receipts/shorter-normal.json
```
