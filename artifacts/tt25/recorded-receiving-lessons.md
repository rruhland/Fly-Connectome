# A learned arm block on previously unseen, recorded table-tennis shots

[Watch all three successful unseen feeds](recorded-nearby-returns.mp4) · [Watch the separate compact two-arm rally](recorded-arm-stock.mp4)

On a regulation-size simulated table, a camera-driven SO-101 landed three of 48 incoming shots from a real human–robot rally dataset. The 48 shots were chosen for physical similarity to six successful simulated demonstrations. A shuffled assignment of the same six joint targets and a stationary paddle both landed zero. On 32 unfiltered new shots the camera pupil landed zero. These tests measure receiving a single mid-rally ball. There was no live opponent reply. [Focused course](recorded-nearby-lessons.json.gz) · [Unfiltered course, teaching trials and controls](recorded-pose-imitation.json.gz)

## What the arm actually learned

The practice teacher could inspect simulator ball position and velocity. It searched physically possible arm postures for ones that might redirect the ball over the net. From a normal home pose the arm had to reach the proposed posture through rate-limited motor commands and land a ball on the far table under the physical referee. Of 24 reach-screened development feeds, six yielded a verified demonstration in 444 proposal trials. Court dimensions, shot speeds, spin and launch positions remained in SI units. The moving links could collide with the table. [Physical teaching and camera pupil](../../scripts/experimental/tt25_pose_demonstrations.py) · [Source data](https://github.com/google-deepmind/competitive_robot_table_tennis)

The pupil retained the early stereo-camera history and joint target from each demonstrated success. On a new shot it read four frames, selected the nearest example in visual features and issued bounded joint rates toward that pose. It continued holding the paddle. It saw the ordinary orange-silhouette detector output, with no incoming-shot state vector, teacher selection or pre-positioned arm. This is a learned choice of **receiving block** from six demonstrated joint targets. It does not establish continuous stroke generation, learning from scratch or a complete game. [Checkpoint](recorded-pose-imitation.npz)

Eight other real shots were selected around each demonstrated launch using position, velocity and spin. Only the course constructor used this information. The three successes came from two of the six neighborhoods. Their initial positions differed by approximately 7, 12 and 18 cm from the corresponding demonstrations, and their velocity vectors differed by 0.20, 0.16 and 0.66 m/s. Each successful camera rollout reproduced the same referee events when rendered again. The camera policy made twelve paddle contacts versus five with its demonstrated joint targets permuted and two with the stationary arm. Three favorable paired shots are a small result. Shot IDs are disjoint, and the second dataset does not provide session IDs for testing independence by match. [All focused launches and controls](recorded-nearby-lessons.json.gz) · [Verified video receipt](recorded-nearby-returns.json)

The repertoire itself limits transfer. On the 32 unfiltered held feeds, trying *every* demonstrated posture with perfect hindsight still produced zero legal returns. Four points admitted any racket contact. Acquiring more varied contact experience and learning to correct the paddle during approach address a limitation that choosing among these six examples alone cannot overcome. [192 physical posture checks](recorded-pose-coverage.json.gz)

## What cheaper practice accomplished

| Procedure | Physical exposure | Result |
|---|---:|---|
| Existing compact camera policy and stationary control | 32 paired recorded incoming balls | 31 first legal near-side bounces. Trained arm made two contacts, stationary one. Neither returned a ball. |
| Reward-ranked fixed ready posture | 33 postures on 12 reach-screened training feeds; 32 other unfiltered test feeds | No legal training return and none on test. Selected posture touched one test ball, default touched two. |
| Perturb the old policy's own two recorded contacts | 128 direct motor-rate variations per contact | 17 and 21 variations retained contact. No legal return. |
| Geometry-assisted training demonstrations, then camera-only play | 444 proposed trials on 24 practice feeds; six successful examples | Three unassisted legal blocks on 48 similar real launches. Zero on 32 unfiltered launches. |

The first two compact-policy contacts left the ball traveling toward the player's end. Contact alone would have rewarded brushes that never reversed play. These procedures used different training budgets and assistance, so this table compares feasibility and failure modes. It is not an equal-budget ranking of teaching methods. [Original physical receiving test](recorded-moderate-receive.json.gz) · [Fixed readiness](recorded-learned-block.json.gz) · [Contact practice](recorded-contact-practice.json.gz)

## Reproduce

Use the pinned DeepMind rallies JSON identified in the physical benchmark, in the repository's MuJoCo environment. The six-demonstration checkpoint and complete evaluation are saved here. [Flight and geometry prerequisites](recorded-benchmarks.md)

```bash
python -m scripts.experimental.tt25_reference_receiver /path/to/rallies.json
python -m scripts.experimental.tt25_learned_block /path/to/rallies.json
python -m scripts.experimental.tt25_contact_practice /path/to/rallies.json
python -m scripts.experimental.tt25_pose_demonstrations /path/to/rallies.json
python -m scripts.experimental.tt25_pose_coverage /path/to/rallies.json
python -m scripts.experimental.tt25_nearby_lessons /path/to/rallies.json
python -m scripts.experimental.tt25_receiving_movie /path/to/rallies.json
```
