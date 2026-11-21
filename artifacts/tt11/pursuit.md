# Tracking a moving target after learning table tennis

The player followed a moving target in 3D while its motor delay and noise changed. In this game the arm earns one frame on target each time its tip is within 4.5 cm. Moving to a new camera mount rotates the two projected views of both markers about the workspace origin. No camera rotation, joint angles, future target path, or switch time enters the policy. The marker identities, shared metric scale, reaching objective, and the algorithm for using movement history were supplied. The arm's action-to-motion map came from table-tennis experience. [Rotation-fitting method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html) · [Environment and optical learner](../../scripts/experimental/tt11_tracking.py) · [Schedules and controls](../../scripts/experimental/tt11_tracking_trial.py)

## First encounter with a moving target

Across 36 sessions per view condition, the player got 24 decisions per 6.4-second course. The analytic orientation fit aligned prior motor predictions with recent observed tip motion. Ordinary online regression corrected the remaining local discrepancies. All models encountered the same smooth moving goals under immediate, two-frame, then one-frame transport delay, with motor noise starting at frame 48. Aligning the map while playing worked without any dedicated motor probes. Each session used a newly sampled path and rotation; the original and rotated conditions shared the same arm, path and commanded-rate disturbance schedule.

| Controller | Familiar view, frames / 6,912 | New view, frames / 6,912 | New view, segments with at least two consecutive on-target frames / 288 |
|---|---:|---:|---:|
| Table map, online alignment with no initial probes | 1,380 | 1,218 | 181 |
| Table map, no alignment or initial probes | 1,422 | 845 | 135 |
| Table map, alignment after two probes | 1,241 | 1,195 | 170 |
| Empty map, probes first, weaker regularization | 896 | 900 | 137 |
| Table map with two probes, no prediction of target motion | 664 | 680 | 146 |

At the changed view, unprobed alignment gained 318 on-target frames over the stronger cold-start player. The paired, site-stratified 95% session-bootstrap interval for the gain was 3.2 to 6.0 percentage points. In the familiar view, adapting the axes cost 42 frames relative to leaving them alone. [Installation A](tracking-offset_a.json.gz) · [Installation B](tracking-offset_b.json.gz) · [Six-joint arm](tracking-rotary.json.gz)

## Camera moves twice during the game

In 30 new sessions the view changed twice at unannounced frames. The third view returned to the first mount half the time. One target jump per session challenged the change detector. Every controller got the same 30 Hz observations and eight-frame decision interval. They kept playing with the same physical arm, pending commands and changing noise. The player looked for a rigid rotation that explained a sudden jump of both visible markers, then rotated its accumulated motor evidence into the new frame. This uses the shared origin of the simulated views. Camera translation, occlusion and unlabeled real images remain open.

| Handling a detected view change | On-target frames before first change | Between changes | Following second change |
|---|---:|---:|---:|
| Rotate and retain the learned motor map | 174 | 277 | 216 |
| Discard local fit and learn it again from the table prior | 174 | 145 | 119 |
| Continue ordinary online updates | 174 | 34 | 20 |

Transporting the map detected 50 of 60 changes on the exact frame with zero extra detections. It stayed on target for 493/3,934 frames after changes, compared with 264 for immediate re-learning and 54 for ordinary updates. All three versions had modest absolute accuracy with only one motor decision every eight frames. All 90 action tapes replayed physically. At each installation, restoring the saved player in the middle of a pulse reproduced the remaining actions and outcomes. The B checkpoint also reproduced all 180 native Pong tapes and 41 earned points. [Five-axis A](remount-offset_a.json.gz) · [Five-axis B](remount-offset_b.json.gz) · [Six-joint](remount-rotary.json.gz) · [Combined player](remount-offset_b-player.json.gz) · [Native Pong retention](remount-pong.json.gz)

## Updating the command at camera rate

On fresh courses the controller evaluated a six-frame-ahead tip and target position using each new frame (30 Hz). The maximum command per channel per tick was 0.25, giving the same eight-frame absolute-command upper bound of two units as the previous pulse controller. Both used the same motor-excursion limit. The higher-rate controller could spread those commands throughout the interval. Data below pool eight fresh courses per installation across two five-axis installations and a physical six-joint design. Each view had 4,608 scored frames.

| Controller | Familiar view | New view |
|---|---:|---:|
| Eight-frame controller, table prior and online alignment | 995 | 813 |
| Camera rate, table geometry and table-fitted actuator timing | 1,997 | 1,739 |
| Camera rate, table geometry and instantaneous actuators assumed | 1,496 | 1,204 |
| Camera rate, table geometry and one mean settling curve | **2,184** | **2,014** |
| Camera rate, table fit without extrapolating the target | 1,107 | 1,030 |

The mean curve is an equal average of the 54 candidate step responses in the pre-existing identification library. It outperformed the individually fitted timing in every installation/view aggregate. Its gain over those fitted curves had a 95% site-stratified paired session-bootstrap interval of 0.6–7.1 percentage points on familiar views and 1.6–10.2 on rotated views. These are separate courses from the mid-game camera-change test. [A](pursuit-offset_a.json.gz) · [B](pursuit-offset_b.json.gz) · [Six-joint](pursuit-rotary.json.gz) · [Predictive servo](../../scripts/experimental/tt11_pursuit_servo.py)

## Does prior experience still help a fast online learner?

A camera-rate local learner updated its action-to-displacement map every four frames in the new task, using the same mean settling curve, motion forecast and per-frame command budget. With exactly the same initial channel probes, the table prior raised time on target from 1,537 to 1,736 frames in the familiar view and from 1,500 to 1,692 in the rotated view. All three bodies improved in both conditions. Using the prior from the very first action and spending no dedicated time probing raised these scores to 2,185 and 1,966. In the rotated view, that unprobed player spent 478/1,152 frames on target in the first 1.6 seconds across courses, versus 225 for the cold learner. The comparison including initial probes isolates prior knowledge from the time spent probing. [Local learner](../../scripts/experimental/tt11_pursuit_learning.py) · [A](pursuit-learning-offset_a.json.gz) · [B](pursuit-learning-offset_b.json.gz) · [Six-joint](pursuit-learning-rotary.json.gz)

## Returning the pursuit result to table tennis

Using the mean settling curve on the same fresh C and D five-axis installations as the timing selection experiment produced four complete losses. On C it scored 2–11 and 3–11. On D it scored 2–11 and 1–11. Across those fixtures the arm landed only 1/24 serves and sustained zero points. The commissioned arm geometry and complete starting state matched the fixed timing comparisons. The mean curve that worked well for continuous pursuit performs poorly on the fast service stroke. Keep the timing fits separate for these tasks. [Full games C](pursuit-table-c-games.json.gz) · [Full games D](pursuit-table-d-games.json.gz) · [Matched per-installation controls](timing.md)

## Reproduction

Run in the hall environment from the repository root. Each runner records action tapes and replays its scores through the physical simulator. Repeat with sites offset_b and rotary for the other two installations.

```sh
python -m scripts.experimental.tt11_tracking_trial offset_a /tmp/tracking.json.gz --count 12 --seed 9125736 --replay
python -m scripts.experimental.tt11_remount_trial offset_a /tmp/remount.json.gz /tmp/player.json.gz --count 10 --seed 12845705
python -m scripts.experimental.tt11_pursuit_trial offset_a /tmp/servo.json.gz --count 8 --seed 9467231
python -m scripts.experimental.tt11_pursuit_learning offset_a /tmp/local.json.gz --count 8 --seed 9467231
python -m scripts.experimental.tt11_pursuit_table c artifacts/tt11 --games 2 --seed 739294612
```
