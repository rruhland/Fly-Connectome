# Gate-and-dock payload transfer after actuator coupling

The TT17 motor correction improved motion prediction under coupling on both arms. In the new task, it reached the first dock 12 frames sooner than cold on average. Both signed maps reached the dock about as quickly as the correction, and all four openings earned every delivery. During later changes, retaining the unaffected model helped over resetting both after motor weakening. The detector also raised false motor alarms in 13 of 16 sham courses. [Full summary](payload-summary.json.gz)

## Opening transfer

| Starting map | Mean RGB tool prediction error | First dock, mean frames | Docks earned | Requested rate-frame units |
| --- | ---: | ---: | ---: | ---: |
| Carried TT17 correction | 38.9 mm | 124 | 32/32 | 1,262 |
| Signed-only optical map | 51.3 mm | 123 | 32/32 | 1,222 |
| Original signed map | 50.2 mm | 124 | 32/32 | 1,217 |
| Cold | 102.0 mm | 136 | 32/32 | 1,608 |

Prediction error uses the 48 common opening pulses and each frozen starting model. The corrected map reduced this error from 42.7 to 27.0 mm on the gantry and from 59.9 to 50.8 mm on the five-axis arm relative to the signed-only map. It reached the first dock sooner than cold in ten courses, tied in four and took longer in two. Four pilot courses on older TT14 fixtures favored the correction over the signed map by 10 frames on average. The two maps differed by one frame in the opposite direction on fresh fixtures. [Gantry receipt](payload-gantry-coupling.json.gz) · [Five-axis receipt](payload-five-coupling.json.gz) · [Pilot summary](payload-pilot-summary.json.gz)

The clock starts with the payload task and includes all three exploratory pulses. Acquiring the corrected and signed-only maps for a one-off attempt adds 92 optical frames per fixture, putting the corresponding first-dock means at 216 and 215 frames. Historic body-model training precedes both clocks. Requested effort sums absolute normalized rates across port-frames during the two opening deliveries. [Frame budgets and map provenance](payload-summary.json.gz)

## Recovery during play

| Hidden change | Reset only flagged links | Reset both links | Keep both fits |
| --- | ---: | ---: | ---: |
| Motor attenuation | 122.0 frames, 32/32 docks | 139.5 frames, 32/32 docks | 123.0 frames, 32/32 docks |
| Puck response rotation | 137.0 frames, 31/32 docks | 137.5 frames, 32/32 docks | 167.0 frames, 29/32 docks |
| Sham | 98.0 frames, 32/32 docks | 100.5 frames, 32/32 docks | 98.5 frames, 32/32 docks |

Each mean times the first delivery after the switch, with misses capped at 232 frames. The score counts both remaining deliveries, which give each policy 464 live control frames. After motor attenuation, selective reset was earlier than resetting both in 11 courses and tied in five. It was earlier than keeping both fits in five, tied in nine and later in two. After puck rotation, either reset recovered faster on average than keeping both fits, and resetting both scored one additional dock. [Per-course comparisons](payload-summary.json.gz)

The selective branch's first alarm in the next delivery identified the motor alone in 16/16 motor changes and the puck alone in 12/16 puck changes. It named both links once and the motor alone three times after puck changes. Motor false alarms occurred in 13/16 sham deliveries after the nominal switch. Nine courses also had an alarm before the switch, so only seven share identical prefixes between all recovery policies. On those seven, selective reset still improved motor-change timing over resetting both, 120 versus 139.4 frames. The results show a cost to discarding a sound model and a benefit to resetting after a large puck change. The false alarms limit the detector's reliability. [Attribution counts and matched-prefix results](payload-summary.json.gz)

The replay regenerated the physical and RGB marker trajectories, action decisions and rewards on all 192 tapes covering 19,488 control blocks. These are within-installation transfers in simulation using calibrated projected cues, a supplied model structure and one change at a time. [Replay and summary code](../../scripts/experimental/tt17_payload_report.py)

## Test

The simulated TT6 gantry and five-axis arms retained the coupled controls from their fresh TT17 installations. A controller moved a planar inertial puck through a visible gate, then stopped it at a dock. The dock marker appeared when the puck's path passed within 5.4 cm of the gate. It earned one point by staying within 3.6 cm of the dock and moving no more than 1.9 cm per update for two consecutive updates. Each of four deliveries had 232 arm control frames, with two pulsed frames and six settling frames per payload update. Arm and puck state continued across deliveries. [Task and learner](../../scripts/experimental/tt17_payload.py)

During the first two deliveries, each opening compared four initial motor Jacobians under identical physics, three exploratory pulses and feedback. The bounded correction came from 68 reference and 24 changed-wiring optical control frames, before any table points on the coupled wiring. The signed-only map came from the same TT17 optical evidence. The original map kept the archived signed assignment, which matched the hardware's port order. The cold map started at zero. The five-axis signed-only optical fit swapped two ports. The separate original-map control tests the value of continuous correction when the discrete assignment is already correct. [Gantry checkpoint](motor-gantry-coupling-bounded-start-player.json.gz) · [Five-axis checkpoint](motor-five-coupling-bounded-start-player.json.gz)

The controller identifies command-to-tool displacement and tool-to-puck velocity as separate linear regressions. It observes projected RGB tool, puck and active marker positions, its own motor commands and the sparse dock reward. The RGB color roles, view alignment and fitted model structure are scripted. It uses the same visual goal controller at the gate and at the dock. All runs learn the puck model within the new task and use ridge regression without backpropagation. The simulator keeps exact poses, the switch schedule and both transformation matrices private. [Implementation](../../scripts/experimental/tt17_payload.py)

After delivery two, the recovery challenge attenuated every coupled actuator command to 47% of its previous value, rotated the puck's hidden acceleration response by 50° to 99°, or left both links unchanged. All recovery branches began with the corrected map. Repeated visual prediction errors could trigger one common change detector. One branch reset only the flagged regression, one reset both, and one kept both. Resetting restored the prior and added the three latest observations. All three branches briefly raised exploration after an alarm and continued both online updates. The detector was enabled after 20 observations and used a five-observation window. The eight held-out courses per fixture use the same physical trajectories before a switch within each branch. [Paired summaries and attribution audit](payload-summary.json.gz)

## Reproduce

Use the [TT6 simulation environment](../tt6/README.md) and the linked optical checkpoints.

```sh
export PYTHONPATH=.:src:scripts/experimental OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for site in gantry five; do
  python -m scripts.experimental.tt17_payload \
    "artifacts/tt17/motor-$site-coupling-setup.json.gz" \
    "artifacts/tt17/motor-$site-coupling-bounded-start-player.json.gz" \
    "artifacts/tt17/payload-$site-coupling.json.gz" --count 8
done
python -m scripts.experimental.tt17_payload_report artifacts/tt17/payload-summary.json.gz \
  artifacts/tt17/payload-gantry-coupling.json.gz artifacts/tt17/payload-five-coupling.json.gz
```
