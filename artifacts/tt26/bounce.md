# Varied SO-101 paddle bouncing

A carried five-joint motor readout practiced on the stock SO-101 in MuJoCo. The varied releases changed all five starting angles independently and varied lateral ball speed, release height, gravity, air drag, paddle contact damping, and actuator gain across three physical settings. The ordinary branch began from the same readout and got the same proposals, number of launches, and horizons. Fewer varied balls stayed in play, so the branches completed different numbers of live control ticks.

Antithetic changes to the carried readout were selected using the physical rebound referee on practice and validation drops. The readout inherited from TT24 already knew bouncing. Training used privileged metric ball and blade observations. The separately held wrist runs used wrist RGB and encoders with the frozen TT24 optical observer. The paddle moved through all five joints with the jaw closed. A scored rebound required a real face contact, an upward exit, and a rise of at least 45 mm. Every held command tape reproduced its original termination and rebound events on replay.

| Held condition | Drops | Before rebounds / drops with ≥2 consecutive | Varied practice | Ordinary practice |
|---|---:|---:|---:|---:|
| Varied, wrist RGB and encoders | 12 | 11 / 0 | **22 / 6** | 11 / 0 |
| Varied, metric ball and blade | 24 | 26 / 1 | **35 / 6** | 26 / 1 |
| Ordinary, metric ball and blade | 24 | 228 / 21 | 144 / 20 | **268 / 23** |

The longest fresh varied wrist streak was four. The varied search roughly halved the number of ordinary held rebounds relative to matched ordinary practice. Each branch had 1,020 practice and 420 validation rollouts. The varied branch accumulated 66,145 live ticks across those rollouts and the ordinary branch 177,614. [Frozen motor policies](bounce-policy.npz) · [Scores, physical events and held action replays](bounce-practice.json.gz)

## Action effects

In separate physical releases, the same inherited actor issued small randomized joint-rate perturbations for two 60 Hz ticks at a time. Each fit used 1,763 encoder transitions drawn from 42 drops. The varied releases produced 42 qualified rebounds and the ordinary releases 63. Ridge regression took current angles, velocities, previously issued servo targets and fresh action as inputs. It predicted joint displacement and next velocity and could compose 60 Hz steps into a 30 Hz prediction. The fit used encoder transitions and issued commands. Metric ball and blade readings drove the source actor during collection.

| Encoder RMS displacement error on held releases | Models from varied practice | Models from ordinary practice |
|---|---:|---:|
| Varied, one tick | **5.42 mrad** | 5.62 mrad |
| Varied, two ticks | **11.99 mrad** | 12.50 mrad |
| Ordinary, one tick | **5.37 mrad** | 5.41 mrad |
| Ordinary, two ticks | 11.95 mrad | **11.89 mrad** |

On held varied releases, substituting a zero action into the varied model raised one-tick error to 7.24 mrad. Fixed encoder velocity gave 11.28 mrad. The source fit estimates short realized motion. Its 60 Hz transitions cover horizontal bounce postures, while a full-size return can move through distant joint angles and long sequences at saturated rates. [Varied fit](bounce-effects.npz) · [Ordinary fit](bounce-effects-ordinary.npz) · [Encoder samples](bounce-encoder-practice.npz) · [Fitting and held prediction receipt](bounce-effects.json.gz)

## Full-size precontact motion

The target replayed all six original TT25 demonstrations on the 2.74 × 1.525 m table and reproduced their legal landings. Each branch then had to track the original joint path under two changes to the simulated servo. The transport applied a mixture of the current and previous requests, with the previous request contributing 45% in the compliant setting and 65% in the damped setting. The tracker read current joint angles, velocities, and its own issued targets. It used the original joint path and rates as its motor reference. The source fits stayed frozen through all target replays.

| Precontact joint RMS error, mean across six paths | Frozen tape | Encoder feedback | Ordinary effects | Varied effects |
|---|---:|---:|---:|---:|
| Compliant servo and lag | 13.58 mrad | 13.34 mrad | 13.37 mrad | **13.28 mrad** |
| Damped servo and lag | 19.21 mrad | 18.11 mrad | 18.05 mrad | **17.91 mrad** |

Every branch contacted all 12 balls. Each landed five of six legal returns under the compliant setting and four of six under the damped setting. The movement errors differed by less than 0.15 mrad between the two learned fits. An earlier pilot applied stronger inverse corrections without anchoring to the original servo target. Both source fits lost legal returns and increased movement error relative to frozen replay in that pilot. [Bounded feedback and physical results](bounce-tracker.json.gz) · [Unanchored pilot](bounce-tracker-pilot.json.gz)

```sh
PYTHONPATH=src python -m scripts.experimental.tt26_bounce_practice
PYTHONPATH=src MUJOCO_GL=egl python -m scripts.experimental.tt26_bounce_score --wrist
PYTHONPATH=src python -m scripts.experimental.tt26_bounce_tracker /path/to/rallies.json
```
