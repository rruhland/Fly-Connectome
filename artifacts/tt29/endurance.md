# A 100-return regulation rally from camera observations

Two different measured openings produced 100 alternating legal returns. The same simulated SO-101 stayed fixed to the near tabletop. Each point began with one recorded ball delivery. For the rest of the point, the two paddles played one continuously simulated ball. The referee required a separate paddle contact, release and landing on the correct half for each return. Each point stopped at its 100th legal return. The original delivery contributes zero.

| Measured opening | Prior successful arm movements | SO-101 / partner returns | Time | Median departure speed | Median stroke interval | Closest audited clearance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7271, practiced | 40 | 50 / 50 | 69.37 s | 5.70 m/s | 0.671 s | 8.71 mm |
| 11292, new to this recorded training sequence | 80 | 50 / 50 | 69.57 s | 5.65 m/s | 0.675 s | 18.60 mm |

Each replay reproduced all 502 referee and contact events. The CAD checks covered 33,297 and 33,393 states at 480 Hz, including the final state. The peak measured joint rates were 6.23 and 6.09 rad/s. The servo's unloaded reference is 5.45 rad/s. Both runs used capped target velocity and 3.35 N·m torque. These are simulator results. For context, the 13,088 measured DeepMind opening shots had a median departure speed of 5.50 m/s. [7271 audit](camera100-physics.json) · [11292 audit](camera100-independent.json) · [Measured speeds and source](../tt25/recorded-secondary-reference.json)

[Watch the complete 100-return point at game speed](camera100.mp4) · [Final landing and both camera inputs](camera100-final.png) · [Video and physics receipt](camera100-audit.json)

## What changed during play

The arm uses RGB from two external cameras, encoders and its issued commands. It receives one positive learning event for landing a legal return. It stores the estimated approach and its own joint targets and retimes those targets on later approaches. The implementation supplies a color detector, calibrated cameras, nominal ball dynamics, a model of the arm, a stroke generator and the landing objective. The opponent sees ball state and drives a finite-mass physical carriage. Training uses no backpropagation. The recorded opening is the only time the arena places the ball during the point. [Controller](../../scripts/experimental/tt29_action_memory.py) · [Referee audit](../../scripts/experimental/tt29_endurance.py)

On 7271, the previous 66-return point ended on a blade-edge contact. Adding a penalty for off-center contact in the controller's predicted strokes produced 100 on the subsequent attempt. The controller used 44 retrieved movements and seven successful searches and finished with 90 stored examples. The independent 11292 run used the existing contact objective, 47 retrieved movements and four successful searches. It reached 100 as the fourth point in a continuous course that had scored 10, 64 and 6 on its first three points. The partner's random seeds differed from the original practice course. Solving 11292 did not require the new center-contact penalty. [Practiced source, actions and checkpoint](camera100-source.json.gz) · [Carried course, actions and checkpoint](camera100-transfer-source.json.gz)

The original four-opening course scored 0, 0, 14 and 66 with memory and 0, 0, 60 and 2 with the same planner and memory disabled. A separate controller using nearest remembered actions without physics verification or fallback search scored 2, 2, 2 and 8. Memory can speed good selections and change which errors the arm makes. The native planner remains an important part of the result. [Original course](camera-action-memory.json.gz) · [Matched memory-disabled control](no-action-memory.json.gz) · [Unchecked recall](pure-recall.json.gz)

On 7271, the near arm landed shots from −0.174 to +0.639 m across the opposite half. The far carriage returned all 50. Its landing points varied from +0.099 to +0.410 m across the near half and from 0.411 to 0.852 m from the net. Its range of placements at the arm was narrower. The arm met the ball over a 70 mm lateral and 79 mm depth span. This is a cooperative forehand drill. A fixed-base SO-101 cannot reach every legal shot on a regulation table. [Per-stroke positions, speed and spin](camera100-physics.json) · [Measured reach envelope](workspace.json.gz)

## Transfer and retention

With a checkpoint containing 40 prior movements, two matched openings gave the following consecutive return counts. The changed configurations did not reveal their physical settings to the learner. In the one- and two-camera cases involving the original wrist board, the controller loaded a wrist-camera estimate from 24 prior solo drops. Each row starts with fresh flight estimates. [Changed conditions](../../scripts/experimental/tt29_conditions.py)

| Change to the live simulator | Opening 7271 | Opening 7345 |
| --- | ---: | ---: |
| Ball mass +8%, changed table and paddle damping and paddle friction | 2 | 42 |
| Paddle mass +13%, elbow +16%, wrist +20%, joint damping +22% | 15 | 12 |
| Two control intervals of command delay, one frame of image delay, command noise | 0 | 14 |
| One external camera | 4 | 0 |
| Original wrist board alone in lateral stance | 0 | 0 |
| Moved external camera and wrist camera, stale pose estimates | 0 | 0 |
| Wrist only, simultaneous camera angle, latency, mass and contact changes | 0 | 0 |

Each pair used offset 289065 plus point ordinal and the original shot ID to seed the far partner. The complete event and command streams are available for the [material](camera-shift-materials.json.gz), [arm](camera-shift-loaded_arm.json.gz), [timing](camera-shift-latency.json.gz), [one outside camera](camera-shift-outside.json.gz), [wrist](camera-shift-wrist.json.gz), [remounted cameras](camera-shift-moved_camera.json.gz) and [compound change](camera-shift-combined_wrist.json.gz) cases.

Longer solo practice did not improve monotonically. On eight held releases, the 0-, 4-, 12- and 24-drop checkpoints made 6, 33, 42 and 25 qualified rebounds. A separate validation course promoted the four-drop checkpoint, which made 10 rebounds versus four cold on another eight held releases. The retained controller reproduced those scores after the table matches. Wrist calibration learned during bouncing improved subsequent optical estimates and beginner feeding results. An initial full-match warm/cold comparison gave equal rally lengths in all four pairs. [Warm-up and retention](tt29_bounce.md) · [Full-game transfer controls](tt29_warm_rallies.md)

[Online adaptation, bounce-to-match transfer and runtime](adaptation.md) · [Single wrist-camera results](tt29_wrist_only.md)

## Reproduce the live learner

The raw opening comes from the pinned CC BY 4.0 DeepMind dataset. The runner checks its SHA-256 before accepting any shot. Run these commands from the repository root with the table-tennis dependencies installed. Compilation of the isolated motion trials requires a C++ compiler and OpenMP. Video export also uses FFmpeg.

```sh
curl -fL https://raw.githubusercontent.com/google-deepmind/competitive_robot_table_tennis/54cf18e41a8f134d98ad38710001ed50aabb8ebc/rallies.json -o /tmp/tt29-feeds.json
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt29_action_memory /tmp/tt29-feeds.json /tmp/tt29-live.json.gz \
    --ids 7271 --checkpoint artifacts/tt29/camera-action-memory.json.gz --robust-contact
python -m scripts.experimental.tt29_vision_replay /tmp/tt29-feeds.json \
    --source /tmp/tt29-live.json.gz --id 7271 --receipt /tmp/tt29-replay.json \
    --fast-partner --movie /tmp/tt29-point.mp4
```

The first program chooses every new joint command online. The second plays the saved commands, audits every physics state, and renders the point. To try a new opponent sequence on the same opening, pass a fresh integer with `--seed-offset` to the first command. [Dataset](https://github.com/google-deepmind/competitive_robot_table_tennis/tree/54cf18e41a8f134d98ad38710001ed50aabb8ebc)
