# One wrist-controlled arm inspects new controls and plays from the resulting posture

One persistent learner played 36 simulated releases in three independent streams. It controlled all five joints from wrist RGB and its own encoders. When it investigated unfamiliar controls it moved the actual arm holding the paddle, read its encoders and played from the resulting physical posture. The simulator never reset or rewound that arm between the inspection and the release. [Runnable learner](../../scripts/experimental/tt24_continual_wrist.py) · [Saved learner](continual-wrist-0.npz) · [Physical course](../../scripts/experimental/tt24_single_arm_course.py)

Each stream began on stock hardware, then moved to a longer altered arm with a different elbow axis, camera mounting and drive strengths. The five motor requests acquired a previously unknown dense mixing. That mixing later changed with no visible cue. Both earlier configurations returned. A change in the wrist image or persistent disagreement between predicted and measured joint movement could trigger inspection. The fixture held the ball during inspection. The learner then got the usual 14–38 ticks of visual preparation. Each release sampled independent mass and drag and could encounter two unsignalled gravity changes. The fixed stock dispenser's position was independent of the arm's five starting angles. The cap was eight seconds.

| Body and controller | Qualified rebounds | Starts with ≥10 consecutive | Full eight seconds |
| --- | ---: | ---: | ---: |
| Stock, persistent learner | 266 | 10/15 | 10/15 |
| Stock, frozen original drive | 269 | 10/15 | 10/15 |
| Stock, true interface given | 269 | 10/15 | 10/15 |
| Altered, persistent learner | 190 | 7/21 | 4/21 |
| Altered, frozen original drive | 9 | 0/21 | 0/21 |
| Altered, true interface given | 268 | 10/21 | 7/21 |

All three controllers underwent the same physical inspection commands and began each game at identical measured joint angles. Across all 36 starts the learner made 456 rebounds versus 278 frozen and 537 with the true mappings. Sixteen inspections took 25.45 simulated seconds total, with a median of 1.41 s and range 0.67–2.85 s. The first invisible change inside the familiar altered installation cost the learner its ball. Across those three releases it made two rebounds versus 75 with the correct inverse. This controller acted on the resulting surprise at the next game break. [Stream 1](single-arm-course-0.json.gz) · [2](single-arm-course-1.json.gz) · [3](single-arm-course-2.json.gz)

The base readout learned bouncing from physical rewards on three donor arms using clean simulator motion and position measurements. On the receiving arm the observer has a previously calibrated model of the tool geometry and visually detected ball. Three consistent sightings move the horizontal control reference by at most 55.7 mm while preserving familiar height and tilt. This horizontal adjustment is supplied. Native Pong reward selected the separate scalar controlling how much prior reference to reuse. The action readout itself continuously produces five joint rates. [Original motor training](tt24_donor.md) · [Reference actually learned from Pong](reference-transfer.md) · [Workspace controls](workspace-stress.md)

## Use the saved policy

Each checkpoint includes the wrist readout, both optical calibrations, motor-interface memory and whether a new inspection is pending. The application can save it between releases. The inspection callback sends a bounded motor request and returns the next measured joint vector. It requires a contact-free setup interval provided by the apparatus. [Stream 1 checkpoint](continual-wrist-0.npz) · [2](continual-wrist-1.npz) · [3](continual-wrist-2.npz)

```python
from scripts.experimental.tt24_continual_wrist import ContinualWrist

learner = ContinualWrist("artifacts/tt24/continual-wrist-0.npz")
if learner.open_release(observation):
    learner.inspect(observation["joints"], send_rate_and_read_joints, seed=17)
    observation = read_live_observation()
learner.begin_play(observation)
inspected_during_play = False
while playing:
    requested = learner.act(observation)
    observation = play_one_tick(requested)
    learner.observe(requested, observation)
    if learner.inspection_requested and not inspected_during_play and launcher_holds_ball():
        learner.inspect(observation["joints"], send_rate_and_read_joints, seed=17)
        observation = read_live_observation()
        learner.begin_play(observation)
        inspected_during_play = True
learner.save_between_releases("my-wrist.npz")
```

An independent replay saved and reloaded the entire learner at every game boundary. It reproduced all 1,527 inspection commands and all 9,854 gameplay and preparation commands, plus the terminal physical records. [Audit runner](../../scripts/experimental/tt24_continual_wrist_audit.py) · [Stream 1](continual-wrist-audit-0.json) · [2](continual-wrist-audit-1.json) · [3](continual-wrist-audit-2.json)

On a second pass the launcher accepted one encoder-triggered inspection during preparation of the current release, as in the example above. This raised the total to 529 from 456 at a net cost of 6.12 simulated seconds of preparation across the 36 releases. On three invisible changes, the current inspection earned 35 rebounds versus four with the same preparation and the stale motor map. [Current-release checkpoint](current-pause-stream-0.npz) · [Paired controls and recall cost](current-pause.md)

On one 12-game replay, computing the motor request and updating the encoder monitor took a median 1.13 ms and 95th percentile 1.62 ms on ordinary frames. The 11 frames that adjusted the workspace took a median 9.41 ms and a maximum 11.96 ms. One of the 3,038 frames exceeded the 16.67 ms budget. Timing excludes rendering, physical stepping and the time spent moving the arm for its inspections. [Timing record](continual-wrist-timing-0.json)
