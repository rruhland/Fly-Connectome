# SO-101 on the end clamp

The benchmark runs a stock-scale SO-101 follower on a rigid clamp at the end of a 1.60 m table. Its external wrist camera sees the physical gripper and paddle. On single-camera runs the arm gets exactly one RGB image stream. The left outside view and wrist view can each drive the arm. The spectator render is added during replay. [Wrist optics and mount](fixture.md) · [Before and after](fixture-wrist-comparison.png)

## What carries between cameras

A short active probe on the first lens can identify a body property that continues to matter when the lens changes. The four-frame motor delay is the cleanest test. The learner commanded 28 probe frames (0.93 simulated seconds) and compared visual motion with an independently rendered nominal response. The outside camera saw the blade move. The wrist camera saw the background move. Each yielded the same four-frame estimate. After transferring that estimate, the destination used only its own images to plan shots. Counts below are first legal returns on fresh opposing serves.

| Source practice → target play | Target with no delay correction | Use source estimate, no target probe | Probe target directly |
| --- | ---: | ---: | ---: |
| Wrist → outside | 0/20 | 7/20 | 7/20 |
| Outside → wrist | 0/20 | 11/20 | 11/20 |

The gains occurred on a bench with a deliberately delayed motor interface. An earlier matched eight-start screen gave wrist-target 0→4 and outside-target 1→1. The 20 starts in each row are a fresh replication. Sensor isolation, destination equivalence and the exact delayed commands are testable from the receipts. The experiments use an independent nominal arm model; the corrected player still has to intercept and legally land the ball under the real simulator referee. [Delay test and independent replication](latency-transfer.md)

The motor convention transferred across lenses too. After an unseen change in which ports drove five axes and in what direction, each lens recovered the map from 35 target frames (1.17 simulated seconds). On one new game seed, using the wrist-learned map through the outside camera changed the score from 0–11 to 8–11. The reverse changed 0–11 to 1–11. The two calibrated branches reproduced the normal-interface motor actions. These are full game scores for one remapping on a known body and lens setup. [Motor mapping and game receipts](motor-carry.md)

[Watch the same incoming ball before and after cross-camera timing transfer](cross-camera-transfer.mp4) · [Exact frame and scoring receipt](cross-camera-transfer-replay.json)

The visual flight model told a different story. A prior learned from wrist-only opponent serves improved the outside camera’s median 200 ms forecast from 51 to 32 mm, yet this changed two legal first returns into one on eight held starts. In the other direction the player gained one return while its forecasts worsened. A better estimate can change which strike the planner chooses and improve or degrade the actual return. [Arc transfer and phase-shuffled control](arc-carry.md)

## Playing through one eye

On 12 identical receiving starts, the wrist player landed 8 legal returns using apparent ball size, with 5 points won. The outside player landed 5 using a causal flight fit, with zero points won. The opposing-service ablation tests a frozen receiver and includes all 48 physical replays. Its role is to establish what each observation stream supports before training. [Receiving ablation](mono-flight.md) · [Metric tracking and camera handoffs](natural-view.md)

On the new clamped arm, each single-camera learner also explored four physically feasible serves using real target tosses, ranked them from practice, then played complete games. It can keep updating the same value memory when it changes views. The setup uses a supplied CAD model, a known orange ball size, court geometry, distinguishable colors, the service role and the clock. Calibration, actuator response and stroke values learned in these trials are specific claims. Independent discovery of the rules and object roles from raw pixels remains open. [Single-camera checkpoints, matched scoring and run instructions](camera-carry.md) · [Sparse reward exploration](opening-exploration.md)

[Wrist-only full game](wrist-only-game.mp4) · [Outside-only full game](left-only-game.mp4)

## Table and paddle dots

Originally twelve surveyed markers helped register camera poses in court coordinates. The paddle rim dots and gripper tip mark provided a separate geometric reference for the arm. The factory player's receiving planner worked from the reconstructed ball, its own commands and its clock. The rim measurements entered body-identification diagnostics and could also estimate the blade face normal.

On this clamp, eight marked then eight unmarked training attempts and sixteen unmarked attempts selected the same serve. Both completed the same entirely unmarked game at 8–11 with 11 legal returns and two points containing multiple learner returns. The zero-practice factory branch reproduced the same actions. Keeping earlier gantry flight and impact physics together produced 11 legal returns in this game versus nine with nominal physics; the latter finished 6–11. On matched visual tapes the table posts improved ball ranging, but the red post confused the blade detector. [Controlled table-marker curriculum](curriculum.md) · [Paddle and gripper marks tested separately](arm-tag-factorial.md)

With only the paddle and gripper dots present, the wrist could recover the blade angle on 116 of 183 frames, with median error of 1.7°. The natural one-eye silhouette leaves two possible face orientations. The existing orange rim dot also looked like the ball. After stripping the tags, a calibration learned on two separately tagged points worsened center estimates on two new points through each camera. [Four-way marker study](arm-tag-factorial.md)

A detector can use multiple plausible color blobs, their apparent size and recent optical motion to recover from those distractors. On 94 held frames containing both kinds of markers, this opt-in change raised ball-depth coverage from 0 to 21 frames on wrist and 4 to 68 outside. Outputs on the paired clean footage were bit-identical. An individual outside depth error still reached 403 mm. [Distractor fix and limits](natural-association.md)

The optional 126° wrist lens saw the toss in 12/24 sampled frames where the 84° lens saw none. It still saw the gripper and blade in all 62 held poses, but the existing controller returned fewer opposing serves with it. Separately, registration on 169 original wrist frames sped up from 22.5 to 36.0 frames per CPU second, with identical fitted poses and decisions. Timing excludes rendering and planning. [Wrist comparison](fixture.md) · [Reproducible timing benchmark](../../scripts/experimental/tt20_wrist_registration_bench.py)

## Longer rallies

A stronger opponent predicts incoming shots with its own private simulator state and searches legal, rate-limited approaches to the ball. The real rival paddle still has to make physical contact and land a legal reply. On 12 matched physical openings on the earlier pedestal, legal rival returns increased from four to eight. The longest alternating exchange increased from three to four legal shots. On the new clamp, the frozen marked player produced few opportunities, and both rivals beat it 11–1. [Opponent qualification](opponent.md)

[Watch four alternating legal returns on the new unmarked clamp](so101-longest-rally.mp4) · [Three-stream RGB and referee replay](so101-longest-rally-replay.json)

Correcting the returner's loaded-arm timing or recomputing the whole dry paddle-path catalogue for the independently identified wrist mass did not improve the seven-of-twelve matched return count. [Blade-path correction](blade-correction.md) · [Rebuilt motion catalogue](direct-receive-loaded-atlas.md)
