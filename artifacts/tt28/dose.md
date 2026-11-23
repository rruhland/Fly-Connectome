# Longer online solo practice on the SO-101

Four drops were enough to put the face-angle estimate within 0.4° on both mounts. The four-drop checkpoints scored 216 qualified rebounds over 24 held releases, compared with 21 at zero drops and 175 after 192. Across 344 original recording–mount combinations, the 192-drop checkpoints made one nearby return that the four-drop checkpoints missed and lost one they made. Their unrestricted outcomes were identical.

Two camera-driven courses played successive solo drops under different paddle mounts. Each course started with the same earlier wrist controller and nominal visual calibration. The changes were a 0.235 rad rotation mostly about the handle in one course and a 0.235 rad mostly transverse rotation in the other. Practice used two alternating ball and servo regimes. The drop registry was fixed before training. Checkpoints at 0, 4, 16, 64 and 192 releases are prefixes of one course per mount.

The wrist policy issued five joint-rate commands from wrist RGB and its encoders. A stationary calibrated stereo pair observed the first flight reversal in each source release. The learner estimated the median change in the directions of those observed reversals in the gripper frame. It also updated four scalar estimates of rebound reward for different tradeoffs between preserving blade position and restoring blade angle. Each release changed those estimates only after the robot had played the ball. The learner used the previous estimates on the next release. No backpropagation was used. The wrist controller's weights, the nominal direction reference, the flight-acceleration estimate, the impact detector and the kinematic model came from earlier experiments. The fixture used the new mount angles to set up the physics. The scoring code used those angles to check the learned estimate after training. [Training implementation](../../scripts/experimental/tt28_dose.py) · [Earlier optical calibration](../tt26/visible-impacts.md)

Each frozen checkpoint played the same 12 fresh solo releases with wider launch and posture variation and the same six-second cap. The held wrist controller used wrist RGB and encoders. The receiving comparison froze the 19 existing visual demonstrations and selected a posture from the first four stereo frames of each incoming ball. The only dose-dependent input to the receiver was the solo-estimated blade rotation. Scoring used the physical referee on the 2.74 × 1.525 m table.

Receiving cohorts per mount comprised 48 new perturbations of the 19 training donors, 76 original recordings selected near those donors and 96 original recordings sampled without a reach screen. The two recorded cohorts excluded the posture bank's training shot IDs and the earlier TT25 and TT26 recorded rosters used by the bank. Some IDs overlapped earlier TT27 held assessments. The 172 recordings were all unfamiliar to this receiver's pose library. The unrestricted denominator includes every sampled launch, including shots that fail the incoming referee or never reach the blade. The primary comparison used zero target reward. [Recorded input and experiment precedent](../tt25/recorded-benchmarks.md)

The continuation experiment let each receiver try the same small set of joint offsets, starting from its own frozen solo checkpoint. Each policy received one shot per trial and the legal return outcome of its own chosen action. A policy used its own past rewards and the distance between current and earlier camera observations when choosing an offset. The continuation used 56 independent practice draws followed by 48 separately drawn perturbations with learning frozen. Both sets used jittered launches from the 19 historical training donors. Each dose had a no-update control with its original transported pose. [Continuation implementation](../../scripts/experimental/tt28_dose_online.py)

## Held solo bouncing

| Practice drops per mount | Handle face error | Handle rebounds / median streak | Canted face error | Canted rebounds / median streak |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 13.43° | 10 / 1 | 9.87° | 11 / 1 |
| 4 | 0.35° | 86 / 2 | 0.38° | 130 / 11 |
| 16 | 1.02° | 64 / 1.5 | 0.38° | 72 / 2 |
| 64 | 0.91° | 52 / 1.5 | 0.66° | 74 / 2.5 |
| 192 | 0.99° | 59 / 2 | 0.58° | 116 / 8.5 |

Compared with four drops on the same held releases, the 192-drop handle checkpoint had a longer streak on one release and a shorter one on four. The corresponding counts on the canted mount were four and five. Each dose had 12 held releases per mount. The best streak at four drops was 22 on the handle and 21 on the canted mount. A qualified rebound required physical blade-face contact and at least 45 mm of rise. The six-second cap can end a streak while the ball is still in play. All 120 held wrist command tapes reproduced their referee outcomes. [Handle wrist results](dose-solo-handle.json.gz) · [Canted wrist results](dose-solo-canted.json.gz)

## Frozen regulation receiving

| Mount | Incoming cohort | Launches | 0 drops | 4 drops | 16 drops | 64 drops | 192 drops |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Handle | Perturbed donors | 48 | 8 | 18 | 20 | 20 | 19 |
| Handle | Nearby originals | 76 | 1 | 4 | 5 | 4 | 5 |
| Handle | Unrestricted originals | 96 | 0 | 1 | 1 | 1 | 1 |
| Canted | Perturbed donors | 48 | 21 | 12 | 12 | 12 | 11 |
| Canted | Nearby originals | 76 | 1 | 1 | 0 | 0 | 0 |
| Canted | Unrestricted originals | 96 | 1 | 0 | 0 | 0 | 0 |

The 192-drop checkpoint made five original-recording returns that the zero-drop checkpoint missed and lost two that it made. Against four drops it gained one nearby return on the handle and lost one on the canted mount. The four- and 192-drop checkpoints made the same unrestricted return on recording 6995. The zero-drop policy made its one unrestricted return on a different recording, 13820, under the canted mount. On the perturbed donors, 192 drops gained three and lost three against four. The same decks were played on both mounts. The incoming referee qualified 47 of 48 donor variations, 68 of 76 nearby originals and 91 of 96 unrestricted originals for each mount. All 179 legal outcomes across the two receiving jobs reproduced under camera-driven replay. [Handle receiving outcomes](dose-target-handle.json.gz) · [Canted receiving outcomes](dose-target-canted.json.gz)

## Online receiving after the solo course

Each column reports a distinct segment of the same matched receiving course. The held column gives the online learner's score and its own no-update control.

| Mount | Solo drops | Legal returns in first 8 trials | Legal returns in next 16 | Legal returns in next 32 | Held legal returns out of 48, updated / frozen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Handle | 0 | 3 | 3 | 8 | 16 / 17 |
| Handle | 16 | 3 | 5 | 16 | 22 / 22 |
| Handle | 192 | 3 | 4 | 15 | 23 / 23 |
| Canted | 0 | 4 | 5 | 16 | 18 / 18 |
| Canted | 16 | 2 | 3 | 10 | 22 / 22 |
| Canted | 192 | 2 | 2 | 8 | 22 / 22 |

The 16-drop updater reproduced its frozen receiver's legal outcome on every held shot. The 192-drop updater gained three legal returns and lost three on each mount. Additional regulation reward practice gave no net held benefit to either transferred checkpoint. The held deck consists of donor perturbations. The broad original recordings are assessed separately with frozen policies. The physical replays verified 347 distinct successful branch outcomes. [Handle receiving continuation](dose-online-handle.json.gz) · [Canted receiving continuation](dose-online-canted.json.gz)

## Solo practice cost

| Mount | Releases | Visually observed first impacts | Qualified rebounds in practice | 60 Hz control ticks | Approximate simulated minutes | Training wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Handle | 192 | 192 | 827 | 20,731 | 5.76 | 617 s |
| Canted | 192 | 192 | 1,244 | 27,935 | 7.76 | 685 s |

Practice capped each release at three simulated seconds. The handle course had spent 1,584 ticks after 16 releases, and the canted course 2,389. Both selected their eventual solo tolerance after four drops and retained it at every later checkpoint. The handle tolerance was 100 mm, and the canted tolerance 55 mm. The two wall times overlap because both courses ran concurrently alongside other simulations. The simulator supplied new releases and reset the arm. An independent run replayed all 384 training command streams, reconstructed all 384 visual first impacts from stored pixel centroids and encoders, and reproduced all ten prefix checkpoints. [Handle practice and held replays](dose-solo-handle.json.gz) · [Canted practice and held replays](dose-solo-canted.json.gz) · [Handle checkpoints](dose-checkpoints-handle.npz) · [Canted checkpoints](dose-checkpoints-canted.npz) · [Replay and reconstruction receipt](dose-verification.json)

## Reproduction

Use the pinned DeepMind launch archive with SHA-256 `300f4bf19556dd8eb69663f46de7f642008048b746321cc98754150a4e0e1237`. The run renders in MuJoCo using EGL. [Archive reference](../tt25/recorded-secondary-reference.json)

```bash
export MUJOCO_GL=egl PYTHONPATH=src OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt28_dose all --feeds /path/to/rallies.json
python -m scripts.experimental.tt28_dose_online /path/to/rallies.json
python -m scripts.experimental.tt28_dose_verify
```

## Resuming the source learner

The compact checkpoint holds the tool rotation, tolerance selection and reward totals for deployment. To continue learning, `restore_learner(receipt_path, count)` in the dose runner reconstructs the full ordered first-impact direction history, the tolerance reward estimates and the exploration generator. Pass the solo JSON receipt and the number of drops already played. The next action uses `learner.select(count, rng)` on a fresh release, and `learner.update(action, reward, direction)` records its observed reward and optical direction. The helper restored every saved prefix on both mounts. For prefixes 0, 4, 16 and 64, applying the immediately following archived outcome reproduced the next state and random-generator state exactly. [Restore implementation](../../scripts/experimental/tt28_dose.py)
