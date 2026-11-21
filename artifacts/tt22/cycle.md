# TT22 physical recurrent rally

The near robot is the clamped SO-101 from TT20. The far robot is the stock four-axis physical rival with a different receiving geometry. The fixture’s built-in seeded rival serve starts each point once. Both controllers issue only normalized rate commands after the reset. Paddle, table and net contacts, aerodynamics, strict service, the scorer and the mechanics that end a point are the stock fixture. The 1,400-frame cap leaves room for long exchanges. [Controller](../../scripts/experimental/tt22_cycle.py)

The two privileged planners rank their shots for a common receiving window about 91 cm from the net, 20.5 cm above the table, 2.6 m/s longitudinally and near the centerline. Each side uses its own physical swing geometry. A separate exact-physics branch confirms a legal landing and a usable next receive window before committing. These trial branches leave the live ball and robot state untouched. The point itself runs forward through every contact and bounce until the physical referee ends it. The planners use simulator ball and mechanism state, so these are feasibility witnesses for cooperative physical rallies.

| Held seed | SO-101 legal returns | Rival legal returns | Alternating total | Control frames | Finish |
|---|---:|---:|---:|---:|---|
| 1909157674 | 12 | 12 | 24 | 447 | floor |
| 728025937 | 5 | 5 | 10 | 228 | floor |
| 530614602 | 16 | 16 | 32 | 604 | floor |
| 1413200957 | 11 | 11 | 22 | 429 | floor |

All four prespecified held points reached ten alternating returns. The longest took 604 control frames at 30 Hz. It began from the fixture’s native seeded rival serve and ended when the final ball hit the floor. All four complete pairs of motor tapes reproduced every strict referee event. The development runs reached 14 and 6 returns after the receiving envelope was tightened. [Held physical tapes](cycle-held.json.gz) · [Development tapes](cycle-envelope-development.json.gz)

## Transfer to fixed stereo

The near RGB receiver retargets swing parameters selected from the physically successful development exchanges. A separate nominal command-driven arm reconstructs its body position, and a natural fixed stereo tracker supplies its metric ball estimate. The receiving gate, velocity estimate and arm movement rely on RGB, its clock and its own previous commands. The far player remains the separate privileged physical cooperative planner. A second condition carries calibration learned from the near player’s own later RGB observations during two new practice points. The residual estimator compares short forecasts with subsequent optical samples. Point reward and strict referee events are stored for evaluation. A frozen imitation control shares the styles, while a fully untransferred control keeps the complete candidate search. [Stereo controller](../../scripts/experimental/tt22_cycle_optical.py) · [Assay](../../scripts/experimental/tt22_cycle_optical_trial.py)

| Near stereo policy | Fresh seed 1134144980 | Fresh seed 602925087 | Total |
|---|---:|---:|---:|
| No teacher, full candidate search | 1 | 4 | 5 |
| Frozen teacher styles | 4 | 2 | 6 |
| Teacher styles with own-RGB calibration | 6 | 0 | 6 |

Nine donor swing combinations came from the actual legally credited development returns. When the restricted first attempt cannot produce a legal model option, the student can re-open the full search. The two self-supervised practice points produced two and six alternating returns and furnished 46 retained RGB forecast samples in 201 control frames. Calibration changed the held result in both directions. The practice budget and matched held data establish a six-return stereo witness with no aggregate gain from that calibration on this two-start pilot. Every stereo action tape reproduced the native camera hashes and complete strict referee events. [Stereo tapes](cycle-optical-held.json.gz) · [Paired 4-to-6 return video](cycle-optical-paired.mp4)

At the final continuation decision in that six-return point, the RGB-based planner predicted a legal return, physically made paddle contact and faulted. A separate evaluator reconstructed the exact same live frame and substituted measured state components into the planning model. Replacing only the ball state produced a physically legal return. Replacing only mechanical state reproduced the fault. The estimated ball center was within 2.1 mm, vertical speed error was 0.096 m/s, and the nominal command-driven blade center was within 0.001 mm of the simulated plant. The dry spin estimate was [74.2, 27.0, −27.0] rad/s versus actual [0.6, 3.2, 3.8]. The diagnostic candidate was tested in a fresh simulator branch with the original prefix. Replacing ball position alone still faulted. Providing exact translational velocity or exact spin separately each led to a physically legal reply at that same decision. The size of the spin error is diagnostic for this rally; it does not identify a unique required correction. [Model-level substitutions](cycle-audit.json.gz) · [Position, velocity and spin substitutions](cycle-spin-audit.json.gz)

## Cross-team bounce filter

The separate bounce-phase estimator was fitted from seven bounces across four earlier optical practice points. This comparison applies it to the same fixed stereo near policy and adds an extra search when the fitted bounce lies in the previous two frames. A model that sees the whole true ball state remains confined to the dated evaluator, and both live branches retain the same privileged far opponent. [Source optical filter](bounce-filter-pilot.json.gz) · [Integrated trial](../../scripts/experimental/tt22_cycle_bounce.py)

| Point | Standard optical observer | Bounce filter and post-bounce replan |
|---|---:|---:|
| 1134144980, already observed diagnostic | 6 | 4 |
| 1551328765, fresh | 4 | 2 |
| 868361554, fresh | 8 | 8 |

The standard and filtered policies both reached eight on the same fresh point. The filter lost two returns on the other fresh point. A more accurate local velocity estimate around the bounce did not translate into a longer rally in this small controller test. A state substitution at the separate six-return point shows that this remaining failure can depend on pre-bounce velocity or spin. [Full-path filter tapes and native RGB verification](cycle-bounce.json.gz)

## Smooth pre-bounce observer on the same failure

A smooth arc fit used only the available stereo positions through frame 125 with a fixed zero-spin prior inside the fit. The command-driven arm state, learned swing library, bounce phase and original planner spin stayed fixed. The measured vertical-speed error fell from 0.096 to 0.043 m/s at this decision. The original candidate touched the ball at frame 134 and faulted. With fitted position and velocity, a new candidate touched at frame 134 and landed a legal return at frame 144. With only fitted velocity and the original position, the return landed at frame 150. These are separate strict physical branches of the same reconstructed prefix, ending at the first new legal landing. [Causal RGB arc and enacted return receipt](cycle-prebounce-arc.json.gz)

## Uninterrupted live test of pre-bounce velocity

The final paired cohort changed only the incoming pre-bounce velocity estimate. The original stereo position, planner spin, body state observer, bounce and refinement rules, teacher swing library and privileged far policy stayed fixed. Both branches carried the same earlier RGB residual state. Four new seeds were generated before testing. Each point ran uninterrupted from an ordinary opponent serve until the native referee ended it.

| New seed | Standard observer | Smooth-arc velocity |
|---|---:|---:|
| 2119291602 | 4 | 4 |
| 265865819 | 1 | 0 |
| 509321423 | 2 | 3 |
| 1213749908 | 2 | 6 |

The smooth velocity version produced 13 alternating returns versus nine, comprising seven versus five SO-101 returns and six versus four rival returns. It lengthened two points, shortened one and tied one. Each branch won one point. The longest point had six returns with the smooth estimate, and neither branch reached ten in these four tests. All eight action tapes reproduced their RGB hashes and referee events. [Fresh matched tapes](cycle-arc-live.json.gz)

## Reproduction

```bash
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt22_cycle_trial --held
python -m scripts.experimental.tt22_cycle_optical_trial
python -m scripts.experimental.tt22_cycle_audit
python -m scripts.experimental.tt22_cycle_audit --components --output artifacts/tt22/cycle-spin-audit.json.gz
python -m scripts.experimental.tt22_cycle_bounce
python -m scripts.experimental.tt22_cycle_audit --prebounce-arc --output artifacts/tt22/cycle-prebounce-arc.json.gz
python -m scripts.experimental.tt22_cycle_arc
```
