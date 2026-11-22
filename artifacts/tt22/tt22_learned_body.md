# Rallying after optical calibration of a new motor installation

The end-clamped SO-101 had a fresh hidden signed permutation of its five arm command channels. The body, gripper, paddle and servo dynamics use the published apparatus. The setup isolates a change to the public actuator convention. The two branches use the same fixed game policy, body model, optical stereo receiver, legal-contact stroke planning, and strict referee. The skill learned in this experiment is the command translation. [Native test and replay](../../scripts/experimental/tt22_learned_body.py)

Five public motor pulses took 35 target control frames at 30 Hz and five resets. These were dedicated probes. The outside and wrist views saw the same movements on one physical tape. Each fit used only its own unmarked RGB, an independent nominal renderer with disjoint ball starts, and a search over every permutation and sign combination. The nominal renderer used the known CAD fixture and took 70 simulated control frames. Both eyes recovered public-to-physical order `[0, 2, 3, 4, 1]` and signs `[−, −, +, −, +]`. The next assignment cost was higher by 0.431 for outside and 0.279 for wrist. Their first 10 motion frames also sufficed retrospectively. Only the full wrist estimate was applied during fixed-stereo play. [Physical probes and per-camera assignments](tt22_learned_body_probes.json.gz)

The comparison reused all 12 predefined wider opponent serves in the prior five-axis paired course. The starts are disjoint from probe and partner-practice starts. The comparison kept the learned far partner's center feed profile, the same opening spread and pace, and the original frozen optical controller. Both branches played live from their own RGB on the same simulated installation and against the same opponent. The learned branch translated the controller's requested rates into public port commands using the inferred wrist map. The unadapted branch submitted the rates directly to the remapped ports. [Pre-existing held cohort](paired-course.json.gz) · [Scored runs](tt22_learned_body_held.json.gz)

## Held results

| Outcome | Unadapted public commands | Wrist-inferred correction |
| --- | ---: | ---: |
| Own legal returns | 0 | 32 |
| Rival legal replies | 0 | 32 |
| Longest alternating rally | 0 | 14 |
| Launches reaching 10 alternating returns | 0/12 | 2/12 |
| Launches reaching 20 alternating returns | 0/12 | 0/12 |
| Points won | 0/12 | 0/12 |
| Frames on which translation changed public commands | 0/505 | 1,515/1,579 |
| Frames on which realized rates differed from controller requests | 468/505 | 0/1,579 |

Corrected lengths in cohort order were `2, 2, 10, 14, 8, 0, 2, 8, 6, 6, 2, 4`. Each of the 24 public command tapes reproduced the full physical point duration, score and referee events. All 12 learned runs reproduced the nominal reference's optical digest, near and far physical commands, score and referee events. The exact six-axis basis verifies the learned command translation, including the untouched gripper. Both branches of held launch `2055314588` also reproduced their original policy RGB. [Scored tapes and replay checks](tt22_learned_body_held.json.gz) · [11.5-second side-by-side replay](tt22_learned_body.mp4) · [Movie verification receipt](tt22_learned_body_movie.json)

The fixed rally controller recovered its earlier behavior after learning a new motor installation. The benchmark covers one wiring change with a matched CAD model and a cooperative far partner. The same calibration method and score have yet to be qualified on a changed linkage or a real arm.

## Reproduce

```sh
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBGL_ALWAYS_SOFTWARE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt22_learned_body
python -m scripts.experimental.tt22_learned_body --phase verify
python -m scripts.experimental.tt22_learned_body_movie
```
