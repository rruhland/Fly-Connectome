# A blind preparation can expose a shallow left stroke, but the full motion loses ordinary returns

The full learned outbound preparation lost all six fresh physical replies. Reset stock returned all six. The marked arm motion is genuinely blind. Its first 56 controls after the shared checkpoint are identical across all six future streams. A smaller elbow-and-wrist compromise preserved the three fresh training returns, while the original difficult reply still beat it. The full motion is not a general default.

## Same outgoing ball, different future draws

The checkpoint is just before the far opponent draws its next aim on the outgoing ball after the 42nd arm stroke. The remaining previous-stroke motor commands are shared. Each alternate continuation receives one preassigned independent random TT37 stream while the profile, physical reactive far carriage, table, clamp and arm are unchanged. For each seed the two branches retain the same actual far impact and near bounce. These are matched continuations of one outgoing ball, not new completed games from launch.

The fixed teaching references come from a clean privileged demonstration on the old terminal, but its contact target, ball trajectory and scoring motion are not inputs to the blind actor. A current outgoing RGB estimate starts one fixed-duration preparation, using the arm's own references. The first incoming estimate cancels it and hands the physically attained body to the unchanged strong stereo/joint controller. The taught endpoint shifts the elbow from approximately −1.69 to +0.48 rad at the handoff. [Frozen streams](protocol.json) · [Six physical branches per side](native.json.gz)

## Training-only after-cue check

| Training reply | Native stock actor | Native prepared actor | Privileged stock after-cue | Privileged prepared after-cue |
|---|---:|---:|---:|---:|
| 1062985728, right | Returned | Missed | Returned and answered | 0/128 modeled contacts |
| 439920736, right | Returned | Missed | Returned and answered | 8/128 modeled contacts, no legal |
| 191026792, left | Returned | Missed | No legal proposal in this screen | Returned and answered |

The last case shows a real alternative reached body state. The same blind material footprint exposed a negative-face contact at x = −0.156 m, y = −1.242 m and z = 1.000 m on a different physical reply. The existing camera controller did not use it. Its privileged after-cue action did land, independently replayed, and had zero full CAD overlaps in 1,368 changed-tail solver states. Neither of the two rightward cases admitted a legal exact-state proposal from the fully prepared body. This is a measured reach tradeoff and an after-cue proposal gap, not solely imperfect camera localization. [Three training-only native interventions](coach.json.gz)

## Smaller changes

The shortened mode only holds the first 24 taught reference targets. The partial mode retains ordinary pan and lift and borrows smaller elbow and wrist changes. Both start on the same optical outbound trigger, and neither sees the next aim. The full-size result above motivated both changes, but outcomes from held draws were not used to choose the profile. The original hard 43rd offer is an explicit training-source control and is not treated as held.

On the three fresh training offers, stock and partial each returned three, and short24 returned two. None returned the original source low under normal camera control. The training-only chooser froze partial because it incurred no ordinary loss. [Compromise protocol](compromise-protocol.json) · [Paired training](compromise-training.json.gz) · [Frozen selection](compromise-selection.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
out=artifacts/tt49_terminal_blind
python -m scripts.experimental.tt49_terminal_blind plan
python -m scripts.experimental.tt49_terminal_blind run
python -m scripts.experimental.tt49_terminal_blind_coach --native "$out/training-only.json.gz" --batch 64
python -m scripts.experimental.tt49_terminal_blind run --plan "$out/compromise-protocol.json" --split training --reuse-stock "$out/training-only.json.gz" --output "$out/compromise-training.json.gz"
python -m scripts.experimental.tt49_terminal_blind_select --held-output "$out/compromise-held.json.gz"
python -m scripts.experimental.tt49_terminal_handoff
```
