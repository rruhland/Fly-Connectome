# Carrying an inferred motor routing into receiving

On a full-size receiving course with unfamiliar joint signs, a one-tick actuator delay and laterally displaced feeds, a carried routing recovered five legal returns out of 12 after two active command and encoder observations. The same-budget reset and frozen controls returned zero. The carried estimator identified the full interface at that budget and tied the known-interface reference. [Paired returns and replays](motor-interface-confirmation.json.gz)

| Controller on the first combined change | Legal returns from 12 feeds |
| --- | ---: |
| Frozen last inferred source interface | 0 |
| Factory command routing | 0 |
| Two observations with the same motor model and uniform routing prior | 0 |
| Two observations with carried routing preference | 5 |
| Known changed interface | 5 |

Across all four initial settings, the carried model landed five of 48 returns at both two and four observations. The gains were confined to the first setting. Reset and frozen scored zero across 48, and the known-interface reference scored 13. The first setting was decoded exactly at two, three and four observations, and the model changed to an incorrect estimate at six. Both carried and reset decoded the fourth setting at eight observations. [Per-setting estimates and outcomes](motor-interface-confirmation.json.gz)

A separate 48-feed course used a fresh seed and added three ordinary command prefixes after each held block to update the routing memory for the next setting. At the two-observation and four-observation budgets, carrying the source routing again decoded the first change exactly and landed its only achievable return. Same-budget reset and frozen landed zero. Across all four settings the known-interface reference landed 11, and each eight-observation adaptive branch landed zero. Retained routing became incorrect after the first new block, and no later block produced a return. The result supports fast reuse when the inferred routing carries over. The chronological update did not keep recovering as the settings continued to change. [Fresh chronological course](motor-interface-chronology.json.gz)

The physical command adapter permutes the five driven axes, flips their signs, and optionally delays the whole request by one 30 Hz control tick before the simulator applies it. The learner knows the family of 7,680 signed permutations and delays, the arm's native command units and its joint limits. The particular assignments are hidden. The first two held settings share the source permutation with new signs, delays and trajectories. Two routes swap in the third setting and remain swapped in the fourth. Held feeds shift laterally by 24–32 mm and their forward speed changes by 2–4%. [Implementation and settings](../../scripts/experimental/tt27_motor_interface.py)

The player selects a normal receiving posture from the first four RGB frames. Its adaptation trace uses regular posture commands and encoder angles and velocities, with zero inserted probe movements or reward updates. It scores each candidate by the motor model's encoder prediction error. The carried branch uses a soft preference for the previously inferred permutation. Both carried and reset estimate all signs and the delay from the same current observations. Each held policy plays the posture through the actual remapped interface. Replaying the issued command stream through that interface reproduces its referee results. [Encoder and command receipts](motor-interface-confirmation.json.gz)

For the first held change, 120 candidates in both courses would have executed exactly the same commands after two active requests. The older order distinguished among those candidates. At eight observations, two to six candidates still executed identical commands across individual held adaptation prefixes. The encoder history alone cannot distinguish interfaces with identical executed requests. [Executed-prefix ambiguity audit](motor-interface-chronology.json.gz)

The initial 60 Hz bounce-effect model alone returned one ball across 48 at the two-observation carry budget. No held estimate was exact within eight observations. The next fit retained the bounce predictor and added a ridge correction from 176 encoder transitions in the prefixes of 16 earlier ordinary match tapes at 30 Hz. Both the reset and carried controls received this fit. The two source remapping settings contributed four ordinary eight-command prefixes each. Combining their encoder likelihoods recovered the common permutation without wiring labels. [Bounce-only pilot](motor-interface-bounce-only.json.gz) · [Calibrated pilot](motor-interface-confirmation.json.gz) · [Earlier bounce model](../tt26/bounce-effects.npz)

```bash
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_motor_interface --bounce-only --output artifacts/tt27/motor-interface-bounce-only.json.gz
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_motor_interface --output artifacts/tt27/motor-interface-confirmation.json.gz
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_motor_interface --continuation --output artifacts/tt27/motor-interface-chronology.json.gz
```
