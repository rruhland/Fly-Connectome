# Solo contact effects on the carried direct-joint actor

The direct-joint transfer did not reproduce the successful geometric-controller rescue on release 9550. The stock direct policy, stationary-face control, textbook moving-face control, and solo-fit moving-face policy each landed one arm return. The added proposals changed the opening stroke, so the second ball was no longer common across arms. The solo variant missed its second contact whereas the other three touched. All four issued-command tapes reproduced exactly and had zero penetrations across 6,240 CAD samples. [Full physical games](live.json.gz) · [Independent command replay](live-audit.json)

The test starts with the same frozen base field and earned optical motor priors as the carried elastic joint controller. It does not add a pose chart, stored solo actions, new table practice or a learned rate-space inverse. At a search with an observed rebound and a camera-projected height between .972 and 1.061 m, it can add a local candidate bank around promising raw and credited direct reference sequences. A pure forward-kinematics derivative and a signed face-relative impulse model propose corrections to where the ball meets the disk, its face angle, and its normal speed. The target asks for sufficient forward *and* upward departure to land beyond the net. The selector leaves a confidently viable incumbent alone. Issued candidates must pass the existing independent model and fresh optical-uncertainty checks. [Learned bounce law](../tt41_bounce/effect.json) · [Direct candidate code](../../scripts/experimental/tt41_bounce_joint.py)

The stationary control and both moving controls have the same width of 64 extra candidates per consultation. They differ only in how the local correction treats blade motion and the inherited two impact coefficients. The textbook law uses a normal coefficient of 1.85 and zero tangential transfer. The solo law uses 1.885 and 0.011. The baseline gets no extra proposals. Across full games the number of consultations differs because played movements change the next optical state. CPU costs include rendered observations, the physical opponent and all planning.

| Single full game at 9550 | Arm returns | CPU seconds |
| --- | ---: | ---: |
| Stock carried actor | 1 | 4.77 |
| Stationary-face extra search | 1 | 5.89 |
| Nominal moving-face extra search | 1 | 6.07 |
| Solo-fit moving-face extra search | 1 | 5.59 |

## Why the clean-face law did not buy a return

The terminal analysis reconstructs all nine moderate-height touches in the audited carried course from fixed stereo and encoders before impact. Native contact time aligns this retrospective diagnostic only. The subsequent physical separation is the validation label. The inferred ball-center offset in the blade plane was 76.5–93.1 mm in every case. The calibration only accepted solo examples up to 71 mm from the disk center, and the nominal blade radius is 76 mm. The optical trajectory fit had 1.15 mm median residual. These are estimates of where the ball center meets the face, not native contact-radius labels. [Precontact dispersion](terminal-effects.json)

| Departure error across those nine touches | Stationary face | Textbook moving face | Solo-fit moving face |
| --- | ---: | ---: | ---: |
| Median absolute forward error, m/s | 1.71 | 0.61 | 0.47 |
| Median absolute vertical error, m/s | 2.97 | 3.72 | 3.77 |
| Median forward/upward error norm, m/s | 3.02 | 3.94 | 4.03 |

The learned law predicted forward pace much better but made the combined estimate worse by often predicting too much lift. The last ball of the 35-return rally physically departed with 3.93 m/s forward and 1.68 m/s upward. The frozen moving-face law predicted 3.45 forward and 2.93 upward. That could mislead a controller into treating the required lift as already secured. The numbers support using the model after achieving an interior contact, not trusting it to steer an extrapolated rim touch. [35-return physical replay](../tt39/carried-35.json) · [Terminal loss audit](terminal-effects.json)

At 9812 the four operators replayed 647 identical command ticks through six previous arm returns and the same opponent strike and bounce. The stationary and textbook corrections changed the outgoing forward/upward pair, from 4.10 / 1.24 m/s at stock to 3.90 / 1.78 and 4.00 / −0.60 m/s, respectively. Neither scored. The solo branch tested extra proposals but declined all of them and reproduced the stock miss. This is a matched failure, not an additional scored action. [Matched commands](fork-9812.json.gz) · [Prefix audit](fork-9812-audit.json)

On the preserved 14390 game, all four arms issued the same 3,457 archived prefix commands, including 35 legal arm returns. Reconstructed camera/encoder history and earned motor memory then handed control back at the first logged search after the final near bounce. Every arm received the same physical far strike, bounce and receive-line crossing. The rearmed stock missed with 4.12 m/s forward and 1.25 m/s up. The stationary correction lowered lift to 0.66 m/s with 4.11 m/s forward and also failed. Textbook and solo declined the extra proposals and exactly reproduced the rearmed stock commands and events. Post-fork computation was 1.89, 2.55, 3.20 and 3.28 CPU seconds respectively, excluding the identical prefix. No 36th return or physical answer was obtained. [Four matched continuations](fork-14390.json.gz) · [Independent comparison of their prefixes and scores](fork-14390-audit.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
out=artifacts/tt41_bounce_joint
python -m scripts.experimental.tt41_bounce_joint live "$out/live.json.gz" --plan artifacts/tt39/endurance-plan.json --deck assessment --limit 1
python -m scripts.experimental.tt41_bounce_joint_audit verify "$out/live-audit.json" --cad
python -m scripts.experimental.tt41_bounce_joint_audit spectrum "$out/terminal-effects.json"
python -m scripts.experimental.tt41_bounce_joint fork "$out/fork-9812.json.gz" --ids 9812
python -m scripts.experimental.tt41_bounce_joint_audit verify "$out/fork-9812-audit.json" --tape "$out/fork-9812.json.gz"
python -m scripts.experimental.tt41_bounce_joint fork "$out/fork-14390.json.gz" --ids 14390 --cache-prefix
python -m scripts.experimental.tt41_bounce_joint_audit verify "$out/fork-14390-audit.json" --tape "$out/fork-14390.json.gz"
python -m pytest -q tests/test_tt41_bounce_joint.py tests/test_tt41_bounce_effect.py
```
