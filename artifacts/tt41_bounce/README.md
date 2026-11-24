# Does the effect learned in solo bouncing help on low table returns?

The transferable part is **the velocity of the moving blade face**, not a solo joint tape or a vertical release in world coordinates. A fit from 37 visible contacts in 39 already played solo drops learned that one extra metre per second of inward normal speed adds about 1.88 m/s to the ball's velocity change along the struck face. The same law predicted optically measured departures from actual table contacts. An independently specified moving-face formula was about as good on the full 3-D velocity, so this finding does not establish that the fitted numbers are necessary. [Solo fit and split](effect.json) · [Blind table contact check](table-audit.json)

## What carries across the tasks

The solo archive includes the actual wrist-controlled attempts before and after earlier regulation receiving. Its 3,238 charged 60 Hz ticks represent 53.97 seconds and 97 qualified rebounds. During those drops the learner had wrist RGB, joint encoders, its issued normalized rates, and rebound reward. The archived fixed stereo exposures provide retrospective incident and departure labels without reading simulator velocities or contact time. The optical blade fit and encoder trajectory locate the struck part of the disk and its signed motion. Only the first optically resolved, face-supported hit per drop contributes to this model. A robust two-coefficient least-squares fit estimates an impulse along the face normal and in its tangent plane. It pools 23 original and 14 continuation hits only after the chronological evaluation. No table labels enter the fit. [Raw practice](../tt29/tt29_bounce_course.json.gz) · [Post-table continuation](../tt29/tt29_bounce_continued.json.gz) · [Fit and candidate ranges](effect.json)

| Real optical assessment | First contacts | Fitted moving face | Independently specified moving face | Stationary face |
| --- | ---: | ---: | ---: | ---: |
| Chronologically later solo drops, median velocity error | 14 | 0.295 m/s | 0.298 m/s | 0.704 m/s |
| Earlier recorded regulation contacts, median velocity error | 17 | 0.204 m/s | 0.198 m/s | 1.793 m/s |
| Earlier recorded regulation contacts, upward component | 17 | 0.071 m/s | 0.100 m/s | 0.752 m/s |

The first row compares the earlier 23-hit fit with a stationary face separately fitted to those same contacts. The next rows compare the pooled solo fit with the nominal 1.85 normal gain applied with or without moving the blade. The table archive contains correlated action contrasts on five releases, not 17 independent deliveries. For example, on identical physical release 5864, a later-issued correction raised optically measured vertical departure by 1.33 m/s with just 0.07 m/s change in forward speed. The frozen solo law predicted +1.25 and +0.07 m/s from the new encoder and optical observations. The stationary-face prediction was +0.70 and −0.58 m/s. That is evidence of a transferable action *effect*, not a prospective claim to have selected that earlier correction. [Paired optical table records](table-audit.json)

The two fixtures use the same shoulder-through-wrist axes with a nominal 5.2 rad/s reference slew, but the table base is 570 mm farther back along the world y axis. Solo permits the full normalized joint command and fixes the gripper. Regulation clips the command at 0.8, keeps the speed–torque envelope, and has a sixth gripper channel. Solo varies the material and servo settings. Table closing speeds on the optically auditable hits were 3.63–4.47 m/s, above the solo sample's 1.75–2.97 m/s. Rotating the calculation with the signed disk face and evaluating current table motion avoids copying the old frame or actuator effort into the new task. [Fixture and closing-speed audit](table-audit.json)

## The receiving test

The low fallback checks the stock stroke only when it finds no nominal legal shot and the camera estimates a low ball with a near receive time. The frozen solo policy estimates the effect of blade motion and proposes bounded, joint-local changes in approach, face angle, and face speed. The stock geometry and independent, sensor-initialized physics model still check every candidate. The equal-budget stationary-face control uses the same seed, initial candidate batch, runtime inputs and number of candidate simulations but removes the moving-face contribution from the directed additions. It still evaluates general joint-rate changes. Each fall-back attempt can test 512 independent nominal motions. The online branch starts with the solo prior and adjusts the directed search's gain on receipt of actual legal-return reward or a fault. No release ID, court truth, future image, opponent target, or prior table outcome labels enter the added action selector. [Physical fallback and matching runner](../../scripts/experimental/tt41_bounce_return.py)

On the first preassigned assessment release, 9550, all five variants issued identical commands up to tick 170 after the same opponent strike and near bounce. The recipient saw a predicted receive height of 0.997 m and acted about 100 ms before contact. The stock ball left at 2.56 m/s forward and 3.16 m/s upward, then faulted. The solo-fit policy put 2.87 m/s forward and 3.18 m/s upward into the same incoming shot. It landed legally and received a physical answer. More forward drive, not a large increase in lift, made this motion useful. The fixed evaluation height projection to −1.53 m places the ball at 1.003 m, immediately above the strict <0.99 m failure band. This is an adjacent-low rescue, not a demonstration that the deepest low balls are solved. [Full matched commands](held-first.json.gz) · [Nominal moving-face control](held-nominal.json.gz)

| 9550, one uninterrupted game per condition | Arm returns | Alternating legal returns | Simulated seconds | CPU seconds |
| --- | ---: | ---: | ---: | ---: |
| Stock fixed | 1 | 3 | 3.64 | 5.68 |
| Equal-budget search, stationary face | 4 | 9 | 7.94 | 15.86 |
| Equal-budget search, nominal moving face | 3 | 7 | 6.90 | 13.29 |
| Equal-budget search, solo-fit frozen | 6 | 13 | 10.94 | 19.97 |
| Equal-budget search, solo-fit online | 6 | 13 | 10.94 | 19.48 |

All four extra searches returned the first shared low delivery legally. The two equal-budget controls selected different motions. Versus stock, the first repaired stroke is a matched causal comparison. The counts after that are the resulting uninterrupted games, with the opponent reacting to different incoming balls, so they do not establish a five-shot direct treatment effect or stable generalization. The online branch earned one observed reward update, but did not beat frozen on this game. The computation includes rendering, the opponent, the stock controller and the added search. The stock, stationary-face and frozen solo tapes independently reproduced every event. The solo replay had zero penetrations in 5,249 CAD states with 15.41 mm minimum measured clearance. [Physical and timing receipts](held-first.json.gz) · [Independent replay and clearance](held-audit.json)

On the five deeper development failures from the feasibility report no new physical return scored. Both extra searches tested 512 nominal motions twice per case. On 8519 the first window supplied one independently chosen stroke per branch, but both physically touched and missed. The four other cases did not produce a nominally legal movement within the search cap, and no alternative was issued. Earlier privileged feasibility work succeeded on two of these cases. [Matched development attempts](witness.json.gz) · [Prior physical feasibility](../tt40/low-feasibility.md)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
out=artifacts/tt41_bounce
python -m scripts.experimental.tt41_bounce_effect "$out/effect.json"
python -m scripts.experimental.tt41_bounce_table_audit "$out/table-audit.json"
python -m scripts.experimental.tt41_bounce_return witness "$out/witness.json.gz" --batch 128 --rounds 4
python -m scripts.experimental.tt41_bounce_return held "$out/held-first.json.gz" --batch 128 --rounds 4 --limit 1
python -m scripts.experimental.tt41_bounce_return held "$out/held-nominal.json.gz" --batch 128 --rounds 4 --limit 1 --modes elastic
python -m scripts.experimental.tt41_bounce_verify --tape "$out/held-first.json.gz" --out "$out/held-audit.json" --cad
python -m pytest -q tests/test_tt41_bounce_effect.py
```
