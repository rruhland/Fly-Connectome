# Anonymous visual timing of recurrent physical strokes

The near SO-101 used two fixed native RGB views. The observer found the moving object from the opening frames, learned its appearance from that motion track, and followed it for the whole point. It sent positions and clock ticks to the motor controller. The motor controller forecast when the tracked flight would cross its strike depth, translated two joint keyframes with factory kinematics, and executed one rate-limited pulse. An observed outward crossing made it ready to act on the next inbound flight. Candidate contact and landing simulations were confined to the separate physical partner. [Controller](../../scripts/experimental/tt22_servo.py) · [Trial](../../scripts/experimental/tt22_servo_trial.py)

The pulse inherited the existing SO-101 motor sequence selected from native referee scores on its earlier clock-driven course. The positive lateral rule carried the strongest earlier Pong coefficients for relative displacement and motion. The matched controls set those terms to zero or reversed both signs. The same height correction, motion-to-plane timing, motor shape and strict referee applied in all three branches. The opening used a narrow stochastic release and the steerable far paddle targeted playable central returns. [Carried motor practice](openloop-practice.json) · [Anonymous tracking](motion-object.md)

## Matched live results

Six lead-and-amplitude choices each played the same two development releases. The twelve points produced eleven contacts and zero legal returns. The specified tie-break selected an eight-frame lead and amplitude multiplier 1.06. The pulse settings therefore remained a starting prior for held play. Six different stochastic starts then compared the three lateral rules. [Practice receipt](servo-practice.json.gz)

| Relative rule | First legal returns | Points with a second pulse | Points with a second legal learner return | Alternating legal returns by start | Wins |
| --- | ---: | ---: | ---: | --- | ---: |
| Follow Pong | 3/6 | 3/6 | 1/6 | 2, 2, 4, 0, 0, 0 | 0 |
| Ignore | 1/6 | 1/6 | 0/6 | 0, 2, 0, 0, 0, 0 | 0 |
| Oppose Pong | 1/6 | 0/6 | 0/6 | 0, 0, 0, 0, 0, 1 | 1 |

The follow rule gained two first returns versus ignore and lost zero. On start `1561323131` the pupil triggered at frames 16, 53 and 84. The physically actuated partner contacted the ball at steps 43 and 76 and landed both replies legally. The pupil contacted the ball again at steps 61 and 92, with its second legal return credited at step 72. The four alternating returns on that point were the longest in this experiment. Starts `948700708` and `1181946049` also re-armed after a physical partner strike. Both replies landed legally before the pupil contacted the ball again. All 30 practice and held branches replayed exactly from the actual commands to both arms. [Matched physical receipts](servo-held.json.gz)

## Bounded alternatives

A second variant added 0.135 rad of loft, kept the fractional part of the optical launch clock, reduced the height adjustment and used the duration-trained Pong coefficients. Four pulse settings on two practice launches each all tied at zero legal returns. The selected variant contacted the ball on every one of 18 fresh branches and landed zero legal returns. This combined variation leaves the individual effects of loft, phase and source coefficients unresolved. All 26 tapes replayed exactly. [Practice](servo-loft-practice.json.gz) · [Fresh comparison](servo-loft-held.json.gz) · [Duration source](duration-pong-player.json)

A prepared-block version used the prior independently qualified arm posture for the opening and a separate anonymous visual push after an observed outward crossing. The partner coached from an independent copy of the block. Newly jittered releases produced zero legal learner returns in four paired complete points, so no recurrent push fired. The twelve practice and complete-point branches replayed exactly. In a separate first-block preflight, jitter at 0.024 of the original release ranges yielded zero legal returns in eight starts and tenfold smaller jitter yielded one. [Prepared-block receipts](servo-block.json.gz) · [Release-width preflight](servo-block-width-preflight.json.gz)

```bash
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt22_servo_trial
python -m scripts.experimental.tt22_servo_trial --loft
python -m scripts.experimental.tt22_servo_block
python -m scripts.experimental.tt22_servo_block --preflight
```
