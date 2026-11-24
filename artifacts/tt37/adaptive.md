# A direct paddle learner calibrated by signed impact

The learned residual turned two practiced first deliveries into legal returns on both new continuation seeds for each. Neither fixed nor sign-reversed control returned any of the 12 anchored balls. On 12 unrelated measured openings, the direct pupil lost one legal return after adaptation. The method improved two local shots but did not extend rallies or generalize here. [Matched results](adaptive-results.json)

The near arm takes one joint-setpoint update per tick from measured stereo ball motion and encoders. It does not compute a receiving point or query a simulator. A new residual tests whether the consequences of a physically executed stroke can teach a better small joint change through a differentiable forward model. The baseline deep-field readout came from earlier visual demonstration frames. [Direct implementation](../../scripts/experimental/tt37_adaptive.py) · [Deep visual field](joint-deep.md)

## Consequences and control

Native candidate families supply paired changes in pose and swing speed and the resulting ball velocity, including signed weak deflections. We mapped their keyed present-time raw states into the RGB observation domain with the prior camera calibration. From 137 native and 248 refined contexts we fitted regularized local matrices for the effect of ten bounded motor changes on the three components of outgoing ball velocity. Another 349 successful native outcomes supply target velocity examples. This prior uses simulator data **in training only**. Its held-action error check split unique actual actions within each native context. Correcting a nearby observed output with the learned Jacobian reduced normalized mean velocity-difference error from 1.209 to 1.117 on 1,981 held pairs. Its changes pointed in the measured direction on 1,448 pairs. That tests a local sensitivity, not prediction on an unseen release. [Model and sources](adaptive-model.json) · [Held-action check](adaptive-model-validation.json)

At play time the adaptor indexes its experience at an observed inbound depth gate. For a nearby previously measured hit, it chooses one successful target velocity and solves a damped inverse for a small continuous motor change. It does not average candidate joint motions. After at least three nearby actual signed observations, it can locally correct the native slope by regression. Visible misses can fit a signed alignment adjustment using actual RGB blade positions and measured encoders. An unsupported or invisible outcome produces no fabricated correction. Earned legal outcomes supply retention evidence. Each issued setpoint is bounded, and every actual outcome enters the session archive only once the physical point ends. No referee event other than a legal-return signal enters that update. [Practice commands and optical feedback](adaptive-practice-learn.json.gz) · [Acquired history](adaptive-memory.json)

| Course | Frozen direct pupil | Learned update | Reversed update |
| --- | ---: | ---: | ---: |
| 18 physical practice points | 0 | 3 | — |
| 12 changed continuations after familiar first deliveries | 0 | 4 | 0 |
| 12 archive-disjoint measured openings | 6 | 5 | — |

In practice the learned branch made three legal returns, one on each exposure to release 15270. The first was proposed from signed outcomes, before any new legal reward. On the same first delivery after training, the new optical velocity was approximately unchanged forward at 2.27 m/s but changed from −2.83 to +3.70 m/s vertically. The approach and final score matched on both fresh continuation seeds. [Frozen practice](adaptive-practice-frozen.json.gz) · [Live practice](adaptive-practice-learn.json.gz) · [Held carry](adaptive-replies-carry.json.gz)

The 18-point acquisition used 41.1 process CPU seconds, with 17 signed impact measurements and one additional miss yielding a body-fit hint. The frozen control reproduced all commands and events from the independent deep joint study on the twelve unfamiliar releases. [Acquisition and frozen-control receipts](adaptive-results.json) · [Independent direct benchmark](joint-deep-held_openings-local.json.gz)

Release 4735 scored zero on all three practice attempts. Its model-derived setting landed the first return on both held seeds. The outgoing forward component measured from camera motion rose from 1.87 to 3.19 m/s, and the vertical component changed from −1.36 to +3.02 m/s. The actual first bounce after the near stroke moved from y = −1.32 m to +0.35 m. Both later balls physically arrived, but neither produced another legal near return. The held unfamiliar course lost the one return on 8599 and kept the baseline's single later return. [Held comparison](adaptive-results.json) · [Sign-reversed commands](adaptive-replies-reverse.json.gz)

A separate first-block challenge did not confirm that the learned actuator directions explore better. Shifting the axes of the model-derived step while retaining its normalized size and the exploration seeds scored two returns on 6208 over the first six points. Aligned scored one on 15270. Thus this small test does not isolate the benefit of the learned Jacobian from other exploratory directions. [Axis-scrambled control](adaptive-practice-scrambled.json.gz)

Command-only replay reproduced the referee outcomes on all five held scoring differences. Full-CAD verification sampled 7,394 solver states, with zero penetrations and 15.7 mm minimum clearance. [Independent replay proof](adaptive-audit.json)

## Reproduce

Use the existing native outcome files and the pinned measured feeds. The twelve unfamiliar assessment cards are the first twelve entries of the untouched TT37 held opening roster.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt37_adaptive fit
python -m scripts.experimental.tt37_adaptive plan
python -m scripts.experimental.tt37_adaptive validate
for mode in frozen learn; do
  python -m scripts.experimental.tt37_adaptive practice "$mode"
done
python -m scripts.experimental.tt37_adaptive practice scrambled --limit 6
for mode in frozen carry reverse; do
  python -m scripts.experimental.tt37_adaptive replies "$mode"
done
for mode in frozen carry; do
  python -m scripts.experimental.tt37_adaptive new "$mode" --limit 12
done
python -m scripts.experimental.tt37_adaptive summary
python -m scripts.experimental.tt37_adaptive audit
python -m pytest -q scripts/experimental/tt37_adaptive_test.py
```
