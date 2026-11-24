# A direct paddle learner calibrated by signed impact

The near arm takes one joint-setpoint update per tick from measured stereo ball motion and encoders. It does not compute a receiving point or query a simulator. A new residual tests whether the consequences of a physically executed stroke can teach a better small joint change through a differentiable forward model. The baseline deep-field readout came from earlier visual demonstration frames. [Direct implementation](../../scripts/experimental/tt37_adaptive.py) · [Deep visual field](joint-deep.md)

## Consequences and control

Native candidate families supply paired changes in pose and swing speed and the resulting ball velocity, including signed weak deflections. We mapped their keyed present-time raw states into the RGB observation domain with the prior camera calibration. From 137 native and 248 refined contexts we fitted regularized local matrices for the effect of ten bounded motor changes on the three components of outgoing ball velocity. Another 349 successful native outcomes supply target velocity examples. This prior uses simulator data **in training only**. Its held-action error check split unique actual actions within each native context. Correcting a nearby observed output with the learned Jacobian reduced normalized mean velocity-difference error from 1.209 to 1.117 on 1,981 held pairs. Its changes pointed in the measured direction on 1,448 pairs. That tests a local sensitivity, not prediction on an unseen release. [Model and sources](adaptive-model.json) · [Held-action check](adaptive-model-validation.json)

At play time the adaptor indexes its experience at an observed inbound depth gate. For a nearby previously measured hit, it chooses one successful target velocity and solves a damped inverse for a small continuous motor change. It does not average candidate joint motions. After at least three nearby actual signed observations, it can locally correct the native slope by regression. Visible misses can fit a signed alignment adjustment using actual RGB blade positions and measured encoders. An unsupported or invisible outcome produces no fabricated correction. Earned legal outcomes supply retention evidence. Each issued setpoint is bounded, and every actual outcome enters the session archive only once the physical point ends. No referee event other than a legal-return signal enters that update. [Practice commands and optical feedback](adaptive-practice-learn.json.gz) · [Acquired history](adaptive-memory.json)

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
