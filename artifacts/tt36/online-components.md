# Did each feedback signal change the returns?

Legal-only and optical-only learning reproduced **every point score** of the combined learner across both held courses. Neither channel added an identifiable legal return beyond the other on these 48 games. [Paired outcome receipt](online-component-results.json)

The component tests reused the same 32 online practice points, executed commands, and action repertoire. They made no new probes and did not change any held controller after its checkpoints were loaded. The only training change was how each logged practice outcome was valued. This measures what each signal adds given the same experience. It does not test how the two ablated controllers would explore or gather experience on their own. [Online experiment](online.md) · [Original experience](online-practice-acquire.json.gz) · [Ablation implementation](../../scripts/experimental/tt36_online_components.py)

- **Legal only** gives one for a referee-awarded return and zero otherwise. It uses exactly the same 48 stored onset observations and selected corrections as the combined learner. It removes the graded distinction between nonlegal strokes.
- **Optical only** never reads the legal-return pulse. It rewards a ball observed moving away after the approach, with larger credit for higher observed speed, or gives smaller credit based on the measured ball-to-blade gap. The away motion is evidence of contact or reversal, not a visual determination that the return would land legally.
- **Combined** gives one for a legal return and uses the graded optical outcome otherwise. **Reset** uses the same direct policy with no learned memory. The **global prior** uses the inherited TT35 reward-trained correction on every approach. All learned variants have access to it as one of four candidate corrections.

There were 57 camera-recognized approaches in the practice recordings. Nine had neither a legal return nor usable optical outcome, so the combined learner did not store them. Both ablations are conditioned on the same retained 48 approaches. The legal-only case does not add those nine as extra zeros. All 26 legal returns in the archive had a measured outgoing path, so the optical-only learner also retained all 48 approaches. It used 30 outgoing paths and 18 ball-to-blade gap measurements. This keeps the amount of practice information matched without treating missing optical observations as failures. [Legal-only bank](online-component-legal-memory.json) · [Optical-only bank](online-component-optical-memory.json)

| Frozen controller | 32 unseen releases | 16 varied continuations |
| --- | ---: | ---: |
| Reset | 11 | 12 |
| Inherited global offset | 9 | 18 |
| Combined signals | 11 | 19 |
| Legal only | 11 | 19 |
| Optical only | 11 | 19 |

The learned variants all made nine opening and two later returns on unseen releases. Reset made eight opening and three later returns. In the varied continuations, all learned variants returned all 16 openings and then landed three more shots on the last 8239 continuation. The inherited global offset also returned all 16 openings and landed two more on that continuation. Both signal variants can account for the added return on that one continuation, but the experiment does not establish a general rally gain. [New-release legal tape](online-component-new-legal.json.gz) · [New-release optical tape](online-component-new-optical.json.gz) · [Continuation legal tape](online-component-replies-legal.json.gz) · [Continuation optical tape](online-component-replies-optical.json.gz) · [Inherited global offset on unseen releases](online-new-prior.json.gz)

Identical scores did not mean identical joint commands. Compared with the combined learner, the legal-only tapes differed on seven points and the optical-only tapes differed on eleven. An action-only replay of each ablation's longest point in both courses checked 12,754 solver samples, with zero CAD penetrations and at least 18.6 mm clearance. [Command and score comparison](online-component-results.json) · [Physical replay checks](online-component-audits.json)

## Reproduction

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt36_online_components prepare
for stage in replies new; do
  for component in legal optical; do
    python -m scripts.experimental.tt36_online_components play --stage "$stage" --component "$component"
  done
done
python -m scripts.experimental.tt36_online_components play --stage new --component prior
python -m scripts.experimental.tt36_online_components summary
python -m scripts.experimental.tt36_online_components audit
```
