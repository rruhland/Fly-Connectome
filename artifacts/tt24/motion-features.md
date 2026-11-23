# Native reward and contextual velocity choice

The native reward fit selected a contextual mix of the freshest and usual tangential velocity estimates. Its held payoff trailed the matched reward shuffle. The study stopped at the registered native gate, before running the altered-wrist course. [Frozen native results](motion-features-native.json.gz) · [Locked protocol](motion-features-plan.json)

Earlier native transfer tests supplied useful constraints. A fixed delay multiplier lost to the simple fixed controls on independent wrist replication. Causal optical intervention credit produced 30 rebounds on the altered body against 29 for the original warm controller. Workspace shifts lost to the frozen native reference. This experiment instead learns which velocity estimate to use at each observed context. [Delay replication](pong-temporal.md) · [Causal intervention results](recurrence.md) · [Native workspace study](pong-workspace-source.md)

The fixed native policy consumes its usual causal velocity estimate blended with a two-frame tangential measurement. The mixture follows a learned logistic readout of the controlled paddle's observed speed, the mover's velocity relative to it along their observed separation, the paddle's contribution to that motion, and observed normal speed. It receives only native raster history and the usual motor and pose conventions. Pairs of signed perturbations to the five mixture coefficients were evaluated on actual native points. Each update uses terminal point score plus 0.43 times the fraction of the active point. The shuffle independently reverses which direction receives each paired score difference. The head, command interface and controls share the same experience budgets. [Implementation](../../scripts/experimental/tt24_motion_features.py) · [Frozen native parent](tt24_joint_pong.md)

One learning stream used 2,112 actual native points. The held round played 480 actual points over 96 distinct launches with two native actuator response times. The fixed controls always use the original estimate, mix estimates equally on valid pairs, or use the freshest valid estimate. All conditions fall back to the existing estimate when a pair of visible ball samples is missing.

| Velocity selector | Held mean native payoff | Starts with an own return / 96 |
|---|---:|---:|
| Trained contextual | 0.35435 | 91 |
| Shuffled score credit | 0.36718 | 92 |
| Ordinary estimate | 0.34376 | 91 |
| Fixed equal mix | 0.34354 | 91 |
| Freshest valid estimate | 0.34340 | 91 |

The gate requires at least 0.012 above all held controls and bounded loss from the usual estimate at each actuator response time. The contextual test fell short against shuffle and ordinary. The scripted follow-up would have changed only the two horizontal velocity features of the fixed reward-trained wrist head, with the familiar origin and all motor parameters held fixed. The registered six altered drops remain unused. [Full source decisions and reserved target starts](motion-features-plan.json)

The missing transfer input is a measure of how observation error depends on the actor's own motion. The fixed camera exposes real relative kinematics, but gives the selector no examples of wrist-camera rotations making the decoded mover appear to accelerate. The stock wrist audit measured that source of error directly, and its supplied full geometric correction already reduced optical velocity error from 0.234 to 0.177 m/s. This fit provides no evidence that a reward-conditioned choice from the native relative tracks improves on that correction or the carried wrist policy. [Wrist egomotion audit](ego-motion-wrist-audit.md)

## Reproduction

Use the repository's table-tennis Python environment from the repository root.

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt24_motion_features
```
