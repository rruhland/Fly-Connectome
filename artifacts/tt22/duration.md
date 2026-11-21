# Duration-trained Pong preference and physical return transfer

## Native Pong

The source learner selected signed bias, displacement and motion gains using episode length as its scalar objective. A point ended at the native score boundary or the 10-second cap. Its controller used randomized RGB palettes, rotations, reflections and command IDs through the frozen role and actuator discovery from TT3. The source selector read elapsed ticks. Scoring events and hit records were retained for evaluator summaries. Training screened 16 broad signed options on 12 matched starts, then tested the four finalists on 16 more starts against the same deterministic opponent at speed 0.56. The frozen winner was `[-0.04631, 1.01418, -0.54695]`. [Native training and held receipt](duration-pong.json.gz) · [Frozen numeric checkpoint](duration-pong-player.json)

| Frozen source on 32 new matched Pong starts | Mean capped duration | First returns | Later returns | Episodes reaching 10 s |
|---|---:|---:|---:|---:|
| Duration-trained | 5.56 s | 19/32 | 17/29 | 8/32 |
| Original point-trained | 5.96 s | 19/32 | 24/28 | 15/32 |
| Broad cold / shortest-duration selection | 2.25 s | 5/32 | 0/5 | 0/32 |
| Permuted duration assignment | 2.30 s | 4/32 | 1/5 | 0/32 |
| Stationary | 2.44 s | 5/32 | 2/6 | 1/32 |

The preassigned cold option was also the option selected by minimizing duration on the search points. The separate permuted control shifted the mapping from duration scores to gain vectors by half the search bank. The duration-trained policy increased capped time by 3.12 seconds relative to stationary and by 3.26 seconds relative to permuted assignment. The point-trained checkpoint lasted 0.40 seconds longer on average, with a paired bootstrap 95% interval of 0.13 to 0.72 seconds for its advantage. Twenty-four point-trained and duration-trained episodes tied and eight lasted longer with the point-trained checkpoint. The two objectives were tested with differently trained checkpoints on the same held opponent.

Four held launches across all six recorded conditions reproduced the native action digests and physical outcomes exactly.

## Frozen SO-101 transfer

Both learned policies carried only the displacement and motion gains. Each target launch used eight native stereo frames under zero commands. The anonymous motion extractor matched moving components across the two cameras using ordinary table registration and produced a usable track on 47 of 48 unselected varied serves. The existing reward-trained clock stroke used the requested lateral goal and vendor inverse kinematics. All branches shared the prefix and physical opponent on a paired start. [Physical transfer receipt](duration-transfer.json.gz) · [Target runner](../../scripts/experimental/tt22_duration_transfer.py)

| Matched target bank | Duration contact / legal | Point contact / legal | Neutral contact / legal |
|---|---:|---:|---:|
| Moderate lanes, 24 | 24 / 14 | 20 / 11 | 16 / 8 |
| Wider lanes, 24 | 23 / 6 | 15 / 6 | 3 / 1 |
| Combined, 48 | 47 / 20 | 35 / 17 | 19 / 9 |

Duration produced 13 contacts the point-trained prior missed, while the point-trained prior produced one the duration prior missed. For legal returns, the exclusive counts were nine and six respectively. The duration and neutral priors produced 14 and three exclusive legal returns. Every matched result reproduces from its recorded RGB estimate and frozen policy. All three branches made at most one legal learner return per point.

## Eight target practice points per branch

Two independent training streams used eight fresh paired starts each. The policies received the same perturbations around their incoming gains and motor parameters. A single weighted update used the actual native learner legal-return count. Frozen endpoints then replayed each stream's original 24 held prefixes. [Adaptation receipt](duration-adaptation.json.gz)

| Legal learner returns | Duration | Point | Neutral |
|---|---:|---:|---:|
| Moderate bank before → after | 14 → 14 | 11 → 13 | 8 → 8 |
| Wider bank before → after | 6 → 8 | 6 → 8 | 1 → 5 |
| Combined before → after | 20 → 22 | 17 → 21 | 9 → 13 |

The carried duration gains produced the broadest initial contact coverage. The legal-return advantage over the point-trained prior was three of 48 before practice and one of 48 afterward. Every adapted arm still made at most one legal learner return per point. This target outcome measures transfer of the first strike. All 144 frozen evaluations, 48 practice episodes and 144 adapted evaluations passed physical replay of the recorded outcomes.

## Reproduction

From the repository root with the MuJoCo environment active, compile the existing accelerated RGB component extractor and run the paired tests.

```bash
c++ -O3 -shared -fPIC scripts/experimental/tt3_components.cpp -o /tmp/tt3-duration-components.so
export TT3_COMPONENT_LIBRARY=/tmp/tt3-duration-components.so
export PYTHONPATH=.:src:scripts/experimental OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MUJOCO_GL=egl LP_NUM_THREADS=1
python -m scripts.experimental.tt22_duration_pong
python -m scripts.experimental.tt22_duration_transfer
python -m scripts.experimental.tt22_duration_transfer --adapt
```
