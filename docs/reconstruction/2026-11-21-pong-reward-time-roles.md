# Finding relevant tracks from Pong reward timing

A derivative-free model learned to identify which anonymous visible track changed at native Pong reward times. The resulting role weights matched the existing supplied-witness fit on the same source observations. A table receiver using the learned attribution continued updating its roles from its own images and public scores. It made legal returns on 18 of 24 held-out points, compared with 17 for the supplied witness and 19 for shuffled source timing. Each receiver won three points.

## Learning from the reward clock

The source played native Pong with random held actions, remapped action IDs and rotated or reflected images. The shared tracker grouped pixels and followed anonymous regions. Each candidate used the previous nine images. Features measured its speed, recent travel and velocity change, apparent area and area change, missing observations, proximity to the frame edge, change in that proximity, and distance to its nearest neighbor. All observed candidate tracks were represented the same way. The training signal indicated when a public point had occurred. It provided no track identity or direction sign.

The model saw 55 eligible reward windows and 240 neutral windows from 60 source episodes. It associated each window’s reward status with a smooth maximum across its candidate tracks. Powell search minimized the scalar classification loss. The model selected the highest-scoring candidate at the reward times, then a separate linear fit distilled the selected tracks into the shared role comparisons. Action-contingency comparisons remained common across conditions. The learned coefficients gave the largest weight magnitudes to area contraction and edge proximity. The supplied-witness condition used the older outward-motion and boundary rule on the same observations. The shuffled condition relocated scored times within each episode before training and distillation.

| Clock training signal | Held-out AUC for real reward times | Agreement with supplied witness |
|---|---:|---:|
| True reward timing | 0.99875 | 20/20 |
| Relocated reward timing | 0.00375 | 7/20 |

The holdout used 20 fresh reward windows and 80 neutral windows. The reference witness was computed for the comparator and attribution audit. The learned model’s inputs were its anonymous visual features and the public clock labels.

## Receiving table

Each condition used the same numeric Pong policy, the same visual motor learner, and the same three eight-point table matches. The two fitted clocks were frozen. At each scored table point, the corresponding receiver used its clock to rank the candidate tracks in its latest views and added new evidence for the selected track. The supplied-witness receiver used the original attribution procedure. Native contact and return records went to the evaluator.

| Receiving role evidence | Points with a contact | Points with a legal return | Points won |
|---|---:|---:|---:|
| Learned reward-time clock | 22/24 | 18/24 | 3 |
| Relocated reward-time clock | 22/24 | 19/24 | 3 |
| Supplied witness | 22/24 | 17/24 | 3 |

The learned clock attributed all 24 receiving rewards, producing 552 role-contrast rows. In a separate check, the legacy witness was set to raise on access. The receiving player completed three scored points and added 63 role contrasts using its own clock. The matched table check supports the witness replacement and gives no performance advantage over relocated timing in these ordinary scenes. Each of the three conditions recorded zero sustained rallies in this course. The four targeted role tests passed, and a repeated run reproduced the optical bags, attribution decisions and table rows.

## Reproduce

The receipt includes episode seeds, all source and held-out feature bags, clock coefficients, attributions, table outcomes and source hashes. [Full experiment receipt](../../artifacts/tt3/discovered-reward-roles.json.gz) · [Role checkpoint](../../artifacts/tt3/discovered-role-model.json) · [Reward clock](../../artifacts/tt3/discovered-reward-clock.json)

```sh
export PYTHONPATH=.:src:scripts/experimental
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
g++ -O3 -shared -fPIC scripts/experimental/tt3_components.cpp -o /tmp/tt3-components.so
export TT3_COMPONENT_LIBRARY=/tmp/tt3-components.so
python -m scripts.experimental.tt3_role_discovery artifacts/tt3/numeric-pong-source-refined.json /tmp/discovered-roles.json /tmp/discovered-roles.json.gz --clock-output /tmp/discovered-clock.json --train 60 --verify 20 --table-matches 3 --table-points 8
python -m pytest -q tests/test_tt3_role_discovery.py tests/test_tt3_visual.py
```
