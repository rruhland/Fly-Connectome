# Reliability selection from live donor wrist images

The optical source screen retained the original donor controller. Averaging the late historical policies tied it on the reliability score, and the other four candidates scored lower. The locked winner therefore carries the original parameters to the held bodies. [Frozen checkpoint](tt24_public_donor_select.npz)

The source screen compares six predeclared 5×20 feedback heads without fitting new motor labels. It uses the original reward-trained head, its development champions at updates 34 and 26, the mean of the five late archived champions, a modest extrapolation away from that mean, and a 35% blend toward a separately broadened reward-trained head. All candidates govern every action of each live drop. The screen chooses one unconditional checkpoint. [Original donor training](tt24_donor.md) · [Fixed candidate arrays and release roster](tt24_public_donor_select-plan.json)

The three screening bodies include two original donor designs at gravity multipliers 0.88 and 1.11, plus a donor with a further 11% link extension, 1.23 gravity, different drag and ball mass, and its own elbow axis and payload. Two matched releases on each body produce 36 source selection trajectories. Each lasts at most four simulated seconds. The ranking assigns half its weight to the average body fraction reaching eight consecutive rebounds, 30% to the worst body's fraction and 20% to the mean streak capped at eight. The original head is eligible as the incumbent. [Selection trajectories and ranking](tt24_public_donor_select-source.json.gz)

Every source controller sees actual live wrist RGB and its own encoders. Static CAD camera and blade transforms are supplied for each source body. The decoder calculates those transforms from the current encoder vector in a separate kinematic buffer. It reconstructs the nominal paddle weld from the measured gripper pose. Ball position and motion come from the raster and its causal history. The body-aware donor optics give the original and every candidate the same sensory support. During held stock and altered-body deployment both controllers use the existing frozen stock wrist and encoder observer, previously calibrated from eight generic stock probe drops. These receiving bodies supply no additional motor practice or reward for choosing the controller. [Source body definitions](../../scripts/experimental/tt24_broad_donors.py) · [Frozen public observer](public-body.md)

The first fixture attempt left the blade's free joint at its model-default position inside the decoder's kinematic buffer, producing approximately 19 cm of registration error. That screen was discarded. The corrected fixture reconstructed the nominal weld and used a fresh seed lock for selection and held testing. A 15-frame audit gave at most 1.35 mm of blade-center error and sub-micrometre camera-position error. The six candidate arrays and selection objective stayed fixed across the correction.

The locked comparison includes eight independent stock drops, eight drops on the altered body held out of selection, and two further drops on each of the three source bodies. All are five-second maximum trajectories. Every complete command tape must reproduce the original physical judge record. [Paired evaluations](tt24_public_donor_select.json.gz) · [Implementation](../../scripts/experimental/tt24_public_donor_select.py)

## Source selection

| Candidate | Light donor | Medium donor | Longer-link donor | Reliability score |
|---|---|---|---|---:|
| Original | 10, 4 | 12, 11 | 2, 11 | 0.642 |
| Donor champion 34 | 12, 4 | 11, 12 | 2, 5 | 0.396 |
| Donor champion 26 | 7, 4 | 5, 8 | 2, 11 | 0.308 |
| Late champion mean | 8, 4 | 13, 11 | 2, 10 | 0.642 |
| Late extrapolation | 4, 4 | 13, 10 | 2, 8 | 0.392 |
| Broadened policy blend | 9, 3 | 4, 13 | 2, 10 | 0.538 |

Entries are longest consecutive rebound counts on the two fixed releases per body. The original and the late mean reached eight on four of six drops, with a worst-body fraction of one in two. Their capped averages also tied exactly. The fixed ordering resolves ties in favor of the original. The report retains all six scored candidates and pairings. [Complete selection traces](tt24_public_donor_select-source.json.gz)

## Frozen held results

| New held course | Mean longest run | Runs of at least eight | Maximum |
|---|---:|---:|---:|
| Stock, eight drops | 9.125 | 5/8 | 17 |
| Altered body, eight drops | 5.750 | 2/8 | 15 |
| Three source bodies, two drops each | 7.000 | 2/6 | 16 |

The original and selected arrays were identical, so each held trajectory was executed once and recorded under both labels. On separate source drops the longer-link body produced streaks of one and three, showing that the frozen controller still has a weak body under the supplied source optics. All 36 source-selection trajectories and 22 independent held trajectories reproduced their complete physical judge record. This small source screen supplies no replacement for the original head. [Frozen paired receipts](tt24_public_donor_select.json.gz)

## Reproduction

Use the repository's table-tennis Python environment from the repository root.

```sh
export PYTHONPATH=src MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt24_public_donor_select
```
