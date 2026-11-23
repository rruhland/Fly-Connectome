# Native Pong motion as a prior over donor-arm exploration

The native optical prior predicted the Pong player's later paddle motion on held games. Its exploration weighting gave the slowest early acquisition across the three paired donor learning streams. The comparison used equal reward budgets and common motor geometry. [Paired learning receipt and frozen weights](tt24_donor_pong.json.gz)

## Source and receiving learner

The source was the archived score-trained Pong actor with its inherited visual, motor and timing systems and visual risk guard. Its 48 fit and 24 held tapes came from live native play and contained 1,449 and 739 eligible optical observations. The source readout predicted the absolute difference between current paddle pose and mean observed pose 0.24–0.42 seconds later. The target was the log of one plus that difference divided by the observed paddle half-width. Native point and contact labels were used for qualifying and reporting this actor. [Source lineage and qualifications](pong-world-strong.md) · [Native optical tapes](pong-world-strong-native.json.gz)

The fitted cues correspond to 11 groups in the existing donor learner. The mapping treats the depth from the native paddle to the ball and motion normal to the paddle as analogues of relative bounce height and vertical velocity. It maps one tangent axis to the two donor tangent axes with equal weights, uses tangent magnitudes, scales image lengths by the observed native paddle span and reference blade radius, and maps median native normal speed to 2.1 m/s. Interaction features follow the existing donor controller. Bias, blade height and blade-normal cues retain unit source weight. Within the singleton and paired groups, absolute source ridge coefficients rank the mutation multipliers. Each group receives the same multiplier spectrum from exp(−0.48) to exp(0.48).

The coherent activity regression predicted held observed native travel with RMS error 0.308 in the transformed units, compared with 0.365 for within-game outcome shuffle and 0.382 for a constant. Held correlations were 0.596 and 0.503 for coherent and shuffled fits. These statistics measure associations with actual travel, with both the supplied geometry and source camera filters applied. [Registered study and optical fit](tt24_donor_pong-plan.json)

## Paired practice on three hinged bodies

Each receiving learner starts with the same zero 5×20 direct joint controller and exploration basis. A single previously specified set of 30 five-frame changes on the three articulated donors measures that basis from blade-pose changes without bounce rewards. Donor feedback policies have access to simulator metric ball position, velocity, blade pose and normal. Reward blends qualified rebound count, length of the longest consecutive streak and recorded survival time. Positions, motor signs and joint rates come from this shared receiving setup and its subsequent reward optimization.

The Pong and shuffle arms apply their respective rank weights only to exploration directions. The random arm assigns the same multiplier spectrum at random within each geometry group and does so independently for each stream. Cold draws the unchanged directions. The exact total perturbation norm for every draw is restored across arms before the common coefficient limits. Every stream uses the same training perturbations and launch starts across its four arms. The common mutation families and motion geometry come from the existing donor learner. [Donor implementation](../../scripts/experimental/tt24_donor_trial.py) · [Paired study implementation](../../scripts/experimental/tt24_donor_pong.py)

Three independent learning streams each use three donor bodies. Per update and arm, six signed directions plus the incumbent receive 39 reward-scored training drops. The three leaders receive 36 separate donor validation drops. A fixed roster of eight other releases per body supplies measured consecutive returns after updates 0, 1, 2, 4, 8 and 12. The primary comparison averages updates 1, 2, 4 and 8. Only the training and donor validation rewards update weights.

| Exploration prior | Update 0 | Update 1 | Update 2 | Update 4 | Update 8 | Update 12 | Early mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native Pong motion | 2.22 | 2.22 | 2.43 | 2.40 | 2.89 | 3.38 | 2.49 |
| Shuffled native motion | 2.22 | 2.26 | 2.44 | 3.08 | 3.53 | 3.97 | 2.83 |
| Cold | 2.22 | 2.35 | 2.56 | 2.57 | 4.31 | 4.22 | 2.94 |
| Matched random | 2.22 | 2.14 | 2.43 | 2.81 | 4.03 | 4.74 | 2.85 |

The primary measurement was mean longest consecutive rebounds over the independent donor releases. Across the three paired streams, Pong minus shuffle was +0.01, −0.24 and −0.80 rebounds. Pong minus cold was +0.05, −1.71 and +0.28. Pong minus matched random was +0.56, −1.63 and −0.03. Across all streams and arms there were 5,616 training drops and 5,328 validation drops, including each zero-policy validation. The acquisition curves account for 1,728 further held evaluations. The gap varies across streams, and the observed averages give no evidence of acceleration from this particular motion weighting. This test stops at the donor benchmark and has zero stock-arm exposure.

## Reproduction

Use the repository's table-tennis Python environment from the repository root.

```sh
export PYTHONPATH=src MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt24_donor_pong
```
