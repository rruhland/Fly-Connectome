# Learning how much to reuse a familiar origin in native Pong

The controller learned one scalar from actual native play with shifted initial paddle positions. It kept the previously reward-trained four-weight visual policy and motor servo fixed. The scalar specifies how far to move the controller's reference origin from the newly observed paddle center toward the median center it saw in its earlier native practice. It changes both the pose feature and the servo equilibrium. An alpha of zero uses the current launch origin, and an alpha of one uses the familiar origin completely. Only alpha is available for transfer to the arm. [Native policy and original practice](tt24_joint_pong.md)

The familiar center was measured from the first native raster of the 90 earlier practice seeds. In each new native game the environment offsets the actual paddle at reset, with balanced signs and magnitudes drawn from 0.17 to 0.33 native screen units. Serve seeds and paddle offsets are independent. Ball physics, the opponent, actuation and camera observations are otherwise the same as in the earlier actor. The only fit inputs are native terminal score and 0.43 times the fraction of the six-second cap spent in active play. [Registered source protocol](tt24_pong_origin-plan.json)

Each of 96 practice starts is played with the same 11 evenly spaced alpha values. The fitted scalar is the bounded continuous maximum of a regularized quadratic fit to the actual mean payoffs across those values. Its shuffled control permutes the outcomes among alpha values independently within each serve, preserving the initial displacement and native opponent for each set. The first source freeze learned alpha 1.0 both from native labels and from its registered shuffle. The paired controller outputs consequently coincide. The actual source payoff increased from −0.088 at alpha zero to 0.168 at alpha one. [First immutable source lock](tt24_pong_origin-lock.json.gz)

| First independent source audit | Own-return points / 64 shifted | Later returns | Losses | Mean native payoff |
|---|---:|---:|---:|---:|
| Familiar and native-selected alpha 1.0, also selected by the first shuffle | 53 | 36 | 11 | 0.197 |
| Neutral alpha 0.5 | 49 | 29 | 18 | 0.069 |
| Current-origin alpha 0.0 | 42 | 27 | 25 | −0.069 |

Across 48 independent ordinary points, all alpha values replayed the same native outcomes. Every point included an own return, and there were 35 later returns. The familiar and current optical origins coincide on ordinary starts. The actor earned no points after own returns on these courses. This evaluation measures interception retention. [First fresh shifted and ordinary scores](tt24_pong_origin.json.gz)

## Independent source fits

Six independent replications each use 96 further native practice starts, identical objective and fitting rules, and a separate within-serve shuffle. The source schedules reserve distinct shifted and ordinary starts for post-lock retention. The bank writes the six paired coefficients and all native practice outcomes before running that assessment. The original tie and its checkpoint remain intact. These files contain only source data. [Bank registration](tt24_pong_origin-bank-plan.json) · [Frozen alpha pairs and source exposure](tt24_pong_origin-bank.json.gz) · [Native retention after all six locks](tt24_pong_origin-bank-held.json.gz)

| Replica | Native alpha | Shuffled alpha | Shifted payoff difference from shuffle |
|---|---:|---:|---:|
| 1 | 1.000 | 0.000 | +0.1795 |
| 2 | 0.913 | 1.000 | +0.0171 |
| 3 | 1.000 | 0.000 | +0.2566 |
| 4 | 1.000 | 1.000 | 0.0000 |
| 5 | 0.939 | 0.000 | +0.2716 |
| 6 | 1.000 | 0.947 | +0.0195 |

| Across six shifted source audits | Own-return points / 384 | Later returns | Losses | Mean native payoff |
|---|---:|---:|---:|---:|
| Native-selected | 305 | 174 | 71 | 0.199 |
| Shuffled | 283 | 143 | 112 | 0.075 |
| Fixed neutral | 290 | 156 | 97 | 0.121 |
| Fixed current | 263 | 122 | 145 | −0.026 |
| Fixed familiar | 304 | 168 | 72 | 0.195 |

Real outcomes consistently chose substantial familiar-origin reuse in this actor. Fixed full reuse achieved almost the same pooled payoff as the learned values. Across 288 ordinary starts, all conditions replayed identical records, with 277 own-return points, 181 later returns and 12 losses. The first fit used 1,056 actual practice points and 591,855 simulated physics ticks. The six new fits used 6,336 actual practice points and 3,748,967 ticks. Verification checked 1,456 distinct starts across the first and replicated protocols, reproduced every scalar from the archived payoffs, and matched the original controller's ordinary native outcomes exactly. [Full source bank](tt24_pong_origin-bank.json.gz)

## Reproduction

Use the repository's table-tennis Python environment from the repository root.

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt24_pong_origin
python -m scripts.experimental.tt24_pong_origin --replicate
```
