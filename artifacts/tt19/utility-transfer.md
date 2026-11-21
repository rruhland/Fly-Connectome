# Can gantry reward help rank the first SO-101 strokes?

The reward predictor changed the receiving stroke selected after 12 and 24 practice points. It yielded zero legal returns in both held tests. No-source made one legal return after 12 practice points and zero after 24. Shuffled rewards yielded one and two. The tested reward transfer gave no early scoring advantage. [Training and held point tapes](utility-transfer.json.gz)

The source used 258 archived gantry points and 11,481 ball sightings derived from their RGB sequences. It fit separate regularized win predictors for 52 recognizable receiving strokes and 27 serving strokes. The predictors used only the first outgoing path and the public point reward. A visible change from a near-side approach or descending toss to outgoing motion identifies that path. Features include launch height and depth, outgoing speed and rise, extrapolated lateral and vertical position at the net, and early progress. Table width, table length and net height set the scale. Later rally segments and referee contact events stay outside the fit. [Predictor and source selection](../../scripts/experimental/tt19_utility.py)

| Gantry validation by held game seed | Receiving AUC | Serving AUC |
| --- | ---: | ---: |
| Actual reward labels | .574 | .572 |
| Rewards shuffled within game and role | .478 | .417 |

The target starts from public SO-101 factory kinematics and joint encoders, nominal flight, and generic adjustments to intercept depth, pace, blade pitch and timing. All branches receive the serving turn. Every branch uses the same local utility based on public point reward and visible outgoing progress. The two source branches add a centered predicted win probability from the actual-reward or shuffled-reward predictor. No previously trained SO-101 stroke archive or gantry motor parameters enter the target. [Target protocol](../../scripts/experimental/tt19_utility_trial.py)

The three branches share the first twelve practice points. A frozen selection is tested on 24 separately seeded native RGB points, twelve in each role. Each learner then chooses its own next twelve practice attempts from the same scheduled seeds and perturbation draws and returns to the same held starts. The held outcomes never enter the learner. Practice uses the fast RGB renderer and held points use native fixed stereo on the strict 0.848 × 1.60 m SO-101 table. A common opening image registers the cameras, with zero dry motor commands. When policies select identical strokes, the held comparison shares the resulting physical tape for that seed.

| Practice points | Method | Legal returns | Legal serves | Points won |
| ---: | --- | ---: | ---: | ---: |
| 12 | Local utility only | 1/12 | 2/12 | 2/24 |
| 12 | Actual gantry rewards | 0/12 | 2/12 | 1/24 |
| 12 | Shuffled gantry rewards | 1/12 | 2/12 | 2/24 |
| 24 | Local utility only | 0/12 | 2/12 | 1/24 |
| 24 | Actual gantry rewards | 0/12 | 2/12 | 1/24 |
| 24 | Shuffled gantry rewards | 2/12 | 2/12 | 3/24 |

The actual-reward and no-source branches selected identical serves at both practice budgets and produced different action tapes on all 12 receiving starts. The shuffled predictor's two-return lead after 24 attempts comes from two discordant starts on one compact table, with an exact paired p = .50. The 24 practice points consumed 1,142 control frames for local utility, 1,161 for actual gantry rewards and 1,040 for shuffled rewards. All 48 distinct practice tapes and 96 native held tapes reproduced encoder readings, physics events and scores. [Paired outcomes and replay data](utility-transfer.json.gz)

```bash
PYTHONPATH=src:scripts/experimental LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m scripts.experimental.tt19_utility_trial
```
