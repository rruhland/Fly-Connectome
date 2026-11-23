# Hindsight optical outcomes and wrist control

Native Pong success exemplars supplied a weak goal distribution. Each example was the optically observed endpoint of the longer of two matched native pulse trials. The goal score compared action-independent post-pulse speed and direction with those exemplars, using optical approach speed and apparent surface extent for normalization. It correctly preferred the longer branch on six of thirteen held native comparisons. A constant score gave 6.5 and shuffled exemplars gave five, counting ties as half. On a separate audit of seven archived arm pairs with different public rewards, the Pong goals agreed on 3.5 pairs and shuffled exemplars on five. These results left the native goal distribution unqualified for arm deployment. [Native and archived target audit](hindsight-source.json)

A target-only alternative tested whether the wrist controller could predict and seek outcomes it had already achieved itself. The common active controller came from 266 earlier target attempts. The new budget was sixteen wrist-only drops, consisting of eight matched warm and perturbed pairs. Both branches issued the same commands through their first reported rebound. Each perturbed branch then added one random bounded five-joint rate vector for eleven control ticks. A local ridge fit related the induced motor changes to the wrist centroid, apparent size and motion observed 16 to 25 control ticks later. The hindsight policy sought nearby optical outcomes from drops with higher public rebound rewards. The reward-only comparison learned a direct reward gradient from the same pairs. Both proposed bounded corrections after the first reported rebound in held play.

The warm branches made 39 rebounds across the eight training starts. Their paired perturbed branches made 15. Every perturbation reduced reward. Four pairs supplied both optical endpoints. The observed action directions spanned four dimensions for a five-actuator head. Held-out optical response error averaged 2.68 normalized units for the ridge fit and 2.67 for persistence, using one omitted pair at a time. The experiment therefore supplied neither rewarded exploration nor evidence that the fitted response could predict a useful new action. The tested inverse applied corrections inferred from these sparse negatives, including reversals of directions whose effects had been observed only one way.

| Controller | Verified rebounds over 16 fresh starts | Mean per drop | Best drop |
|---|---:|---:|---:|
| Wrist hindsight model | 34 | 2.13 | 4 |
| Same-practice reward-only update | 31 | 1.94 | 3 |
| Frozen warm controller | 42 | 2.63 | 6 |

The hindsight mean was 0.50 rebounds below the warm controller, with a paired 95% bootstrap interval from -1.19 to 0.19. Its advantage over the reward-only update was 0.19, with an interval from -0.19 to 0.63. The pulse and observation window are specific choices in this test. The result gives a direct coverage check on what sixteen attempts bought under those choices. All 64 practice and held action tapes replayed through the evaluator. [Wrist experiment and exact learned memory](hindsight-arm.json.gz)

## Reproduction

Use the table-tennis environment from the repository root. Set `PYTHONPATH=src`, `MUJOCO_GL=egl`, `OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`.

```sh
python -m scripts.experimental.tt24_hindsight
python -m scripts.experimental.tt24_hindsight_arm
```
