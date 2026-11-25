# Sensor-checked commitment with optical history and outgoing scoring

This test adds only the TT42 checked-execution rule to the pre-season TT46 player. The motion history still supplies the same optical fallback, and the same outgoing risk model still reranks legal candidates under the existing quality and robustness bounds. Search ticks and how long an accepted motion stays installed are the intervention. The trained readouts, joint candidate family, starting memory, arena, native opponent, actuator limits and exact rollout reuse are held fixed. Assessment is frozen in both arms, with full RGB history from launch and no evaluator measurements available to either controller. [TT42 calibration and held results](../tt42/direct-execution.md) · [Original TT46 season](../tt46_season/README.md)

The comparison covers the first preassigned TT46 assessment cycle, with one independently seeded game on each of eight distinct releases. All eight are release-disjoint from the TT42 execution calibration and course. The later two reply-seed cycles are not part of this comparison. A freshly run first stock game checks command and referee equivalence with the TT46 archive. All paired starts use the same original `learning-48` checkpoint and native target draws. Physical events are checked before the first different command, and first changed offers are separately reported where their physical incoming ball still matches. [Input and source receipts](protocol.json)

## Paired physical results

The checked arm scored **68 legal near returns versus 58** and **144 alternating legal returns versus 124**. Four games lengthened, three shortened, and one tied. The peak rose from 22 to 27 near returns, but on different releases. The 20-return gain on 13344 was nearly offset by the 19-return regression on 12192. The result is a modest net change on eight starts, with sizeable changes in either direction.

| Release | TT46 stock | Checked | First different command |
| --- | ---: | ---: | ---: |
| 10762 | 5 | 14 | 178 |
| 15541 | 9 | 3 | 256 |
| 13344 | 7 | 27 | 80 |
| 5212 | 9 | 3 | 166 |
| 7398 | 0 | 0 | 80 |
| 12246 | 4 | 12 | 94 |
| 12192 | 22 | 3 | 198 |
| 14429 | 2 | 6 | 160 |

All eight initial offers matched, and native event prefixes were exact before every first changed command. Opening legality was seven of eight in both arms. Among the seven identical incoming offers in play at the first changed command, both arms touched all seven and landed six. The checked policy preserved 76 due calls and made ten early replans in 360 inspections. Measured full-search cadence fell from 2.50 to 2.33 per simulated second. The checkpoint restored at serial 56 (48 prior games plus eight assessment games) with all 768 starting fragments unchanged. [Pair, contact, and checkpoint audit](comparison.json) · [Treatment actions](checked.json.gz) · [Checkpoint](checked.npz) · [Fresh control](fresh-control.json.gz)

## The sensitive 7398 opening

On the original full-history 7398 stream, both players chose the same weakly robust demonstration transport at tick 72. The checked arm rejected it on a fresh discrepancy and switched to a different weakly robust credited transport at tick 80. It checked and rejected that motion again at tick 84. The first changed issued command came at tick 80 after eight exact native events. Both touched the identical incoming offer at control step 88 and neither landed a legal return. Extra checks changed the motion but did not rescue this offer. This uses full RGB from release and does not repeat the earlier short-warmup intervention or identify a velocity-error threshold. [Earlier 7398 matched intervention and full-history control](../tt48_exact_repair/README.md)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt46_checked_execution run
python -m scripts.experimental.tt46_checked_execution report
python -m pytest -q tests/test_tt46_checked_execution.py
```
