# Check accepted joint execution before replanning

The frozen execution rule uses the stock 30 Hz stereo and joint encoders to check whether an already issued motion still matches its short private rollout. It can keep a corroborated buffer past a due search, or check an incumbent early near impact if a fresh image disagrees or robustness was weak. It asks the unchanged ranker earlier only if that check no longer finds a legal return. The ordinary joint repertoire, ranking, uncertainty perturbations, optics, native opponent, actuator limits, physical readiness and pretraining bank are unchanged. Changing search ticks can still change which warm branch and random variations the stock ranker sees. This is a comparison of execution policies, not a pure change to clock frequency.

The fixed scales were 4.19 mm translation, 0.0575 m/s velocity, and 0.022 rad across the first five joints. They came from a componentwise 85th percentile of residuals on 46 samples from successful archived practice, with fixed floors and ceilings. The thresholds were exceeded on 9 of 46 successful and 17 of 70 unsuccessful samples. Joint residuals were essentially zero at the median for both, and this small fit contrast is not evidence of a reliable miss classifier. A three-game treatment smoke test scored 4, 5 and 0 legal returns, exercised three preserved due calls and one early replan, and did not update motor memory. Those unpaired practice scores do not measure treatment benefit. [Calibration](direct-execution-fit.json) · [Practice actions](direct-execution-practice.json.gz)

The held comparison consists of all 24 preassigned starts against the native opponent, with independent draws for each card. Both arms start every game with the same 321 stored fragments. No held result is used for score-gated learning. The report audits score against native event tapes, physical assignment and exact opponent draws, exact prefixes through the first different issued command, and the restored point-boundary checkpoint. Later offers may change as different legal returns reach the opponent. [Protocol receipt](direct-execution-held.protocol.json) · [Stock actions](no-more-practice.json.gz)

## What changed in the 24 held games

Checked execution raised legal near returns from **102 to 171** and the longest game from **12 to 32**. Alternating legal returns were 225 vs 363. Six games improved, six shortened, and twelve tied. The four largest gains (9550, 7812, 6411 and 8056) added 79 returns. The other twenty netted ten fewer. Neither arm won more opening returns (18 each), and the controller found no legal nominal motion on 12 terminal offers in each arm. The original job was stopped after game 13. The restored checkpoint replayed no completed game, and the remaining 11 went from 47 stock to 86 checked.

| Release | Stock | Checked | Release | Stock | Checked |
| --- | ---: | ---: | --- | ---: | ---: |
| 9550 | 5 | 20 | 8544 | 1 | 1 |
| 15665 | 12 | 11 | 6411 | 4 | 32 |
| 11884 | 4 | 3 | 8863 | 0 | 0 |
| 3401 | 1 | 1 | 3786 | 4 | 1 |
| 14760 | 0 | 0 | 12199 | 12 | 5 |
| 7901 | 11 | 6 | 8056 | 9 | 25 |
| 8772 | 1 | 1 | 9665 | 5 | 5 |
| 9965 | 5 | 7 | 6765 | 0 | 0 |
| 3905 | 3 | 3 | 12068 | 0 | 0 |
| 15253 | 2 | 2 | 11961 | 2 | 9 |
| 7812 | 10 | 30 | 4802 | 11 | 9 |
| 12198 | 0 | 0 | 10294 | 0 | 0 |

All 21 delivered openings matched and the three starts without an offer matched exactly. There were 16 changed command tapes, all with matching native event prefixes. On the 15 first changed offers whose incoming ball still matched exactly, both policies made contact on every offer. Stock landed 14 and checked execution landed 13. The only changed first-offer legality was a regression on 3786. Both arms' final checkpoint bytes matched with 321 fragments and serial 24. [Exact-prefix and first-offer audit](direct-execution-audit.json) · [Paired scores](direct-execution-results.json) · [Treatment action archive](direct-execution-held.json.gz) · [Checkpoint](direct-execution-held.npz)

The treatment preserved 217 due calls and made 22 early replans across 889 incumbent inspections. The longer rallies meant more total searches, but the rate declined from 2.51 to 2.25 searches per simulated second. Inspection added 19.8 CPU seconds. The tail gains cannot be attributed to extra search alone or to newfound reach: most first changed strikes already landed and simply sent a different ball back. The sharp next mechanism test is to hold the same candidate pool and clock up to a matched pre-impact prefix, then compare preservation alone with early replacement alone. This would tell whether keeping a corroborated motion helps on the same ball or merely steers the subsequent rally into a better reply path.

## Reproduce

Run from the repository root with the project environment and the archived measured launches available. Restore the exact course source to a temporary file if the working implementation has since changed. The replay wrapper verifies the eight archived hashes before it can launch or resume.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
runner_dir=$(mktemp -d)
git show 1d82db9:scripts/experimental/tt42_learning_course.py > "$runner_dir/tt42_learning_course.py"
python -m scripts.experimental.tt42_replay_direct_execution --course-source "$runner_dir/tt42_learning_course.py"
python -m scripts.experimental.tt42_direct_execution_audit
python -m pytest -q tests/test_tt42_direct_execution.py tests/test_tt42_camera_precision.py
```
