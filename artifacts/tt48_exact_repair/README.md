# An earned early correction transfers to one exact new offer on the strong player

The 9519 correction returned an unchanged physical ball from a different release, 7398, on the stronger pre-season carried player. The reset stock arm and guarded selector both contacted it and faulted into their own half. The deliberately unguarded nearest of two independently earned motor examples landed. The ordinary held probe is evidence for a transferable motor move, not for adopting the nearest-example policy in full rallies.

The strong exact-offer comparison replays the original physical prefix, including native referee and opponent state. It clones the solver after the far separation, verifies the archived continuation from the clone, then runs three fresh camera-initialized branches with the same 768 prior fragments and unchanged temporal, direct, and late coverage code. No branch sees the true ball state or the archived event labels. The physical far impact and near bounce are retained on every branch. The reset controller does not in general reproduce the archived controller’s prior solver or planner history. Its score and the original score are reported separately.

| Native offer | Height projected at the ordinary receive depth | Archived | Reset stock | Gated local | Nearest earned |
|---|---:|---:|---:|---:|---:|
| 7398, first | 1.006 m | 0 | 0 | 0 | 1 |
| 10762, first | 1.016 m | 0 | 0 | 0 | 0 |
| 14429, third | 0.989 m | 0 | 1 | 0 | 1 |
| 12246, third | 1.134 m | 1 | 1 | 1 | 1 |

The 14429 outcome is not a learned low rescue. Reset stock already lands, and the gated local policy regresses it. Thus there is one demonstrated same-ball transfer and one learned regression on this four-offer diagnostic. No terminal miss from these held releases was used to fit the two motion examples. [Actual commands, optical traces, and outcome receipts](transfer.json.gz) · [Original training interventions](../tt44_continuous_credit/training.json.gz)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt48_exact_repair
python -m scripts.experimental.tt48_exact_repair --limit 1 --full-observer --output artifacts/tt48_exact_repair/full-7398.json.gz
python -m scripts.experimental.tt48_exact_repair_audit
python -m scripts.experimental.tt48_terminal_repair
```
