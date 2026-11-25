# An earned early correction can rescue one exact new offer, but depends on the optical history

The 9519 correction returned an unchanged physical ball from a different release, 7398, on the stronger pre-season carried player. The reset stock arm and guarded selector both contacted it and faulted into their own half. The deliberately unguarded nearest of two independently earned motor examples landed. This rescue held only with the common 22-tick optical warmup. Both optical histories picked the same six parameters at the same control tick, with identical estimated position and only a 0.007 m/s difference in initial along-table velocity. The later replanner accepted three robust updates with the short history and one with the full history. On a second branch that observed the complete original launch, neither reset stock nor the borrowed action landed. The shorter-warmup physical witness is reproducible, not robust deployment transfer.

The strong exact-offer comparison replays the original physical prefix, including native referee and opponent state. It clones the solver after the far separation, verifies the archived continuation from the clone, then runs three fresh camera-initialized branches with the same 768 prior fragments and unchanged temporal, direct, and late coverage code. No branch sees the true ball state or the archived event labels. The physical far impact and near bounce are retained on every branch. The reset controller does not in general reproduce the archived controller’s prior solver or planner history. Its score and the original score are reported separately.

| Native offer | Height projected at the ordinary receive depth | Archived | Reset stock | Gated local | Nearest earned |
|---|---:|---:|---:|---:|---:|
| 7398, first | 1.006 m | 0 | 0 | 0 | 1 |
| 10762, first | 1.016 m | 0 | 0 | 0 | 0 |
| 14429, third | 0.989 m | 0 | 1 | 0 | 1 |
| 12246, third | 1.134 m | 1 | 1 | 1 | 1 |

The 14429 outcome is not a learned low rescue. Reset stock already lands, and the gated local policy regresses it. Thus there is one demonstrated same-ball transfer and one learned regression on this four-offer diagnostic. No terminal miss from these held releases was used to fit the two motion examples. An independent replay from launch reproduced all six audited tapes without CAD penetration in 9,701 solver states. On 7398, the successful altered strike used the same deep negative face with better solver alignment, −0.991 instead of −0.921. [Native replay and contact-face audit](audit.json) · [Full-RGB-prefix negative control](full-7398.json.gz) [Actual commands, optical traces, and outcome receipts](transfer.json.gz) · [Original training interventions](../tt44_continuous_credit/training.json.gz)

## The original 43rd offer on release 2741

The record game delivered one more legal incoming ball after the arm’s 42nd successful return. That same physical prefix and the original terminal failure were replayed bit-for-bit. Because no subsequent return or miss update changed the memory, the verified final 768-fragment bank is also the bank at the fork. The original received path projected below the earlier ordinary regime, at 0.953 m. The closest sampled sphere-to-disc clearance was still 34.5 mm. The ball was at x = −0.120 m, y = −1.569 m and z = 0.936 m. Tangential offset was 128.6 mm and the elbow was already at about −1.692 rad. This is an actual native-record diagnostic, not a global reachable-set bound.

The fresh strong stock replay, its guarded local reader, two separately earned ordinary corrections, and two variants that sustain those corrections through rebound tested the same live physical delivery. None touched. None increased the original 42 legal arm returns. [Six native matched counterfactuals](terminal-43.json.gz)

A separate privileged diagnostic gave a bounded native teacher exact state at the first inbound RGB cue. It screened both signed faces across 13 admitted contact windows. Across 288 private dynamics trials none touched, so no physical stroke was issued. This does not rule out earlier preparation or other contact families. It rules out blaming the optical state alone for the tested post-cue branch. [Exact-state post-cue comparison](terminal-native-coach.json.gz)

A privileged earlier fork began immediately after the previous near stroke, giving 1.25 s before the post-bounce terminal search. Two signed approaches actually moved the physical arm on clear routes and preserved the exact far impact and bounce. Their subsequent private dynamics screen found 296 contacts in 384 proposed commands, but no legal landings. Contrast the same state search beginning at first inbound detection, which found zero contacts. Earlier motion buys contact opportunities in this family. A useful departure still has to be learned and demonstrated. [Causal physical preparation and private contact comparison](terminal-earlier-prepare.json.gz)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt48_exact_repair
python -m scripts.experimental.tt48_exact_repair --limit 1 --full-observer --output artifacts/tt48_exact_repair/full-7398.json.gz
python -m scripts.experimental.tt48_exact_repair_audit
python -m scripts.experimental.tt48_terminal_repair
python -m scripts.experimental.tt48_terminal_coach
python -m scripts.experimental.tt48_terminal_prepare
```
