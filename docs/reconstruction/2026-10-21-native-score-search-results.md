# Native-score search: no established additional winning skill

The valid, independently audited screen failed. The last complete policy won
18/64 fresh serves versus 16/64 for its acquired starting policy. Both total-win
and own-hit-win lift over the strongest comparator were **3.125 pp, paired 95% CI
[−6.25,12.50]**, below the registered 10 pp/positive-lower-bound criterion. The
incremental comparison against the starting policy is identical. This does not
establish native-score acquisition or show that winning is impossible in this class.

![Audited native-score adaptation](/workspace/m1b-experiments/native-score-parent-audit-20261021/native-score-learning.png)

The [protocol](2026-10-21-native-score-search-protocol.md) warm-started the
predetermined first qualified return policy, retaining its frozen generic vision,
transform, cap and readout class. It used win-only terminal fitness against the
fixed .35 opponent, 12 shared fresh training serves, eight mirrored directions,
top four updates and score-free relative noise .50R/step cap .10R. There were no
hit labels in fitness, privileged policy inputs, teacher initialization, backprop,
intermediate DEV selection or final-stream use.

| Frozen policy | Wins /64 | Own-hit wins /64 |
|---|---:|---:|
| Endpoint theta5 | 18 | 8 |
| Acquired starting policy | 16 | 6 |
| Original initial policy 1 | 10 | 0 |
| Original initial policy 2 | 11 | 1 |
| Constant down | 11 | 1 |
| No-op | 12 | 2 |
| Constant up | 10 | 0 |
| Random1 | 10 | 0 |
| Random10 | 10 | 0 |

All nine cohorts had the same ten natural initial-serve wins. Consequently the
two registered win views give the same paired difference, not independent evidence.
The bootstrap resampled 64 shared serves 10,000 times and recomputed the strongest
of eight comparators each draw. Return behavior persists: endpoint/start made
46/86 and 41/84 returns, with 26/54 and 25/54 first returns. The ten-second horizon
limits repeated-return counts; these are not long-rally mastery results.

Five populations completed before the camera cap. Training took **1,735.691 s
(28.93 min)**, below the 40-minute wall limit: **199,853 transition cameras =
165,212 used fitness + 20,588 unused incomplete-population fitness + 14,053 anchors**.
The unused rows are 159 completed rollouts from g05; 33 jobs were never started.
No partial population updated the policy. Theta5 is the last-complete endpoint;
the unused draw is skipped and the retained next global index is 6. No failures
were recorded. The generic starting return procedure had previously observed
199,676 training cameras, including 178,890 used fitness rows.

Reward variation existed: the five populations had 2/4/4/8/5 nonzero mirrored
contrasts and 65 own-hit wins in 960 completed fitness rollouts. Cached
leave-one-serve-out update cosines had minima .439/.708/.884/.940/.768, with no
zero updates. Pre-update center/start anchors were mixed: 1/12 versus 0/12 at
theta2, then 1/12 versus 4/12 at theta4. Different centers and rotating serves
make these training diagnostics, not a learning curve. Fixed score-free
calibration actions changed on 8.30% of rows; action variation proves neither
competence nor actual visual necessity.

Evaluation/scoring took 657.841 s, **2,393.532 s (39.89 min) overall**. There were
85,498 new evaluation transitions: **285,351 total new transitions, 1,755 additional
initial images and 244,439 encoded images**, with zero visual-learning updates.
All 1,755 observed jobs completed; initial/encoded counts use retained receipts.
Scoring separately reexecuted 85,498 physics rows. Preflight separately used
162 old camera reexecutions and four initial images; its preserved fixture failures
did not change the engine or candidate allocation.

Own verification passed in 34.551 s: all completion pins, ordered records/state/
chains, saved optimizer/RNG/mappings/anchors and seven direct recoveries were exact,
with zero new observations or replay. The independent audit passed in 120.565 s:
all 285,351 physics rows, five manual updates, pending draw, budgets and paired
statistics matched. Audit wall measurements overlapped, after candidates exited;
they are not candidate runtime benchmarks. Launch source **4f974bb**, integrated
unchanged into main as **35cd0e9 + f6447b0**; no source/test differences remain.

Artifacts and both proofs:

- [Run and completion verification](/workspace/m1b-experiments/pong-native-score-search-20261021-r1/)
- [Independent audit](/workspace/m1b-experiments/native-score-parent-audit-20261021/summary.json)
- [Cached training diagnostics](/workspace/m1b-experiments/native-score-training-signal-20261021/)
- [Cumulative ledger](/workspace/m1b-experiments/cumulative-research-ledger-20261016.json)

Next is the reviewed epsilon .02 pendulum intervention with its audited .2 control
reused, unchanged competence gate and fresh matched evaluation. A subsequent Pong
proposal uses actually executed children and fresh training validation, rather
than another small averaged update extension; it remains a separate prospective
allocation. Neither this negative screen nor the earlier proxy successes complete
M1B. Final streams remain untouched.
