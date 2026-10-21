# Lower exploration did not establish further pendulum learning

The complete, independently audited epsilon .02 continuation failed the unchanged
five-second competence screen and did not outperform the reused .2 control.
On 64 fresh common starts, .02 averaged **2.502 s with 4/64 five-second balances**;
.2 averaged **2.758 s with 6/64**. The paired .02-minus-.2 mean effect was
**−0.255 s, 95% CI [−0.654,0.141]**; the cap-fraction effect was −3.125 percentage
points [−12.50,6.25]. These intervals establish neither improvement nor equivalence.

![Fixed epsilon comparison](/workspace/m1b-experiments/pendulum-epsilon-parent-audit-20261021/pendulum-epsilon.png)

The [protocol](2026-10-21-pendulum-low-exploration-protocol.md) changed only epsilon
from .2 to .02. Both started from the predetermined experienced E20k head and owned
visual memory, with new episodes/reset traces and the same environment/exploration
roots. The .2 training/checkpoints were reused exactly from the audited prior
condition. Later states and random-number consumption can diverge; this is one
counterfactual training root, not an independent-seed population comparison.
Vision learned throughout the new dose; native elapsed survival, alpha .001,
lambda .995, gamma .95 per second, transform/cap and physics stayed fixed.

| Frozen endpoint on common 64 starts | Mean survival | Reached 5 s |
|---|---:|---:|
| Shared E20k start | 2.567 s | 10/64 |
| Epsilon .02 at 40k | 2.502 s | 4/64 |
| Reused epsilon .2 at 40k | 2.758 s | 6/64 |
| Strongest simple control, no-op | 1.340 s | 0/64 |

Both endpoints retain gains over simple controls: .02 lift 1.162 s [0.923,1.426],
.2 lift 1.417 s [1.154,1.700]. Neither reaches the registered mean ≥3 s and ≥50%
cap conditions. Incremental mean effects versus E20k are −0.065 s [−0.492,0.351]
and +0.190 s [−0.194,0.583]. Thus previous reward-based B improvement remains,
but neither continuation establishes additional stable balancing skill.

The fixed first-32-state curves were evaluated after training, without selection:

| Total training dose | Shared start | 25k | 30k | 40k |
|---|---:|---:|---:|---:|
| Epsilon .02 mean seconds | 2.272 | 2.206 | 3.192 | 2.618 |
| Epsilon .2 mean seconds | 2.272 | 2.393 | 2.265 | 2.619 |

The .02 midpoint is descriptive; the registered endpoint remains 40k. Its cap
rate at that midpoint was only 6/32, also below the competence condition.

The original Pong readout was retained without A retraining. On the explicitly
reused 64 development serves, .02 kept every first-return outcome, 36→36/64;
all-return change was −4.63 pp [−10.22,0.92]. The reused .2 control had 36→34/64
first returns and an inconclusive all-return effect. This is empirical reused-
cohort retention evidence, without a prospective noninferiority guarantee.

Training completed **20,000 new transitions in 798.417 s (13.31 min)**, including
171 eager initial images and **20,171 new visual-learning images**. Cumulative
visual exposure was 42,453 images. All fixed 5k/10k/20k exports were retained.
Evaluation added 53,768 B and 15,923 A transitions: **89,691 total new transitions,
875 initial images and 77,751 encoded images** across 705 jobs. Reused training,
A controls and baselines add zero new observations. No failures or recovery
reexecutions were recorded. Total allocation time was **1,738.553 s (28.98 min)**,
within the prospective 20/30/50-minute train/evaluation/overall caps.

End-to-end heartbeat median/p95 latency was 35.98/52.33 ms for training,
39.98/61.55 ms for frozen B and 54.66/73.00 ms for A; these include snapshots and
exclude process/recovery boundaries. The 20 ms target remains unmet. Worker
process lifetime peak RSS was 384,676 KiB. Existing profiling does not justify
another speculative optimization before the next learning attempt.

Own verification passed **32.302 s**: all 705 pins/89,691 ordered record-state-
chain rows, seven checkpoint maps and eight actual zero-replay recoveries;
source, inputs, reused raw controls and original wall eligibility were unchanged.
The first verifier attempt accessed an intentionally omitted default attribute;
its script/error are preserved, with zero new/replayed observations and no
candidate change. The independent audit passed **18.535 s**, verifying all raw
physics, TD equations, 875 initials, mappings, reuse and paired statistics.
Scoring separately reexecuted 15,923 A physics rows; the independent audit
reexecuted 89,691 saved rows. No final streams were expanded.

Launch source **c36ebdd**, integrated unchanged as **dc40c48**; coordinator exec
25486/PID 113443 and all workers exited 0. Source is released after both audits.
After verification, all 705 journals were archived losslessly: 4,778,004,480 to
921,031,179 bytes in 40.192 s, with every original byte reconstructed and SHA
verified before removal. No SQLite sidecars remained; the sidecar-only utility's
empty-input assertion was preserved and made no changes. No candidate clock
overlapped storage work. Readouts, memories, source and metadata remain accessible;
the artifact's `JOURNAL-RESTORE.md` documents exact restoration. The old .2 raw
training/A controls remain plain for retained evidence verification.

- [Run, failed-attempt preservation and own verification](/workspace/m1b-experiments/pendulum-low-exploration-20261021-r1/)
- [Independent audit](/workspace/m1b-experiments/pendulum-epsilon-parent-audit-20261021/summary.json)
- [Cumulative ledger](/workspace/m1b-experiments/cumulative-research-ledger-20261016.json)

Next execution is the reviewed Pong native-win experiment selecting exact
executed children with fresh training validation, at fixed .35. A later B proposal
tests native complete-episode survival actor optimization with immutable visual
memory; it remains unallocated and would not establish online-vision competence.
No epsilon sweep, best-midpoint selection or representation-impossibility claim
follows this result. Stable B, score learning and final/B2 evidence remain open.
