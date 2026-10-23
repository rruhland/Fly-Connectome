# Selection: preserve accumulated interception learning through balanced practice

## Question and alternatives

The opponent-return study independently verified a39.06pp acquisition gain and
35.94pp ordinary first-return gain versus the original global reference. Its direct
retention gate failed, but the interval included zero; definite damage is unproven.
This is strong acquisition with incomplete joint evidence, not a null experiment.

Astra challenged spending the next24.5minutes on exact opponent-return replication:
that would strengthen acquisition evidence but leave integration untested. Astra
initially favored balanced practice from the original direct scaffold, which could
independently demonstrate acquisition under a different training distribution.
Sol challenged discarding the newly acquired opponent-return updates. Carrying them
lets us test continued improvement while preserving accumulated behavior, consistent
with the user's scaffold progression. Astra accepted that argument and added ordinary
game retention against the immediate start, since comparison only with the weaker
original global reference could conceal loss of the new ordinary-game gains.

Selected: continue from RIGHT r1theta6 with6fresh direct and6fresh opponent-return
starts per population. Retain first-contact binary fitness and the original
8-direction, mirrored±.25, top4, capped.10 rule. No model or reward change.
The carried head is an experimental candidate, not a qualified global promotion.
Exact opponent-return replication is deferred, not retroactively declared passed.

## What would change our mind

Require new equal-weight phase-averaged learning against the immediate start:
>=10pp and stratified paired95%lower>0, with64fresh starts per phase. Report each
phase separately. Require opponent-return retention versus that start, direct
retention versus the designated direct policy, opponent-return retention versus
the original global reference, ordinary first-return gain versus global, ordinary
all-return retention versus global, and both ordinary first/all retention versus
the immediate start. Noninferiority lower bound must exceed−5pp. The provisional
80%absolute target in each phase and ordinary games remains separate.

A pass supports a useful combined interception scaffold, pending independent
training replication. A balanced mean gain with a failed retention gate is a
tradeoff. Preservation alone is not new learning. A null does not prove incompatible
representations: fewer opponent-return starts per population and changed realized
update counts remain alternative explanations. No automatic rate or mixture sweep.

Keep160k training transitions/1200s,900s evaluation/2100s total, max12/min5complete
populations; last complete endpoint only. Three heads per cohort permit all tests
in576evaluation jobs: LEFT endpoint/start/direct; RIGHT endpoint/start/global;
FULL endpoint/start/global. Prospective driver protocol governs exact seeds and
bootstrap definitions; all outcome checks must be fixed before launch.

## Outside evidence and limits

Astra read Rolnick et al., *Experience Replay for Continual Learning*, introduction,
§3 and§4.1: https://arxiv.org/html/1811.11682 . The simultaneous/sequential comparison
motivates testing continued exposure before attributing loss to capacity. Their
method includes off-policy learning and behavioral cloning; neither is adopted.
Sol independently read the introduction and§4.1 and reached the same limited use.
Astra also read Isele and Cosgun's introduction:
https://arxiv.org/pdf/1802.10269 . Its exposure-distribution argument supports keeping
past situations in practice, without establishing a result for our search method.

Fresh native starts in two selected phases are simultaneous practice assistance,
not stored-experience replay or CLEAR, and not an autonomous continual learner.
Phase labels select training starts only; they never enter policy observations.
Every episode's fresh interaction is charged. Approximate phase symmetry does not
justify asserting identical sample or update dose across experiments.

## Independent stage review

Root `/workspace/m1b-experiments/pong-balanced-interception-20261022-r1`.
Same source68d15a7. Nine focused checks passed3.56s; Sol preparation2.2403s
and own stage review48.7ms. Astra independently regenerated seven-root schedules,
336accepted canonical seeds,278rejected draws and38576exclusions, checked exact
carried/reference weights, all192first-population jobs, perturbations and RNG.
Parent stage passed223ms with zero new observations/replay. Driver SHA256
`5e79af3cc9cee2a264b9c7bf16e99680e31e5a78fa85657354729a33c745cf46`;
protocol `da239d4de84e8110a101287e0cb6fff590f85c856f067a0354c35be73ec67113`;
manifest `6757187646f25bca85a08da94eec47c5d667c541f2b0d4f782bb735c58fdbd3f`.
Cleared for one bounded launch; exact receipt recorded separately. All three remain
frozen until both post-exit checks. Parent checker scripts and outputs live under
`/workspace/m1b-experiments/pong-mechanism-review-20261022/`.

## User pause

After launch at2026-10-22T19:39:25.537299559Z (exec9708), the user requested:
“find a good pausing point and stop.” Orchestration stopped initiating work and
requested prompt clean shutdown of the newly launched allocation. The last fully
verified research result remains opponent-return acquisition with unresolved direct
retention. Preserve partial balanced exposure/checkpoints and original clock; do not
label interruption a null or silently restart the clock. HANDOFF/live state record
final process and accounting receipts. Resume only on renewed user instruction.

Shutdown confirmed: exec9708 exited130 after coordinator SIGINT; all workers and
resource tracker exited. Durable receipt reports14879training transitions and161
initial images, zero complete populations/updates/evaluation. This includes291
transitions drained after interrupt but absent from the coordinator cost log.
All partial exposure is unused for learning; theta_at0 remains exact RIGHTtheta6.
Receipt: `/workspace/m1b-experiments/pong-balanced-interception-20261022-r1/pause-receipt.json`.
Parent independently checked no matching training/worker processes remained.

## Bounded resumption on2026-10-23

User renewed orchestration for at most two hours, clarifying “Run for up to two
hours from now.” Session starts06:09:32UTC and stops by08:09UTC, with ten minutes
reserved for closure. No new35-minute allocation after07:24UTC. Reuse Sol6.1XHigh.
The same selected balanced experiment receives a fresh prospective allocation,
new independent roots and exclusions including the interrupted allocation's whole
published schedule. No old driver restart, clock reset or partial-fitness reuse.
The previous14879unused training transitions remain separate paid research cost;
the carried learned state is unchanged RIGHTtheta6. Review results before selecting
any further experiment; no automatic parameter sweep or architecture change.
