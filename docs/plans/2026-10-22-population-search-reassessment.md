# Reassess the learner before another architecture intervention

Ongoing user authorization applies. No new major architecture is proposed.
The [direct-serve actor trial](../reconstruction/2026-10-22-scaffold-reassessment.md)
failed incremental learning while preserving ordinary-game performance. Astra
challenged whether the recent local actor trials had actually exercised policy
changes comparable to the original successful population learner. Sol independently
measured the historical steps and challenged competing explanations.

## Evidence changing the next research choice

On the SAME1024-row legal calibration cache and exactT, Sol compared saved heads
from both successful acquisition lineages, failed population refinement and local
actor trials. No observations, journal reads or replay;0.506s. No cache rows hit
the norm cap, so historical and current function-RMS metrics agree.

Original successful population steps were approximately.095 RMS; acquisition-phase
net displacements were.222/.234 and later full-rally phase displacements.183/.180.
The latest direct-serve actor's62 nonzero updates had median.000364 and total net
change.003796. Its median step was roughly260times smaller; all carried local
actor stages together moved only.00984 from the qualified original population head.

The failed population refinement is relevant counterevidence but used a different
scale:4updates, median.01464/net.02858, perturbation.06017 and cap.01504, compared
with original perturbation.25 and cap.10. It did not test the original successful
search regime. Therefore neither that failure nor the recent actor failures rules
out useful progress with the demonstrated population mechanism.

A correction to an earlier diagnostic interpretation: successful population
net/root-sum-square displacement ratios were also around.96–1.05. A ratio near one
cannot by itself reject useful learning directions or establish a zero gradient.
Small net movement is a measurement, not proof that increasing it will help.

Artifacts: `/workspace/m1b-experiments/pong-search-reassessment-20261022/`
contains `compare_dose.py` and `dose-comparison.json`.

## Selected proposal, not yet launched

Return to the demonstrated greedy population update with its original perturbation
and update rule, on ordinary initial-leftward training starts. Preserve original
ARtheta112, visual representation and physics. A concrete bounded protocol and
cost estimate are being prepared; no automatic parameter sweep or best-checkpoint
selection. Fresh direct-serve acquisition and ordinary-game retention remain
separate. Independent replication required after positive learning.

This changes more than update magnitude: coherent parameter exploration evaluates
greedy controllers directly, whereas the local actor optimizes expected return
under noisy action sampling and is evaluated without that noise. A gain would not
isolate step size from objective alignment or exploration. Larger perturbations
could also erase prior skill or average across narrow good policies. Compare this
proposal against larger local updates, preconditioning and missing-information
explanations before final launch selection; none is ruled out by inspection.

Astra read Mania/Guy/Recht2018 (intro,2.1,3/Algorithm2),
https://arxiv.org/pdf/1803.07055. It motivates a fair parameter-search baseline and
attention to seed variability, not a guarantee on this task. Existing implementation
already resembles that family; no wholesale algorithm import, supervised labels
or backpropagation is needed.

## Selected protocol for implementation and independent stage review

80,000 new training camera transitions,900s training;600s evaluation;1500s overall
from an immutable original clock. Max12 complete populations; at least5 complete
populations for eligible assessment, with actual nonzero updates/movement reported
separately. Last complete endpoint only. All partial observations and initial
images count; no result-dependent extension.

Use the original phase0 rule without retuning:8 centered Gaussian directions,
normalized to unit function RMS on the fixed generic cache; paired perturbations
plus/minus.25;12 fresh ordinary initial-leftward starts shared across all16greedy
candidates.192rollouts/population, K1, strong.65opponent, first-hit10s episodes.
Fitness is mean binary first-own-hit success. Original top4directions by maximum
sign fitness, stable ties, selected-return standard-deviation normalization and
.10 aggregate function-RMS cap. Original zero-update behavior for zero spread or
zero contrasts; no invented fallback. Keep exact original ARtheta112/T and vision.

At46–47cameras/rollout, one population costs roughly9,000transitions plus192initial
images;80k affords approximately8–9 populations. The original successful first-hit
phases completed5/6. Correction to Astra's earlier rough estimate:40k would afford
only about4populations, not9–18. Timing is plausible from retained greedy throughput
but short-episode overhead remains uncertain. A budget-limited partial dose is not
a learning failure and does not authorize extending the run.

Endpoint and carried control:64freshLEFT first-contact episodes plus64separate
freshunfiltered20s games,256jobs total. PrimaryLEFT lift>=10pp and paired95%lower>0;
absoluteLEFT80% separate. Full first-return AND all-return retention each require
paired95%lower>-5pp. Fixed10k bootstrap, separate cohorts, DEV blind until exit.
No full-task80% claim from restricted success; independent replication before
promotion. No concurrent training control or pure-dose causal claim.

Astra selected after Sol's measured comparison and independent challenge. Larger
local-actor steps remain a legitimate deferred alternative; tiny carried movement
leaves that hypothesis alive, but enlarging it would retain uncertain noisy-action
to greedy-policy transfer. ExistingT already ridge-whitens generic moments, so
another raw feature preconditioner is not an obviously missing ingredient. Input
information loss remains possible. None of these alternatives is ruled out.

Reuse the already checked greedy first-contact evaluator and explicit interval16
research execution. No architecture or environment-source change is needed.
Focused coordinator validation covers complete/partial populations, paired fitness,
optimizer RNG/checkpoint progression, seed isolation and budget accounting. Frozen
source/driver/protocol and independent stage approval are required before launch.


## Independent stage accepted; allocation live

Source `/workspace/Fly-Connectome-m1b-pong-leftward-population`, commit
68d15a7ed658347da9f6f5a7b2c4d5b4da644616; core/worker/math unchanged and clean.
Allocation `/workspace/m1b-experiments/pong-leftward-population-20261022-r1`.
Driver8200c75ff4529a76c6827e24e64aaf370c13ac8b0203dffe6b60be09705bd853;
protocolaafc37de93dae44c1196444f769fecdef005632ed969624d7139f91ba7a9b236;
prelaunch manifestc7bee37ddbe1d220f90cf768b96d5b7e15f0e5608b8c7abc4c4a335bdbf77656.

Sol7focused checks PASS1.51s, zero observations; unchanged greedy execution proof
reused, no redundant replay. Stage1.772s plus own verification41ms. Parent reviewed
310-line coordinator, independently regenerated conditional seed schedule including
227rejections, all8normalized Gaussian directions and192candidate cells/mapped heads,
source/file pins and exact carried policy. Parent check191ms, zero observations.
272fresh canonical environment IDs,35,899prior IDs excluded, no final stream use.

Launch exec71247, original started_wall_ns1792690225066585708. Source/driver/protocol
frozen through both postexit checks; DEV blind.80k/900s training,600s evaluation,
1500s overall, no extension. Reused Sol supervises; Astra independently reconstructs
final updates and learning/retention evidence. Completed prior direct-serve actor
journals archived3.292GB to.481GB in34.548s after both checks, zero observations or
replay and all bytes reconstruct-verified, before this launch.


## Checked first result: substantial acquisition, preservation not established

Both checks PASS; frozen source/driver released.9complete populations/9nonzero
updates.79,829train (78,432used+1,397unusedpartial);38,851eval (5,716direct+33,135full),
2,014initial images.936.362s total=662.624train+273.738eval. No recovery/extension.

| Test | Carried ARtheta112 | Endpoint | Paired change,95% CI |
|---|---:|---:|---|
| Direct first returns |29/64|56/64|+42.1875pp,[+29.6875,+54.6875]|
| Ordinary first returns |31/64|39/64|+12.5pp,[-4.6875,+29.6875]|
| Ordinary all returns |68/128|48/112|-10.2679pp,[-21.1275,+2.5298]|

**Primary learning and provisional direct-serve80% competence PASS.** Ordinary
first-return retention passes, but all-return retention fails. Games with at least
2/3/4returns fall15/9/7→6/2/1. The all-return interval does not prove a population
mean decline at95%, but clearly does not establish preservation. Do not erase the
strong acquisition result or promote this as a combined competent learner.

310/864 direction/serve pairs supplied contrast. Steps.0769–.10, net RMS.311016,
328generic-cache greedy changes; precontact actions changed in64/64 of both cohorts.
This exercised substantial behavior change. The frozen visual representation and
linear class can support high direct interception. It does not establish visual
sufficiency for later rallies or isolate which population-mechanism change helped.

Sol's source/RNG/direction/update/greedy-action/cutoff verification PASS14.054s;
Astra independently reconstructed all9updates, raw counts/rewards and paired outcome
gates in2.551s. Zero new observations/replay in both checks. Artifacts:
`/workspace/m1b-experiments/pong-mechanism-review-20261022/population-parent-result-check.json`
and `population-result.png/svg`; allocation `completion-verification.json`.

Storage overlap: old completed Pendulum ON/OFF706journals archived4.6076GB→.9024GB,
58.005s plus.293s empty-sidecar preservation, during training. Both dependent epsilon
reused-control checks had already passed, satisfying the old plain-file retention
condition. All bytes reconstruct-verified, no observations/replay or clock shift.

## Selected independent replication before integration

Exact same source, algorithm, constants,80k/900+600/1500budget,max12/min5populations,
cohorts and gates, from original ARtheta112. Change only prospectively declared
independent seed roots and allocation path. Exclude all r1 accepted/rejected/unused
schedules and canonical aliases. Pin r1manifest/results and both checks in immutable
replication provenance. Reuse verified coordinator and checks; no new architecture,
mixture, rate tuning, redundant tests or replay. Independent stage review before launch.

Predeclare before r2outcomes: if acquisition replicates, r1lastcomplete theta9 stays
the designated direct-interception scaffold candidate regardless of which seed
scores better. R2 tests reproducibility, not endpoint selection. Neither is promoted
as a combined learner while preservation remains unproved. The original full-rally
population stage and mixed-task training are integration contenders after replication;
no integration allocation selected yet. The ultimate continuous-learning objective
and major-architecture review boundary remain unchanged.


## Exact replication staged and launched

Root `/workspace/m1b-experiments/pong-leftward-population-20261022-r2`, same source
68d15a7. Driverdiff is exactly oneROOTSline; fixturebyte-identical, no new tests or
replays. Training3158167291,LEFTDEV1740069383,FULLDEV4021736819,optimizer2185392077,
bootstrap947630153; sameTAG29128.272freshcanonicalstarts,217rejectedcandidates,
36,404priorIDs excluded including all r1unused/rejected schedules.

Driver0e57ce62f9f00983572b09cea923e4919fe1a9146d8f5e8872200f646c6a36fa;
protocolc6876e0c6c25a2b814734577e779dcd4fc63204a657d6926d54d15f3bf65731d;
prelaunch manifest2192ad19f7fefd61f4eab6672453708850ed7594b13ba04c78c777b714810f0a;
replication-lineage8ac553730745d5ddf2aef97acd1f04631741fa5ed215f3b21e6a3a58a6226199.
Provenance explicitly designates r1theta9 regardless of r2score and pins both r1checks.
Solstage PASS37ms after1.855s prep; Astra independently regenerated seeds,
8directions/all192candidate cells and checked the exact driverdiff/pins in202ms.
Zero candidate observations at staging.

R1journals archived after bothchecks/rawread completion:2,014DB5.8557GB→.8124GB,
65.018s plus1.622s empty-sidecar preservation. All bytes reconstruct-verified,
zero observations/replay; heads/metadata remain plain. Archive completed BEFORE r2.

R2 launch exec65584, originalstarted_wall_ns1792691458745935191. Same80k/900train+
600eval/1500overall, noextensions. Source/driver/protocol freeze through bothchecks;
DEVblind until exit. Reused Sol supervises; Astra's independent checker is prepared.

## R1 retention interpretation while r2 remains blind

Astra's metadata-only posthoc split of R1 ordinary games: among34direct openings,
first returns13→30; among30opponent-first openings,18→9. Later returns/opportunities
were13/25→8/38 for direct openings and24/39→1/10 for opponent-first openings.
These conditional later exposures differ with policy; this is descriptive analysis,
not a new gate or causal estimate. Cost0.3ms after reading plain metadata, no new
observations, raw journals or replay. Artifact
`/workspace/m1b-experiments/pong-mechanism-review-20261022/population-r1-direction-retention.json`.
The direct skill generalizes to the direct subset inside ordinary games; the policy
specializes away from opponent-return openings and later play. First-hit cutoff
provides no postcontact training experience, motivating full-rally training as an
integration contender after replication. It does not prove that one fixed policy
can learn all situations or that architecture changes will be unnecessary.
No r2outcomes read, protocol changes or new training allocation.

## Checked replication: acquisition repeats, combined skill remains unproved

R2 COMPLETE, both checks PASS and source freeze released. Nine complete/nonzero
population updates; 79,825 training transitions, all used; 37,687 evaluation
transitions (5,876 direct + 31,811 ordinary), 1,984 initial images. Wall 907.916s
=631.124s training +276.792s evaluation, no extension or recovery.

| Test | Carried ARtheta112 | R2 endpoint | Paired change, 95% CI |
|---|---:|---:|---|
| Direct first returns |27/64|49/64|+34.375pp,[+18.75,+50]|
| Ordinary first returns |34/64|39/64|+7.8125pp,[-10.9375,+26.5625]|
| Ordinary all returns |59/122|47/111|-6.0183pp,[-16.8198,+5.5303]|

**Learning replicates; the 80% absolute competence threshold does not** (76.5625%).
Both ordinary retention tests fail to establish preservation. Games with at least
2/3/4 returns fall 14/6/4 to 8/0/0. Descriptive adverse rally change is consistent
with r1; intervals do not establish population mean harm at 95%.
327/864 direction/serve pairs contrast; net function RMS .319505, 326 cache action
changes. All 64 games per cohort change precontact actions, median first divergence 0s.

Sol's verification PASS12.682s; Astra independently reconstructed all nine updates
and fresh raw outcome counts/paired bootstrap in 1.761s. Zero new observations or
physics/vision replay in either check. Parent artifact:
`/workspace/m1b-experiments/pong-mechanism-review-20261022/population-r2-parent-result-check.json`.

As predeclared, r1 theta9 is the direct-interception scaffold candidate, independent
of which seed performs better. Original ARtheta112 remains the qualified global
reference; neither specialized endpoint is a combined learner. These trials refute
the claim that the current visual/linear class cannot learn strong direct interception,
but do not isolate exploration scale from objective alignment or establish autonomous
continuous learning.

## Selected next research question: preserve interception while learning rallies

Restore the original full-rally population stage from designated r1 theta9, using
ordinary mixed starts and full-episode own-hit count fitness. No new architecture,
backpropagation or privileged observations. First-contact training omitted later
states entirely; merely mixing opening directions would still omit those states.
Sol independently challenges/stages a bounded protocol; Astra reviews before launch.
Fresh evaluation must compare the endpoint to both its specialized start and the
original global ARtheta112. Use a per-game rally measure to avoid interpreting a
policy-dependent opportunities denominator as the sole progress test. Direct-skill
preservation is a separate required gate. No new allocation launched yet.

### Integration protocol decision before staging

Original full-rally rule: 8 directions, +/- .25 perturbations, top 4, .10 update cap;
6 shared fresh unfiltered starts per population; greedy K1, opponent .65, 10s games,
mean total own-hit fitness. Budget80k/900s train, max12/min3 complete populations;
600s eval/1500s overall, no extension. Earlier full-rally stage cost roughly20k
cameras/population, so expect3–5 updates; report actual dose separately from gates.

Three heads (DIRECT=r1theta9, GLOBAL=original ARtheta112, ENDPOINT) evaluated on64
fresh direct first-contact starts and64 separate ordinary20s games:384jobs,
288k maximum evaluation transitions. Fixed last complete endpoint, no checkpoint
selection or DEV peeking. Primary: fraction of ordinary games with >=2 returns;
ENDPOINT minus DIRECT >=10pp with paired95%lower>0. Combined progress additionally
requires the same against GLOBAL. Preservation: direct first returns versus DIRECT
lower>−5pp; ordinary first/all returns versus GLOBAL each lower>−5pp. Direct gain
versus GLOBAL >=10pp with lower>0 is separate. Mean own hits/game and >=3/4 returns
are secondary. Combined carry requires all gates; any positive needs replication.

Interpret outcomes separately: rally gain/direct loss means unresolved retention;
recovery over DIRECT but not GLOBAL means recovery, not overall progress; no rally
gain despite substantive changes weighs state coverage/generalization, not an
automatic rate/noise sweep. Natural mixed starts rehearse direct situations but
cannot guarantee preservation. No new architecture or training allocation yet.

Outside-source check: Astra read Kaplanis, Shanahan & Clopath (2019), introduction
and section3, [Policy Consolidation for Continual Reinforcement Learning](https://proceedings.mlr.press/v97/kaplanis19a/kaplanis19a.pdf).
Their discussion identifies within-task experience distribution changes and the
distinction between parameter memory and behavioral memory. This supports checking
retained behavior rather than small weight movement. Their PPO/distillation cascade
is not adopted, and the paper does not establish our failure's cause. First test
full-task exposure with the already demonstrated population rule before adding a
consolidation mechanism. The paired outcome plot is
`/workspace/m1b-experiments/pong-mechanism-review-20261022/population-replication.png`.

R2 storage after both checks:1,984 journals5.767GB→.802GB,62.918s, all bytes
reconstruct-verified, zero observations/replay. Parent sequencing error overlapped
cold archive start with sidecar script's final post-removal DB hash check; that check
raised missing-file because compression had removed a DB. Pre-removal sidecar and
main-DB verification plus original hashes were already durable. Separate recovery
reconstructed all1,984 affected databases against those original hashes and rechecked
all3,968 archived sidecar members, PASS14.370s. No experimental/source changes or
lost data; recovery artifact `sidecar-recovery-verification.json`. Next training
has not launched, so no overlap with live training.
