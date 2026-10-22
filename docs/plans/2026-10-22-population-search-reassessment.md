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
