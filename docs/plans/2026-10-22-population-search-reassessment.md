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
