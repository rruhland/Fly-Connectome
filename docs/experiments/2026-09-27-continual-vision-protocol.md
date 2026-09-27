# Remaining M1A.5: registered continual visual audit

Test production parameters, not a new memory mechanism. Five seeds. Each segment
has 2,000 camera samples in 100 independently reset 20-sample scenes. Learned
parameters persist; only actual scene resets clear transient observations. A has
constant motion, a five-step speed pattern (1,2,1,3,2), and circular motion with
turn +/- .12 radians/sample. Shared B changes motif/polarity/position/angle while
keeping those dynamics. Changed B changes the speed pattern to (3,1,2,1,2) and
turn magnitude to .28, with familiar motifs. All are generic visible patterns.

Arms: continuous production learner, frozen production baseline, and fresh
production baseline reset at each switch. All start with the same bundled prior;
fresh does not mean an unfair blank observer. Run A then branch to shared B and
changed B, then return to A. Evaluate original A immediately before/after B
without return-A training; then measure adaptation during the return segment.
Evaluation uses fixed independent rendered streams and scoring only. No domain,
shape, trajectory or simulator identity labels enter learning.

Assess every 500 training samples, including sample zero. Predeclare the target
mean NLL <=3.5 over horizons 4/8 with >=95% usable forecast coverage, independent
of the test results. If both arms already satisfy it at zero, acquisition speed
is unidentifiable for that comparison, not evidence of faster learning. Report
all curves, per-horizon scores and paired seeds, not just time-to-threshold.
Forward-transfer screen: >=20% fewer observations than fresh to the same target;
retention screen: return-A mean NLL increase <=.2 and coverage loss <=.05. Keep
calibration separate: learned marginal rank intervals, not joint existence claims.

Cache the causal observer's actual same-ID positions once per stream for paired
learner comparisons. Dynamics do not feed into the tracker. Verify cached-observation
credit matches the production full loop on a real rendered stream before using
this optimization. Credit at the observed endpoint only, using the distribution
issued earlier; never feed truth or propagated positions to learning. Cache scoring
truth separately. This audit tests dynamics retention; context memory needs its own
relation-retention check and is not implicitly covered by fully visible motion.

If this exposes a material failure needing deeper investigation, preserve the
baseline and report what failed/why to the user rather than quietly tuning it.
Compute is reported, not a hard pass/fail budget.
