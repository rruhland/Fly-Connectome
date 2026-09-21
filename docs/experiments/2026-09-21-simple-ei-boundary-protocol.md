# Simple local E/I boundary diagnostic

Question: can a small explicit current boundary express the timing information
found by the offline neighbor probe? This is diagnostic analysis, not a neural
gate implementation, biological claim, or change to the model/local learning.

Let E be existing excitatory prediction current and H the positive magnitude of
existing inhibitory prediction current. Inspect existing state-plane observations.
Use seed9023 trials1-25 for numeric bounds,26-35 for calibration,36-50 for
retrospective evaluation. Seed9024 has already been inspected and is also
retrospective. Generate new50 frozen trials with blank seed9025 only after
selecting boundaries; do not change candidates based on those results.

Fixed families: two polarity-associated scalar windows on E, H, E+H, H-E,
E*H, min(E,H), or H/(E+H); two rectangles on (E,H), or (E+H,H/(E+H)).
Zero total current has ratio0. All expressions are algebraic functions of the
two existing currents. Each family ORs two windows/rectangles; polarity labels
are used only in fitting, never in detection. No fitted projection or hidden
layer. Scalar windows need4 bounds; rectangles need8.

For each polarity, generate bounds from its training event quantiles:
lower q in {0,.025,.05,.1}, upper1-q independently from the same set.
For a rectangle, the same lower/upper quantile choice applies to both axes.
This gives16 candidate regions per polarity and256 unions per family.
Choose each family's union on calibration by maximizing minimum ON/OFF recall,
then their sum, then minimizing quiet false positives, subject to <=5% quiet
false positives. If no candidate qualifies, report failure without relaxing
the cap. Select the overall candidate before new data using the same ordering,
preferring fewer bounds and earlier listed family for exact ties.

Report all families, exact selected inequalities, event recall by polarity,
quiet phase errors, and separate branch calls on ON/OFF/quiet. The branch
association is not proof of correct signed prediction. Compare the prior fixed
neighbor probe on the new batch. Report selected-boundary sensitivity to1%
training-standard-deviation perturbations of E,H and to all bounds expanded
or contracted by1% of their training feature standard deviations. Perturbation
checks are not parameter fitting. Show new observations against selected regions.

Evidence criterion stays >90% recall in both polarities with <=5% quiet errors;
also inspect near-event errors and blank transients. Success supports a simple
expression mechanism only on this periodic trajectory, not learning the bounds
locally, arbitrary-motion generalization, or M1A. No additional family search
after new evaluation in this experiment.
