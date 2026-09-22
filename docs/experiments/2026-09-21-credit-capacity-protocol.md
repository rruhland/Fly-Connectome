# Frozen-history capacity comparison

Authorized diagnostic only: no fitted value is written to a model/checkpoint,
no artificial prediction head, no backpropagation or neural training.

Use saved runs/context-audit-v1 states and its exact mixed-trained baseline.
Verify source checksums, +8 target alignment and reconstruction by all 12
existing incoming predictive eligibility columns. Drop trial0 at every tempo
because the observer lacks the neural warmup's eligibility history; require
reconstruction error <=1e-6 thereafter. Preserve every remaining quiet sample.

Compare shared magnitudes with two nonnegative magnitudes per existing edge,
selected by forecast-time sensory state >0 versus <=0. No intercept, future
label, tempo, phase, or other-neuron trace is an input. Magnitudes bounded by
the existing maximum_weight=10. The diagnostic selects contexts at forecast;
it does not specify physical transmission or prove a viable biological gate.

Two fixed least-squares objectives: ordinary per-frame MSE and equal total
ON/OFF/quiet category weight. Fit on trials1-14 of dwells2/4/6 only. Retain
trials15-19 unused; test20-29 at these tempos and all29remaining trials of
unseen3. No hyperparameter search or held-out selection.

Report MSE, ON/OFF anticipation/error, quiet alarms at0.1, clipped prediction
counts and optimizer status. Compare recorded baseline, zero and persistence.
Fit the raw linear signal; score through the existing [-1,1] encoding. This is
an optimistic frozen-history capacity probe, not a universal impossibility
bound on the recurrent/clipped system: changing physical weights changes
spikes/histories, and a least-squares optimum need not optimize every threshold.
If the simple fit fails, do not claim every possible coefficient choice fails.

Test nested context design, nonnegative bounded recovery and score semantics
before fitting real data. Keep raw coefficients/provenance and findings saved.

Exploratory follow-up after the four least-squares fits: directly test finite-
history feasibility of both polarity means >=0.1 and <=5% quiet alarms at each
training tempo. Use exact [-1,1] clipping for event means. Relax the unflagged
quiet boundary to <=0.100001 (actual alarm is >=0.1), so infeasibility also
excludes the stricter original requirement. Keep the same bounds and features.
Use a mixed-integer feasibility solver with15s per case; timeouts are inconclusive.
No held-out labels participate. After joint shared/split infeasibility, test
split magnitudes independently at2/4/6 to isolate within-tempo conflict. These
are optimistic diagnostic coefficients, never local online learning outcomes.

Verification correction: the installed mixed-integer solver's presolve falsely
rejected an independently checked known-feasible no-quiet-cap control. Those
initial results were discarded and preserved only as rejected raw output.
Added the actual multiscale feature matrix as a regression fixture; reproduced
the failure, disabled presolve, reran all cases and validated the feasible
control. Use only the presolve=false outputs in final conclusions. Nine new
probe/solver tests and36existing relevant tests pass (45total).
