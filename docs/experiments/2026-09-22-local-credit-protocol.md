# Frozen-history causal local-credit replay

This is a diagnostic for how amplitude credit interacts with context and the
causal timing gate. It does not install an update in the neural model. Keep the
measured 12-edge basis, signs, nonnegative magnitude bounds 0..10, original
initial magnitudes, saved frozen histories and fixed timing reference. No
backpropagation, additional connections, future target at issue time, or
validation/test examples in learning updates.

Train only on the original trials 1..14 of tempos 2,4,6, in recorded source
order. Evaluate trials15..19 for learning-rate selection, trials20..29 for
known-tempo tests, all retained tempo3 trials for unseen-tempo transfer, and
the newer continuous standard/omission neural histories as untouched
challenges. Gate for each row is obtained by chronological causal replay of
the complete original recording, including unscored first cycles. Trials from
different fixed-tempo recordings are *not* claimed to form one neural
trajectory; this is an offline replay over their locally recorded samples.

Compare four local rules at forecast issue/confirmation (+8 neural ticks):

1. Original shared scalar: p=clip(x dot w); update eta*(y-p)*x.
2. Split ungated credit: select one of two nonnegative magnitudes per existing
   edge with sign of the *issued* sensory state; p=clip(x_context dot w);
   update eta*(y-p)*x_context. Evaluate through the unchanged timing gate.
3. Split gate-consistent credit: p=gate*clip(x_context dot w); update
   eta*(y-p)*gate*x_context. Closed-gate surprises update the timing reference
   from the observed event but make no amplitude update in this candidate.
4. Same gate-consistent rule, with event updates weighted by a capped causal
   local quiet/event count ratio. Count histories start at one each, are read
   before the current update, and are advanced only afterward; event gain is
   min(8,max(1,quiet_count/event_count)), quiet gain is one. This is a bounded
   candidate to address class imbalance, not a claim of biological mechanism.

For all rules use the encoded/clipped issued prediction in the error and clamp
updated magnitudes into 0..10. Use one pass over the training rows, then up to
five repeated passes as repeated presentations of the same training trials.
Evaluate after pass1 and pass5. Select eta from .01,.03,.1,.3,1 using only
validation samples after pass5: prefer a model meeting ON/OFF >=.1 and quiet
alarms <=5%, then lower MSE; otherwise maximize the weaker ON/OFF amplitude
subject to the quiet cap, then lower MSE. Never select against tempo3 or the
continuous challenges. Report all candidates and both pass counts, not only
the selected winner. This pass/eta sweep is a diagnostic of optimization,
not an online deployment protocol.

Follow-up nested control added after the first four replay outputs, before
choosing an architecture: repeat the same grid with **shared** edge magnitudes
and gate-consistent credit, both with and without causal event balancing. This
tests whether the gate alone resolves the conflict and whether two magnitudes
per anatomical edge are actually necessary. Report it beside the original
four rules, including negative results.

Measure each rule's training trajectory, validation, known-tempo tests,
unseen3, standard/omission continuous challenges, and initial values.
Retain first-pass prequential scores using each forecast before its own update,
both overall and by training tempo, so post-training test scores are not
mistaken for performance while acquiring weights.
Compare the saved offline split-balanced fit as a capacity upper reference,
not an initialization. A frozen-history replay cannot prove closed-loop neural
learning: updating magnitudes would change future spikes and eligibility.
