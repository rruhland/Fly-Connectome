# Matched camera-frame supervision experiment

User-authorized comparison with temporal-visual-v1: retain eight neural ticks per
camera frame, but replace eight one-tick visual targets with one forecast scored
at the next camera observation, eight ticks later. This is a separately versioned
experimental learning schedule, `frame-horizon-v1`, not a production default.

At neural ticks 0,8,16,..., first confirm the previous boundary's prediction using
the current local observation. Apply the existing equation
`eta * (observation_now - prediction_at_issue) * eligibility_at_issue` to the
same measured visual edges. After processing current arrivals, retain the new
forecast and the signed causal eligibility present at issuance. No future
observation or future arrival enters this snapshot. The final unconfirmed
forecast is discarded. Intermediate ticks advance every neural state, live
eligibility, behavioral R-STDP and homeostasis, without visual error updates.

This schedule applies to all visual predictive edges in this cropped experiment.
Sensory cells use camera increments; other cells retain their local feedforward
arrival observation, sampled at the boundary. It does not aggregate internal
arrivals over the interval. The eight-tick cadence is externally synchronized to
the camera for this diagnostic; it is not claimed to be a biological timing
mechanism. Private retained local state is the bounded change being tested.

No changes to topology, signs, initial weights, neuronal dynamics, input,
prediction encoding, learning rate, synchronization cadence, bounds, homeostasis,
behavioral learning, or no-backprop constraints. No event-only gating: quiet
camera observations are still supervised. The existing per-tick eligibility
diagnostic records pre-observe live traces: the post-observe snapshot at issue
tick `t` is stored in `eligibility[t+1]`, not `eligibility[t]`. Actual update sums
use the retained issue-time trace. Production checkpoint/resume does not support
this experimental class.

Use exactly the prior 250-neuron/1664-edge crop, 200 training trials (seed 9022),
50 frozen evaluation trials (9023), and paired frozen precursor/blank preflight
(9021). Preserve the original scoring, including quiet frames, first-cycle cue
exclusion, ON/OFF metrics and fixed pass gate. Compare to the saved tick-v1 run;
verify the untrained preflight and frozen baseline arrays match exactly.

Run with the project Python:

```
scripts/temporal_visual.py --visual-schedule frame-horizon-v1 --output runs/temporal-frame-horizon-v1
```

Tests must show no visual update on the seven intermediate ticks; tick eight
must use the original prediction and eligibility despite intervening arrivals;
quiet confirmation must penalize a nonzero forecast; behavioral updates and live
traces must match the old rule. Run relevant tests before the experiment, then
the full suite before committing. Wall-clock experiment budget remains 600 s.

If the fixed gate fails, report the effect size and remaining failure directly.
Do not extend training or launch full Pong merely to search for a positive result.
Use the matched comparison to decide which architectural limitation warrants
the next bounded proposal.
