# Longer-horizon and next-event frozen diagnostics

The trained network's existing predictive current does not beat zero-event
prediction at 30, 50 or 100 ms, or when scored against the next local event in a
bounded future window. This extends the earlier approximately 17 ms timing
investigation. It does **not** test training with a different horizon, a different
local eligibility rule, or information in other internal states. Those hypotheses
remain open; no biological or learning-rule change was made.

## Fixed horizons

Use the unchanged initial and 10,000-frame checkpoints, 500 scripted frames and
development seeds 1101/1102. Graph and configuration hashes match the prior
diagnostics. The lead is measured from the recorded neural tick to the future
target tick; the requested 30 ms becomes 29 neural ticks, or 30.208333 ms. The
50 and 100 ms leads are exactly 48 and 96 ticks. All three use the same 488 target
frames, excluding unavailable history and the artificial initial zero forecast.

If target frame `f` begins at neural tick `f*N`, a lead of `L` ticks reads
`timeline[f*N-L+1]`: timeline entry `i` is recorded after neural tick `i-1`.
Persistence uses the latest camera frame actually observed at that tick,
`floor((f*N-L)/N)`. No future frame or post-event response enters a forecast.

| Lead | Initial MSE | Trained MSE | Zero MSE | Causal frame persistence MSE |
| --- | ---: | ---: | ---: | ---: |
| 30.208 ms | 0.001149044348 | 0.000944900116 | 0.000931584781 | 0.001586832581 |
| 50 ms | 0.001158241210 | 0.000946058722 | 0.000931584781 | 0.002115776775 |
| 100 ms | 0.001161077631 | 0.000946581117 | 0.000931584781 | 0.001831019553 |

The zero score differs from prior all-frame reports because these comparisons
use a common interior window. Training beats initialization and persistence, but
loses to zero at every lead. Supported OFF-event MSE is respectively 1.000606866,
1.000587924 and 1.000756156 after training, versus a zero baseline of 1. Supported
ON-event MSE is slightly below 1 (0.999065920, 0.998768475, 0.999167397).

With a fixed shuffle (seed 421), trained all-sample MSE becomes 0.000945009578,
0.000946085978 and 0.000946715069 respectively. The unshuffled advantage is small
and does not establish successful anticipation. One permutation is descriptive,
not a significance test; no best-performing horizon is selected for acceptance.

## Next meaningful local event

Operational definition: at the end of every camera frame, predict the polarity
of the first subsequent nonzero signed input to the **same sensory neuron** in
the next 12 frames. Use zero if none occurs. A local input is the projected,
possibly aggregated camera event, not an object, collision or simulator label.
All incomplete tail windows are excluded. The current frame cannot confirm its
own prediction. Future observations are used only to construct offline labels.

The 12-frame window is nominally 100 ms. Because a prediction is recorded after
the final neural tick of the decision frame, the last sampled future event is
92.708333 ms later; it is not an exact 100 ms fixed lead. The mean future-event
offset is 5.316030 frames, or approximately 37.009 ms from the recorded tick.

There are 488 decision frames, yielding 2,612,752 neuron/environment/windows:
17,068 contain a future event and 2,595,684 contain none. Windows overlap and can
repeat the same future event, so these counts are not independent observations.
Scoring all windows keeps false alarms visible; supported ON/OFF groups use the
same anatomical sign masks as the preceding audits.

| Predictor | All-window MSE |
| --- | ---: |
| Zero | 0.006532575614 |
| Initial network | 0.006756310803 |
| Trained network | 0.006545652694 |
| Shuffled trained forecast | 0.006546501436 |
| Current-frame event persistence | 0.007624910439 |
| Last observed nonzero polarity | 0.068201650980 |
| Opposite of last observed nonzero polarity | 0.068082236661 |

Last-polarity controls use only events already seen, including the decision
frame; before any event they output zero. They are deliberately simple controls,
not fitted competitors. Their many quiet-window false alarms demonstrate why
event-only polarity accuracy would be insufficient.

The trained next-event readout still loses to zero. Supported ON-event MSE is
0.998883026; supported OFF-event MSE is 1.000336525. Quiet-window MSE is
0.000015869501. Small positive alignment exists, but does not offset prediction
power enough to beat zero across all windows. Conditional or shuffled comparisons
alone cannot supply an M1A claim.

## Scientific consequence

The timing investigations now test the existing frozen current over longer fixed
leads and a bounded event-based target. Neither rescues its predictive performance.
Combined with the C2 restoration experiment, these results discourage both blind
timing reinterpretation and simply restoring firing as fixes.

They do not reject the user's broader hypothesis that the **learning objective**
should operate over longer time or meaningful events. This network was trained
against a one-tick local target, not either diagnostic target. Before approving a
new training target, the next useful evidence would be whether existing signed
edge-local traces have predictive alignment with such future targets, and whether
a causal local mechanism could retain that information without invented edges,
backpropagation, future labels in online updates, or an artificial readout.
That is distinct from claiming the present current already predicts those targets.

No longer training run, parameter search, horizon-specific retraining, topology
change, sign change or optimization experiment occurred. Final seeds 1201-1204
remain unused and M1A remains unmet.

## Reproduction and verification

```powershell
.venv/Scripts/python scripts/audit_long_horizons.py checkpoints/event-v1-combined-rate-initial.pt --output docs/experiments/2026-09-20-long-horizons-initial.json
.venv/Scripts/python scripts/audit_long_horizons.py checkpoints/event-v1-combined-rate-10000.pt --output docs/experiments/2026-09-20-long-horizons-10000.json
```

The new analysis reuses the existing frozen forecast collector through an optional
diagnostic callback. Baseline output and dynamics are unchanged when it is omitted.
JSON bundles include the original strict scores, source hashes, configuration hash,
common target frame indices, group denominators and all controls. Source checkpoint
bytes were verified unchanged.

Tests cover exact earlier-tick and causal-persistence indexing, exclusion of the
current event, incomplete-tail censoring, no-event windows and past-only polarity
controls. The existing collector's ordinary-evaluation comparison also passes.
Fresh review found no blocking issues; the distinction between frame-window size
and exact tick lead is documented above. Full suite: 190 passed, 4 CUDA skips.
