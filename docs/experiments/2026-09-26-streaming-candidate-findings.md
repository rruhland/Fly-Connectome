# Streaming candidate: preserved strengths and acceptance failures

Status: **experimental, not accepted or promoted**. Production `src/` remains
unchanged. The frame+event sensor revision must be declared before promotion;
event-only noisy sensing has not passed.

## Reusable contextual state

A single global covariance metric cannot retain both previously successful
relations: joint experience gives spatial 16/16 but appearance 10/16 in all four
rotations. The context-dependent covariance successor instead scores **16/16 on
both relations in every rotation**, using one 256-example memory with no task
labels. The integrated streaming state reproduces those results.

Context-dependent selectivity also improves the direct dynamics development
score: eight-frame mean error 1.81 pixels versus the best fixed control's 8.53.
Straight-motion error is .70 versus fixed two-step replay 2.28. These are
development results, not sufficient acceptance evidence.

## Streaming interface and sensor evidence

`streaming_visual_state.py` exposes current observations separately from continued
hypotheses and forecasts. Delayed credit uses actual observed positions only.
Missing camera samples are censored; expired tracks are compacted while public
identities remain stable within a scene. Forecasts issued before a gap can remain
available with explicit origin and evidence age. Histories, calibration buffers,
associative memory, prototypes and pending deadlines are bounded.

The original binary-only front end fails grayscale transfer completely. Pure
contrast normalization recovers motion but is rejected: independent blank-image
noise creates hundreds of tracks and stalls throughput. A robust noise floor and
event/image corroboration restore the simple contrast/illumination/outage tests
to 12/12 identities each. Sensor events must be raw corroborating evidence, not
the learned observer's already-downweighted output; a reproduced regression
exposed that integration error. Established tracks can use visible continuity
through event dropout; the earlier every-sample hard gate is also rejected.

Simple two-entity streaming measurements were roughly 150–325 frames/second;
broader online training processed 3,840 frames in 31–33 seconds. These exclude
measured-graph execution and are not directly comparable under differing host
load. A 600-frame blank image plus false-event audit still produced one false
track at a time (mean .1), so this observation model is not a final noise result.

## Uncertainty and checkpoint review

A balanced matched-prefix fixture yields equal .5 reidentification probability,
identical predicted locations before the split, and Brier .25. This is a correct
base-rate/causality result, **not broad uncertainty calibration**. The probability
means future visual reidentification of a track, not hidden existence. Radius is
an empirical conditional residual quantile, not a guaranteed coverage bound.

Review found the first integrated checkpoint discarded learned calibration;
regeneration now transfers it and checks the round trip. Review also reproduced
silent changes to injected component types/horizons on reload. Version 2 records
and validates both, with a regression test. The original provisional checkpoint
is preserved under `checkpoints/m1a5/provisional-v1-candidate.pt`. Regenerating
training after the raw-event fix gives identical dynamics tensors, so the issue
affected held-out observation coverage, not those training weights.

## Fresh transfer rejects the stored-displacement forecaster

The first 216-scene transfer run changes motif, direction, scale, phase, contrast,
illumination and sensor noise. Eight-frame error rises to 13.10 versus fixed
two-step replay 11.22. The bounded broader-experience comparison trains 192
streams with the same learning constants and 128 prototypes; it improves but
still fails. After the dropout correction, four-frame error is 5.29 versus 6.46
(less than the required 20% gain); eight-frame error is 7.46 versus 10.74.
Straight motion regresses by 1.64 pixels, exceeding the registered one-pixel
tolerance, and observation coverage is 94.5%, below 95%.

An asymmetric-motif scoring issue was also corrected: truth must be the motif's
geometric center, not its arbitrary rendering origin. Symmetric development
scenes are unaffected. Provisional outputs are preserved rather than overwritten
without an audit trail. The corrected broader run still fails its gates.

Close prototype/readout tuning. The registered next comparison is the separate
2x2 [local-generative dynamics and statistical observation experiment](2026-09-26-generative-dynamics-state-protocol.md).
Do not substitute development scores for that acceptance result.

## Measured connectome contribution

The paired T4/T5 input audit verifies unchanged graph weights and source/graph
identity. Horizontal context scores are 16/16 with and without measured features.
Both vertical arms score 0/16 after anisotropic resizing to the fixed 32x64 retina;
that transformation moves the context beyond the original local receptive field,
so it cannot establish absence of useful vertical motion information. The audit
establishes no additional benefit for this task, not that the graph is useless.
It costs 690 seconds for 5,504 samples including setup (about eight samples/second).
Do not imply the faster standalone visual state includes this graph cost, and do
not add it to the critical path merely to claim connectome involvement.

Latest completed full suite before the 2x2 additions: 683 passed, four skipped.
The new generative/statistical components passed focused tests; full verification
and their experiment are ongoing. The milestone is not complete.

## Subsequent bounded comparisons

The full 2x2 comparison is complete. Statistical observation is the retained
component: 36/36 identities in clean/noisy/outage scenes, zero final center error,
180/180 outage location hits, and zero confirmed tracks in 600 blank frames with
both Gaussian image noise and false events. Both forecasting principles clear
aggregate/coverage gates but still fail the straight-motion family gate.

Learned forecast competence reduces eight-frame error to 4.53, versus 5.03 with
no gate and 5.74 with shuffled credit, but straight-motion error remains 3.64
versus 1.41 for mean recent displacement. The entity-local online autoregressive
alternative improves four-frame error (2.21 versus 3.15) but not eight-frame
error enough (7.72 versus 7.90). Neither is promoted. Stop point-readout variants.

Next reassess the **future spatial belief** objective explicitly: retain alternative
future outcomes and score a learned distribution against strong calibrated
controls. This is a new investigation, not permission to erase the failed point
gate or declare M1A.5 complete. A production proposal must disclose any revised
acceptance contract. Latest completed full suite: 686 passed, four skipped;
the entity-local addition and relevant focused tests subsequently passed.
