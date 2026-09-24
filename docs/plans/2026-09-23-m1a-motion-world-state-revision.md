# Proposed M1A motion and world-state revision

**Status:** Proposed for user review; no production architecture has changed.

## Decision

Use a hybrid, no-backprop M1A visual path. Keep the measured MaleCNS graph and the successful opt-in T5c/d OFF response available as biological reference. Add an explicit retinotopic local event-correlation pathway as the primary small-object motion cue, followed by an event-derived object state and an online local-error transition model. This separates *where/how the object moves* from *what will happen next*. M1B remains reward-based motor learning; M1A may later use a fixed controller only to show that its forecast is actionable.

This revises the original requirement that directly observed L1/L2/L3 neurons learn their own next-frame activations. It also relaxes graph-only computation: the proposed pixel-neighbor temporal correlations are new computational subunits, **not measured MaleCNS chemical edges**. No internal measured edge is rewritten, no transmitter sign is flipped, and no backpropagation or learned dense encoder is introduced.

## Evidence and alternatives

The [front-end comparison](../experiments/2026-09-23-m1a-motion-front-end-findings.md) shows a spatially transferable T5 OFF foothold, but four-way 3×3 dot direction from four T5 subtype rates reached only 7/12 held-out cases, and T4 soma spikes were nearly absent. Keeping graph-only T4/T5 as the sole motion source risks another narrow parameter search. A direct event-state tracker forecasts free flight well, but by itself bypasses the optic-lobe motion computation and currently relies on a Pong-specific paddle crop. The [state/motion follow-up](../experiments/2026-09-23-m1a-state-motion-findings.md) shows that a fixed local event-correlation motif resolves controlled dot direction, supplies sparse Pong direction evidence, and that delayed local visual error can learn a useful wall transition on held-out seeds. A hybrid uses those strengths while keeping each unproven piece explicit.

## Components and data flow

1. **Retinotopic motion map.** A stateful ON/OFF event trace decays over eight camera frames. For each pixel and polarity, directed one- and two-pixel coincidences with new events produce signed horizontal and vertical motion evidence. The implementation must use bounded tensor shifts, keep spatial maps rather than only global sums, and expose a confidence/support mask. It introduces no trainable convolution and no backpropagation. Existing T4/T5 graph activity can be recorded beside this map; it is not overwritten or silently relabeled as the source of the new signal.
2. **Visual object state.** Maintain event-derived occupancy, associate compact moving components through local spatial and temporal continuity, and emit position, velocity, and confidence for the tracked ball. The current interior-column crop is a diagnostic ceiling and must not become the production object selector. Start with a fixed transparent component-association rule; if it fails at paddle overlap or across position, leave the revision opt-in and test a local learned association rule. No simulator coordinates, velocity, collision flags, or action labels enter inference.
3. **Local world state and forecast.** Use the visual object state to issue 8/16/32-frame position forecasts. Free-flight velocity comes from local temporal state. An event-triggered wall feature may use the known image boundary; its gain updates only when the delayed visual observation arrives, via local prediction error times the stored feature. Add paddle-collision features only after the visual association and wall stage pass. Store uncertainty/visibility and abstain when evidence is absent rather than fabricating a confident target. M1A updates use no reward or motor credit.

The interfaces should separate `observe(events) -> motion/occupancy`, `estimate(motion, occupancy) -> visual state`, and `forecast/confirm(visual state) -> prediction/local update`. This makes the source of every forecast inspectable and permits frozen, motion-off, and learning-off comparisons.

## Incremental implementation and gates

1. Add the local motion-map component behind an explicit M1A opt-in switch. Verify exact polarity separation, directional shifts, empty/static behavior, state reset, CPU vectorization, and reproducibility. Re-run the square-dot position/speed screen and Pong direction coverage against the frozen experimental result.
2. Add event-only object association behind the same switch. Compare its ball coverage/localization with the 89.3%/0.0083 diagnostic ceiling on held-out Pong seeds, including paddle approach and reset cases. Do not use fixed paddle x columns as the production selector. The integrated motion/state path must run at least 50 camera frames/s for one CPU environment in a release-mode benchmark; report the complete pipeline timing, not just the motion kernel.
3. Add delayed local-error world-model updates. Require paired-sample forecast error below persistence at 8/16/32 frames; at 16 and 32 frames require held-out wall-reversal error at least 25% below observed constant velocity without more than 5% free-flight degradation. Compare frozen and learned gains, and report paddle reversals separately. Verify that all learned values update causally after the observation, stay bounded, and use no autograd.
4. Run a fixed, nonlearning controller diagnostic using the predicted intercept. This is an M1A actionability check; reward-modulated motor learning is deferred to M1B.

After each chunk, run focused checks and the repository suite, save versioned results, and keep the branch testable. The switch remains opt-in until all M1A gates pass. A failed gate calls for a new evidence-backed proposal rather than silently relaxing the criterion or changing measured graph signs/topology.

## Current limits

The tested local correlation emits x evidence on 45.2% of Pong frames at 98.5% sign accuracy and y evidence on 20.8% at 78.2% accuracy when active. It needs persistent visual state and confidence handling. The tested local wall model learns one scalar gain for a known geometric reflection feature; it has not learned paddle-collision dynamics or general object motion. The proposed object association is not yet demonstrated, and the new local motif is a biological hypothesis rather than a reconstruction of measured T4/T5 dendrites. These are explicit risks to resolve before calling M1A working.
