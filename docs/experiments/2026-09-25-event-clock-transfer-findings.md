# Event-indexed execution does not rescue the frozen native-rate model

**Status:** Opt-in timing test under the [event-clock protocol](2026-09-25-event-clock-transfer-protocol.md), with [results](2026-09-25-event-clock-transfer-results.json). No production change.

The same four native-rate Pong event-camera streams were converted to the four-frame local-trace code. The frozen sensory encoder processed every physical camera frame in both arms, so the scored target latent codes were identical. One arm advanced the recurrent transition every frame; the other preserved recurrence on empty-code frames and advanced only on a nonempty code event. Both were scored against the *next nonempty code event*, not a next-camera-frame target. No Pong state or labels entered the model.

Across 287 scored code events and 509 target spatial sites, the learned transition's event-indexed occupancy F1 was **0.013**. Advancing every frame scored **0.035** on those same event targets; holding the previous observed code event scored **0.081**. Physical intervals between successive scored code events were 1 frame in 134 cases, 2 frames in 129, and 3 frames in 24. Skipping empty frames changes the model clock but does not yield a useful forecast at this native cadence.

The generic transition is not simply suffering from eight invisible camera ticks or excessive state advancement between rare code events. Stop frozen-clock adjustments. Next train the same local transition rule on generic moving patterns sampled at fractional as well as one-/two-pixel-per-frame speeds, using the local trace input, then evaluate zero-shot on native Pong and check for regression on independent/crossing generic scenes. This is an opt-in learning experiment; production remains unchanged.
