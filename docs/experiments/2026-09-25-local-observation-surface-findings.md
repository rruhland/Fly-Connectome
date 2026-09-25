# Local event integration supplies persistent evidence without inventing unseen pixels

**Status:** Opt-in sensory-primitive audit under the [registered protocol](2026-09-25-local-observation-surface-protocol.md), with [results](2026-09-25-local-observation-surface-results.json). Production unchanged.

A local two-channel surface retains the last observed ON/OFF value at each pixel and marks all untouched pixels unknown. It receives event-camera maps only. Rendered scene frames were used solely to audit pixels with known event history. No shape, object, velocity, Pong, or simulator label enters the surface.

Across 128 generic moving-pattern episodes (unseen shapes/dot, four directions, fractional and whole-pixel speeds, both contrasts), all **68,232/68,232 known-pixel observations** matched the rendered audit frames. All **704/704 camera-quiet scored frames** retained nonempty known state. Across four native-rate Pong streams, all **27,871/27,871 known-pixel observations** matched and all **177/177 camera-quiet frames** retained known state. In these episodes, all currently changed foreground pixels had an event history and were therefore known.

The surface does **not** reconstruct the whole image: average known fractions were only **1.86%** of generic pixels and **2.93%** of Pong pixels. Untouched background remains explicitly unknown. Exact known-pixel consistency follows from the binary event-camera semantics; it is an implementation and information-flow check, not evidence that visual motion or future events have been learned. It does show that meaningful local visual evidence remains available through quiet camera frames, unlike the adjacent-frame coincidence code that was active on only 7.1% of native frames.

The next substantive test is to let a locally plastic latent population consume this persistent evidence alongside local motion correlations, and compare delayed four-frame versus next-code-event prediction on generic variable-cadence scenes and zero-shot native Pong. The observation surface alone is not a production M1A system; no in-place revision has been approved.
