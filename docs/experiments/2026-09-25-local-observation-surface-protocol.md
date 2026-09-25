# Persistent local observation surface

**Status:** First opt-in chunk of the [observation-state and horizon proposal](2026-09-25-observation-state-and-horizon-proposal.md). No production change.

Maintain at each retinal pixel the last ON/OFF value observed from the event camera and an explicit known/unknown mask. An ON event sets known-on; an OFF event sets known-off; a camera-quiet frame retains the state. Untouched pixels stay unknown, so the system does not infer an unobserved initial background. This fixed local operation is a sensory primitive, not the final motion representation, and receives no object, velocity, Pong, or simulator label.

Check generic moving single-pixel and multi-pixel shapes at fractional/whole-pixel timing and native Pong camera streams. Rendered frames may be used **only** to audit the state at pixels whose event history makes them known; they never enter the surface. Report exact known-pixel consistency, known fraction, and retention on camera-quiet frames. The chunk passes only if known values remain consistent with the camera evidence and known state persists through quiet frames without filling unknown pixels. Then test a learned predictive latent on top of this persistent observation with delayed local credit; do not promote the surface alone as M1A.
