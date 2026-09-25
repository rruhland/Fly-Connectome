# Next opt-in M1A direction: persistent observation, learned predictive state

**Status:** Architecture investigation proposal, not a production revision. The user has authorized experiments without per-experiment approval; an in-place architecture change still requires approval.

The current local event-correlation code learns useful motion transitions on generic one-/two-pixel-per-frame scenes, including independent and crossing patterns. At a native 120 Hz camera rate, adjacent-frame correlations occur on only 7% of frames. A short local event trace raises code availability to 62%, but next-frame latent prediction remains below persistence. Training the same recurrent weights on fractional and whole-pixel speeds erases some fast-motion ability without learning native timing; adding a parallel decaying hold spreads activity. The [evidence chain](2026-09-25-local-persistence-findings.md) suggests that one event-triggered code is serving two jobs it cannot yet reconcile: what is currently visible between camera events, and what hidden motion will produce at a useful future horizon.

Use two local states in the next **opt-in** experiment:

1. **Persistent observation state:** Integrate each pixel's signed ON/OFF event history locally to retain the latest evidence between events. This is a low-level event-camera primitive, not a fixed object, direction, speed, or Pong feature bank. Keep uncertainty about unseen initial intensity; do not feed rendered frames or privileged simulator state to the learner.
2. **Learned hidden predictive state:** Let a locally plastic recurrent population consume both the persistent observation and learned motion code. Its state can represent a moving pattern through missing evidence without itself emitting an event. Train with delayed local prediction error at a preregistered meaningful horizon (initially four camera frames, roughly one-pixel native motion), and compare against a next-event target. Keep no backpropagation, shared local plasticity, shuffled-credit and frozen controls.

Run three decisive checks in order:

- Verify that the observation integrator tracks generic moving dots/shapes and native camera evidence through quiet frames using event maps only; report coverage and ambiguity, not object accuracy. Stop if it requires Pong-specific priors.
- Train the learned hidden state on generic variable-cadence scenes with four-frame and next-code-event local targets. Require a gain over persistence on held-out generic patterns and zero-shot native Pong, with independent/crossing performance retained. If neither horizon works, reconsider the recurrent state and local eligibility rather than tuning the readout.
- Only if the hidden forecast transfers, revisit a separately calibrated visible-evidence likelihood on matched-prefix uncertain histories. The prior site/field heads' aggregate Brier gains did not satisfy quiet-frame or 50/50 pair calibration, so they cannot simply be reattached.

Before any production promotion, specify how the learned state interfaces with the measured optic-lobe graph and its motion-selective pathways, and seek approval for that concrete revision. M1A remains perception/world-state; reward-modulated motor learning is deferred to M1B.
