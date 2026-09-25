# Hidden-state transfer beyond one moving pattern

**Status:** Opt-in experiment. The visibility-likelihood heads failed calibration and remain separate from this test. Production architecture is unchanged.

Freeze the one-pass locally learned transition from the [hidden-state experiment](2026-09-25-local-hidden-transition-findings.md). On the existing generic robustness suite, evaluate one-frame latent spatial occupancy against the actually observed latent code for independent simultaneous patterns, crossing patterns, abrupt speed changes, and noisy frames. Compare learned transition, shuffled-credit transition, frozen transition, and a fixed persistence forecast that holds the previous observed latent occupancy. Score each family separately and score the speed-change frame and subsequent recovery frames separately. No simulator trajectory or object identity enters inference or learning.

For disappearance scenes only, pair the event-camera sequence with an unoccluded version for evaluation and compare the model's hidden state against the counterfactual latent code during the gap. Report occupancy F1 and active-site ratio. The paired unoccluded code never enters model inference.

Require learned next-state F1 to exceed the persistence baseline by at least 0.10 on independent and crossing scenes, exceed frozen/shuffled on every family, and recover after an abrupt speed change within two visible frames. If these fail, the transition is a controlled single-pattern motion result, not yet a generic reusable world state. Do not promote production architecture on this result alone.
