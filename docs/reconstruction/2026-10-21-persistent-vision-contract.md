# Optional persistent visual learning: implementation draft

This isolated implementation is preparation, not a registered experiment. No
training roots, observation budget or acquisition claim is introduced here.
The active full/state-only comparison uses a separate immutable checkout.

`PersistentVisionSession` keeps one existing visual model across actual Pong
episodes, with an unchanged fixed linear readout and scaler. `learn_enabled=True`
passes each genuine initial and transition image through the existing visual
learner. Observer weights stay fixed; context associations, short/long dynamics,
their private RNGs and all calibration evidence may change. The full mutable
graph participates in every durable camera hash and checkpoint. Only observer
weights/bias (and a projection if present) use immutable bank references. The
default evaluator and learning-session contracts remain frozen and unchanged.

At an episode reset the existing `reset_state()` clears observer traces, tracker
state, per-slot context endpoints/confidence, entity histories and unfinished
forecast credit/cache. It retains learned context keys/values/metric, dynamics
keys/values/seen/capacity/private generators, branch and ContextBelief marginal
calibration, and endpoint reidentification/residual calibration. Initial images
count separately from transition cameras, including learning on those images.
An explicit episode seed schedule constructs a fresh camera phase each episode;
without it, the inherited continuing physics decision cadence is preserved.

`retained_vision()` deep-copies the entire model, then applies this same scene
reset. The snapshot owns its tensors, storage, RNGs and calibration deques;
subsequent live learning cannot modify it. Its content/source/model/configuration
hashes are checked before every defensive clone. A frozen-snapshot probe pins
all learned memory/calibration as well as the actor and observer. A journal
checkpoint instead preserves the complete live scene and pending credits for
exact continuation, without applying a scene reset.

The first proposed use is fixed-policy closed-loop visual exposure, followed by
fresh frozen before/after snapshot probes and a separate prequential stream with
learning enabled. These are distinct measurements: changed exposure trajectories
are part of the treatment, and a confidence interval crossing zero does not
establish retention or noninferiority. Exact canonical camera replay and joint
policy/vision training are deliberately outside this initial implementation.

Before allocation, deferred tests must verify frozen-default parity, genuine
learned-bank/credit replay across an episode boundary, complete snapshot ownership
and tamper rejection, and no observer/policy/scaler drift. A bounded runtime and
memory preflight must inform a prospective dose/wall limit; no experiment clock
starts merely by importing or constructing this optional class.

## Focused implementation checks, between the forecast arms

Nineteen focused tests passed in 24.43 s. They cover the optional contract and the
existing fresh evaluator, with actual learned short-bank/calibration growth and
pending horizon credits replayed across an episode reset. Snapshot storage/RNG/
calibration independence, pickle roundtrip, frozen continuation and tamper checks
passed. No drift experiment or performance preflight has been allocated.

The first attempt correctly exposed a recovery defect in the new optional path:
Torch restored a mutable context key with local version 1 while its valid cached
variance retained numeric version 0. Changing only that local hint recovered the
original whole-state hash exactly. Optional session/snapshot serialization and
copying now preserve the existing fingerprint's logical cache-validity bit,
retargeting it to the copied tensor's local version. Cached values and graph
aliases remain intact; the digest format and default frozen contracts are
unchanged. Both valid and invalid cache states are covered. The failed source,
checkpoint and initial 143 camera fixtures remain retained externally.

Across both attempts: 703 diagnostic camera transitions, 624 actual visual images,
61 initial visual images, 1,612 physics ticks and 6 replayed transitions (a subset).
These are implementation fixtures, not novel candidate acquisition experience.
Evidence and the documented skipped-perception counting correction are in
`/workspace/m1b-experiments/persistent-vision-preflight-20261021/`.
