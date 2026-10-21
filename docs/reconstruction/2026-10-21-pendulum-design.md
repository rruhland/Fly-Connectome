# Distinct visual survival task — static preparation, no allocation

No pendulum roots, calibration, physics/vision tests or learning runs are allocated
yet. The live fixed-policy online-vision study and both audits come first. These
files add a small environment/session, not a new visual representation or learner.

Use an unstable upright one-DOF pendulum, with three torques −1/0/+1, 120 Hz RK4
physics and the existing 50 Hz 2/3-tick camera cadence. Ordinary 64×64 binary pixels
show a fixed pivot, a one-pixel 24-pixel rod and radius-two bob. No angle, velocity,
controlled-object identity, balancing error or teacher action enters the policy.
Dynamics are θ¨=8 sinθ+12u−.1θ̇. Initial signed magnitude is uniform over
[.03,.12] radians and angular velocity independently uniform over [−.2,.2].
Failure at |θ|≥π/3 prevents a trivial hanging pose. Reward is actual elapsed
survival time, including a partial final camera interval. Failure is terminal;
the five-second horizon is a truncation of continuing balance, bootstrapped from
the actual final image before reset, with explicit gamma .95 per physical second.

The new policy head starts from zero. Existing true-online SARSA and its audited
camera-rate update/reset logic are reused; existing vision learns on every image
and persists across episodes. Scene reset clears traces/tracks/pending scene credit
but retains learned visual banks, private RNG and calibration. Eager reset observes
and learns the next initial image inside an ending transition, even if the training
dose stops there. Report that observation and unused pending action explicitly;
the retained scene-reset snapshot includes all learned updates, not that scene.

Any B comparison uses one common score-free random-action cache generated from
the same generic prior, with its frozen/plastic mode declared before collection.
Fit the common generic DCT/support transform once from that cache, without reward,
teacher labels or selective learned calibration injection. Preserve the original
Pong head/transform and separately count B calibration, B-only reward experience,
and total prior experience. Fix alpha only after norm/stability preflight; the
current .001/.995/.2 constructor values are unallocated hypotheses, not a sweep.

Before candidate learning, test RK4 against an independent high-accuracy solver,
reward-time conservation, failure/truncation/render/reset boundaries, and actual
encoded feature responsiveness to subpixel tilt/motion/history. Visibility evidence
must not assume that angular velocity is observed or fit an oracle readout. Then
register one bounded first learner condition, held-out initial-state cohort and
matched owninitial/constant/no-op/random controls. Decide native-survival acceptance
and any necessary single alternative prospectively; no early checkpoint selection.

For A→B→A, probe retained Pong skill immediately using B-updated vision before any
A retraining. New B heads do not demonstrate policy transfer. Fixed-W A degradation
would indicate compatibility/nonstationarity, not intrinsic vision inferiority.
Persistent-stream episodes are not independent samples. Retention/noninferiority
criteria must be set before data; CI crossing zero alone proves neither.

Snapshot source signatures include all control files. An explicit bridge therefore
pins ancestor snapshot bytes/source and the complete reviewed control-only change
map. It rejects any visual/Pong code change and preserves the full visual content
hash, calibration, private RNG and cumulative learning count in an owned copy.
There is no silent source relaxation or replacement with a default visual model.
