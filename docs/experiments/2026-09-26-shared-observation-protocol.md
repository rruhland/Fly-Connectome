# Shared observations during entity overlap

The camera-state audit retained 34/36 identities clean, 32/36 under image noise,
and 23/36 after a camera outage. It preserved all 180 outage location probes but
failed reacquisition identity: errors of 38 or 53.74 pixels match exchanging the
two crossing trajectories. Do not optimize a distance threshold around this.

Test a structural observation-model change: a connected visible region
containing the predicted centers of two already confirmed moving histories is
shared evidence. It cannot uniquely update either one's position/velocity or
create a new entity. Keep their separate motion hypotheses until observations
separate. No object labels, identity truth or collision rule is used; the rule
handles observational overlap only. Birth/association parameters, learning,
receptive fields and all gates remain fixed.

Implement as a separate opt-in tracker/state. First run the same complete camera
audit unchanged, then rerun positional/appearance context regressions if it
passes. Require the original 90% identity, <=2 pixel mean error and <=1 extra
hypothesis gates in every condition. Do not add a chain of overlap thresholds
or per-scene exceptions after the result.
