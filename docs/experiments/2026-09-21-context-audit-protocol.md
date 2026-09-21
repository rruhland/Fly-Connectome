# Existing local context across tempos

Freeze mixed-tempo weights from multitempo-v1. Collect30 trials per dwell2/3/4/6
using matched blanks seed9070, original collector and +8tick targets. The first
cycle remains excluded and all other quiet frames retained. Fit on trials1-15
of2/4/6, calibrate16-20, test21-30. All30 dwell3 trials are held-out-tempo tests.
No trial/tempo/phase label enters features or neural execution.

Fixed15-neighbor offline probes, training-only standardization, four subsets:
E/I; E/I plus voltage/adaptation/refractory/own spike/posttrace/rate; E/I plus
incoming eligibility/arrival traces/current arrivals; all50 recorded local states.
Incoming histories are distributed synapse-local state, not automatically
accessible as a postsynaptic vector. This diagnostic is not a neural readout.

Calibration maximizes worst ON/OFF event recall across all three calibrated
tempos, subject to <=5% quiet false positives separately at each tempo; ties
use average summed recall then lower worst quiet error then higher threshold.
Report each tempo independently. Also report signed confusion: if event is
called, choose the most frequent nonzero polarity among neighbors (ties ON).
This is separate from binary event detection. Stress-test features with1% of
their training SD, seed9071; shuffled-label all-local control seed9072.

For direct ambiguity inspection, find each test sample's closest E/I training
sample from a different tempo, without looking at future labels. Define close
as standardized RMS E/I distance <=.01 (1% of pooled training feature SD).
Then report how often their future labels differ, whether local-history probes
correctly predict the test sample's signed future, and representative pairs'
E/I and history distances. Labels select disagreements for reporting only.
This matching is descriptive, not independent pairwise hypothesis testing.

Existing-state sufficiency would require high recall of both event signs and
low quiet errors across held-out trials AND tempo, especially on conflicting
E/I matches. Probe failure is not proof that all mechanisms fail. Do not add
traces, change dynamics, train neural weights, tune further probe families,
or deploy thresholds. If context still appears insufficient, discuss a bounded
local temporal-context proposal with explicit biological/efficiency tradeoffs
and remaining uncertainty before implementation.
