# Fixed policy projection implementation and preflight

Prepared in isolated `/workspace/Fly-Connectome-dct-sarsa` from fddda83 while the
stationary curriculum was frozen. Only static checks ran during that allocation;
all executed fixtures below followed its completion. No candidate training has
launched at this implementation checkpoint.

`LearningSession(policy_projection=descriptor)` owns the exact float64 T by
file+array hash, computes T*x then cap256, and retains learner/pending traces in
the output dimension. Raw transport, SARSA rule/defaults and simulator inputs
remain unchanged. Separate policy_scaling is rejected to make cap order explicit.
FrozenCodec anchors embed T, preserving complete per-camera byte hashing and
alias mutation detection. Recovery verifies embedded content/config without
reading the original external path. There are no cache/factorization shortcuts.

The worker's optional projection_from training descriptor uses the existing
durable/warm-session path. Checkpoints export reduced theta, full W=theta*T and
cap metadata with exact T/theta/W/config hashes. The ordinary independent fresh
evaluator consumes W and the exported cap via weights_from/scaling_from. Positive
radial factors preserve greedy argmax, but training/evaluation Q scales differ.
Same-boundary export gaps can be rebuilt; missing/inconsistent past exports fail
closed. Default jobs retain the existing checkpoint helper and state structure.

Ten projection/worker tests passed4.59s: ownership, blank/mapping/cap order,
content mismatch, immutable/NumPy-alias mutation, path-independent persistence,
crash replay/full graphs/traces/RNG, write gaps and actual mapped fresh evaluator
handoff. Two three-seed allocation/control-gate tests passed1.13s. The fixed
coordinator and prospective protocol separately specify candidate allocation.

Real-T preflight `/workspace/m1b-experiments/dct-sarsa-preflight-20261016/report.json`
took27.17s, executing640diagnostic cameras+23existing replay, zero candidate data.
Raw128-step mean3.740s versus projected4.705s is1.258×cost; this is early-state
throughput, not a guarantee. Raw256record strings matched the archived raw-cap
run. Duplicate projected256records/full graphs and fresh-process128record/state/
chain/final graph equality passed, including nonzero TD updates. The fixed1024
calibration transformed norm² maximum233.63 and0cap activations are reported.
Using the actual nonzero preflight theta and full T, all1024calibration greedy
actions matched mapped-full/raw-cap inference; score discrepancy after accounting
for positive cap factors was2.61e-16 maximum. This cached check added zero data.
