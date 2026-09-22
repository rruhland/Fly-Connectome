# Live local timing comparator: zero-learning parity

The approved experimental `TimedLocalPrediction` subclass is implemented for
the measured L3 visual target. It leaves the physical recurrent current in
the membrane calculation, and gates the local issued prediction and
eligibility used for confirmation. It stores one local past-event sensory
reference plus prior quiet/event counts. Non-target visual synapses receive no
updates in this experiment. The production plasticity and network code were
not changed.

Before training, eta was set to zero and the new rule was run on the exact
411-frame continuous stimulus already saved in the prior experiment, with the
same frozen neural weights and warmup. The following arrays match the saved
baseline **exactly**: every neural spike, every frame's raw prediction,
target and sensory state, the causal reference and gate, the gated issued
prediction, and all final weights. The new rule also matches an independent
causal replay of the sensory reference and gate. Thus the comparator has not
changed the neural dynamics before learning.

The source mixed-weights SHA256 is
`6b518d9021c1dd7df8f1d4c4a5c8d28b043bde27a68262ef70162ced04cd7e53`;
the saved baseline neural-history SHA256 is
`64dfb51e63987c93b49ae7fad5edc3639c25afdd7eb81ea46667f4f337f88b6e`.
The parity output in `runs/live-local-timing-v1/parity.npz` has SHA256
`80a66107ecc12755aa95c6c0a711896df42ae47173aee5b44cefbe473ad59501`.
The run itself took 9.83 seconds after model loading. `parity.json` beside the
array has exact field names and source identities.

Unit tests cover closed-gate surprise, reference acquisition, gate masking,
prior-count event gain, target-only bounded update, and no non-target update.
The 14 directly relevant tests passed. This is a parity gate for the next
chunk, not evidence of successful live learning.

```powershell
.venv/Scripts/python scripts/live_local_timing.py parity
.venv/Scripts/python -m pytest tests/test_timed_local_prediction.py tests/test_frame_prediction.py tests/test_local_credit_probe.py -q
```

The parity command refuses to overwrite its saved run; preserve or move the
existing run directory before reproducing it. The next chunk will add live
training and held-out evaluation while retaining the parity fixture.
