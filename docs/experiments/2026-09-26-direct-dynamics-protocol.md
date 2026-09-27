# Direct delayed-credit dynamics experiment

The one-step prototype learner compounds rasterized observation error during
autoregressive rollout. Training diversity did not fix this; causal online
adaptation helped but still regressed on straight motion. Close that branch.

Test independent local displacement associations at horizons 1, 4 and 8.
Credit the stored observed history only when its endpoint is actually observed.
No hidden trajectory or forecast-generated position can be a training target.
Keep the same 24 development streams, 36 transfer streams, competitive rule,
prototype limit and fixed controls. Report frozen and causal online operation,
including shuffled endpoint credit. No horizon-specific tuning.

This changes the learned temporal relation rather than its input or sensor.
Retain only if it improves the registered average horizon-4/8 errors without
the large straight-motion regression. Family errors remain mandatory.
