# Approved production promotion

The user approved the hybrid/probabilistic contract on 2026-09-27. The runtime is
now shipped under `fly_connectome.vision`, with the exact approved checkpoint as
package data and a generic `vision-run` CLI defaulting to it. No experiment-script
imports, renderer labels, motor policy or reward rules enter that runtime.

Only runtime definitions were extracted into observation, association, memory,
dynamics and state modules. Unused experiment classes/generators and intermediate
candidate constructors were excluded. Full tensor/dictionary and checkpoint parity
tests cover frozen inference, online learning and missing observations. Production
sensor validation rejects invalid shapes/ranges before state changes and converts
valid input to the approved float32 computation. Code review found float64 events
were initially accepted but incompatible with float32 convolution; a reproduced
regression and explicit normalization resolved that defect.

The 216-scene production replay reports maximum score difference 2.384e-7 versus
the reviewed experiment, identical learned parameter/calibration/RNG fingerprint,
and 86.2 samples/s for `step` on this run. That timing includes interval export
but excludes sensor acquisition and graph execution. Focused online parity and
an installed-wheel online smoke test pass; this is not a new broad online-learning
speed or continual-transfer claim.

The wheel was installed into a temporary directory outside the checkout. It loaded
the bundled checkpoint, produced online forecasts, saved/reloaded learned state,
and imported no research modules. Reproduction:

Final complete repository verification: **700 passed, four skipped**. The bundled
checkpoint SHA-256 is pinned in a regression test; skipped tests remain explicit
environment-dependent checks, not passing runtime evidence.

```
python -m pytest tests/test_production_vision.py -q
python -m pip wheel . --no-deps --no-build-isolation -w runs/production-wheel
```

See the production parity/wheel JSONs beside this report and the current roadmap
`docs/plans/2026-09-27-m1a5-to-m1b-roadmap.md`. Existing graph trainer/worker commands
remain the legacy research path; the new M1B state-to-action bridge is planned,
not secretly enabled by this promotion. The new production vision API is the
baseline for that work. Old event-only and point-forecast failures remain recorded.
