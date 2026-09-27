# Joint reconstruction: reject both registered implementations

The event-only space-time dictionary and sparse-frame local linear reconstructor
both fail the unchanged joint state gate. Do not change atom counts, temporal
windows, learning rates, or sensor cadence to tune this result.

All noise scores below use fresh seed base 50000. They are not directly paired
with earlier reports using seed base 5000.

| Observation arm | Clean context /16 | Familiar noise /16 | Heavy noise /16 | Constant /16 | Two movers /16 | Continued /16 | Vanished false /16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Learned joint event patches | 0 | 2 | 2 | 7 | 8 | 2 | 0 |
| Data-initialized event patches | 5 | 4 | 0 | 3 | 4 | 4 | 1 |
| Learned sparse-frame reconstruction | 16 | 8 | 1 | 16 | 16 | 14 | 2 |
| Early/frozen sparse-frame reconstruction | 16 | 11 | 2 | 16 | 16 | 16 | 0 |
| Learned sparse-frame, .01 frame-pixel flips | 16 | 9 | 4 | 16 | 14 | 13 | 1 |

Localization is top-32 except the two-mover top-64 regression. Continued and
vanished are the clean matched-pair post-reveal strength checks; image noise is
also present there in the final row. All pre-reveal pair differences are zero.
The sparse learner's shuffled-context arm retains the clean regressions but has
4/16 clean contextual hits and only 1/16 familiar-noise hits. Context learning
still matters; the observation learner does not improve the relevant robustness.

The event dictionary reconstructs a joint event volume from learned local atoms,
rather than assigning an independent credibility to each arriving event. This
particular implementation nevertheless destroys enough clean input that the
downstream association receives only two learning updates. The hybrid learner
uses only every-eighth current visible image as a teacher and observed anchor,
yet performs worse than its early/frozen reconstruction on the same noisy input.
These results reject the implementations, not the broad theoretical families.

Training uses a predeclared mixture of observed motion cadences and contains no
hidden masks, object labels or per-frame intensity teacher between anchors.
The frozen controls are data-initialized after the first training episode, not
zero dictionaries. Tests cover learning freeze, reset persistence, and sparse
teacher availability. The complete shared suite after this chunk passed
656 tests with 4 skipped.

Reproduce: `.venv/Scripts/python scripts/run_joint_reconstruction.py`.
[All seven arms and timing](2026-09-26-joint-reconstruction-results.json).

The next substantial architecture replaces the remaining benchmark-specific
above/below context comparator with generic locally learned patch associations.
It must pass physical rotations under the positive visible-frame sensor control
before its event-only results can be interpreted as a reusable representation.
