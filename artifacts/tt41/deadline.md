# How early must the body move for the three difficult low returns?

The three cases are previously declared development releases, not held-policy
tests. The original stock commands and exact simulator event sequence are kept
through every stock fork. The target inbound ball is the same ball driven by
the same continuously randomized, reacting opponent. None of the searches is
an actor or exposed to the pupil.

## Stock body at the first usable cue

The calibrated 30 Hz stereo gate needs three consecutive observed centers with
two outbound changes to arm and, later, three centers with two inbound changes.
It does not use simulator events or states. For the test harness, watching
starts just after the preceding near separation to isolate one exchange. Far
separation is a separate, evaluator-only gate one full control tick after its
recorded time. Exact flight, including spin, is handed to the offline stroke
screen at either gate. Earlier physical CAD-clean examples supply only
shot-specific geometry and reference-tape seeds. The tested family varies both
signed faces where available, early-rising depths, movement and strike phase,
bounded wrist motion, and compressed older movement tapes.

Targets are emitted as direct rates limited to ±0.8 of the calibrated joint
increment, with at most 0.32 normalized command change per 60 Hz step. The
private comparator screens contact, landing, and guard state. Only a command
tape that scores an answered return under the original physical opponent and
full CAD would qualify. In the source protocol an answer is a legally delivered
next opponent bounce, not an arbitrary contact or outbound velocity flag.

| Release / leg | RGB after far hit | Best reference-time deficit at far separation / RGB | Direct search from far separation / RGB |
| --- | ---: | ---: | --- |
| 14843 / 1 | 52 ms | +106 / +123 ms | 0 / 0 touches |
| 4322 / 5 | 67 ms | +49 / +74 ms | 0 / 0 touches |
| 8302 / 3 | 77 ms | −23 / +27 ms | 0 / 0 touches |

There were 1,152 private candidates per case per gate, 6,912 total. The deficit
is the minimum across the seeded poses of the difference between the fastest
componentwise command-reference travel and time to that pose's aimed contact.
It does not account for actuator following lag or clearance-safe detours, and
is not a lower bound for all poses or for every possible legal stroke. The
negative value at the audit gate on 8302 did not produce a usable stroke. The
RGB gate was designed on these development cases and its outbound arming
threshold was corrected to handle 4322's far impact near the net instead of
using an event oracle.

This is not a pre-bounce failure of the private comparator. On every gate,
an already-prepared clean demonstration was replayed to the same timestamp.
All eight gate/demo combinations retained bitwise-identical stock ball
position, velocity, and spin, and all eight produced a legal shadow result.
At stock start the closest virtual blade-center approach over all tested
curves was 0.163 m.

## Sources and reproducibility

The raw receipts are scoped to the supplied development release tape and
include input hashes, optical position traces, exact audit labels, candidate
counts, and all winning physical action tapes. Only internal comparators use
the privileged state. Run from the repository root with its experimental Python
environment.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt41_deadline --batch 192 --rounds 6
python -m scripts.experimental.tt41_deadline_warm --paired-only --output artifacts/tt41/deadline-ordinary.json.gz
python -m scripts.experimental.tt41_deadline_warm --low-only --batch 192 --rounds 6
python -m scripts.experimental.tt41_deadline_margin
python -m scripts.experimental.tt41_deadline_margin --audit-native artifacts/tt41/deadline-stock.json.gz --output artifacts/tt41/deadline-native-control.json
pytest -q tests/test_tt41_deadline.py
```
