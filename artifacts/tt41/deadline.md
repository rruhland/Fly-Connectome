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
At stock start the closest virtual blade-center approach in those six
full-budget searches was 0.163 m.

For 8302, the first 60 Hz tick after actual impact was already separated,
so a follow-up uses that earlier stock anchor, just 10.4 ms after impact.
Another 576 smooth curves still did not touch. The best approach was
0.150 m from the blade center. The other two releases were still touching
the opponent blade at their first post-impact command, and the isolated
native twin is not used to freeze an unresolved opponent collision.

## A fixed, lateral-neutral early stance

A single outbound rule read the normal stereo RGB and started to prepare
after two consecutive outward displacements with the ball past y = −0.65 m.
It moved only shoulder, elbow, and wrist roll to the 42% blend of stock ready
and the median approach for the three proved opposite-face returns. The target
was `[-1.372, -0.923, 0.483]` rad on those joints. It retained stock pan, wrist
tilt, and grip references. The common stance was released upon the inbound
visual turn or a 60-control-tick timeout. In the low branches the replacement
stroke remained an exact-state evaluator at the causal RGB gate.

| Low release | Outbound preparation before far hit | Warm native touches / tested | Native legal |
| --- | ---: | ---: | ---: |
| 14843 | 448 ms | 467 / 1,152 | 0 |
| 4322 | 767 ms | 643 / 1,152 | 0 |
| 8302 | 556 ms | 754 / 1,152 | 0 |

Each warm prefix was physically played through the same causal turn without
changing the existing far event or penetrating the full CAD. The neutral
stance lets the coach find contact in all three private screens, but not
the orientation and impulse needed for a legal return. No candidate was
promoted as a physical rescue.

Nine ordinary development opportunities were paired, three per projected
height band. They were sampled without inspecting score. After the warm
interval the comparison returned to the original absolute joint references
at bounded rates. This measures adding preparation to the archived control,
not a retrained policy's adaptation. Both sides of all nine pairs had
identical incoming far contacts and near bounces. Two of the three high pairs
timed out of preparation rather than detecting an airborne turn. Every
solver step in the comparison window was audited and neither arm penetrated
the full CAD.

| Projected arrival | N | Stock legal / answered | Warm legal / answered |
| --- | ---: | ---: | ---: |
| 0.99–1.04 m | 3 | 1 / 1 | 1 / 0 |
| 1.04–1.09 m | 3 | 3 / 3 | 3 / 3 |
| At least 1.09 m | 3 | 3 / 3 | 3 / 3 |
| All ordinary opportunities | 9 | 7 / 7 | 7 / 6 |

## Retiming known shot-specific preparations

A separate full-dynamics proof left stock intact until a sampled time
50–1,000 ms before the far hit, then tried to catch the original clean
demonstration's absolute reference timeline with the new rate and slew
bounds. The grid does not assume monotonicity and is not a global latest
deadline. The same incoming far impact and table bounce were preserved.

- **4322.** A departure from stock 1,017 ms before the far hit scored and was
  answered. Minimum full-CAD gap was 2.475 mm across 4,800 solver steps, with
  no penetration. At 817 ms the ball still landed, but the opponent did not
  answer. The previous witness started at 1,033 ms before the far hit.
- **14843.** The 415 ms trials can land, but neither signed tape got an answer
  at any tested lead through 515 ms. The earlier already-verified
  solutions started 648 ms before the far hit.
- **8302.** This constrained tape retiming touched at 106–656 ms leads,
  without landing. Its earlier verified solution started 773 ms before the
  far hit.

**Decision.** Within these movement families the cameras saw the incoming
ball soon enough to detect the reply, but not soon enough to begin from the
stock posture. Earlier preparation helps reach. A single neutral untuck is
not a demonstrated scoring solution, and the renewed contact is not enough
reason to replace stock readiness on every shot. Use early outgoing-flight
or opponent cues to select a preparation, then use incoming RGB for strike
correction. Three developmental counterfactuals cannot set a universal
motor deadline or imply that any learned policy will improve on held games.

## Sources and reproducibility

The raw receipts are scoped to the supplied development release tape and
include input hashes, optical position traces, exact audit labels, candidate
counts, and all winning physical action tapes. Only internal comparators use
the privileged state. Run from the repository root with its experimental Python
environment.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt41_deadline --batch 192 --rounds 6
python -m scripts.experimental.tt41_deadline --cases 8302 --only first_tick_after_hit --batch 144 --rounds 4 --output artifacts/tt41/deadline-contact-8302.json.gz
python -m scripts.experimental.tt41_deadline_warm --paired-only --output artifacts/tt41/deadline-ordinary.json.gz
python -m scripts.experimental.tt41_deadline_warm --low-only --batch 192 --rounds 6
python -m scripts.experimental.tt41_deadline_margin
python -m scripts.experimental.tt41_deadline_margin --audit-native artifacts/tt41/deadline-stock.json.gz --output artifacts/tt41/deadline-native-control.json
python -m scripts.experimental.tt41_deadline_margin --audit-native artifacts/tt41/deadline-contact-8302.json.gz --output artifacts/tt41/deadline-contact-8302-native.json
pytest -q tests/test_tt41_deadline.py
```
