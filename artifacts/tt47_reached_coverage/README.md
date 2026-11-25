# Does the sending posture predict next-return coverage?

The tested recovery-aware readout did not beat the corresponding reply-only
readout. Even giving it the actual own encoder state on the next approach
slightly worsened release-disjoint prediction of a legal return. Approximating
the sending posture and marginalizing the same empirical distribution of
possible replies did not recover a benefit. This mechanism was stopped before
changing commands on a physical assessment.

| Release-disjoint comparison | RGB and encoder sample | Larger reference-proxy sample |
| --- | ---: | ---: |
| Actually offered next balls | 162 from 10 releases | 792 from 66 releases |
| Reply-only Brier error with the realized incoming flight | .08022 | .06962 |
| Same flight plus reached body and motion | .08109 | .07016 |
| Marginalized possible replies, reply only | .10455 | .09605 |
| Same possible replies plus projected body | .10462 | .09614 |
| Next-joint RMSE, outgoing signature, reply and elapsed time | .05266 rad | .04802 rad |
| Same forecast plus the earlier sending joints | .05297 rad | .04793 rad |

The release-weighted bootstrap intervals for the marginal Brier gain both
straddled zero. The additional sending information did almost nothing for
the recovery forecast. In the authentic encoder sample, achieved joints
were about .118 rad RMS away from home at the observed approach. Knowing
the outgoing signature, next flight and recovery delay reduced that to .053 rad
prediction error. That does not establish that posture never matters. It
says this measured sending-state proxy adds little information over the
next ball under these existing motion and recovery policies.
[Grouped results and provenance](qualification.json)

## Qualification

The primary sample uses the actual early RGB track, own five joint encoders
and six-tick encoder difference from exact archive replays. The selected
approach must follow a physically scored stroke with a recorded causal
outgoing prediction. A timing screen rejects archived predictions that
look inconsistent with when the preceding physical stroke occurred. The
sending-state proxy is the issued reference about 200 ms after that stroke,
which is available from the arm's own controls. It is not the actual
encoder state at that instant.

The larger diagnostic obtains both arm states from issued references, and
uses the evaluator's realized receive-line flight instead of a camera
estimate. It is a separate proxy test, not 792 camera-and-encoder packets.
Neither sample treats unavailable future variables as signals the live
arm can read.

For each held release, a ridge model predicts the achieved recovery state
from the outgoing signature, possible reply flight and elapsed time. Adding earlier sending
references tests whether anything useful survives the recovery period.
A second model estimates next landing from sensed flight, with or without
the reached joints and their recent movement. The prospective input to a
reply-distribution estimate would be only the current private simulated
outgoing signature. In this qualification we use that recorded signature
and enumerate up to 24 physically realized neighboring practice replies
from other releases. Both scoring arms use identical reply/time cases and
weights. There is no sampling of the true held opponent target. The score
is an estimate of the existing player's physical landing probability,
not a certification of all feasible motor trajectories.

All descendants of a measured release stay together in the same fold.
This sample conditions on already landed, answered strokes for which
a suitable prediction was archived. It is not representative of
opponent faults, missing optical tracks, unusually close recoveries,
or larger changes to stroke and standby policy. A new policy could
change those distributions. The qualification required enough evidence
of a meaningful body contribution before spending further simulator or
assessment time.

## Reproduce

```sh
export PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1
python -m scripts.experimental.tt47_reached_coverage
python -m pytest -q tests/test_tt47_reached_coverage.py
```
