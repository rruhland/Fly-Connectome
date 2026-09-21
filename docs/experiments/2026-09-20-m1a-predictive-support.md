# Return to M1A: signed predictive support

Throughput experiments are closed per the user's scope limit. The next M1A
question is why the 10,000-frame learner still loses to zero-event prediction,
not whether another unchanged long run can be scheduled faster.

The frozen structural audit uses the pinned graph and exact scripted camera/Pong
stream for development seeds 1101/1102, 500 frames. It performs no neural training,
weight updates, fitted prediction, parameter changes or final-seed evaluation.
It counts all measured predictive edges, including currently zero-weight edges.
Thus it asks what signs are structurally possible, not what the current activity
or weights can actually produce.

| Population | Both predictive signs | Negative only | Positive only | Neither | Events lacking required sign |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 | 871 | 13 | 5 | 3 | 129 / 834 (15.5%) |
| L2 | 612 | 281 | 0 | 0 | 231 / 844 (27.4%) |
| L3 | 304 | 574 | 0 | 14 | 402 / 834 (48.2%) |

The signed contrast target is negative for ON and positive for OFF. A recurrent
current receiving only negative signed inputs cannot become positive from zero
initial current with nonnegative magnitudes and the existing decay equation.
Intrinsic resting current is excluded from this predictive readout. The same
argument applies to negative targets when no negative predictive edge exists.

Across populations, 762 of 2512 events (30.3%) lack structural sign support. This
gives an optimistic event-conditioned MSE lower bound of about 0.3033, or an
all-sample bound of 762 / 2,677,000 = 0.00028465. The denominators are deliberately
separate: M1A sensory-event MSE includes all neuron/frame observations, whereas
event-conditioned MSE includes only nonzero events. First-frame events are included.

These are lower bounds, not achievable forecasts or learned results. Most events
remain structurally supported, and the bound is well below the zero predictor's
all-sample MSE of 0.00093836 on this stream. Therefore polarity constraints alone
do not establish that M1A is impossible or explain its complete failure to beat
zero. No signs, topology, target or acceptance criteria should be changed on this
evidence alone.

Next: compare initial and trained forecasts separately on supported ON and OFF
events, then inspect whether their local eligibility carries predictive signal
at the required time. Distinguish unavailable sign, inactive pathways and weak or
mistimed local credit. Do not blindly extend the existing 10,000-frame checkpoint.

```powershell
.venv/Scripts/python scripts/audit_predictive_support.py checkpoints/event-v1-combined-rate-10000.pt --steps 500 --seeds 1101,1102 --output runs/predictive-support.json
```

Raw report: `2026-09-20-predictive-support-10000.json`. The source checkpoint hash
remains `6358107d489ee859f1644105eb274c29b226e6cdc71ae7e58d65a8bf6c925e58`.
Regression tests distinguish predictive from feedforward edges, retain zero-weight
structural support, verify mixed signs and verify both error denominators.
