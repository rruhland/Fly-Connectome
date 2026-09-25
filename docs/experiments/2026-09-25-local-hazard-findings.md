# Local age-conditioned hazard improves generic scores, not native prediction

**Status:** Opt-in M1A experiment. No production model, measured graph, motor learner, or backpropagation changed.

The [event-centric study](2026-09-25-event-centric-competition-findings.md) found a transferable four-frame *location ranking* but too many missed events and quiet-time alarms. We tested a factorized local head on the same recurrent state and 64 generic fractional-motion training episodes. One shared 5×5 synapse learns the signed spatial distribution when an event arrives. A separate local event-count rate learns from every due target, including quiet targets, conditioned on the winning unit and its own recurrent age (0, 1, 2–3, 4–7, or 8+ frames). The two factors multiply at inference. Credit arrives exactly four frames later; all updates use local source activity and observed camera events. A temporally shuffled hazard and the previous competitive head were trained in parallel. The optional sparse output budget was calibrated only from generic training event counts and current forecast mass, then frozen for held-out generic and native Pong tests.

| Frozen four-frame evaluation | Hazard F1 | Previous competitive F1 | Shuffled hazard F1 | Hazard top-eight signed recall |
| --- | ---: | ---: | ---: | ---: |
| Held-out generic fractional motion | .064 | .047 | .014 | .144 |
| Native 120 Hz Pong | .068 | .059 | .072 | .210 |

The direct 0.5-threshold hazard F1 was .003 generic and .020 native. Under generic-calibrated sparse output, native hazard produced 42 true positives, 677 false positives, and 472 misses; the shuffled hazard produced the same 42 true positives with 613 false positives. Native quiet alarms were 0.92 versus 0.85 for the prior competitive head. The hazard's mean native forecast mass was 1.67 on event-due frames and 0.72 on quiet frames, but that separation was not sufficient to beat the matched shuffled-credit output. Held-out generic improvement is real relative to shuffled credit, yet does not meet the zero-shot transfer gate.

**Decision:** Do not promote the age-binned hazard, budget calibration, or event-centric head into production. Stop this family of scalar timing-gate variants. A useful next *opt-in diagnostic* is to test whether a causal local patch of recurrent activity, age, and observed event history can distinguish four-frame event-due from quiet cases across generic and native streams at all. If a frozen capacity probe separates them, the local credit/objective needs a richer factorization; if it does not, the recurrent temporal state needs to change. Do not tune more thresholds or age bins before that question is answered.

Reproduction: `python scripts/run_local_hazard_probe.py`; full counts, masses, gains, and controls are in [the result JSON](2026-09-25-local-hazard-results.json).
