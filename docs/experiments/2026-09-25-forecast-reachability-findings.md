# Four-frame failures are limited by absent local sources

**Status:** Frozen opt-in M1A diagnostic. No production architecture, synapse, or learned weight changed.

The [age-conditioned hazard test](2026-09-25-local-hazard-findings.md) failed to turn a transferable four-frame location signal into a reliable event forecast. Before adding another plasticity mechanism, we measured whether any causal event-centric recurrent source could physically influence each future target through the existing shared 5×5 readout. The recurrent state was trained only on 64 generic fractional-motion streams, then frozen. For a forecast issued at frame *t*, the audit compared source locations at *t* with signed camera events at *t+4*. This is an upper bound for any choice of those local readout weights, not a forecast score.

| Frozen target set | Current 5×5 source ceiling | Current camera-event ceiling | Source-free future event pixels |
| --- | ---: | ---: | ---: |
| Held-out generic fractional motion | 55.4% | 42.5% | 556 / 2,834 |
| Native Pong at 120 Hz | 49.8% | 45.5% | 84 / 514 |

Among the 122 native target event pixels whose forecast originated on a quiet camera frame, only 11.5% were reachable from the current recurrent state. Increasing the hypothetical readout radius from 2 to 16 pixels raised overall native coverage only from 49.8% to 65.4% and quiet-origin coverage from 11.5% to 24.6%. A wider readout alone cannot address the missing state; 84 target pixels followed frames with no active hidden source anywhere.

We then computed a *causal memory ceiling* by retaining the union of recent hidden-source locations at forecast time, without fitting weights or looking at future events. A four-frame source history raised native 5×5 coverage to 72.0% and quiet-origin coverage to 59.0%. An eight-frame history raised them to 95.3% and 93.4%. Held-out generic coverage rose from 55.4% to 99.2% with eight frames. This does not prove a learner can pick the correct target from old locations or avoid false alarms, but it identifies a local temporal state that makes those predictions geometrically possible without adding long-range readout edges.

**Decision:** Stop widening the readout and tuning scalar hazards. The next opt-in experiment should keep a bounded, age-tagged local memory of recently active learned hidden units, then learn four-frame signed spatial outcomes from that state with delayed local credit and a shuffled-credit control. Require improved held-out generic and native F1 *and* quiet-frame behavior before any production proposal. Persistent camera observation remains separate evidence; no Pong identity or motion label enters the model.

Reproduction: `python scripts/run_forecast_reachability.py` and `python scripts/run_memory_reachability.py`. Their JSON results include each native seed and the radius sweep.
