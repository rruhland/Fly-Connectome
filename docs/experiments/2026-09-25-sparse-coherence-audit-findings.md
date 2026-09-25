# Local spatial competition preserves most causal support

**Status:** Opt-in M1A state experiment. Production architecture, measured graph, and motor learning are unchanged. The frozen source encoder and recurrent transition were trained on generic scenes with local rules and no backpropagation.

We tested a causal state that retains winning hidden units with 0.9 per-frame decay and inhibits other sites within one pixel of each winner. This is an experimental local state/readout condition, not a learned representation or a production architecture change. A 2×2 comparison isolated retention from competition. All arms used the same learned recurrent model, four-frame target horizon, 5×5 local readout support, held-out generic scenes, and zero-shot native 120 Hz Pong.

| Domain and source state | Target support | Quiet-origin support | Active source sites/frame |
| --- | ---: | ---: | ---: |
| Generic held-out, current | .554 | .276 | 3.16 |
| Generic held-out, competition only | .544 | .265 | 1.58 |
| Generic held-out, retention only | .990 | .998 | 12.08 |
| Generic held-out, retention + competition | .964 | .963 | 3.59 |
| Native, current | .498 | .115 | 1.48 |
| Native, competition only | .498 | .115 | .92 |
| Native, retention only | .922 | .869 | 6.96 |
| Native, retention + competition | .922 | .869 | 2.94 |

The combined state kept the native support gain from causal retention with fewer than half as many active sites as retention alone. It also retained most generic support. Generic independent and crossing cases were less favorable: target support was .356 and .819, respectively, versus .358 and .914 without competition. Quiet-origin support was zero in both of those cases, so the state still has a domain limitation.

**Decision:** Proceed to a matched delayed local-learning comparison of current, retained, and retained-plus-competitive states with a time-shuffled-credit control. Support is only a geometric ceiling: the previous eight-frame age-tagged learner had high support and still failed native forecasting. Require a held-out and zero-shot native gain over the current-state and shuffled controls before proposing any production change. The tested decay and one-pixel inhibition are experiment settings, not hardcoded object or motion semantics.

Reproduction: `python scripts/run_sparse_coherence_audit.py`; [raw results](2026-09-25-sparse-coherence-audit-results.json).
