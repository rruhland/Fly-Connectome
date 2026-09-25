# Sparse retained state did not improve transferable forecasting

**Status:** Opt-in M1A learning experiment. Production code, measured optic-lobe graph, and motor learning remain unchanged. All trained updates were local and used no backpropagation or privileged scene labels.

The [frozen state audit](2026-09-25-sparse-coherence-audit-findings.md) found that decaying retention plus one-pixel nearby-site inhibition preserved 92.2% native four-frame target reachability with only 2.94 active source sites/frame. We then trained matched four-frame competitive local readouts from the current state, retained state, and retained-plus-competitive state. A fourth head received the same competitive state but time-shuffled target credit. The source transition and all heads trained on 64 generic native-cadence scenes; event-budget gains were fitted only on that training set and frozen for held-out generic and zero-shot native 120 Hz Pong. Fixed repeat-event and zero-output baselines were scored in the raw results.

| Held-out domain | State | Generic-calibrated F1 | Top-eight signed recall | Quiet false pixels/frame |
| --- | --- | ---: | ---: | ---: |
| Generic fractional scenes | Current | .047 | .112 | 1.80 |
| Generic fractional scenes | Retained | .027 | .109 | 1.96 |
| Generic fractional scenes | Retained + competitive | .050 | .160 | 1.92 |
| Generic fractional scenes | Shuffled competitive credit | .014 | .052 | 2.21 |
| Native Pong | Current | .059 | .195 | .85 |
| Native Pong | Retained | .009 | .039 | 1.65 |
| Native Pong | Retained + competitive | .020 | .088 | 1.79 |
| Native Pong | Shuffled competitive credit | .006 | .023 | 1.78 |

The competitive retained state did learn something on generic scenes: its F1 and ranking exceed its shuffled-credit control. The improvement does not transfer. On native streams it is worse than the current state on F1 and ranking, while quiet false alarms roughly double. Independent and crossing generic tests also do not show a useful gain. More target reachability and post-hoc sparse inhibition therefore do not solve the forecast objective. In particular, spatial competition applied **after** the sensory dictionary and recurrent transition were learned cannot force those units to encode stable, ordered local motion.

**Decision:** Do not promote retention or post-hoc inhibition. Stop tuning its decay, radius, or budget. The next high-value question is whether spatial competition must be part of learning the latent itself: select a sparse set of locally matched spatiotemporal units before Hebbian dictionary and recurrent updates, then test position/speed and native transfer. This remains an opt-in architecture experiment. The result should be compared with the existing unconstrained dictionary, current-state head, and shuffled credit; if it does not beat them, move away from this per-site dictionary family rather than tuning another local parameter.

Reproduction: `python scripts/run_sparse_coherence_forecast.py`; [raw results](2026-09-25-sparse-coherence-forecast-results.json) include signed counts, source density, controls, mass, ranking, and elapsed time.
