# Per-site learned sequence units were dense and failed native transfer

**Status:** Opt-in M1A architecture experiment. Production code, measured optic-lobe graph, and motor learning remain unchanged. No backpropagation or privileged scene labels were used.

The [lagged-memory test](2026-09-25-lagged-local-memory-findings.md) showed that old source locations make future targets reachable but independent age-conditioned synapses cannot select them. We tested a distinct learned representation: 24 locally competitive dictionary units received a causal eight-frame stack of raw signed camera events. The stack was a low-level sensory memory, not a fixed direction or object bank. A local Hebbian dictionary learned conjunctions over 5×5 space and recent time from 64 generic fractional-motion episodes. A locally plastic recurrent transition then continued that code. Matched four-frame local heads read either the directly observed learned sequence units or the recurrent hidden state; a time-shuffled recurrent-credit head controlled for learning. All output budgets were calibrated on generic training data and frozen before held-out generic and native 120 Hz Pong evaluation.

| Held-out domain | Direct four-frame F1 | Recurrent F1 | Shuffled recurrent F1 | Direct top-eight signed recall |
| --- | ---: | ---: | ---: | ---: |
| Generic fractional multi-entity scenes | .015 | .015 | .004 | .067 |
| Native Pong | .000 | .000 | .008 | .018 |
| Generic independent objects | .000 | .000 | .000 | .000 |
| Generic crossing objects | .002 | .002 | .008 | .005 |

The recurrent head was essentially identical to the direct head. The lag stack kept many sites active: 13.4 source sites per held-out generic frame and 8.6 per native frame, versus about 1.5 native sites for the prior event-centric state. Its generic-calibrated output produced 2.39 false event pixels per quiet native frame. The fixed local dictionary learned repeatable generic patterns, but per-site winning units did not produce a sparse, transferable trajectory representation. Training the dictionary took 21 seconds; the complete experiment took several minutes, so another input-window sweep is not justified by these results.

**Decision:** Do not promote the temporal stack or its learned dictionary. The next architectural hypothesis must address *spatial competition and coherent recurrent state*: local inhibition should select a limited set of active hypotheses across nearby sites, and local plasticity should maintain ordered motion information through quiet frames. Fixed traces may remain sensory primitives, but the useful latent must be learned. Before another long run, specify how this state interfaces with measured optic-lobe motion pathways and a test that can falsify the hypothesis on held-out generic scenes and zero-shot native Pong. No production revision is authorized by this result.

Reproduction: `python scripts/run_learned_sequence_units.py`; [raw results](2026-09-25-learned-sequence-units-results.json) contain signed counts, quiet alarms, ranks, source activity, controls, and elapsed time.
