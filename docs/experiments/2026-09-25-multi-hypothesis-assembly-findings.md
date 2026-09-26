# Delayed path competition preserves motion but does not learn entity selection

**Status:** One-shot opt-in M1A architecture comparison under the [registered protocol](2026-09-25-multi-hypothesis-assembly-protocol.md). Production architecture, measured graph, and motor learning remain unchanged. No backpropagation or object/direction labels entered the sensory code, transition-affinity learning, or path generation.

The previous greedy assembly tracker fragmented a speed-two mover into 18–20 short tokens and scored 0/16 direction. This experiment retained several causal local continuations per candidate, carried each through quiet frames, and selected a sparse set of non-overlapping paths after the 18-frame sequence. The learned local Hebbian affinity, proximity/smoothness-only control, and temporally shuffled-affinity control received exactly the same full event-camera candidates. Synthetic object paths were used only **after inference** to match and score outputs.

| Held-out set | Learned | Proximity only | Shuffled credit | Output paths/case |
| --- | ---: | ---: | ---: | ---: |
| Position, 16 objects | 16/16 | 16/16 | 16/16 | 4 |
| Speed ×2, 16 objects | 16/16 | 16/16 | 16/16 | 4 |
| Plus shape, 16 objects | 16/16 | 16/16 | 16/16 | 4 |
| Separated movers, 16 objects | 16/16 | 16/16 | 16/16 | 8 |
| Crossing movers, 4 objects | 4/4 | 4/4 | 4/4 | 8 |

Matched path coverage was 100% in every arm and set. These perfect numbers are a **generous path-capacity ceiling**: the evaluation matcher chose one correct path from roughly four alternatives per object. The model itself did not resolve which path represented the entity. The exact equality of aligned, fixed, and shuffled arms shows no useful contribution from the learned transition affinity on this suite. Fixed temporal continuity, together with delayed path selection, explains the gain over immediate greedy assignment. This result also shows the event-linked sensory candidates are not inherently missing the speed-two motion path; the failure lies in binding and selecting a coherent state online.

**Decision:** Do not promote the path bank as an M1A representation, and do not tune beam width or scoring coefficients. The next architecture must learn *competition and consolidation*, not merely retain alternatives. A biologically/local direction would use recurrent assemblies with local excitatory support among compatible event fragments, inhibition between incompatible explanations, and delayed prediction-error credit that strengthens the assembly whose continuation is confirmed. Its evaluation must score the model's own chosen state (one active explanation per entity-like stream, without an oracle path picker), alongside coverage, duplicate assemblies, speed/position transfer, quiet frames, and shuffled-credit controls. If learned consolidation cannot beat fixed continuity, the representation mechanism should change again before any forecasting or production integration.

Reproduction: `python scripts/run_multi_hypothesis_assemblies.py`; [raw results](2026-09-25-multi-hypothesis-assembly-results.json) contain per-object decisions, coverage, path counts, controls, and elapsed time.
