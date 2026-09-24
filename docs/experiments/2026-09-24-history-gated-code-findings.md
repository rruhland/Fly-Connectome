# Gating a learned history code restores transferable temporal distinction

The [registered opt-in representation test](2026-09-24-history-gated-code-protocol.md) changed the temporal code, not its sensory ingredients: raw first-sighting events gated **where** an eight-unit history population could fire, while unit identity was selected from the decaying trace of the existing learned motion population alone. Two local Hebbian/homeostatic passes used the same 96 clean plus 32 generic interruption/noise episodes. Direction labels were absent from training and entered only a frozen post-hoc probe. The run took 62 seconds; production M1A was unchanged.

| Representation check | Learned history dictionary | Random history dictionary |
| --- | ---: | ---: |
| Different units on matched opposite-motion pair | **13/13** common event sites | 12/13 |
| Post-hoc held-out direction probe | **47/48 (97.9%)** | 36/48 (75.0%) |
| Active held-out event sites | 608 | 608 |

The matched pair had identical raw t=10 input, yet all 13 learned source-unit assignments differed after opposite histories. Across three unseen shapes, two speeds, both contrasts, and four directions, a simple probe of each episode's unit histogram transferred at 97.9%, exceeding the random dictionary by 22.9 percentage points. Both registered representation gates passed. The random dictionary also carried considerable direction information, so the learned dictionary's advantage matters; this is evidence of a usable temporal code, **not** yet evidence of useful event prediction or of object tracking.

The previous concatenated raw+trace dictionary produced zero different sources on the same matched pair. Separating raw onset as a gate prevents its activation from dominating competition over history. The next bounded experiment should train a separate **local delayed-event readout** from this frozen learned code, with a matched random-code readout and a trace-reset ablation. Require an exact reappearance gain, direction-dependent forecasts on the matched pair, clean transfer, and controlled quiet false alarms before considering production integration. No offline probe weights belong in that learner.
