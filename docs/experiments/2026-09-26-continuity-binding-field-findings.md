# Learned continuity made the sparse state less coherent

**Status:** Final bounded opt-in test of the sparse paired-unit M1A family under the [registered protocol](2026-09-26-continuity-binding-field-protocol.md). The fast predictor and paired sensory code were frozen. Aligned and time-shuffled local temporal/common-fate associations fed back into the *live state*, with a frozen self-continuity arm and the original sparse paired state as controls. No object coordinates, directions, speeds, labels, rewards, Pong semantics, or backpropagation entered training or inference. Production M1A was not changed.

The decisive autonomous-state measures worsened. On separated-mover scenes (two moving entities), the aligned field occupied **5.19 connected components per active frame**, with **3.19 excess** above the entity count, and **54.9%** of significant activity outside offline entity neighborhoods. The original sparse paired state had 3.44 components, 1.44 excess, and 30.4% outside activity; the time-shuffled field had 3.86 components and 34.5% outside. The aligned field therefore created more fragments and spatial spillover, not coherent moving state. On doubled speed it had 56.2% outside activity and 2.86 components for one entity. These synthetic object regions were used only after inference to score the autonomous state.

The fixed offline 3×3 spatial direction probe also lost transfer: aligned continuity scored **12/16** at doubled speed and **10/16** on changed shape, versus 14/16 on both for the sparse paired state. The frozen self-feedback arm scored 16/16 and 14/16 respectively, so the degradation is associated with the learned feedback rather than a need for feedback per se. A decodable feature at a known location is still not proof of autonomous tracking.

Full-frame next-meaningful-event top-32 forecast recall:

| Held-out set | Fast only | Aligned binding | Shuffled binding | Frozen feedback | Sparse pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic scenes | .360 | .376 | .368 | .358 | **.433** |
| Doubled speed | **.480** | .385 | .432 | .289 | .362 |
| Changed shape | **.542** | .506 | .518 | .412 | .473 |
| Separated movers | .312 | .302 | **.325** | .250 | .293 |
| Crossing movers (2 scenes) | .354 | **.422** | .417 | .385 | .396 |

Aligned binding beat shuffled binding on only **2/16** speed and **2/16** shape cases, and on **0/8** separated-mover cases. It beat fast-only on just 2/16 speed cases. Its speed top-8 recall (.182) exceeded fast-only (.151), but top-32 recall and precision were lower (.385/.140 versus .480/.175), so the extra high peaks do not amount to a useful full-frame forecast. The learned aligned lateral kernel received only 56 co-motion counts over training, versus 3,162 local temporal counts; the sparse winners offered little concurrent evidence to bind. Parameter tuning of feedback gains or association thresholds would not answer the larger representation question.

**Hard decision:** Reject the sparse paired-unit family as the current M1A architecture. Three substantive variants now show the same pattern: transferable motion can be read offline, but live state is fragmented or spills outside moving evidence, and prediction does not transfer across speed and shape. This final state-level continuity/binding attempt made those failures worse. Do not extend the family with another decoder, recurrent gain, or winner-count sweep.

**Pivot:** Keep the fast locally learned predictor as a useful sensory primitive, but replace the globally capped sparse slow winners with a spatially distributed recurrent predictive sheet. Every locally supported site should be able to carry multiple weak latent features; local inhibition/homeostasis, rather than a global winner budget, should control activity. Feedforward and lateral/recurrent synapses should learn from delayed *signed* local fast prediction error and local eligibility, with no backpropagation. The state must be updated by its own prediction and observation, not merely decoded after the fact. This tests whether preserving distributed local evidence allows multiple moving entities and speeds to coexist. A first opt-in comparison should use aligned, time-shuffled, frozen, and fast-only controls on the same registered transfer/coherence metrics, with no Pong semantics or oracle regions in inference. Promote a concrete production revision only after it establishes both autonomous state quality and useful forecast transfer.

Reproduction: `python scripts/run_continuity_binding_field.py`; [raw results](2026-09-26-continuity-binding-field-results.json) retain per-scene hits, top-8/top-32 recall and precision, probe scores, activity/component measures, kernel counts, and runtime (about 62 seconds).
