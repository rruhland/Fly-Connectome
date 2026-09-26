# Representation-first M1A screen: seven local principles

**Decision:** None of the seven one-run instantiations is ready to become M1A's world-state or to receive a prediction head. Stop this experiment batch and reassess the representation objective and benchmark with the user. This screen does **not** prove that any broad principle is impossible. It establishes that these bounded, label-free, no-backprop implementations do not improve the joint representation criteria over the frozen learned observer and causal coincidence control. Production M1A is unchanged.

The [registered protocol](2026-09-26-representation-first-family-protocol.md) fixed 64 unlabeled mixed-cadence training episodes (5,120 camera frames), the observer and low-level input planes, and one held-out audit for all families. Labels and object masks were used only for frozen offline probes. The occupancy probe was one full-frame local linear classifier calibrated on separate single-object scenes; direction was an **oracle-region capacity** probe, not autonomous object recognition. Separated-component and crossing-identity checks, noise discrimination, and three missing-evidence frames tested harder structure. Each family was compared with its own frozen-initialization control. No future-event learning or motor learning was attached.

| State | Occupancy F1: position / speed / shape | Direction: position / speed / shape | Two separated components | Crossing identity | Noise margin | Missing gap margin | Active fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen sensory + coincidence | **.964 / .946 / .598** | **1 / 1 / 1** | **1.00** | **1.00** | .841 | .293 | .003 |
| Slow feature | .338 / .178 / .274 | .625 / 1 / .875 | 1.00 | .50 | -.210 | .095 | .162 |
| Common fate | .742 / .599 / .437 | 1 / 1 / 1 | 1.00 | 1.00 | .782 | .473 | .006 |
| Sparse reconstruction | .542 / .502 / .413 | 1 / .875 / .875 | .75 | 1.00 | .840 | .309 | .003 |
| Attractor assembly | .236 / .184 / .208 | .625 / .750 / .625 | 1.00 | .50 | -.084 | .381 | .029 |
| Synchrony/phase binding | .539 / .367 / .293 | .250 / .750 / .438 | 1.00 | 1.00 | .843 | .445 | .003 |
| Relational graph | .261 / .170 / .218 | 1 / 1 / 1 | 1.00 | .00 | **.882** | **.752** | .013 |
| Fixed reservoir + local readout | .621 / .477 / .348 | 1 / 1 / 1 | 1.00 | .00 | .792 | .654 | .012 |

Noise margin is clean/corrupted latent cosine similarity minus clean/unrelated similarity; higher means noise preservation without scene collapse. Missing gap margin is the clean/interrupted similarity at the end of the three-frame gap minus clean/unrelated similarity. The registered pass/fail rule used the raw missing similarities plus a non-collapse and transfer-retention check; reporting this unrelated-adjusted margin makes the interpretation clearer without changing that rule. The control is unusually strong on this small suite because the frozen observer's contrast state and causal motion coincidence already expose the relevant evidence. The crossing score has only two scenes, so its 0/1 values are a useful warning, not a precise population estimate.

The individual keep/reject calls are:

- **Slow feature — reject this state.** Raw clean/interrupted similarity reached .997, but unrelated scenes were also .902 similar. Its position occupancy fell from .964 to .338 and noise margin became negative. Temporal stability mostly erased discriminative content.
- **Common fate — reject this implementation.** Direction and crossing capacity survived, but occupancy transfer dropped on every split. The learned-link model and its frozen counterpart were effectively identical on scored measures, so this run did not demonstrate learned binding.
- **Sparse reconstruction — reject as a standalone state, retain as a weak research lead.** Its local dictionary improved noise margin by .071 and recovery similarity by .098 over its frozen initialization, but merely matched the sensory control's noise margin and lost much of its occupancy and some direction transfer. It cannot yet support a downstream world model.
- **Attractor — reject this state.** Raw gap similarity was .996, but unrelated scenes were .614 similar, noise margin was negative, and occupancy/direction transfer collapsed. Pattern completion was not selective enough.
- **Synchrony — reject this implementation.** Learned coupling changed no reported representation metric relative to frozen initialization. Its motion-direction capacity was worse than the sensory control; separated components and crossing alone do not rescue it.
- **Relational graph — reject as the M1A state, retain a partial memory signal.** It had the strongest noise and missing-gap margins while preserving oracle direction decodability. Yet 4× spatial pooling reduced position/speed/shape occupancy to .261/.170/.218 and crossing identity to zero. The graph captured scene history but discarded needed spatial detail.
- **Reservoir — reject as the M1A state, retain a partial memory signal.** Fixed recurrent dynamics preserved oracle direction and improved missing-gap margin over the sensory control. The learned local readout barely changed transfer, occupancy lagged badly, and crossing identity was zero. These results do not support learned recurrence as necessary, but this reservoir is not a usable world-state.

The highest-value reassessment is now the **representation test itself**: the sensory control solves the simple direction and separated-object probes, while the learned candidates that retain hidden history often discard spatial detail. A next M1A proposal should explicitly require added information beyond the observer/control under longer occlusion, ambiguous overlapping patterns, and independently moving entities, with a decoder that does not receive oracle object regions. That is a new architecture/benchmark discussion, not another parameter sweep of these seven runs.

All results were written after each family and are in the [raw result file](2026-09-26-representation-family-results.json). The entire shared preparation and seven-family comparison took 195.1 seconds. Reproduce with `.venv/Scripts/python scripts/run_representation_family_audit.py`. The observer's rendered-intensity training teacher remains an experimental assumption, as in the prior plan.
