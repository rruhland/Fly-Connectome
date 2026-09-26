# Slow assemblies retain spatial motion information, but their local objective does not learn the desired state

**Status:** One-shot opt-in M1A architecture test under the [registered protocol](2026-09-26-two-timescale-assembly-protocol.md), followed by one explicitly [post hoc spatial-probe check](2026-09-26-spatial-state-audit-protocol.md). Production M1A, measured connectome, and motor learning are unchanged. The fast field from the preceding experiment was frozen after local no-backprop training. Sixteen shared slow assembly motifs learned to reconstruct current and previous fast activity from local patches; a locally error-trained recurrent kernel predicted the next active assembly. No object region, direction, speed, label, or Pong-specific feature entered training or full-frame inference.

The slow circuit did not preserve the fast predictor's transferable forecast. Top-32 recall at the **next meaningful event** was:

| Held-out set | Fast only | Aligned slow | Frozen slow | Shuffled slow |
| --- | ---: | ---: | ---: | ---: |
| Generic held-out shapes, 16 scenes | .360 | **.376** | .266 | .216 |
| Doubled speed, 16 scenes | **.480** | .197 | .177 | .139 |
| Changed shape, 16 scenes | **.542** | .251 | .275 | .180 |
| Separated movers, 8 scenes | **.312** | .184 | .148 | .141 |
| Crossing movers, 2 scenes | **.354** | .286 | .224 | .208 |

Aligned slow credit beat its slow controls on generic held-out scenes in 14/16 and 16/16 cases, respectively, but the absolute +.016 gain over fast-only is small and did not transfer to the registered speed, shape, or multi-entity cases. Thus the slow circuit has a learned predictive signal, but not a useful replacement for the fast forecast.

The registered direction probe averaged each state channel over offline object neighborhoods after full-frame inference. It scored aligned slow assemblies 16/16 on held-out position and 16/16 on separated movers, but the frozen and shuffled slow controls also scored 16/16. Doubled-speed transfer was 6/16 aligned, 8/16 in each slow control; changed-shape transfer was 4/16 in all slow arms; crossings were 1/4 aligned versus 3/4 frozen. On doubled-speed scenes, 44.5% of significant aligned assembly activity lay outside offline object neighborhoods, versus 42.2% frozen. Single movers produced 1.43 occupied assembly components per active frame on average. These are evaluation measurements; no object mask or target count shaped the model state.

Because a channel average can erase a spatial trail, a **single post hoc** capacity probe retained a fixed 3×3 arrangement of state activity relative to the offline evaluation center. It used the same training configuration and nearest-centroid calibration, with no crop-size or classifier search:

| Offline spatial probe | Fast only | Aligned slow | Frozen slow | Shuffled slow |
| --- | ---: | ---: | ---: | ---: |
| Doubled speed | 12/16 | 14/16 | **16/16** | 14/16 |
| Changed shape | **10/16** | 6/16 | 8/16 | 6/16 |
| Separated movers | 14/16 | 16/16 | 16/16 | 16/16 |
| Crossing movers | 2/4 | 3/4 | 2/4 | 3/4 |

This revises one inference: **motion information is spatially present in the field**, and mean pooling substantially understated its speed capacity. It does **not** rescue the registered architecture result. Spatial decoding used oracle centers after inference; aligned slow learning did not beat frozen recurrence on speed, failed changed-shape transfer, and weakened the forecast where it most needed to generalize. These two-scene crossings are too small to rank architectures confidently.

**Decision:** Do not promote or tune this two-timescale assembly variant. Preserve the fast local predictor as a useful no-backprop dynamic primitive and the spatial-probe result as evidence that a downstream reader must retain spatial organization. The next major architecture question is the *learning objective for the slow state*: predicting the next arbitrary sparse winner is apparently misaligned with preserving coherent, reusable motion. A distinct follow-up should let a persistent assembly earn local credit for reducing future fast sensory prediction error over an event interval, with competition among explanations and an autonomous entity-coherence check. Merely increasing decay, spatial crop, assembly count, or recurrent gain would not answer that question.

Reproduction: `python scripts/run_two_timescale_assemblies.py` and `python scripts/run_spatial_state_audit.py`. [Registered raw results](2026-09-26-two-timescale-assembly-results.json) and [spatial-probe raw results](2026-09-26-spatial-state-audit-results.json) retain the control scores and per-scene forecast counts.
