# Distributed local state fixes fragmentation geometrically, not through learned binding

**Status:** One-shot opt-in pivot under the [registered protocol](2026-09-26-distributed-predictive-sheet-protocol.md). The frozen fast predictor and 64 unlabeled generic training streams were unchanged. All sheet arms shared learned graded feedforward templates and a signed future-error decoder; only local recurrence differed (aligned, time-shuffled, frozen self-recurrence, or zero). Production M1A and the measured graph were not changed. No object labels, oracle regions, direction targets, reward, Pong semantics, or backpropagation entered training or inference.

The distributed state removed the *component-count symptom*: each sheet arm had 2.00 occupied components per active frame for two separated movers, versus 3.44 for the rejected sparse paired state. But **zero recurrence had the same 2.00**, so this is a consequence of local spatial support/graded occupancy, not evidence that learned recurrence discovered entity binding. At crossings the occupancy formed only about 1.06 components for two movers, so the component count alone overstates object separation. Aligned recurrence placed **40.9%** of activity outside offline mover neighborhoods on separated scenes; zero recurrence placed 27.7% outside. The learned feedback increased spillover rather than improving spatial selectivity. Offline coordinates were used only for this evaluation.

Full-frame next-meaningful-event top-32 recall:

| Held-out set | Fast only | Aligned sheet | Shuffled sheet | Frozen sheet | Zero recurrence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic scenes | **.360** | .312 | .314 | .315 | .314 |
| Doubled speed | **.480** | .370 | .365 | .338 | .375 |
| Changed shape | **.542** | .513 | .529 | .500 | .535 |
| Separated movers | **.312** | .263 | .266 | .268 | .266 |
| Crossing movers (2 scenes) | .354 | .344 | **.359** | .323 | .344 |

Aligned recurrence beat shuffled recurrence on 12/16 speed cases but only **2/16** shape cases, and it beat fast-only on **0/16** speed cases. It did not preserve fast-only speed recall within the registered three-point margin. The zero-recurrence sheet nearly matched fast-only on changed shape (.535 versus .542), suggesting the spatially distributed sensory code can retain some shape generality; learning the current recurrent rule did not improve it. Top-8 recall on speed was .141 aligned versus .151 fast-only, and top-32 precision was .135 versus .175.

The fixed offline 3×3 direction probe found 16/16 speed decoding for every sheet arm, but aligned changed-shape decoding was **10/16**, below the registered 14/16 gate; frozen scored 12/16 and zero recurrence 4/16. These readings require known object centers after inference and cannot establish autonomous entity tracking. The sheet's behavior reflects a distributed visual field with some transferable information, but no learned, stable object-level state.

**Decision:** Reject this distributed-sheet implementation as an M1A world model. It improved spatial coverage without a learned binding advantage and missed prediction transfer, spatial selectivity, and shape-code gates. Do not tune local normalization, recurrent gain, or decoder footprint against these held-out scenes. The next architecture should make **persistent, generic entity hypotheses** explicit: group local event evidence by learned temporal compatibility, maintain an identity/state through missing events and crossings, and learn each hypothesis's future sensory consequences from local prediction error. Slots or object files would be an inductive bias, but their content, associations, dynamics, and number must arise from visual evidence rather than Pong labels, hand-coded motion directions, or oracle crops. The fast field can remain a sensory primitive. Compare true online association and forecast with shuffled/frozen controls, and score identity continuity as well as spatial coverage; do not mistake two occupied blobs for two tracked entities.

Reproduction: `python scripts/run_distributed_predictive_sheet.py`; [raw results](2026-09-26-distributed-predictive-sheet-results.json) retain per-scene forecast counts, recall/precision, spatial probes, activity/coherence measures, and elapsed time (about 36 seconds).
