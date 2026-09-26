# Local affinity fixes visual fragmentation, but does not solve world-state prediction

**Status:** Completed one-shot opt-in comparison under the [registered protocol](2026-09-26-local-affinity-binding-protocol.md). No production architecture, connectome topology, transmitter sign, motor learner, or backpropagation changed. The [raw results](2026-09-26-local-affinity-binding-results.json) contain per-case forecast counts, live-file counts, crossing identity, stress cases, learned couplings, and the matched prior fast-only reference.

The prior generic overcount was primarily a **proposal-geometry failure**. The four-neighbor component rule split diagonally touching pixels in held-out shapes into separate files on the first frame, and those files lasted through the episode. A shared, signed local affinity learned from 64 unlabeled generic streams activated all four diagonal relations; time-shuffled credit activated none. Same-sign aligned co-occurrence counts were 623–941 per diagonal relation versus 11–21 under the shuffled null. The aligned conditional weights were .085–.117, versus .0007–.0022 for shuffled credit. No object identity, shape label, true count, or held-out threshold entered this fit.

| Generic held-out measure | Four-neighbor | Learned affinity | Shuffled affinity | Fixed eight-neighbor |
| --- | ---: | ---: | ---: | ---: |
| Wrong count, two expected regions | 213/447 | **10/447** | 213/447 | **10/447** |
| Wrong count, three expected regions | 190/304 | **4/304** | 190/304 | **4/304** |
| Mean files, two-region scenes | 2.46 | **1.98** | 2.46 | **1.98** |
| Mean files, three-region scenes | 3.80 | **2.98** | 3.80 | **2.98** |
| Top-8 next-event recall | .246 | **.293** | .246 | **.293** |
| Top-32 next-event recall | .273 | **.331** | .273 | **.331** |

The trained fast-only field still achieved **.360** generic top-32 recall. Aligned affinity beat four-neighbor on 6/16 generic cases; it did not improve the controlled speed (.857), shape (.843), separated (.789), or crossing (.823) top-32 scores because those shapes were already connected. Clean crossing identity switches remained **four** frames. Static-noisy entity count stayed wrong on **13/13** frames, ending at five files for one intended entity. The different-shape crossing kept two files on 13/13 frames in every arm. The expected generic region count is the generator's moving motifs plus optional static mark and is approximate during overlap; it is used only for scoring after inference.

**Decision:** Retain this opt-in local affinity as a promising generic sensory primitive, but do not promote the entity-file system to production M1A. The learned arm exactly matched fixed eight-neighbor grouping on every measured outcome, so the strong count improvement is not evidence that a learned *entity representation* has emerged. Better local connectivity removes fragmentation while leaving forecast coverage, noisy false births, and crossing identity as separate failures. Do not sweep connection thresholds or neighborhood radii on the held-out sets.

The next substantial hypothesis is that the entity file needs a **shared, locally plastic prediction/reliability mechanism** on top of its now coherent sensory proposals: learn which local sensory transitions an existing file can explain and make unmatched evidence compete for birth based on predictive confirmation. Compare its full output with fixed geometry, shuffled credit, and the fast-only field. If it cannot improve generic next-event coverage and noise/crossing stability without losing clean transfer, this entity-file family should be reconsidered rather than extended through small gates.

Reproduce with `.venv/Scripts/python scripts/run_local_affinity_binding.py` (about 17 seconds on this machine).
