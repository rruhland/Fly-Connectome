# Direction is already locally readable from individual learned history units

The [registered post-hoc locality probe](2026-09-24-history-code-locality-protocol.md) retrained the same unlabeled history-gated visual code and froze it. Direction labels fit only discarded nearest-mean evaluation probes on 64 generic interrupted training-shape episodes. The probes then tested 608 reappearance event sites across 48 unseen-shape episodes. No labels, probe weights, or gradients changed the visual model. The run took 82 seconds.

| Held-out direction accuracy | Learned history code | Random history code |
| --- | ---: | ---: |
| Individual event-site unit | **599/608 (98.5%)** | 269/608 (44.2%) |
| 5×5 local unit neighborhood | **98.5%** | 66.4% |
| 9×9 local unit neighborhood | **98.5%** | 68.9% |
| Whole-episode unit histogram | **47/48 (97.9%)** | 36/48 (75.0%) |

For the learned code, the individual-unit probe was correct on 143/152 up sites and **all 152** sites in each of down, left, and right. Nearby population pooling gave no improvement. This rejects the previous hypothesis that direction is readable only from a broad histogram and unavailable to individual local emitter synapses. The representation is locally legible under this probe; the online per-source emission rule's zero true positives for three directions in the 144-case transfer test must arise downstream of code selection. The probe does not show that future event location and polarity are linearly predictable from one unit; it only isolates direction availability.

The next bounded experiment should change **one** local credit normalization while preserving the learned code, event target, data, threshold, and baseline: average eligible error separately for each presynaptic unit rather than dividing every source's update by the total number of active sources across the frame. This is a local synaptic update, not backpropagation. Compare the old and new rules, trained random-code control, and trace reset on the full 144-case interruption suite. If the new rule still fails to transfer across directions or causes quiet false alarms, stop this rule and reassess the event objective; do not sweep gains or thresholds.
