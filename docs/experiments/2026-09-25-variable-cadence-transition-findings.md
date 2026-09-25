# Mixed-cadence local updates overwrite fast motion without learning native timing

**Status:** Opt-in one-pass learning test under the [variable-cadence protocol](2026-09-25-variable-cadence-transition-protocol.md), with [results](2026-09-25-variable-cadence-transition-results.json). Production remains unchanged.

The original learned sensory dictionary was frozen. The same local recurrent update rule trained on 160 generic event-camera episodes spanning five shapes (including a dot), four directions, both contrasts, varied positions, and speeds 0.25, 0.5, 1, and 2 pixels per camera frame. Both an arm initialized from the successful fast-motion transition and a zero-initialized arm learned for one pass from the four-frame local event-trace input. No simulator speed, shape, object, Pong state, or gradient entered learning.

| Held-out test, spatial occupancy F1 | Frozen fast transition | Variable-cadence adapted | Variable-cadence from zero | Persistence |
| --- | ---: | ---: | ---: | ---: |
| Native-rate Pong code-active frames | **0.035** | 0.006 | 0.006 | **0.073** |
| Generic independent patterns | **0.749** | 0.662 | 0.661 | 0.342 |
| Generic crossing patterns | **0.683** | 0.617 | 0.618 | 0.306 |

The adapted transition changed by 40.9 summed absolute weight units relative to the frozen fast transition, and both learned arms ended with almost the same total weight (37.0 versus 36.4 frozen). Native predicted/target active-site ratio fell from 1.44 to 0.39, and only 2 of 509 native target sites were hit. This is a failed learning outcome, not a failure to execute plasticity. The arm also regressed by 0.087 F1 on independent patterns and 0.066 on crossings, exceeding the registered 0.05 tolerance.

The update only receives a next-state mismatch where local visual evidence exists. That protects an occluded pattern from being falsely trained away on a blank camera frame, but it also gives no direct target for the subpixel intervals when the pattern persists without a new event. **Inference:** one translation-shared transition weight bank may be asked to move immediately after fast events and to retain state through slow/event-sparse intervals, without a local cue in its current update to decide which. The data do not prove this is the only cause; the frozen dictionary and synthetic/native appearance gap may also matter.

Do not promote variable-cadence training or tune its scalar threshold. The next useful opt-in architecture test should add a genuinely local temporal-context state at the hidden transition—such as site-local persistence/adaptation or short-term synaptic state—and compare it against the current fast transition on native and generic scenes. It must keep hidden state distinct from visible-event emission and train without backpropagation or object labels. If a bounded local-state test cannot preserve both cadences, revise the M1A architecture proposal rather than continue this rule family.
