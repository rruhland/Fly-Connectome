# Simple local hold broadens hidden state without native prediction gain

**Status:** Opt-in architecture test under the [local persistence protocol](2026-09-25-local-persistence-protocol.md), with [results](2026-09-25-local-persistence-results.json). Production remains unchanged.

The successful fast-motion transition weights were frozen. At each hidden site without current sensory evidence, one arm used the usual learned moving continuation; the other took the stronger of that continuation and a decayed copy of its own previous state (`exp(-1/4)` per camera frame). The additional branch was local, label-free, and never emitted a camera event.

| Held-out occupancy F1 | Unchanged transition | Local hold | Previous-code persistence |
| --- | ---: | ---: | ---: |
| Native Pong, code-active frames | **0.035** | 0.026 | **0.073** |
| Generic independent patterns | **0.749** | 0.729 | 0.342 |
| Generic crossing patterns | **0.683** | 0.682 | 0.306 |

At native timing, the local hold predicts **2.90×** as many active sites as the target, violating the registered 2× bound without gaining accuracy. On paired generic three-frame gaps, last-blank hidden-state F1 falls from **0.297** to **0.260** while the active-site ratio grows from **1.39×** to **2.66×**. Holding position and advancing a moving branch together creates spatial spread, not a useful single forecast. The arm does preserve most ordinary independent/crossing performance, but this does not compensate for the native and gap failures.

Stop simple hold/decay variants. Across the native-rate tests, the immediate correlation front end was starved; a four-frame local trace wakes it, but the frozen transition, event-indexed execution, mixed-cadence local weight updates, and an added local hold all fail to predict the native code reliably. This calls for a changed temporal representation and prediction target, not another scalar threshold. The [next architecture investigation](2026-09-25-observation-state-and-horizon-proposal.md) separates a local persistent observation state from a learned hidden world state and tests a forecast horizon tied to meaningful visual change. No in-place architecture change is approved or made.
