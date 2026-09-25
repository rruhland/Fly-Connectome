# Competition during latent learning reduced density but not credit ambiguity

**Status:** Opt-in M1A architecture experiment. Production architecture, measured optic-lobe graph, and motor learning remain unchanged. The experimental dictionary, recurrent transition, and forecast heads use local updates without backpropagation or privileged scene labels.

The [post-hoc sparse-state test](2026-09-25-sparse-coherence-forecast-findings.md) preserved geometric support but did not improve native forecasts. We therefore moved one-pixel nearby-site competition into the learned sequence dictionary itself. Each candidate hidden unit matched a local 5×5 patch of the current and previous signed eight-frame event stacks; the strongest nearby site won **before** local Hebbian dictionary credit and recurrent updates. The previous per-site dictionary was the matched unconstrained control. All models trained on 64 generic fractional-motion episodes and were evaluated on held-out generic scenes and zero-shot native 120 Hz Pong. A temporally shuffled recurrent-credit head checked whether aligned future targets caused any gain. Output budgets were calibrated only on generic training data.

| Held-out domain | Head | Spatial-competitive F1 | Previous per-site F1 | Spatial-competitive top-eight recall | Previous top-eight recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Generic fractional scenes | Direct | .029 | .015 | .076 | .067 |
| Generic fractional scenes | Recurrent | .030 | .015 | .077 | .067 |
| Generic fractional scenes | Shuffled recurrent credit | .032 | .004 | .089 | .022 |
| Native Pong | Direct | .022 | .000 | .105 | .018 |
| Native Pong | Recurrent | .022 | .000 | .095 | .014 |
| Native Pong | Shuffled recurrent credit | .035 | .008 | .097 | .062 |

Spatial competition cut native direct source activity from 8.64 to 3.74 sites/frame. It increased the dense sequence code's native direct ranking, but the shuffled-credit head matched or exceeded the aligned head on both held-out domains. Native quiet false alarms remained high at 2.43 pixels/frame for the aligned recurrent head, close to 2.39 for the previous per-site model. Independent and crossing generic cases also remained weak. Recurrent prediction added essentially nothing over the direct sparse code.

**Decision:** Do not promote this spatial dictionary or its output. Stop tuning this per-site lag-stack family: sparsity alone cannot establish a coherent predictive latent when time-shuffled credit performs as well or better. The next architecture question should compare *learned recurrent motion state* against simple local sequence dictionaries on a controlled, domain-shifted representation task before adding another forecast head. The latent should be tested for stable motion information across position, speed, and concurrent entities, with evaluation-only probes and randomized-credit controls. Measured T4/T5-related responses should be included as a candidate local sensory substrate, while the useful latent remains learned. Only a representation that survives those tests warrants another production proposal or end-to-end M1A forecast run.

Reproduction: `python scripts/run_learned_spatial_competition.py`; [raw results](2026-09-25-learned-spatial-competition-results.json) include source density, signed counts, quiet alarms, ranking, and controls.
