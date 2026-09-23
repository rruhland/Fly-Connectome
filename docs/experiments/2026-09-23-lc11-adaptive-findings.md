# A local 250 ms baseline reduces blank release but does not recover LC11 detection

The [bounded opt-in follow-up](2026-09-23-lc11-adaptive-readout-protocol.md) used the same measured T2/T2a/T3→LC11 contacts, signs, weights, one-tick delay, and frozen upstream network as the fixed-reference test. Its only change was a presynaptic baseline trace updated from each source's own voltage with a fixed 250 ms time constant. The [results](2026-09-23-lc11-adaptive-results.json) identify the exact input tables and ignored per-tick release/current/spike artifact.

| Maximum release/source/tick | ON excess LC11 spikes | OFF excess | Blank spikes/neuron/tick | Mean blank release/source/tick |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0 | 0 | 0.00069 | 0.01529 |
| 0.10 | 0 | +1 | 0.01097 | 0.03059 |
| 0.25 | -1 | +3 | 0.04181 | 0.07647 |

The adaptive trace lowered blank activity relative to the fixed warm reference (at `f=0.10`, 79 versus 140 blank LC11 spikes over the 96-tick transit window), but it still produced no ON excess at any tested fraction, and the two larger fractions exceeded the preset blank-rate limit. The smallest fraction was quiet enough but had zero dot-evoked spike excess. No fraction passed; no second-location, direction, bar, or static readout was run. A further release-gain or baseline-time-constant sweep is not justified by this result.

The opt-in readout deliberately included only the excitatory T2/T2a/T3 branch. In the measured LC11 graph, other selected source classes supply substantial input, including inhibitory classes. Fly experiments find LC11's GABA-sensitive inhibition important for small-object detection and local-surround response shaping ([Keleş et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7210872/)). The next *different* hypothesis is therefore to replay **all measured selected-source LC11 inputs with their fixed signs** and test whether native inhibition/competition can suppress autonomous release while retaining the dot's local signal. That should be one bounded opt-in comparison, not another scalar gain sweep. If it also fails, an LC11 point-soma sum is unlikely to be the right next in-place architecture; retinotopic dendritic subunits or upstream temporal filtering would need a separate proposal. No production model or learning rule changed here.
