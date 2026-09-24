# Raw events restore first-sighting activity but disrupt local forecasting

The sensory blind spot is deterministic in the frozen occlusion suite: across eight cases, frame 10 reappearance produced **116 raw ON/OFF events** and **zero** same-polarity prior/current coincidence activations. The coincidence-only 24-unit latent was silent at that frame. Adding raw ON/OFF events as two input channels to the same competitive locally learned circuit made the augmented latent active at all **116** reappearance event sites. First appearance at frame 2 has the same upstream property. No object or motion labels entered the new circuit.

The registered primary forecast still decoded **only predicted coincidence channels**, leaving predicted raw channels out of event scoring. All 24 units, local rules, 96 training episodes, seeds, 5×5 fields, and next-frame horizon were otherwise unchanged.

| Held-out event F1 | Coincidence-only learned code | Raw + coincidence learned code | Raw + coincidence random code |
| --- | ---: | ---: | ---: |
| Unseen single patterns | 0.669 | **0.422** | 0.263 |
| Independent movers | 0.681 | **0.404** | 0.249 |
| Crossings | 0.583 | **0.359** | 0.255 |
| Speed changes | 0.634 | **0.414** | 0.215 |
| Brief disappearance | 0.430 | **0.123** | 0.162 |
| Sensor-bit noise | 0.487 | **0.122** | 0.146 |

At the exact reappearance-to-next-frame transition (t=10→11), the augmented learned forecast reached only **0.045 F1**; the coincidence-only learner was zero because it had no t=10 input, but the augmented random-code control reached 0.138. The learned augmented code retained post-hoc direction readability on **45/48** unseen single-pattern cases versus 47/48 for the prior code, so direction information was not simply erased. Quiet-target false-alarm pixels were 92 versus the previous 97. The registered improvement and non-regression gates failed decisively. This training/evaluation run took 147 seconds.

Do not add raw channels to production or conclude that raw events are unhelpful. The result separates **sensory access** (fixed) from **using the added evidence predictively** (failed). The immediate next diagnostic should freeze the augmented code and fit the same evaluation-only local linear future-event decoder used before, alongside its training F1. If offline capacity transfers well, redesign local prediction target/readout to use the available code. If capacity itself collapses, prevent raw first-sighting events from competing with ordinary motion correlations in the learned dictionary, for example through a separately learned low-level re-anchoring pathway. This is a diagnostic branch, not a final architecture. Production M1A remains unchanged.
