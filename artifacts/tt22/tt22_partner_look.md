# Choosing replies for the actual RGB pupil

The fixed rival chooses physical shots from geometric placement and strict scoring. The contextual rival considers the same physical motor proposals on the same five-axis arm. It also estimates what the frozen RGB pupil will do on each proposed incoming arc. The pupil keeps its ordinary two fixed RGB inputs and its existing controller. Each point has one initial release and runs until the strict referee ends it. [Implementation](../../scripts/experimental/tt22_partner_look.py)

The archive contains 67 legal far replies from 14 earlier five-axis point seeds. Forty-five led to a legal pupil return and 22 did not. Every arc provided features at a common depth before the pupil's contact, including lateral position, height, three velocity components, three spin components, bounce depth, net height and flight time. These evaluator features came from exact physical replay of the original two motor tapes. The model also uses the total legal exchanges remaining in each point, capped at ten in the training objective.

A Gaussian kernel weights archive episodes by trajectory similarity and balances the contribution of whole point seeds. Leaving out each source seed gave a next-return Brier score of 0.261, compared with 0.266 for the unconditional rate. The rival uses both predicted next-return probability and expected subsequent legal returns to rank its proposals. Up to six undergo exact checks using the same referee. The rival rescores the resulting physical trajectory before choosing. The model fits before the independent matched assessment and stays fixed through it. [Model, held starts and two-arm tapes](tt22_partner_look.json.gz)

| Fresh start | Fixed legal alternating returns | Contextual legal alternating returns | Fixed point ends | Contextual point ends |
| --- | ---: | ---: | --- | --- |
| `166544231` | 2 | 4 | Own-half bounce | Own-half bounce |
| `1323299260` | 4 | 4 | Floor | Fixture contact |
| `734127872` | 4 | 2 | Own-half bounce | Floor |
| `655133517` | 2 | 2 | Floor | Floor |

Each arm earned six legal returns under each condition, and the learner finished 0–4 in both. Neither condition reached ten returns. The far controller changed its motor tape on all four points. Initial learner contacts and pre-contact commands were identical across counterparts. All eight points independently reproduced every referee event from the two motor tapes.

The fixed and contextual rivals each ran 36 exact candidate checks. Median wall time per confirmed planning decision was 3.08 seconds fixed and 4.06 seconds contextual in this host run. Similarity support for the chosen contextual trajectories ranged from 0.13 to 0.90 seed-weight equivalents. On this small independent sample, the learned ranking changed individual exchanges and left their total unchanged.

```bash
python -m scripts.experimental.tt22_partner_look --count 4 --output /tmp/tt22-partner-look.json.gz
```
