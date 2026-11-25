# Carrying the direct joint learner across 24 unfamiliar games

One motor memory was carried across each complete course. Every arm began with the same 12 games of prior practice. Before running, we fixed 24 opening recordings absent from both fitted readouts and earlier direct training and assessment. The opponent tracks the physical incoming ball and strikes it with a finite-mass paddle. Its continuously sampled instructions span ±0.28 m laterally, a 0.12 m depth range, a 0.10 m height range and 3.15–3.98 m/s in longitudinal pace. The resulting physical ball depends on both sides' actions. Two opening assignments never produced an incoming legal ball. All 24 were retained. [Assigned releases, draw seeds and training exclusions](fresh.json)

| Feedback available | Legal arm returns / actual offers | Median | Peak | Points reaching 10 arm returns |
| --- | ---: | ---: | ---: | ---: |
| Original direct controller | 101 / 123 | 2 | 20 | 4 |
| Short visual and encoder history | 141 / 162 | 3.5 | 24 | 6 |
| Learned ranking of outgoing choices | 74 / 96 | 2 | 12 | 2 |
| History plus outgoing ranking | 197 / 218 | 7 | 35 | 10 |

The combination extended 14 assigned games, shortened three and tied seven versus the original. One of the longer points went from two to 35, but the advantage persisted on other releases. The combined player returned 16 of 26 incoming balls below 0.95 m at the deep receiving line, 77 of 85 between 0.95 and 1.00 m, and 104 of 107 above 1.00 m. These are the physical balls that reached each band under this policy. Receiving different balls after returning differently is part of the task. [All scores, paired differences and native opportunities](report.json)

The short-history readout and the outgoing predictor were fitted before these games and frozen in play. The direct motor memory kept learning from each scored return within and across games. The outgoing predictor's earlier training used the next real miss and a diagnostic physical height label. The live player received stereo-derived ball flight, encoders, its own issued movement and scoring feedback. Its private forward search uses calibrated body and ball dynamics. This result establishes improved play on an unfamiliar continuous stream. It does not isolate the contribution of updating the motor memory inside each particular rally. [Combined actions and learned state](combined.json.gz) · [Matched original](stock.json.gz)
