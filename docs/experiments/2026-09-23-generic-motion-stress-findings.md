# Generic motion stress findings

The opt-in stress script froze the local-competition learner and tested next **clean** event prediction from visual events. It supplied no object, path, speed, or game metadata to any predictor. Event F1 was:

| Scene | Learned competitive | Fixed competitive | Fixed raw coincidence |
| --- | ---: | ---: | ---: |
| Clean translation | 0.977 | 0.977 | 0.955 |
| Direction change | 0.848 | 0.848 | 0.815 |
| Speed change | 0.857 | 0.857 | 0.784 |
| Same-polarity crossing | 0.855 | 0.855 | 0.729 |
| Passing an occupied region | 0.784 | 0.784 | 0.652 |
| 0.1% false events/polarity/pixel plus 10% dropout | 0.888 | 0.888 | 0.865 |
| 0.5% false events/polarity/pixel plus 10% dropout | 0.831 | 0.831 | 0.783 |

Noise results aggregate five seeded event streams. Competitive F1 ranged 0.845–0.944 at the lower rate and 0.803–0.874 at the higher rate. The fixed and learned competitive predictors made the same binary forecasts in all cases; learning has not yet earned a role. These scenes are synthetic and small, and event F1 alone cannot verify persistent entity identity, occlusion reasoning, or long-horizon dynamics.

Local displacement competition is a promising **generic front-end hypothesis**. The next architectural question is whether local visual evidence can form stable, reusable coherent-entity state and whether a causal, local rule can learn context-dependent transitions better than this fixed front end. An occupied region is the clearest stress failure to target with persistent visual context, without encoding any particular game boundary or collision type.
