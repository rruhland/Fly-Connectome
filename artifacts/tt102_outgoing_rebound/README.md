# TT102 · Evaluate a return by the far-table bounce it would cause

**The rebound reranker shortened the ordinary rallies.** In six matched complete TT80 games, the frozen controller scored 30 legal returns and the reranker scored 16. Three games changed. Both players landed the first different stroke on the same exact incoming in all three, so the immediate first-return gate was not the main failure. Three other games reproduced the original commands and event tapes unchanged.

| Release | Frozen | Reranked | First changed stroke and the common next target |
| --- | ---: | ---: | --- |
| 14645 | 7 | 2 | Both first shots and both next balls landed. Actual next projected arrival 10.8 mm lower with reranking. |
| 12265 | 15 | 2 | Both first shots landed. The planned +8.2 mm next-height change became −0.6 mm when run through the model on the achieved far arcs. The actual next ball was 35.6 mm lower, in the same central lane. Frozen landed it and reranked missed. |
| 4888 | 4 | 8 | Both first shots and both next balls landed. Actual next arrival 26.5 mm higher despite the achieved-arc model predicting a modest drop. |
| Three identical games | 4 | 4 | No changes. |

The differences in projected arrival refer to three separate common-seed comparisons. They are not estimates of the expected distribution of future shots or broad performance. The full games had the same starts, carried frozen state, opponent goal streams and referee. We stopped at this bounded negative result without retuning the rule. [Actual regular games](games.json.gz) · [Summary and exact first afterstates](summary.json)

## Separate the flight forecast from the choice

The actual pre- and post-bounce arcs forecast the next arrival **height** better than they forecast which branch the arm will return. Across 17 independently matched pairs of actual scored outgoing strokes, the predicted relative height correlated 0.759 with the real mean height from six common future draws per pair. The predicted relative next-return probability correlated 0.481 with actual scoring, but picked only four of the nine directions with a nonzero real score contrast. It was worth about +0.015 next legal returns over uninformed routing, with a release-bootstrap 95% interval from −0.039 to +0.078. That does not qualify an achieved-stroke tactical gain. [Matched achieved-stroke pairs](achieved-pairs.json)

Flight prediction was not simply broken. Two disjoint shadow games reproduced 13 and 6 returns exactly, giving 19 time-matched planned-to-camera rebounds and eight candidate reranks that were not issued. A small fixed median correction, bounded in each feature dimension, was fitted from those observed rebounds. In held games, 15 later camera rebounds could be aligned to an issued plan. The projected **expected** next-reply height from the plan differed by a median of 2.91 mm from the same predictor on the actual camera cue, or 2.12 mm after correction. But on 20 matched regular-game camera rebounds the predictor disagreed by a median of 29.0 mm with the **single next ball actually drawn**. The physical reply variation and errors in ranking two very close alternatives matter far more than the typical scalar plan-to-camera gap here. [Native shadow and calibration](calibration.json.gz)

The 2,593-transition teacher uses the TT94 two-arc geometry and TT82's receiving-skill estimate. It excludes all prospective TT80 releases and all held paired releases. In play, the actor sees nominal and two existing optical uncertainty trajectories from the ordinary public RGB/encoder-initialized twin. It only reranks the existing legal candidates and never accepts a decrement in the pre-existing legality votes. No future goal or live opponent simulator state enters the decision. The received optical rebound remains useful for anticipation even though this ranker did not improve shooting.

[Frozen protocol](protocol.json) · [TT94 optical calibration](../tt94_optical_rebound/README.md) · [TT82 independent future draws](../tt82_partner_model/README.md)
