# Testing what an actual hit or miss teaches

These simulated games use a finite-mass opponent paddle that moves to strike the actual incoming ball. Its first and subsequent target positions and speeds are independently continuous draws. Each score below is a legal arm landing. The longest point counts the arm's own returns. All 12 or 24 assigned games count, including any failed opponent opening.

## Corrections after hits and misses

The carried direct controller observes calibrated stereo RGB and encoders, remembers its own issued joint targets on rewarded approaches, and screens movements using a camera-initialized private model. No network gradients are involved.

| Same 12 starts | Total legal arm returns | Longest point |
| --- | ---: | ---: |
| Carried direct controller | 109 | 35 |
| Reconstruct a corrective movement after a terminal miss | 71 | 17 |
| Credit predicted stroke effects for surviving the next incoming ball | 74 | 14 |
| Adjust joint rates using paired local velocity estimates | 69 | 12 |
| Associate scored joint movements with subsequent rally length | 101 | 21 |
| Warp scored arm sequences to match current flight and body motion | 85 | 23 |
| Offer geometric tutor actions only when the direct search is weak | 109 | 35 |
| Let the tutor compete with qualified direct candidates | 69 | 13 |
| Change readout on the first camera exposure showing the table bounce | 104 | 20 |

Some changes rescued individual incoming balls. In the camera trial, 12259 went from one to 16 returns. The first altered joint command came after an identical physical prefix and the arm returned the common next ball. The same readout shortened the 35-return game to 17. Reach and execution continue to limit performance even when an optical motion change is detected early. [Stock](../tt39/credit-carry.json.gz) · [Miss corrections](after-misses.json.gz) · [Continuation value](continuation-value.json.gz) · [Rate correction](rate-secants.json.gz) · [Future reward on past movements](return-memory.json.gz) · [Motor warp](motor-warp.json.gz) · [Tutor](joint-tutor.json.gz) · [Competing tutor](joint-compare.json.gz) · [Camera comparison and causal replay](early-optics.md)

## Direct policy without a rollout search

On the same 24 full games, the original direct action-value learner made 69 returns when updating and 38 with a frozen readout. Adding examples from 48 more games of geometric motor practice produced **46 with updates and 34 frozen**. The larger training set retained some benefit from online updates and reduced overall performance. A second, planner-free policy that phase-aligned scored motor sequences made 33 frozen and 22 online. Both results identify interference from the way recent examples change execution. [Original controls](../tt39/direct-action-comparison.json) · [Additional earned examples, updating](earned-direct-live.json.gz) · [Additional examples, frozen](earned-direct-frozen.json.gz) · [Sequence readout, updating](sequence-online.json.gz) · [Sequence readout, frozen](sequence-frozen.json.gz)

The full 24-game comparison of repeated joint replanning against committing once to a stroke gave 223 returns with replanning, 117 with commitment and 120 with commitment plus fixed carried practice. All three longest points replayed with physical scoring and zero audited CAD penetrations. The original replanner took 350 CPU seconds in search across that comparison, versus 52–55 for the commitment runs. This favors investing computation when the arm acquires new evidence during the approach. [Complete 24-game comparison](../tt39/commitment-replication.md)

In a separate 192-game course, retention varied sharply across groups of unfamiliar openings. That test used the geometric controller and a held-out evaluation at successive training checkpoints. [Longitudinal scores and controls](../tt40/continual.md)
