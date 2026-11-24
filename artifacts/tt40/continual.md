# What survived continued practice on random rallies

The arm and opponent physically play every point. Each new far reception draws a continuous placement and speed, then the opponent tries to realize that request against the actual incoming ball. The score below counts successful arm returns. Each comparison uses all twelve assigned measured openings, even if the opponent never gets an opening ball to the learner.

## Carrying what the arm did, and carrying how it valued it

A single learner practiced through a sequence of 96 measured openings with independently randomized shots. It made 338 arm returns in its first 96 games. Both its bank of successful joint corrections and its estimates of which predicted outcomes would score were allowed to change. At the indicated checkpoints we froze them and played independent releases. Both comparators used the same pretrained visual and motor machinery.

| Condition | First 12 assessment openings after 48 practice games | Different 12 assessment openings after 96 practice games |
| --- | ---: | ---: |
| Original motor bank and value estimate | 36 returns, longest 12 | 40 returns, longest 7 |
| Carry joint corrections and keep the original value estimate | 42 returns, longest 14 | 66 returns, longest 15 |
| Carry joint corrections and updated value estimate | 25 returns, longest 8 | 43 returns, longest 8 |

The first assessment gave 32 with only the updated value estimate. Restricting its changes to similar local visual contexts gave 29 with the local model frozen, and 34 when it continued updating during assessment. On the second assessment, carrying joint corrections alone won seven paired games, tied three and lost two. These independent checks favor retaining successful movements and provide evidence of interference when continuing to refit the action ranking. They do not establish monotonic improvement during a single rally. [Practice and independent lineup](../tt39/endurance-plan.json) · [48-game motor carry](../tt39/endurance-motor-only.json.gz) · [96-game initial](initial-second.json.gz) · [96-game motor carry](motor96-second.json.gz) · [96-game joint and value carry](both96-second.json.gz)

## Can a broad bank of paired corrections repair new shots?

We sampled 171 short joint variations at each of 111 naturally reached approach states from actual practice games. Independent native physics forks supplied the hypothetical legal-landing labels. Camera estimates, encoders, issued movements and the predicted consequence in a camera-initialized private model supplied the input. The unmodified movement scored on 71 of the 111 states. At least one variation would have scored on 82. With every example from the query release held out, the conservative selector changed zero actions and scored 71. Its available improvements did not transfer under that rule. [Practice receipt](shared-practice.json) · [Release-held decisions](shared-validation.json)

On twelve independent games, a direct joint controller with a larger library retrieved successful movements for 93 total returns and a longest rally of 19. Using local regression to reshape those movements before the same private checks gave 72 and 15. Extending the direct search horizon up to 39 ticks to prepare the arm earlier gave 60 and 13 in a separate trial, versus 109 and 35 in the matched previously carried short-horizon controller. Increasing the planning horizon alone did not supply the low-body postures that the training examples lacked. [Direct retrieval](joint-library.json.gz) · [Local regression](joint-generalize.json.gz) · [Longer preparation](early-joints.json.gz) · [Carried original](../tt39/credit-carry.json.gz)

## Learning during a point from the same starting memory

We replayed each of the twelve games in the 35-return study with adaptation disabled only inside that point. Before each pair the arm had exactly the same practice and earlier-game memory. Reconstructing updates from the archived stereo images, encoders and issued commands reproduced every live memory hash. Every command before the first earned update matched between conditions.

The live arm made 109 returns, versus 95 with within-point updates frozen. Nine games tied. Updating improved one from 10 to 13, improved the record game from 17 to 35, and shortened one from 24 to 17. The largest gain began after the two policies had issued the same first 1,144 commands. That comparison shows that a learner can benefit during one sustained point and that the same update rule can hurt another. The aggregate gain does not establish that the arm improves predictably with every hit. [Matched interventions and command tapes](matched-within.json.gz)

## Two passes with new physical replies

The geometric controller completed all 192 games, carrying its motor and reward memories without a reset. The second pass repeated the 96 measured launch recordings in a different order, with a new independent opening request and new random subsequent requests in every game.

| Pass | Arm returns / deliveries | Median per assigned game | Games reaching five | Longest |
| --- | ---: | ---: | ---: | ---: |
| First | 338 / 429 | 3 | 31 / 96 | 18 |
| Second | 313 / 402 | 2 | 28 / 96 | 24 |

On the second pass the arm improved on 37 recordings, tied on 12, and regressed on 47. The longer best rally coexisted with a lower total. Opponent replies changed, so this is a longitudinal workload comparison. The frozen, identical-assignment tests above isolate the effect of carrying a checkpoint. Training used 1,731 CPU seconds for 1,418 simulated seconds, including both cameras and both physical players. [Final game block and motor memory](../tt39/endurance-08.json.gz) · [Final reward model](../tt39/judgement-08.json.gz)
