# 63 legal arm returns on new opponent draws

A clamped SO-101 made 63 consecutive legal returns. The simulated opponent landed 64. Both stayed in the same point for 100.15 seconds. A second simulation replayed all 6,009 arm commands, checked every random target against its assigned stream, and reproduced all native referee events and placements. The minimum arm clearance across 48,070 physics steps was 15.29 mm. Median post-impact ball speed was 5.04 m/s. [Watch the uninterrupted point](watch-63.mp4) · [Physical replay receipt](random-63.json) · [Complete command and event tape](random-63-tape.json.gz)

## Did the arm learn while the ball was in play?

With within-point updates enabled the arm scored **63**. Starting from precisely the same motor memory, freezing updates for this point produced **14**. The first different joint command came at tick 353, after three successful arm returns. Both simulations received identical external random draws for the common prefix and produced identical physical events until their first different command. After that, the opponent reacted to the different ball it actually received. The live arm added 272 scored movement fragments over 63 updates. [Matched freeze and full controls](within-63.json.gz)

On a separate 24-game schedule the live player made 179 returns compared with 119 when frozen independently at the true start of each point. Seven games improved, two shortened, and fifteen tied. Both conditions had a median of two. This comparison reconstructs the live motor memory before each new game, so it isolates what the model learns *within* a rally. [All 24 matched points](../tt46_season/within24.json.gz)

## How typical was that point?

It was the tenth of 32 preassigned releases, all with new independent first and subsequent placements. The arm entered the course after 48 practice games that explored early joint movements. During the 32 games it used ordinary direct joint control and continued adding rewarded movements to its motor memory. It returned 207 balls, with a median of four per game. Eight games ended with zero arm returns, sixteen reached five, five reached ten, and one reached twenty. The 63-return point included four successful offers below 1.00 m in the evaluator's deep-height estimate. The fifth such offer ended the point. [Prospective draw schedule](streams.json) · [Full course](live-plain.json.gz)

The actor used two calibrated outside RGB cameras and joint encoders, a carried motor library, and a camera-initialized dynamics model to compare joint commands. Earlier practice included demonstrations. Each opponent placement drew four continuous quantities, then its physical paddle acted on the incoming ball. This experiment covers lateral requests from −0.28 to +0.28 m on the clamped arm's side. Whole-table coverage, wrist-only control at this level, general forward transfer from this single point and a 100-arm-return rally remain open.
