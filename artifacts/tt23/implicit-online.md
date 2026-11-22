# Learning when three fixed lateral moves are the only opening choices

A native Pong learner saw camera-derived movement, tried fixed motor displacements and used elapsed point time to learn a preferred direction. The entry point on the arm supplied exactly three lateral choices, left, center and right. A flat-prior learner had the same three actions and trained arm swing. It started centered, with no direction preference from Pong. Both chose from optical context, carried their experience across points and updated only the action they had taken from the elapsed time of their own point. A common practiced stroke took over after frame 25. [Native direction study](direction.md) · [Online protocol](implicit-online-design.md)

Four new eight-point schedules supplied 96 complete physical branches. Motors and referee were verified on all three alternatives. For each learner, the three-alternative archive reproduces the specific point its own decision selected; the unchosen alternatives are unavailable to the learner. The 256 paired exploration sequences per stream describe uncertainty about a policy’s choices on those 32 actual physical starts. The table scores a failure to reach ten in eight points as nine launches. [Scored arm command tapes](jog-bank-design.md)

| Eight-point stream | Probability of ten with Pong / flat | Mean launches to ten with Pong / flat |
|---|---:|---:|
| 1 | .64 / .37 | 4.54 / 6.43 |
| 2 | .35 / .35 | 8.30 / 8.30 |
| 3 | .98 / .85 | 3.92 / 5.14 |
| 4 | .95 / .73 | 2.43 / 4.49 |

Across streams, the Pong prior required 4.80 capped launches versus 6.09 with the flat prior, and reached ten with probability .729 versus .575. It made a mean 14.67 arm returns per eight-point stream versus 8.75. Fixed center averaged 11.25 and reached ten on three of the four streams. Deploying the frozen native direction without online exploration also reached ten in three of four, with a mean of 16.75 arm returns. Every one of the five independently trained native direction fits gave the same four-stream aggregates. The prespecified pooled source-label shuffle happened to prefer the same actions on this court and reproduced the source outcomes. The fixed shuffled comparison supplies no attribution by itself. [All scored conditions, retention and attainable outcomes](implicit-online-results.json.gz)

The evaluator could achieve an opening contact on all 32 starts by selecting the best of the three moves in hindsight. Only 21/32 could produce a legal opening, 18/32 could produce a second legal arm return and 7/32 could reach ten alternating returns. Only three starts could reach twenty. Those counts bound what this particular opening-only experiment and continuation can accomplish. The learner never observes the best alternative.

On an additional eight-point stream, both agents played directly and updated only from selected actions. Both first reached ten on point six and both produced a 32-return point. The Pong-trained learner accumulated 29 legal arm returns to the flat learner’s 20. Its longer earlier points meant it used 24.4 seconds to reach ten versus 18.6. All 16 scored episodes passed physical command replay. [Direct chronological deployment](implicit-online-live.json.gz)

The source learner still made 15 first returns on 48 held native Pong starts after the eight arm updates, precisely its pre-arm score on all four streams. Its frozen evaluation chose the same native actions; this retention test supplies no evidence of an additional benefit on returning to Pong. The receiving benchmark's narrow movement distribution is consequential: on independent crossed flight paths, holding the arm centered often beat following the source direction. [Independent trajectories](crossflight.md) · [Direct sixteen-stream confirmation protocol](implicit-confirm-design.md)
