# Sealed arcade transfer qualification

Evaluate held-out rotated Pong first. Freeze the Breakout candidate and record its commit before revealing or playing the independently sealed Breakout starts. The schedule digests are recorded alongside the fixed Pong checkpoint. The schedule generator uses separate random 256-bit keys for the two experiments and stable SHA-256 derivations for environment, policy, layout and distractor seeds. The keys and generated schedules become public at the corresponding candidate freeze.

## Rotated Pong

Pair 24 hidden Pong starts under every camera rotation. There are six starts in each combination of initial longitudinal and transverse motion signs. Use independently seeded static distractors off the paddle rails. Compare online role and axis discovery with a frozen original Pong orientation, a passive paddle, the native Pong reference and an unrotated run through the same online adapter. Use the evaluator's physics to score first and later incoming opportunities and to audit the selected paddle, ball and axis against the applied camera rotation. Log role lock time and performance losses, including starts where role binding fails. An axis supplied to the controller belongs to a labeled diagnostic arm.

## Breakout

The primary ordinary cohort consists of eight independent 16-life courses. In each course eight starts require leftward movement and eight require rightward movement. The first direction alternates across courses. Randomized held-out brick layouts and visual distractors appear from the first life. Ordinary control has the arena's default activation and no added command delay. A separate stress cohort has four independent 12-life courses. Each course has four lives under each of a fixed nonidentity button map, a delay of eight camera commands with 0.15 s activation, and both together. Every stress condition has two starts on each side. Stop a life on a loss, board clear or 1200 native physics ticks. No early stop on controller performance.

Each arm runs matching courses and starts with the same policy seed for a given course. Policy and environment seeds are independent. A new controller starts each course, with online knowledge retained across its lives where the arm permits. The evaluator knows life conditions and outcomes. Every active controller acts on its own public frame and ordinary transition feedback. The passive arm keeps physical drive at zero even in the remapped conditions.

| Arm | Initial knowledge and updates |
| --- | --- |
| Full transfer | Sealed Pong motor, timing, interception and score state with online updates |
| Motor and arrival | Same inherited motion and interception state with a blank score table and online updates |
| Frozen transfer | Fixed Pong motor, timing, interception and scores with per-life visual role alignment and no cross-life learning |
| Frozen Pong orientation | Frozen transfer with fixed original Pong orientation, as a stricter diagnostic |
| Fresh | Same model family, visual observer and online update rules with blank motor order, interception direction and score values |
| Fresh visual | Fresh acquisition with no inherited numeric motion constants, as a diagnostic |
| Passive | Zero physical drive |

Log the checkpoint byte hash, candidate source hashes, initial and final controller state hashes, requested-button digest and evaluator outcomes. Record all opportunities with return or miss and native tick and camera frame. Report the first five slots for every life, including explicit unreached slots, and the first five chronological opportunities pooled across lives within each course. These fixed denominators count missed and unreached slots as zero returns. Also report the number reached. Count the first opportunity of each completely cold course separately. Report the life, opportunity, and accumulated camera frames of the first successful return for each course, with failures explicit.

A sustained rally is three consecutive returned opportunities within one life. Report cumulative frames and ticks from the start of each course until the third return. Include all prior failed lives in elapsed time, and report unsuccessful courses with their observed follow-up time. For a single paired time comparison, assign an unsuccessful course the fixed maximum of its scheduled lives times 1200 native ticks. Also report simple opportunity and return learning curves across life ordinals, with every scheduled loss included.

Rapid transfer requires an ordinary-cohort advantage for full transfer over fresh and passive on the fixed-denominator first five chronological opportunities, with gains present in a majority of the independent courses. Sustained rallies must occur at least as reliably as fresh and sooner on the paired capped-time analysis. Report the paired course differences and uncertainty, plus the corresponding motor-and-arrival contrast. Report stress outcomes separately, including each stress condition. Interpret a later brick-score advantage separately from evidence for rapid return transfer.
