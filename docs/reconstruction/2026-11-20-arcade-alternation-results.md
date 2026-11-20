# One learner alternating between Pong and Breakout

Each course switched games five times at the usual life boundary. The learner saw the 64×64 binary camera, its own requests, elapsed time, and native score. It continued updating visual timing and button effects in both games. At the first frame of each life, a visible array of rectangular targets selected the Breakout placement update and two opposed paddle rails selected the Pong placement update. Both reused the same canonical interception model. Ambiguous geometry used center interception. The rule was designed for these scene structures and applied to all matched arms. [Registered method and criteria](2026-11-20-arcade-alternation-protocol.md)

## The original learner could lose both games after one miss

The original alternating session started from the sealed Pong checkpoint and the practiced score-only brick model. In both courses it missed the very first Pong return and reversed its approach direction after the short scoreless life. The reversal persisted. The shared learner returned 0/16 Pong and 0/18 Breakout balls in the ordinary course, and 0/15 Pong and 0/18 Breakout balls in the conflicting course. Separate Breakout continuations returned 108/109 and 108/110 respectively and earned 98 and 94 bricks after a return. The isolated Pong streams also started with the miss and failed to recover. Retrieving the correct role and motor belief did not repair the wrongly revised direction. The contextual arm selected the correct map and lag by camera 4 at all four conflicting re-entries, then still missed its first five incoming balls in each block.

An evaluator-supplied correct role on each exact initial Pong start moved readiness from camera 19 to camera 3. This changed both opening misses to returns with identical pre-contact ball trajectories, and preserved the previously learned approach direction. The intervention was a diagnostic. The ordinary and conflicting learner streams had to discover or recall their roles from the camera throughout.

## Update the approach direction from observed movement

The revised credit rule looks for a sustained intended request, verified own paddle motion, a stable visual actuator estimate, and agreement between the visible ball and projected interception point. Two separate failed lives with supported wrong-way evidence are needed to reverse a carried direction. An independent development test began with an intentionally inverted direction. The learner reversed after the first two misses and returned the next six openings. The locked inverted comparison returned zero of eight. The revisable rule uses the same public camera and score. [Pursuit credit](../../scripts/experimental/arcade_pursuit_credit.py) · [Inverted-direction comparison](../../artifacts/m2/2026-11-20-arcade-pursuit-revision.zip)

The fresh alternation courses used the independently qualified visual-rebound and score checkpoint in every arm. The guarded and unguarded streams had the same role recall, and both could also recall a previously verified motor-order and timing belief in the conflicting course. The isolated control played the same starts in two uninterrupted, guarded streams. These comparisons isolate the credit rule within the new courses. The earlier score-only runs establish the original failure, and the new course starts and brick checkpoint differ from those runs.

## Performance on the two re-entries per game

| Controls | Learner | Pong returns | Wins after a return | Breakout returns | Bricks after a return |
|---|---|---:|---:|---:|---:|
| Ordinary | Guarded, alternating | 13/14 | 9/12 | 77/79 | 64 |
| Ordinary | Unguarded, alternating | 11/13 | 8/12 | 77/79 | 64 |
| Ordinary | Guarded, isolated | 13/14 | 9/12 | 78/80 | 65 |
| Reversed buttons plus lag in Breakout | Guarded, alternating | 21/21 | 9/12 | 82/82 | 80 |
| Reversed buttons plus lag in Breakout | Unguarded, alternating | 1/12 | 0/12 | 0/12 | 0 |
| Reversed buttons plus lag in Breakout | Guarded, isolated | 21/21 | 9/12 | 82/82 | 80 |

Both guarded courses met the registered retention criteria. Ordinary play retained 92.9% of Pong and 97.5% of Breakout returns on re-entry, within 0.1 percentage point of the isolated references. The conflicting course retained 100% in both games on re-entry. Its guarded alternating and isolated learners made exactly the same requests on 35/36 lives, including all re-entry lives. A first Pong miss in that course reversed the unguarded learner for the rest of the stream. The guarded stream retained its direction. More returns create more opportunities later in a rally, which accounts for the different denominators.

| Controls | First five incoming balls in the four re-entry blocks | Re-entry first-return camera | Re-entry role ready | Re-entry correct map and lag |
|---|---|---|---|---|
| Ordinary | 5/5, 5/5, 4/5, 5/5 | 40, 51, 41, 49 | 3, 3, 3, 3 | 4, 4, 4, 4 |
| Reversed buttons plus lag in Breakout | 5/5, 5/5, 5/5, 5/5 | 44, 46, 47, 45 | 3, 3, 3, 4 | 4, 4, 4, 5 |

Chronological re-entry order is Pong, Breakout, Pong, Breakout. The guarded learner returned the first incoming ball in each of the eight re-entry blocks. In the full ordinary course it returned 24/25 Pong balls and 117/119 Breakout balls and earned 15 Pong wins after an own return and 93 Breakout bricks after a return. In the full conflicting course it returned 29/30 and 116/117 and earned 12 such wins and 106 such bricks. Breakout yield on re-entry was 5.33 bricks per life under ordinary controls and 6.67 with the tested conflict, versus 5.42 and 6.67 in the isolated references. Each course used different starts, so those yields describe retention within each course.

Learning continued in both domains. The ordinary guarded course recorded 16 Pong score updates, 100 visually observed Breakout rebound updates and 95 brick score updates. The conflicting guarded course recorded 15, 104 and 100. The results cover only two re-entries per game per course, nine-brick layouts, a .35 Pong opponent, and one combined reversal and six-camera delay. The role and timing memory was recalled from these visible structures at normal life boundaries.

The offline audit replayed every recorded request from the 468 original and fresh course lives. It reproduced all 166,176 public camera transitions and rewards, checked the matched first-contact geometry and verified the learner state hashes across consecutive lives. The archive includes both initial checkpoints, the actual practice and held-out board masks and frame hashes, the missed starts, and the committed implementations used for each round. [Complete native receipts and checksums](../../artifacts/m2/2026-11-20-arcade-alternation.zip)

## Continue from the guarded endpoint

Use the guarded ordinary session after 36 lives as the default portable learner. Its state includes both learned role priors, the shared sensorimotor model, Pong shot values, the brick rebound and score models, the pursuit credit state, and the random generators. On both guarded endpoints, loading the saved JSON reproduced the same actions and final learned state on the next Pong and Breakout starts as the in-memory session. The ordinary version then played another disjoint board with a 75-second cap, cleared all nine bricks in 35.65 seconds with 11/11 returns, and won the immediately following Pong point after one return. Two independent loads produced the same action and state hashes on that continuation.

```python
import json
import sys
from pathlib import Path

sys.path[:0] = ['src', 'scripts/experimental']
from arcade_alternation import AlternatingSession

path = Path('artifacts/m2/2026-11-20-arcade-continued-session.json')
session = AlternatingSession.from_state(json.loads(path.read_text()))
```

For each new life, call `session.begin_life()` and reset the returned life with the opening camera frame. Call `choose()` before each camera transition and `observe(transition)` afterward. At termination or timeout, call `session.end_life()`. Reuse the session for the next game and call `session.state()` between lives to save further learning. [Continued session JSON](../../artifacts/m2/2026-11-20-arcade-continued-session.json) · [Complete native receipts and checksums](../../artifacts/m2/2026-11-20-arcade-alternation.zip)

## Fixed learner under unannounced control changes

A single session used the actual conflicting-course final state throughout six more alternating blocks. Its guarded direction and both recall options stayed enabled. Pong used ordinary controls throughout. Breakout began with ordinary controls, switched to reversed outer buttons and six commands of lag, then returned to ordinary controls. All three Breakout blocks had the same visual scene type. Each block used four fresh lives, a 20-second limit, and balanced serve directions. The exact twelve Breakout brick masks and opening frame hashes were distinct from the earlier alternation, practice and clearance boards. [Registration](../../artifacts/m2/2026-11-20-arcade-deployment-stress-plan.json)

| Block | Returns | First incoming ball | Wins after a return or bricks after a return | First life’s correct map and delay |
|---|---:|---|---:|---:|
| Pong | 7/7 | Returned | 3 wins | Camera 4 |
| Breakout, normal | 19/20 | Missed | 19 bricks | Camera 30 |
| Pong | 10/10 | Returned | 4 wins | Camera 4 |
| Breakout, reversed with lag | 21/22 | Missed | 14 bricks | Camera 25 |
| Pong | 10/10 | Returned | 4 wins | Camera 4 |
| Breakout, normal restored | 21/22 | Missed | 18 bricks | Camera 30 |

The learned timing prior was stale on entry to each Breakout block. The learner identified the current button order and lag during the first life, missed its first ball, and returned every later ball in the block. It returned 61/64 in Breakout, with 51 bricks after a return, and 27/27 in Pong, with 11 wins after a return. One pass through 24 lives yielded 14,803 camera transitions and a continuous state hash chain. The archived fixed-policy endpoint includes these additional updates and uses the same loading API. [Fixed-policy session JSON](../../artifacts/m2/2026-11-20-arcade-fixed-policy-session.json) · [Native receipts](../../artifacts/m2/2026-11-20-arcade-deployment-stress.zip)
