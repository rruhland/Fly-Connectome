# Paddle-game transfer checkpoints and receipts

The [source Pong checkpoint](../m1b/2026-11-19-pong-after-mixed-course.json) comes from 96 score-practice points followed by a 64-point mixed course. Its SHA-256 digest is `d09f1050ad2e5f04fbe12572d506b55cc30bb765db040720607633076c8ecb0e`. The transfer experiment used a separate instance of that saved state for each course. State continued across all lives within a course.

## Play two consecutive Breakout lives

Run this from the repository root with the project and its data dependencies installed. The first life begins in the state acquired in Pong. The second resumes the resulting state on a different board. The paddle and ball move in a 64×64 binary camera and bricks disappear when struck.

```python
import json
import sys

sys.path[:0] = ["src", "scripts/experimental"]
from arcade_transfer_session import TransferSession
from breakout_arena import play_breakout

session = TransferSession("full", policy_seed=42)
for seed in (901, 902):
    life = session.begin_life()
    receipt = play_breakout(life, seed, layout_seed=seed, layout_split="heldout",
                             max_episode_ticks=1200, distractors=True)
    session.end_life()
    print(receipt["own_returns"], receipt["own_opportunities"], receipt["post_own_bricks"])
    saved = json.dumps(session.state(), allow_nan=False)
    session = TransferSession.from_state(json.loads(saved))
```

The carried state includes motor timing, action-order belief, selected interception relation, score estimates, role binding and policy RNG. A new life starts with an empty camera history and physical actuator queue. The arena supplies the camera, elapsed time, rewards and termination. The controller discards the native hit count. Opportunity and post-return brick counts in the example are measured by the evaluator.

## Sealed comparisons

The role and transfer comparisons have separate schedules. Each schedule can be regenerated from its revealed key and checked against its earlier digest. The protocol defines the first-five measure, the three-return rally and the time penalty for a course without a rally. [Qualification protocol](../../docs/plans/2026-11-20-arcade-transfer-qualification.md) · [Seals](2026-11-20-qualification-seals.json) · [Rotated Pong receipts](2026-11-20-rotated-pong-role-transfer.zip)
