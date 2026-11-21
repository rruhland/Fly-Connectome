# Learning which motor experience to use during table tennis

A player used the public outcome of each point to choose between its original actuator timing and the timing it learned during a cursor task. It chose separately for serving and receiving after recognizing the role from opening stereo frames. A second player recovered a previously learned arm model after an unannounced change in how motor ports were wired. Both learned without backpropagation.

## Timing choices that carried into another game

Each five-axis installation played an adaptation fixture, then a disjoint held fixture. During adaptation the player tried each timing model twice in each role and used point outcomes to guide later choices. The held controls all started from the adaptive player's same postgame checkpoint, including its updated serving and rally state. We also tested reversing the two role assignments on that checkpoint. Each branch kept the usual within-game rally and registration updates. [Full 14–12 first-person game](online-c-adaptation.mp4) · [Choices, returns and replay audit](online.md)

| Installation and stage | Learned choices | Always use inherited timing | Always use cursor timing | Reverse the learned role assignment |
|---|---:|---:|---:|---:|
| C, adaptation | **14–12** | 9–11 | 6–11 | — |
| D, adaptation | 11–13 | 10–12 | 8–11 | — |
| C, held | 8–11 | 6–11 | **11–9** | **12–10** |
| D, held | **16–14** | 8–11 | 8–11 | 7–11 |

D's held win depends on which model served and which received. On C reversing those choices was better. Both installations changed their receiving preference at point nine during adaptation. Matched receiving starts against the initial dry-swing choice yielded the same number of wins on each site. The selected configurations therefore show a useful outcome on D and substantial uncertainty in learning the right receiving strategy across fixtures. Both role-reversal games replayed their full physical points, and each action decision used visual role recognition. The D player preserved all 180 reference Pong action tapes and 41 points. [C reversal](role-reversal-c.json.gz) · [D reversal](role-reversal-d.json.gz) · [D checkpoint](online-d-player.json.gz) · [Native Pong](online-d-pong.json.gz)

## What survived a change in the physical body

- **Four-axis gantry with failed service.** Matching two short RGB motor probes to earlier arm experience raised legal serves from 1/16 to 14/16 on disjoint tosses. Changing just the stored geometric coefficients gave 12/16, changing just timing gave 1/16, and swapping only the tracker reference gave 1/16. These tests used the same serving algorithm. Across two full mixed-condition fixtures, the player increased legal returns from 11 to 21 and sustained rallies from two to six; the two versions each scored five points. Starting from the 2–11 loss and carrying its match updates, the gantry subsequently won both disjoint fixtures (11–6 and 11–8); a reset player won 11–5 and lost 7–11 on the same fixtures. [First two held tosses, side by side](gantry-motor-recall.mp4) · [Component comparisons and game results](motor-recall.md)
- **New gantry and six-joint rotary arm.** The earlier eight-swing timing rule selected ordinary timing on the new gantry and cursor timing on rotary. The chosen policy won one of four games; both fixed alternatives also won one. Rotary first returns on opponent service rose from 2/12 to 6/13, and the selected player still lost both rotary games. The carried player returned to the gantry and lost 9–11 versus a fresh player's 11–7 on the same schedule. This revisit used an assigned saved motor map; independent visual probes did not pass the strict recognition threshold. [Individual fixtures, service controls and checkpoints](body.md)
- **Transfer into another game.** Under changing motor delay and noise, the two-probe transferred player tracked a moving 3D target for 4,755/9,216 frames, versus 3,619 for a learner starting in the target game. Learning a new map from the same two table-tennis probes reached 4,724. These are calibrated RGB marker displays with a supplied tracking objective. [Transfer methods and controls](motor-recall.md)

The next table-tennis target is recovering more points *after* landing a legal serve, while retaining the now-competent service stroke. The gantry receipts include matched service trajectories and independently qualified motor maps. They allow a receiving intervention to be tested on the exact playable replies before qualification in new full games. [Research handoff](../../docs/reconstruction/HANDOFF.md)
