# Where does the Pong lateral preference help during a rally?

On eight fresh starts the SO-101 served at ordinary speed against the learned five-axis practice partner. Each policy issued exactly the same physical serve. During later strokes the high-level policy chose an eight-centimetre lateral region. The same table-tennis controller handled the orientation, timing, speed and contact location within that region. Both arms moved real physical paddles throughout every point. All 24 referee and motor tapes replayed. [Prospective design](after-service-design.md) · [Tapes](after-service.json.gz)

| Policy after the identical serve | Consecutive legal returns on each start | Arm returns | Points ≥10 | Points ≥20 |
|---|---|---:|---:|---:|
| Native score-trained Pong lateral gains | 6, 0, 5, 5, 11, 0, 7, 1 | 15 | 1/8 | 0/8 |
| Zero lateral gains | 11, 0, 7, 5, 41, 0, 37, 15 | 55 | 4/8 | 2/8 |
| Opposite of the native gains | 3, 0, 1, 5, 1, 0, 3, 7 | 7 | 0/8 | 0/8 |

Zero lateral gains tell the stroke planner to work in the region around the arm's current lateral pose. The arm still moves all its joints to serve and strike. The 41-return point lasted 797 control frames, or 26.6 simulated seconds. The SO-101 made 20 legal returns and its partner made 21. At the same start the active Pong policy made five legal returns and its partner made six. The Pong policy won one of the eight points; the zero-gain control lost all eight. Winning a point and sustaining a cooperative rally score different outcomes.

Twelve different *receiving* starts, using anonymous stereo motion, a partly varied far feed and the same region size, produced 17 near returns with the Pong preference, 26 with zero gains, and seven with opposite gains. None of the three reached ten on these receiving starts. On ten other receiving starts, removing the geometric lateral correction from the learned stroke selector gave longest rallies of 8 under native-duration Pong and 10 under zero gains. These are distinct policies and launch distributions. Carrying a learned lateral preference into every stroke proved less effective than using it for opening contact. [Receiving course](workspace-intent.json.gz) · [Direct lateral control](branch-motor-direct-held.json.gz)

The 41-return video shows the broadcast view and both real stereo inputs. Physics, each camera frame and referee calls reproduced from the two saved command tapes. [Watch all 41 returns](after-service-neutral.mp4) · [Replay check](after-service-neutral-video.json)

## How much came from the arm's own practice?

The zero-gain controller was replayed on all eight starts again with its seven learned optical corrections removed. The motor search, edge and net penalties, assigned motor region, physical service and partner stayed the same. Without practice its rally lengths were **3, 0, 15, 21, 3, 0, 1, 7**. The arm made **22** legal returns, reached ten on **2/8** starts and twenty on **1/8**. With the four practice points it made **55** legal returns, reached ten on **4/8** and twenty on **2/8**.

On the same two starts, practice took the arm from 3 to 41 and from 1 to 37 alternating returns. It also shortened one point from 21 to 5 and another from 15 to 7. This zero-practice comparison was added after the 41-return point was seen and includes every start. The two arms' commands and strict physical outcome replayed for all eight controls. A separate complete game with both services also favored the practiced arm, 28 to 19 own returns. These results establish a repeatable contribution from target optical practice within the supplied physical controller. [Zero-practice controls](../tt23/after-service-zero.json.gz) · [Complete game](ordinary-game.md)
