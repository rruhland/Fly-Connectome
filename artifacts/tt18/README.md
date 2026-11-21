# TT18 · Table tennis in changing physical environments

**The most reliable transfer was keeping track of the same ball and blade from new views.** With moving cameras alone, the player recovered to a 9–11 game, close to its performance with stable cameras. A location-dependent bounce estimate improved held rebound predictions and helped win one game on a changed table. A motor fit from a familiar gantry helped on one combined schedule, lost its advantage on a second, and failed to return a ball after a switch to a four-joint articulated arm. [Watch the combined game](changing-table-and-views.mp4) · [Mixed-change results and checkpoint](combined.md)

| Question | Evidence |
| --- | --- |
| Do mass, workspace, camera payloads, table size and bounce actually change trajectories? | [MuJoCo physical audit with matched actions and drops](plant.md) |
| Can several views of the same objects be registered when cameras move, including a camera on a moving link? | [RGB registration and 3D position error](view-registration.md) |
| Can the player still serve and return when cameras move, disappear and come back? | [Three complete games with camera controls](viewplay.md) |
| How long can one surviving camera support depth estimates and does that change actions? | [Short dropouts, close tosses and a matched live game prefix](mono-depth.md) |
| Does learning local rebound from real play help with a table whose two halves swap bounce properties? | [Two independent source streams, four held games per control](surface.md) |
| How fast can the motor fit recover through ordinary scored play on altered hardware? | [Twenty-two full games on compact gantry and five-axis hardware](online-body.md) |
| What happens when body, table and cameras change together during the game? | [Twelve scored games, independent schedule, model choices and Pong retention](combined.md) |

Parallel simulation provided independent source experience and matched comparisons. Each target learner played one chronological stream. The surface experiment transferred only two estimator settings chosen on 5,286 source frames. Its source-selected settings reproduced the prespecified local baseline exactly. The camera learner used known landmark positions and object colors. The motor experiments reused a practiced table-tennis controller. All experiments here ran in simulation without backpropagation. Real-time execution on a physical arm and quick adaptation to a novel articulation remain open. [Research plan](../../docs/plans/2026-11-21-continual-world-and-body-learning.md)
