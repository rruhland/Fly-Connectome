# Learning a delayed visual forecast during ordinary wrist play

A fixed prediction gain learned in native Pong helped on an initial delayed-video course and lost to simple fixed gains on a replication. Here the five-joint arm starts with that frozen gain of two and accumulates its own visual experience on three independent 12-drop streams. Camera packets arrive with public capture timestamps, sometimes 67–133 ms late. The current proprioception stays available. When an image arrives, the learner can compare the current visual estimate against older forecasts whose outcomes are now observable. It stores predictions for different ages in bounded memories of 768 prior cases per age. It uses eight nearby observations to correct the native linear forecast in position and velocity. Inputs come from the public optical decoder. Game scores, hidden ball motion and contact times are excluded from learning. [Native gain and independent replication](pong-temporal.md) · [Online learner](../../scripts/experimental/tt24_temporal_memory.py)

| Wrist controller | First 4 releases per stream | Next 4 | Final 4 learning releases | 18 fresh releases |
| --- | ---: | ---: | ---: | ---: |
| Carry optical memory between games | 45 | 56 | 64 | 94 |
| Clear optical memory every game | 38 | 46 | 59 | 54 |
| Frozen Pong forecast | 31 | 38 | 41 | 49 |
| Prompt camera packets | 119 | 139 | 137 | 229 |

Values sum the longest qualifying streak on each release. On the 18 fresh drops the carried memory was frozen and the per-game learner could still update within each release. The carried learner reached ten on 4/18, compared with 1/18 for the reset learner, none for the frozen Pong forecast and 15/18 with prompt frames. Individual stream held sums of 59, 9 and 26 versus 22, 12 and 15 for frozen forecasting reveal uneven benefits. The three acquisition histories took 24.65, 32.82 and 32.93 simulated seconds. These are online target interactions even though no scores were used. The arm still loses substantial performance to camera latency. [Stream 1](temporal-memory-0.json.gz) · [2](temporal-memory-1.json.gz) · [3](temporal-memory-2.json.gz)

Reloading the stored forecasts at the boundary between acquisition and held play reproduced all 2,826 held commands and every scoring record. [Stream 1 memory](temporal-memory-0.npz) · [2](temporal-memory-1.npz) · [3](temporal-memory-2.npz)
