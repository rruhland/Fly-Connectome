# TT22 reciprocal optical rally pairing

The ordinary `RallyPlayer` controls the clamped SO-101. It retains its stereo-only RGB detector, neutral waiting posture, nominal flight law and physical shot shortlist from the four-axis course. The adapter passes only its own camera estimates and control clock. The arm tracks its commanded rates in its independent nominal model. This pairing starts with empty return memory and uses no target updates. [Ordinary RallyPlayer](../../scripts/experimental/tt22_rally.py)

The far controller is the same privileged four-axis physical cycle planner that previously sustained 24, 10, 32, and 22 legal returns with the privileged near controller. The comparison replaces the near player with the earlier stereo cycle receiver, retaining its demonstrated swing library and timing residual fitted from its own two practice points. Its previous best fresh rally against this far controller was eight returns. The two intact optical stacks use different component association and stereo gating rules. [Physical cycle and earlier optical results](cycle.md)

Four fresh seeds were fixed before play. Each branch receives the same unmodified native serve, uses the same far controller and strict contacts, and plays until the referee ends the point. Every result counts, including any point without a return. Each action pair is replayed with both original camera streams and its complete event list verified.

| Ordinary start | Prior cycle observer | Ordinary RallyPlayer |
|---|---:|---:|
| 1758044470 | 0 | 2 |
| 270973880 | 0 | 0 |
| 210681131 | 2 | 6 |
| 1831566172 | 4 | 1 |

The ordinary RallyPlayer increased the total from six to nine across the four matched starts. Its SO-101 landed five legal returns versus three, and the far robot landed four versus three. It lengthened two rallies, shortened one and tied one. The longest rally rose from four to six, with three returns by each robot in the six-return point. The prior policy lost all four points, and RallyPlayer won one. Additional calibration was omitted from this comparison.

All eight loops reproduced their native stereo RGB digests, scorer results and referee events.

[Native optical and physical tapes](cross-rally.json.gz) · [Reproduction script](../../scripts/experimental/tt22_cross_trial.py)
