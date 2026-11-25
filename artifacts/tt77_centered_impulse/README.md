# TT77 · Teach the centered arm an outgoing impulse from its own misses

The centered body learned a better outgoing direction on exact central-low release 9254, but did not earn a return. The best pace/direction candidate reached the net at 0.814 m high with a 73.5 mm blade contact. The one clean interior candidate added lift but was still too sideways and slow. None of the 24 real attempts landed legally, so no new stock motor lesson or held result was recorded.

| Actual contact on the same incoming | Radius | Ball departure `(vx, vy, vz)`, m/s | Result |
| --- | ---: | --- | --- |
| TT72 reached face, centered | 56.3 mm | `(1.68, 1.17, 1.48)` | Bounce on own half |
| TT77 interior candidate, +2.88 ticks prep | 57.3 mm | `(1.86, 1.41, 2.15)` | Floor beyond side of table |
| TT77 forward candidate, +3.45 ticks prep | 73.5 mm | `(0.60, 2.92, 1.86)` | Hit net at y = −0.024 m |
| TT72 actual translated teacher, for context only | 16.5 mm | `(0.41, 2.83, 2.27)` | Legal and ordinarily answered |

The learner was not trying to replay the shifted paddle's tangential twist. It started with six **actual centered-stock** failed schedules. It advanced their preparation and fitted smooth perturbations in stock joint-reference coordinates. The training loss used a band of outgoing velocities inferred from the current contact's depth and speed, not the donor's departure. Six rounds of 96 model trials used the signed error on separation misses to credit commands. For four distinct proposals per round, actual native egress replaced the model label at the tried command before proposing the next round. All 24 had exact far impact and near bounce, matched receive line when reached, normal stock signed rates, and zero sampled CAD penetration. Their terminal outcomes were 21 own-half bounces and three floor misses. Two of the last four real attempts hit the net before bouncing on their own half.

The positive gate was an actual repeatable full-face interior, legal far landing at least 0.38 m beyond the net, forward departure at most 4.35 m/s and an ordinary native answer. There were zero model legal landings and zero real legal landings. This does not establish a stock impossibility. It does establish that the bounded earlier-approach correction improved pace and direction by moving contact back out to the rim, and the interior face-speed deficit remained. Independent nearby releases were not opened, and no RGB policy was deployed. Native impulse labels were development-only.

Run `python -m scripts.experimental.tt77_centered_impulse dev` under the usual one-thread EGL settings. The ledger is resumable by round. The setup pilot is separate and excluded from counts.

[Protocol](protocol.json) · [Practice and actual motion ledger](development.json.gz) · [Prior whole-trajectory retarget](../tt72_retarget_trajectory/README.md)
