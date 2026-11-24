# Far carriage returns after short net deflections

The optional rescue added one legal far return to each of two archived points that previously ended on a second bounce. The replays reproduced every archived contact and scoring event before the failed second bounce. The 22-return control point reproduced its entire event trace with the option enabled. [Matched native replay records](tt29_partner_short.json.gz)

| Recorded opening | Original alternating returns | Returns with rescue | Far strike y from net | Legal near bounce y from net | Minimum blade-to-table clearance |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7345, motor envelope | 95 | 96 | +0.415 m | −0.869 m | 1.59 cm |
| 11292, varied opponent | 21 | 22 | +0.439 m | −0.911 m | 3.27 cm |
| 7271, varied opponent control | 22 | 22 | — | — | — |

After a slow forward net deflection, the far planner starts again as soon as the ball separates from the net. It samples closer post-bounce contact windows and uses a bounded acceleration, cruise and brake along the carriage rail. It checks proposed contact and outgoing flights in an independent native solver, with the usual blade clearance and live referee. It tries the ordinary practice lane first, followed by two slower alternatives. The regular planner remains the default. [Carriage planner](../../scripts/experimental/tt29_partner_fast.py) · [Bounded stroke trajectory](../../scripts/experimental/tt29_partner.py)

The reported extension uses the full original near command tape and then supplies zero rates for the 13 or 22 additional ticks until the far return lands. Ball state was initialized at the archived opening and then advanced by the live physics throughout. Peak measured carriage speeds were 2.98 m/s and 3.21 m/s, both within the 3.45 m/s command limit. [Replay and clearance verification](../../scripts/experimental/tt29_partner_short.py)
