# History-gated local readout works narrowly but does not transfer across direction

The [online prediction protocol](2026-09-24-history-gated-prediction-protocol.md) trained a separate zero-initialized local ON/OFF event-emission bank from the learned eight-unit history code, using one pass of delayed local event error on 96 clean plus 32 generic interruption/noise episodes. The same readout trained from a random history dictionary and a trace-reset copy were controls. The archived coincidence predictor remained intact; forecasts combined through a fixed pixelwise maximum. No offline probe weights, backpropagation, direction labels, or Pong semantics entered learning.

On the **original narrow** held-out disappearance suite (eight rightward diamond/ring cases), the candidate met the registered gate:

| Exact t=10→11 F1 | Archived | Learned history | Random history | Trace reset |
| --- | ---: | ---: | ---: | ---: |
| Original eight cases | 0.000 | **0.138** | 0.015 | 0.000 |

The learned branch predicted 10 of 124 target event pixels with 11 false positives. It preserved ordinary unseen-shape F1 at 0.669 and quiet-target false alarms at 97 pixels, exactly matching the archived predictor. Matched opposite-motion histories gave different forecasts (8.736 total absolute pixel-probability difference) and 0.174 pair F1 versus zero after trace reset. This is a real but sparse context-dependent forecast on that subset. Training interruption cases, however, had **zero true-positive pixels** at the fixed threshold, which made generality uncertain.

The separately [registered transfer check](2026-09-24-history-gated-transfer-protocol.md) reran identical training and evaluated 144 held-out interruptions: three unseen shapes, four directions, two speeds, two contrasts, and three positions. It rejected generality decisively:

| Exact t=10→11 F1 | Learned history | Random history | Trace reset |
| --- | ---: | ---: | ---: |
| All 144 cases | **0.027** | 0.008 | 0.000 |
| Up | **0.000** | 0.000 | 0.000 |
| Down | **0.000** | 0.000 | 0.000 |
| Left | **0.000** | 0.000 | 0.000 |
| Right | **0.100** | 0.032 | 0.000 |

Only rightward motion generated true positives, and the overall learned-minus-random margin of 0.019 missed the registered 0.05 transfer gate. The original success was a directional subset effect, not a robust M1A predictor. The history code's earlier 47/48 direction readability remains a valid **post-hoc population probe**, but this local emission rule has not converted it into direction-general online prediction. A plausible explanation is that direction lives in a pattern of local unit activity while each emission kernel credits individual source units independently; that mechanism has not yet been isolated and should be tested before changing plasticity. Do not promote this readout or tune its threshold/learning rate to rescue the current score. Production M1A remains unchanged. The expanded run took 238 seconds.
