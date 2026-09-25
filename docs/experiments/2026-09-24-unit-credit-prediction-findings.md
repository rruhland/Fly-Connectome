# Per-unit local normalization does not rescue direction-general prediction

The [registered opt-in experiment](2026-09-24-unit-credit-prediction-protocol.md) kept the 24-unit learned motion code, eight-unit learned history code, generic episodes, one-pass delayed event target, learning rate 0.5, readout field, and fixed 0.5 threshold unchanged. It changed only the event-emission update denominator: each unit's eligible local error was averaged over its own active source sites instead of all active sources across the frame. The matched random-code readout received the same new rule. No backpropagation or production model change occurred.

| Exact t=10→11 on 144 unseen interruptions | Archived | Learned/global rule | Learned/per-unit rule | Random/per-unit | Learned/trace reset |
| --- | ---: | ---: | ---: | ---: | ---: |
| F1 | 0.000 | **0.027** | **0.027** | 0.103 | 0.000 |
| True positives | 0 | 30 | **30** | 153 | 0 |
| False positives | 0 | 33 | **33** | 666 | 0 |

The learned per-unit rule was **identical at the registered threshold** to the original global-denominator rule on every direction: zero true positives for up, down, and left; 30 for right. Its training interruption subset also had zero true positives. Clean held-out next-event F1 remained 0.669, and quiet-target false-alarm pixels remained 97 for all candidates. The random-code per-unit rule produced some true positives in every direction but at very low precision (0.187) and 666 false positives on event frames. Its higher pooled F1 is broad, unreliable firing, not an effective motion forecast.

The new rule fails the registered all-direction and control-margin gates. Normalization by unrelated active units is therefore not the demonstrated cause of the learned readout failure. A possible reason for the near-identical learned outputs is that most sources in a frame share one strongly direction-selective unit, making the two denominators similar; that occupancy was not measured here and remains an inference. The next step should revisit the **event-learning objective and credit assignment across temporal/polarity contexts**, using the locally legible history code as a fixed input. A decisive candidate would separate learning on actual future ON/OFF events from quiet/no-event credit while keeping each update local, then require nonzero true positives in all directions and a clear advantage over reset/random controls on the full 144-case suite. Do not sweep the present update's rate or threshold. Production M1A remains unchanged.
