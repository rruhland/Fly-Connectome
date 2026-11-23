# How accurately must Pong recover the motor drive for competitive table tennis?

On sixteen fresh two-arm points the maps learned during ordinary native Pong play produced zero legal near-arm returns. The same learned maps had helped in solo bouncing on both drives. A separate 1.45-second motor inspection per drive brought the two-arm game to eleven legal returns and eleven complete exchanges. With the exact drives, the match controller made 21 returns and its longest point contained 18 alternating returns. [Immediate transfer receipts](rolling-match-summary.json.gz) · [After physical inspection](rolling-match-inspected-summary.json.gz)

| Sixteen identical serves | Own legal returns | Completed exchanges | Starts with an own return | Longest alternating run |
| --- | ---: | ---: | ---: | ---: |
| Directly from the reward-trained Pong player | 0 | 0 | 0 | 0 |
| Original drive | 1 | 1 | 1 | 2 |
| Measured in 1.45 s of extra body motion | 11 | 11 | 6 | 6 |
| True drive supplied | 21 | 21 | 5 | 18 |

The five motor responses learned in Pong had 13.3% and 5.3% relative matrix errors. The separate inspection used the existing five-direction encoder procedure in a stock no-ball setup with the same drive. Its errors were 1.31% and 0.80%. It collected 87 real simulated control ticks per drive. This explicit inspection replaces the earlier map for this test. The table-tennis motor readout had previously learned from demonstrations and played the actual opponent using two camera images and joint encoders. All conditions were frozen before the match, started from identical paired camera frames, and were independently replayed for the physical referee. Across the two comparisons this verified 96 condition tapes, including repeated original and true-map controls. [Match code](../../scripts/experimental/tt24_rolling_match.py) · [Earlier calibration on other drives](settled-match.md)

The result sets a limit on the current transfer claim. The same motion experience can support meaningful early solo bouncing, yet its remaining control error can cost the arm an entire table-tennis return. Acquiring a useful drive model in one game and meeting the impact precision of another are separate measurements.
