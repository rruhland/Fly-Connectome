# Does Pong experience shorten online practice for ten-return rallies?

The same 24 physical starts tested three opening policies. The arm moved its paddle laterally during the first stroke, then handed every successful or unsuccessful opening to the same trained table-tennis controller at frame 25. Native-duration Pong gave 23 first contacts, neutral gave 11, and shuffled native scores gave zero. Native and neutral each reached ten on **2/24** points. Both first reached ten on start 13. Native used 1,187 cumulative control frames, compared with 884 for neutral. Their longest points lasted 12 and 16 returns. All 72 actual point tapes replayed. [Physical experiment](tt23_sequence.md)

## Allow the arm to change its opening online

An online selector received the duration of just the opening it chose on each complete table-tennis point. The selector could choose from the same three motor options under every prior. Replaying these single-choice updates against the measured physical branches is exact for this selector: every point starts the same physical arm, only the selector's statistics carry, and the other two outcomes on that start are hidden from its update. Pong, uniform and reversed priors used the same sampling draws. Simulating 512 choices of exploration randomness on each fixed 12-point stream estimates uncertainty about which action a learner would take. It adds no independent physical starts.

The exact three motor-displacement rules were rescored on 18 native Pong starts with the bias component set to zero in all three. Mean durations were 6.08 seconds for the source option, 3.00 for neutral and 2.30 for the permuted option. Those ranks, with the strength of two prior observations, were the only difference between the source and cold selectors. Actual Pong contacts and simulator return labels were excluded from both updates. [Design](sequence-learning-design.md) · [Native scoring](sequence-learning-native.json.gz)

| Twelve-point training stream | Prior | Chance of reaching ten by the last practice point | Average launches to ten, capped at 12 |
|---|---|---:|---:|
| A | Pong | 90% | 7.74 |
| A | Uniform | 81% | 8.39 |
| B | Pong | 42% | 10.73 |
| B | Uniform | 32% | 11.05 |

The native prior bought less than one practice launch on either bank. By 20 simulated seconds the uniform prior had a higher chance of reaching ten on both banks. At 30 seconds Pong led on both. On the opposite held bank after 12 practice points, the Pong versus uniform selectors averaged 0.79 versus 0.62 ten-return points in one direction and 1.33 versus 1.45 in the other, out of 12. These two small banks support a head start in selecting useful openings but do not establish a significant reduction in time to sustained rallying. [All learning checkpoints and time budgets](sequence-learning.json.gz)

## Retention

On 18 independently drawn native Pong starts the source-selected rule averaged 6.25 seconds per point. Deploying the same selector after its 12 table-tennis updates yielded 5.57 seconds after bank A and 5.20 after bank B, averaged over its final choices. Updating a single shared preference thus lost some source performance. The underlying three movement rules stayed frozen. [Held native episodes](sequence-learning-retention.json.gz)
