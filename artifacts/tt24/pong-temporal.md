# Pong reward and delayed wrist prediction

A prediction gain selected from native Pong point rewards raised the sum of longest streaks from 23 to 31 across 12 delayed wrist starts. Unit extrapolation scored 26. The gain left every command identical on prompt wrist input. Prompt delivery scored 125, so the adjustment recovered eight of the 102 streak counts lost to delay. [Physical wrist trials](pong-temporal-wrist.json.gz)

The same scalar multiplies the age of the most recent camera packet and the observed velocity of the moving object. The controller adds this displacement to the last observed position. The delay varied across the trial from prompt to 67–133 ms and back. Capture timestamps were public. Pong kept a current paddle image, and the wrist used current encoders with calibrated gripper geometry. The wrist recovered the moving object's position by decoding delayed RGB in the corresponding earlier encoder frame. Both existing motor heads remained frozen. [Native source runner](../../scripts/experimental/tt24_pong_temporal_source.py) · [Wrist runner](../../scripts/experimental/tt24_pong_temporal_wrist.py)

Earlier studies learned an unknown image clock from stock wrist play and a velocity readout from native Pong optical tracks. The present fit uses native point rewards to select the strength of extrapolation at a supplied camera age. [Unknown wrist delay](latency.md) · [Earlier optical transfer](wrist-transfer.md)

An independent six-start diagnostic scored 59 with prompt images, 16 with delayed images and no extrapolation, and 21 with full linear extrapolation. These scores passed a fixed diagnostic gate. The gain fit used 720 native points across 80 launches and nine gain settings. A regularized quadratic fit to terminal point scores chose 2.0, the upper grid boundary. Reassigning point rewards among gains within each launch chose 0.0, which duplicated the cold baseline. Wrist rewards and action labels supplied zero examples to the fit. [Diagnostic and replays](pong-temporal-diagnostic.json.gz) · [Source plan](pong-temporal-plan.json) · [Frozen source fit](pong-temporal-source.json.gz)

On 64 held delayed native starts, the learned gain produced a net score of −1 with 62 returns, compared with −4 and 59 returns for unit extrapolation and −6 and 56 returns for shuffled and cold. A return here is a start on which the controlled paddle returned the ball at least once. Most points timed out, and the paired 95% bootstrap intervals for native point score differences included zero. All gain variants reproduced the original native policy's command tapes on all 32 prompt native starts. [Native held records](pong-temporal-source.json.gz)

| Wrist conditions | Summed longest streaks | Best streak | At least ten |
| --- | ---: | ---: | ---: |
| Prompt baseline and prompt with gain enabled | 125 each | 17 | 8/12 |
| Delayed with Pong gain 2.0 | 31 | 7 | 0/12 |
| Delayed with fixed gain 1.0 | 26 | 6 | 0/12 |
| Delayed with shuffled and cold gain 0.0 | 23 each | 6 | 0/12 |

The paired wrist gain over cold or shuffled was +0.67 streak counts per start, with a bootstrap interval from +0.25 to +1.08 and six improved starts, six ties and zero declines. The gain over fixed unit extrapolation was +0.42, with an interval from 0.00 to +0.83 and five improved starts, six ties and one decline. The 48 delayed command tapes also matched their prompt counterparts before the first delayed frame. All 90 diagnostic and held command tapes replayed exactly in MuJoCo. [Initial wrist results](pong-temporal-wrist.json.gz)

## Independent source fits and fresh wrist starts

Four additional fits used disjoint native training and holdout rosters and a separate reward shuffle per fit. Together the fits used 2,880 native training points. Each fit supplied a gain to six fresh paired wrist starts. The delay schedule, fitting procedure, source policy and wrist policy were the same as in the initial experiment. Fixed gains of 1.0 and 2.0 were explicit comparators. Exact numerical duplicates used the same executed condition. [Registered protocol](pong-temporal-repeat-plan.json) · [Native fits and held play](pong-temporal-repeat-source.json.gz) · [Replication runner](../../scripts/experimental/tt24_pong_temporal_repeat.py)

| Source cohort | Pong gain | Shuffled gain | Pong | Cold | Shuffled | Fixed 1.0 | Fixed 2.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.000 | 0.000 | 13 | 8 | 8 | 9 | 13 |
| 2 | 2.000 | 0.933 | 29 | 13 | 16 | 30 | 29 |
| 3 | 2.000 | 0.803 | 8 | 13 | 17 | 14 | 8 |
| 4 | 1.267 | 1.406 | 12 | 21 | 15 | 17 | 18 |
| **Sum across 24 starts** | | | **62** | **55** | **56** | **70** | **68** |

The initial reward fit and three of the four repeat fits selected the upper boundary of 2.0. The remaining fit selected 1.267 and scored 12 on its cohort, compared with 18 for fixed gain 2.0. The paired difference against fixed gain 2.0 was −0.25 per start with a percentile interval from −0.50 to −0.04, four declines and 20 ties. The comparison with fixed gain 1.0 had an interval from −1.83 to +1.13. Comparisons with cold and shuffled included zero. The Pong-selected gain and each fixed nonzero gain reached ten rebounds once in the replication. All 96 additional physical command tapes replayed exactly. [All paired results and exact replay count](pong-temporal-repeat-summary.json)
