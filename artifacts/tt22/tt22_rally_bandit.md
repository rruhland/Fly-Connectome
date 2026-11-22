# TT22 motor search using complete rally length

The clamped SO-101 retained the ordinary fixed-stereo controller, neutral waiting posture, arm model, and physical candidate planner. The only searched motor parameter was the multiplier on the forward pace of every generated stroke, including later strokes. The three candidates used 0.72, 1.00, and 1.28. The physical five-axis Cartesian rival and gentle serve distribution were common to all conditions. Gentle serves use 30% of the full native launch spread and 82% of the native launch speed. [Optical controller](../../scripts/experimental/tt22_rally.py) · [Five-axis opponent](precision.md)

The search used the strict referee’s alternating return count after each complete physical point. It ranked paces by the sum over four starting states drawn from two independent training schedules. Each pace played every training start, with seed-dependent order. A tie retained the ordinary pace. The selected pace was frozen before 12 new held starts. Both held conditions played the same previously unseen openings. The learner used native RGB, its clock, and its own normalized motor requests. No per-shot contact measurements entered policy inference, and held results supplied no updates.

## Training budget

| Pace multiplier | Four practice lengths | First seed block total | Second seed block total | Returns | Control frames |
|---:|---|---:|---:|---:|---:|
| 1.00 | 4, 4, 8, 4 | 8 | 12 | 20 | 467 |
| 0.72 | 6, 4, 5, 4 | 10 | 9 | 19 | 486 |
| 1.28 | 7, 2, 2, 2 | 9 | 4 | 13 | 364 |

The first seed block favored 0.72×. The second favored stock pace. The pooled search retained 1.00× from the 12 complete episodes. Both held labels therefore resolve to the same policy and share a single verified tape per start.

## Held comparison

| Seed | Alternating returns for stock and selected |
|---|---:|
| 1041155415 | 4 |
| 404509396 | 2 |
| 1726127918 | 8 |
| 1066773233 | 6 |
| 1314711354 | 2 |
| 1349112526 | 4 |
| 1305997524 | 2 |
| 1381448516 | 3 |
| 948202879 | 0 |
| 931902106 | 11 |
| 12364646 | 4 |
| 768259252 | 6 |

Both labels accumulated 52 total returns, 27 by the SO-101 and 25 by the far robot. One point reached ten (seed `931902106`, 11 returns), zero reached twenty, and the SO-101 won two points. The exercise shows a new 11-return physical exchange with the stock optical player and five-axis rival. The pace search retained the original controller. The 12 physical training episodes consumed 1,317 control frames and the 12 distinct held episodes consumed 1,349. All 24 tapes reproduced native RGB, both robots’ actions and complete referee events.

[Scored tapes and parameter receipt](tt22_rally_bandit.json.gz) · [Reproduction code](../../scripts/experimental/tt22_rally_bandit.py)
