# Observed-only opening scaffold test

Removing the first own-hit gate from the same learned endpoint increased TRAIN32 returns from 130 to 144 and first returns from 29 to 31. Games reaching at least two returns increased from 23 to 31; at least three increased from 22 to 29. Overall return fraction rose from 130/142 (91.55%) to 144/157 (91.72%). Later returns rose from 101 to 113, but opportunities rose from 110 to 125, so later fraction fell from 91.82% to 90.40%.

The branch did **not** pass its declared rule requiring later fraction to be no worse. Fresh tag 63329 stayed reserved and unused; no policy was promoted. The opening improvement remains an experimental branch. Mean hit difference was +0.4375/game with paired bootstrap 95% interval [−0.40625, 1.34375]; 10 games gained hits and 11 lost hits. Timeouts fell from 20 to 19, and maximum streak stayed 6 within the fixed time window.

Cost was 27,586 transition cameras and 32 initials in 77.227 seconds, within the 180-second / 32,000-transition cap. All 27,586 raw-Q and stateful decisions replayed exactly. Weights, source and reference hashes stayed unchanged. No fitting, coefficient or binding changes, or additional opening variants occurred.

- Frozen learned weights SHA256 `026ff8da73457238209708a03060bcaf10a5f94bbb83805d9f7a60c0644beedf`.
- Spec SHA256 `af010c32d8269716d5c9c1aa83ed1229b35b0d55305b01eda858318bd883149c`.
- Receipts/outcome `/workspace/m1b-experiments/pong-relational-opening-20261029-r1/summary.json` and adjacent `train32/`.
- Source `2c7c156`; eight core/opening tests passed before launch. The original gated 131-return result on fresh tag 63313 remains a separate frozen endpoint.
