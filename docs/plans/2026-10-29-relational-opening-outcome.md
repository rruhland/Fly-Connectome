# Observed-only opening scaffold test

Removing the first-own-hit gate from the same learned relational endpoint increased TRAIN32 returns130→144 and first returns29→31. At least two returns increased23→31 games, and at least three22→29. Overall return fraction increased130/142(91.55%)→144/157(91.72%). Later returns increased101→113, but later opportunities increased110→125, so later fraction fell91.82%→90.40%.

The branch therefore did **not** pass its predeclared non-worse-later-fraction rule. Fresh63329 stayed reserved and unused; no policy was promoted. This preserves a useful experimental opening improvement with a conditional-later tradeoff, without relabeling the strict gate as passed. Mean hit difference was+0.4375/game with paired bootstrap95% interval[−0.40625,1.34375];10 games gained hits and11 lost hits. Timeouts20→19 and maximum streak6 show continued fixed-window censoring.

Cost27,586 transition cameras/32 initials in77.227 seconds, within180 seconds/32,000 transitions. All27,586 raw-Q/stateful decisions replayed exactly. Weights, source and cached reference hashes remained unchanged. No fitting, coefficient/binding changes or additional opening variants occurred.

- Frozen learned weights SHA256 `026ff8da73457238209708a03060bcaf10a5f94bbb83805d9f7a60c0644beedf`.
- Spec SHA256 `af010c32d8269716d5c9c1aa83ed1229b35b0d55305b01eda858318bd883149c`.
- Receipts/outcome `/workspace/m1b-experiments/pong-relational-opening-20261029-r1/summary.json` and adjacent `train32/`.
- Source `2c7c156`; eight core/opening tests passed before launch. The original gated131-return fresh63313 result remains a separate frozen endpoint.
