# One bounded Astra / Sol research pass — complete

Latest scope: one research → hypothesis comparison → experiment → interpretation pass, then stop. No automatic continuation is authorized by this pass.

## Research and selection

Astra independently proposed interception-versus-shot-placement, sparse-win selection noise, and insufficient relative/motor-state readout as competing explanations. Sol6.1 XHigh independently inspected the actual code/results and reached similar explanations, additionally identifying that the win-indicator objective treats losses and timeouts identically. Neither claimed a demonstrated representation limit.

Astra inspected primary ARS and reward-shaping papers; Sol inspected ARS, ES, and the original policy-gradient proceedings discussion. Astra subsequently checked the ES derivation. Their relevant lessons were: intermediate rewards need not preserve the objective; survival bonuses can favor unhelpful deterministic policies; random-search results vary substantially; perturbed-policy fitness is not proof of an averaged greedy policy's quality. These sources motivate hypotheses, not proof of a local defect.

We selected a fresh evaluation of existing, previously unevaluated learning instead of another training allocation. The final c7 incumbent was fixed regardless of outcome; no best-checkpoint selection. Its inherited cost is87,241 used plus1,513 unused partial training transitions, seven completed cycles/two accepted updates. The original interrupted allocation remains incomplete and its clock/source were not changed.

## Experiment

Three frozen policies ×64 untouched, prospectively reserved development serves. Same opponent .35,10-second horizon,action hold1,vision and action class. Primary endpoint-minus-native-start win rate; practical screen≥10pp and paired95% bootstrap lower>0. Original acquired policy is a secondary comparator.10,000 paired bootstrap draws.10-minute/96k transition cap; no extension, new training, final-stream expansion or architecture change.

| Policy | Wins | Wins after own hit | Losses | Timeouts |
|---|---:|---:|---:|---:|
| Original return-acquired |13/64|4|45|6|
| Native-win starting policy |14/64|5|43|7|
| Last committed exact-child policy |17/64|8|45|2|

Nine natural initial-serve wins are common to all three policies. Primary difference+4.6875pp,95%CI[−4.6875,+15.625]; seven endpoint-only wins versus four start-only wins. Secondary versus original+6.25pp,CI[−1.5625,+15.625]. The practical screen **fails**. This is not equivalence, proof of no learning, or proof of a representation ceiling; it is insufficient fresh evidence of a useful winning improvement.

Descriptively, endpoint has fewer own returns40/85 versus46/89 but more wins following its hits8 versus5. This suggests a behavior tradeoff worth distinguishing from improved interception, but different trajectories/opportunity denominators prevent a causal shot-placement conclusion. More wins alongside more losses and fewer timeouts also makes a simple defensive-improvement claim inappropriate.

Cost:389.31seconds,39,383 new evaluation transitions+192initial images,192/192jobs,zero new training/visual learning and no failures. Existing annotation reconstructs physics. Parent independently recomputed terminal outcomes from every raw transition, compared all192policy/serve cells and paired statistics, checked frozen inputs/original snapshot; PASS in0.497seconds with zero new observations. Separate completion/source verification is retained with the experiment.

## Decision and stop

The accepted training gains remain unconfirmed on fresh games. Do not automatically extend this optimizer or switch to survival reward. This pass closes the missing generalization evidence and leaves the causal hypotheses unresolved. Any next allocation needs a new explicit selection; no further experiment is running from this pass.

Artifacts: `/workspace/m1b-experiments/pong-one-pass-20261021/` contains independent initial hypotheses, both research briefs, preregistration, evaluator, semantic verification, raw results, independent check and `one-pass-result.png`/`.svg`.

Sources: [Mania et al.2018](https://arxiv.org/pdf/1803.07055), [Ng et al.1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf), [Salimans et al.2017](https://arxiv.org/pdf/1703.03864). Source-extraction scope and limitations are recorded in the external briefs.
