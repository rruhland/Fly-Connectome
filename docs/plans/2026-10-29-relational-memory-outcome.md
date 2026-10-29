# Public visual memory: fixed 2×2 TRAIN comparison

Using the same learned parameters, the ungated public-memory controller returned all 180 of 180 incoming opportunities on TRAIN32: first returns 32/32 and later returns 148/148. Every game reached the 20-second limit. The gated memory arm returned 159/162, with first returns 29/32 and later returns 130/130. These are censored TRAIN results; fresh and longer independent validation are still needed.

| Controller | Returns | First | Later |
| --- | --- | --- | --- |
| Baseline 40069 | 83/108 | 29/32 | 54/76 |
| Observed only, gated | 130/142 | 29/32 | 101/110 |
| Observed only, ungated | 144/157 | 31/32 | 113/125 |
| Public memory, gated | 159/162 | 29/32 | 130/130 |
| Public memory, ungated | 180/180 | 32/32 | 148/148 |

The memory controller reanchors the learned horizontal pair whenever the required three objects are observed. During an observation gap it retains those same IDs only while the existing visual tracker still publishes them. IDs are public visual lineage keys, not simulator identities or semantic roles. Missing IDs, invalid inputs and excess observations clear the binding. No new predictor, age cutoff, coefficient, binding choice or fit was introduced.

Cost was 61,261 transitions and 64 initials in 188.119 seconds, within 300 seconds / 64,000 transitions. Every policy decision replayed exactly after reloading all public metadata and raw baseline inputs. Source, original weights and assets remained frozen. The ungated arm made 6,570 decisions using retained visual identities; 309 decisions used baseline fallback because no binding was available or a selected ID disappeared.

- Experiment: `/workspace/m1b-experiments/pong-relational-memory-20261029-r1/summary.json`.
- Ungated model SHA256 `75530410f2c985a3037e9aa264447b8d172e5e9297a9e556bbb74d2c511dc7cc`; gated model SHA256 `5466b30c0de02f5774aa8cd3bd9b6c1bf31356b3ca1036f27286073fd10d9f13`.
- Spec SHA256 `1df9f58476d6d01845ca7b053df2c891e08175af70d8855a4b2ce56628c708be`; prior unlaunched spec preserved as `spec.pre-exact-copy.json`.
- Core `cb38dcf`; public harness `a9a2cb0`, `9473571`. Twenty focused tests passed before launch. Fresh 63345 was reserved for the selected ungated procedure; 63329 remained unused.
