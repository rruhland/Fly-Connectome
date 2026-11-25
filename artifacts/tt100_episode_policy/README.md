# Whole-point credit for a compact search policy

TT46 context search changed a local joint residual and credited each offered shot. TT41's episodic critic still ranked individual motion fragments. TT45's tactical planner and TT57's shot strategy chose outgoing effects. TT42's checked execution kept or reopened a particular forecast after observing innovations. None of those directly optimized a common search policy over complete games while holding the memory and ranking law fixed. TT100 tests two such weights and gives them no credit until a point ends.

The first weight shifts when stereo-derived inbound y first permits ordinary search. The second scales the random direct-joint proposals used by that search. The existing proposal budget, hardware, joint and slew limits, stereo estimator, incumbent interval, uncertainty tests, ranker, opponent and reference feedback are unchanged. A zero setting is the literal ordinary actor. Each pair runs from one measured opening per arm. Common assigned continuous partner seeds reduce shared noise, but later delivered balls can legitimately differ after different successful outgoing shots.

The installed motor library was paid for on the .12 m translated TT73 clamp. Both policies play back on the original centered fixture with that same frozen bank. Six calibration release IDs are outside the installed motor practice and both TT97 and TT80 release pools. The first six untouched TT97 assignments with distinct release IDs form the held comparison. Every TT80 release appears in installed/source training, so those streams stay untouched. [Preassigned episodes and rules](protocol.json)

Three antithetic blocks each run two native launch assignments with plus and minus whole-point policies. A scaled difference of log-return utilities moves the common weights only between blocks. Opposite signs across releases shrink the step. No per-shot reward shaping, synthetic success, simulator truth, rerun lottery, GPU training, or backpropagation enters the update.

Each native point runs to its referee outcome, with a 240 alternating-return safety cap and the stock 24,000-control limit. Evaluator-only logs report actual own returns every twenty if reached. A second withheld deck of four distinct release recordings was fixed before any primary held outcome. It is used only if the primary paired gain is at least six total returns, at least three points improve, and fewer get worse. The weights are never changed by either held deck. [Optional continuation schedule](extension-protocol.json)

The plus and minus trials differed by `(+2, −4)`, `(0, +1)` and `(−5, −5)` legal returns in the three practice blocks, respectively. The same combined shift toward later search onset and wider proposals won both pairs in the last block. The resulting policy starts searching at −.5603 m instead of −.49 m, with direct-joint exploration radius multiplied by 1.0258. The signs of the first two blocks show why tiny practice samples alone are not sufficient for deployment. [All 12 native practice points](practice.json.gz) · [Frozen weights](policy.json)

## Untouched full points

| Opening release | Ordinary own returns | Episode-trained own returns |
| --- | ---: | ---: |
| 8831 | 10 | 10 |
| 2741 | 17 | 3 |
| 11242 | 17 | 17 |
| 11809 | 9 | 3 |
| 9114 | 7 | 8 |
| 10053 | 4 | 2 |
| **Total** | **64** | **43** |

The medians fell from 9.5 to 5.5. One pair improved, three regressed, and two tied. Every point ended naturally and the opponent won all twelve. Neither policy exceeded 17 own returns. The optional deck was not opened because the frozen policy missed its advancement rule. [All held commands and referee tapes](held.json.gz) · [Paired measurements and replay witnesses](report.json)

All six pairs had identical first incoming racket contact and table bounce. Both policies returned that first ball legally in every pair, but the actual near racket contact differed each time. The regression on 2741 was therefore not a failure to open the rally. Changing the global search policy altered which legal balls were sent back and which subsequent rallies occurred. This result rejects deployment of these two learned weights, not the use of whole-point reward in general. The next experiment should favor perturbations that preserve known-good receiving and show repeatable benefits over multiple distinct physical continuations before updating a general policy.

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt100_episode_policy plan --feeds /tmp/tt25-rallies.json
python -m scripts.experimental.tt100_episode_policy plan-extension --feeds /tmp/tt25-rallies.json
python -m scripts.experimental.tt100_episode_policy practice
python -m scripts.experimental.tt100_episode_policy held
python -m scripts.experimental.tt100_episode_report --verify
python -m scripts.experimental.tt100_episode_policy extension
```

The protocol embeds the measured initial balls. Re-running from the sealed plan does not need the original launch export.
