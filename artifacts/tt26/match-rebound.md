# Does paddle bouncing teach a better receiving search?

The stock-paddle rebound fit produced the same physical receiving results as the existing cold reflection rule on every held shot at every practice budget. The fitted rule returned seven of 96 similar recorded feeds at the largest budget, as did cold. The shuffled fit returned four. All three returned zero of 32 unrestricted feeds. [Paired TT25 trials and camera checkpoints](match-rebound.json.gz)

## Stock paddle

We replayed 16 actual wrist-camera bounce episodes from TT24 and reproduced all 113 referee-verified rebounds. Around each contact the recorder read ball velocity, spin and blade velocity from MuJoCo. It fit two coefficients in blade coordinates, one for normal restitution and one for tangential velocity lost to surface slip. The source actions came from the wrist-camera and proprioception policy. The subsequent collision fit used privileged simulator measurements and rebound labels. The corresponding shuffled fit permuted measured contact responses across the same impacts. The existing TT25 cold ranker supplied a restitution coefficient of 0.85 and zero slip correction. The stock bounce fit found 0.8485 restitution and zero slip correction. Shuffling found 0.8269 and zero. [Source action archive](../tt24/bounce-ident.json.gz) · [Measured impacts and fits](match-rebound-source.json.gz)

All conditions used the same 24 reach-screened TT25 development feeds and the same bank of rate-limited postures. The teacher ranked proposals with known ball flight. Only the predicted rebound changed across conditions. The physical court tried the leading proposals and admitted a demonstration only after a legal landing across the net. The camera pupil then matched four early stereo images to a successful development demonstration. The held course used the two established nearby cohorts of 48 distinct recorded feeds and the established unfiltered cohort. Held successes were verified by repeating the actual image-based action selection and comparing the referee result. The opponent held its home posture. [TT25 receiving protocol](../tt25/recorded-receiving-lessons.md) · [Trial implementation](../../scripts/experimental/tt26_match_rebound.py)

| Maximum ranked probes per feed | Actual probes per condition | Development feeds with a legal return, bounce / cold / shuffled | Nearby held legal returns out of 96, bounce / cold / shuffled | Unfiltered held legal returns out of 32, bounce / cold / shuffled |
| ---: | ---: | --- | --- | --- |
| 4 | 92 | 1 / 1 / 1 | 2 / 2 / 2 | 0 / 0 / 0 |
| 12 | 268 | 4 / 4 / 4 | 3 / 3 / 3 | 0 / 0 / 0 |
| 24 | 532 | 6 / 6 / 6 | 7 / 7 / 4 | 0 / 0 / 0 |

Two development feeds had only three or one eligible proposals. The same number of probes was available to each condition. The stock bounce and cold rules selected the same camera demonstration at all 384 held shot and budget combinations. The cold ranking reproduced all 24 original TT25 proposal orders. [Per-shot trials, budgets and selected demonstrations](match-rebound.json.gz)

## More heavily damped paddle

The higher-damping paddle gave the bounce fit a materially different response. The intact fit still found no legal TT25 practice action in 532 probes, and the camera pupil returned none of the held feeds. The cold ranker found one demonstration and returned two nearby held feeds at both larger budgets. The two held successes came from separate nearby cohorts. [Paired soft-paddle trials and camera checkpoints](match-soft-rebound.json.gz)

We set the MuJoCo direct-format damping parameter to 85 for the learner's blade in both tasks, with the rest of the trial protocol kept fixed. The previously trained wrist-camera policy completed 24 fresh solo drops on this setup. Its command tapes replayed exactly and produced 130 verified rebounds for the collision fit. The intact fit estimated 0.6602 normal restitution and 0.1155 slip correction. The shuffled fit estimated 0.6304 and zero. Cold retained the stock analytic values of 0.85 and zero. This material change is a simulated intervention. [Wrist action tapes, physical impact measurements and fits](match-soft-rebound-source.json.gz)

| Maximum ranked probes per feed | Actual probes per condition | Development feeds with a legal return, bounce / cold / shuffled | Nearby held legal returns out of 96, bounce / cold / shuffled | Unfiltered held legal returns out of 32, bounce / cold / shuffled |
| ---: | ---: | --- | --- | --- |
| 4 | 92 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 12 | 268 | 0 / 1 / 0 | 0 / 2 / 0 | 0 / 0 / 0 |
| 24 | 532 | 0 / 1 / 0 | 0 / 2 / 0 | 0 / 0 / 0 |

## Reproduce

Use the pinned DeepMind rally states and MuJoCo 3.17.0. [Dataset identity and provenance](../tt25/recorded-benchmarks.md)

```bash
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt26_match_rebound /path/to/rallies.json
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt26_match_rebound /path/to/rallies.json --soft
```
