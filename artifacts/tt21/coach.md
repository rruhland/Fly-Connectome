# Reply reachability and a cooperative practice rival

The comparison replays eight first learner strikes from the unmarked TT20 clamped game, balanced across learner-served and learner-received starts. The openings and their physical contact events are identical under all rival conditions. After the archived stroke, the learner uses fresh wrist and both fixed native RGB feeds with the existing factory-serve and rollout policy. The regular rival provides the standard reference. The physical search rival provides the stronger reference. The cooperative rival applies the same joint-rate limits and ranks its physical replies for a usable, varied central interception window after the receiving bounce. The strict referee is shared across all three conditions. [Source game](../tt20/curriculum-frozen-2082473549.json.gz) · [Practice rival](../../scripts/experimental/tt21_coach.py)

The reachability evaluator branches after the first physical rival strike, once the ball reverses and separates from the far blade. A copy of the actual simulator state and referee state supplies the initial arm posture, requested joints, and ball motion. An independent blade-free flight probe identifies potential post-bounce windows. A bounded search proposes up to 270 rate-limited swings and tests up to 12 promising candidates with the strict referee in a separate arena. The far arm stays on its side and returns home during the exact counterstrike trial. A return counts only when the ball contacts the learner's paddle after the near bounce and earns a legal far-side landing. The RGB learner receives neither the evaluator's actions nor its state or outcome. The resulting count lower-bounds playable replies for the tested aperture and search budget. [Evaluator](../../scripts/experimental/tt21_coach_oracle.py)

The first successful independent witness came from archived point 2 (seed `890890078`). The standard opponent struck at tick 41 and landed at tick 52. The RGB learner earned no credit on that reply. A separate branch made legal learner contact at tick 53 and earned the far-side landing at tick 72. The pilot retains the full counterfactual command tape and strict events. [Pilot receipt](coach-replies-pilot.json.gz)

| Opponent | Shared openings | Real opponent paddle replies | Legal opponent replies | Legal counterreturns witnessed by evaluator | Legal counterreturns by the RGB learner |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 8 | 7 | 3 | 2 | 0 |
| Strong search | 8 | 8 | 7 | 3 | 0 |
| Cooperative practice | 8 | 8 | 7 | 6 | 1 |

All 24 trials preserved their original opening action prefixes. A second physical replay reproduced the branch state and exact referee traces for the first witnessed return under each opponent. On held point 12 (seed `1963576530`), the cooperative rival's reply landed legally at tick 63. The RGB policy struck it at tick 64 and earned its legal return at tick 79. The cooperative mode doubled the number of exact reachability witnesses relative to the strong rival on these eight openings. The frozen RGB policy returned one of those feeds. This supports using the coach to collect more physically returnable practice attempts. Competition results remain separately measured against the standard and strong opponents. [Matched RGB, action and referee receipts](coach-replies.json.gz)

The tape records include the observation stream, released reply tick, and exact evaluator command and event sequence for each witnessed return. The trial runs with native MuJoCo EGL in the repository environment.

```bash
python -m scripts.experimental.tt21_coach_trial --limit 8 --output /tmp/tt21-coach-replies.json.gz
```

[Trial harness](../../scripts/experimental/tt21_coach_trial.py)
