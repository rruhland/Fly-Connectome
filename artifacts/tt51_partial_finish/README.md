# Reversible receive preparation and body-conditioned finishing

The original clamp, disk paddle, servo limits, launch hardware, optics, broad TT37 opponent, and referee are used throughout. The shared motion adds at most .48 rad to elbow, .105 to forearm and .22 to wrist. It begins only after optical evidence of an already-played outward ball. The arm attenuates the move using the previous played body, encoder speed and reference lag, then releases it as the new incoming ball becomes visible. It does not assume a left or low next return. An optional early far-paddle risk forecast can reduce the movement without adding a pose switch.

To learn the follow-through, first run several exact, previously earned native outbound states on unrelated releases. The prior return must remain unchanged. A native training case only supplies a turning waveform if its own actuation lands the ball and passes the independent CAD scan. A private teacher may propose a few controls for a failed training ball. Those controls must physically return the same offer and be independently replayed while recording real RGB and encoders. No private teacher runs in held assessment.

The arm starts from the existing strong direct controller. Additional waveform candidates depend on current optical flight, the attained joint state, local encoder velocity and issue-to-body lag. It retimes nearby motion traces using its own stereo trajectory prediction and screens them with the same independently camera-initialized model. A candidate must continue to work under a perturbation before it can replace the native fallback, and cannot displace a currently supported stock commitment. The policy does not receive the actual ball, current draw, score to come, or the later referee placement.

[Fixed school/held fork roster](plan.json) · [Predeclared fully native game roster](whole-plan.json)

## Physical school

Stock and partial mode both returned the same four of six native offers. All twelve traces retained the prior native far hit and near bounce without a single penetration sample. Four ordinary receives from projected heights 1.024–1.057 m produced genuinely rewarded shallow-preparation waveforms, with observed elbow changes up to .48 rad. The two .978–.980 m low arrivals were missed from both preparations. A finite after-cue teaching pass from the altered native body screened 96 controls for each low and found 3 and 65 touches, respectively, with no legal return. This is evidence for moderate readiness at normal height, not evidence that it reaches the full deep-low branch. [Native school actions and observations](school.json.gz)

## Held native offers

The same three-way test on six disjoint releases scored **4/6 stock, 4/6 partial and 3/6 partial plus the learned finish**. All 18 branches received the same physical far hit and near bounce as their paired reference. The arm generated a new held, legal contact on 12866, which stock also returned. The learned finish turned another stock success, 5212, into a bounce on our side of the table, despite both model perturbation checks passing. Two other low held offers were still missed while a third low was returned by all conditions. Zero penetrations were recorded across 18,208 changed-tail CAD samples. The safe reversible preparation transferred more readily than the policy for deciding when to replace a future stock strike. [Complete held test, including rejected and selected motion candidates](matched.json.gz)

## Complete points on broad independent streams

The preregistered four-game pilot used the frozen kernel risk model `4478d3b7…`, only to scale the size of the small readiness move. It made 172 actual precontact stereo forecasts. The two controllers scored **43 stock vs 41 with preparation and learned finishing**, so there is no improvement over stock in this pilot. [All native controls, forecasts and reward-triggered adaptation](whole.json.gz)

| Release | Stock returns | Prepared plus finish | Exact common initial offers |
|---|---:|---:|---:|
| 7901 | 6 | 9 | 2 |
| 12068 | 17 | 1 | 2 |
| 11884 | 3 | 16 | 1 |
| 10294 | 17 | 15 | 1 |

The action choices produced 35 new scored, observation-derived waveforms. Points 7901 and 11884 later reused a waveform acquired during that same point and physically scored again. The first action difference in 11884 was an opening stroke before any new readiness motion, so the improvement cannot be assigned to the forecast or preparation alone. On 12068 the second physical ball was still identical and the preparation-plus-finish actor scored nothing from it. The selected learned stroke had passed both simulator perturbations. Later exchanges follow the different ball the opponent actually saw, not identical trajectories or matched outcomes.

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt51_partial_finish plan
python -m scripts.experimental.tt51_partial_finish school
python -m scripts.experimental.tt51_partial_finish held
python -m scripts.experimental.tt51_partial_live plan
python -m scripts.experimental.tt51_partial_live run --n 4 --predictor artifacts/tt50_opponent_vision/risk-model.npz
python -m scripts.experimental.tt51_partial_live_audit
python -m scripts.experimental.tt51_partial_live_audit --isolate 12068
python -m pytest -q tests/test_tt51_partial_finish.py
```

Whole-point comparisons retain the stock player's ordinary within-point memory updates in both conditions. The ready branch can also retain its own previously scored waveform using only the native scalar reward and observed stereo velocity reversal, its encoder history and issued joint references. Each pair starts from the same carried checkpoint and the same independent physical stream. No best-of replays are counted.
