# Paired practice found one same-ball rescue, not a broad early-control advantage

On the same ordinary delivery, two different camera-driven preparations turned the stock arm's own-half fault into legal landings. Neither found a second distinct rescue on this 12-offer practice batch. Across the three fixed controllers, a perfect choice would have scored **5/12**, one more than stock. All four moderate lows remained losses. The new credit source is useful because it measures interventions on the same ball, not because it demonstrates a major performance gain.

## Exact reset and command boundary

The fixed training plan uses prior native tapes from ordinary unmodified TT37 random play. It includes four archived easy returns, four ordinary archived misses, and four archived misses in the 0.95–0.99 m projected-height band. Each release contributes one offer. The previous three extreme low cases and the current TT44 fresh roster are excluded. This is a stratified training set, not an estimate of offer prevalence. [Frozen plan](plan.json)

The original commands are replayed from launch to the first control boundary after the opponent's physical separation. One full MuJoCo solver workspace and the evaluator's referee and partner histories are then copied. The archived command continuation is replayed from that copy before allowing experiment branches. All 12 snapshot verifications reproduced the recorded events exactly. Each branch also retained the recorded far impact and near bounce, within a 1e−8 numerical tolerance, and used the original continuous random opponent target draws. No launch is synthesized and no future observation is passed to an actor.

The same short RGB history warms the fit for all three branches. Every teacher chooses its first policy on the same observed row before issuing a differing command. Stock applies the existing instantaneous direct field. The temporal teacher blends a short-history prediction of five joint references into that field. Four-tick preview queries the ordinary direct field on a short forward prediction of the RGB flight. All three use the same normal boot, clamp, live opponent, rate limits, late joint MPC and candidate budget. No score or policy change occurs inside a branch until the target ball physically scores or the point ends.

## Causal movement and landing credit

| Practice offers | Stock | Temporal | Four-tick preview | Any teacher |
|---|---:|---:|---:|---:|
| Easy archived successes, 4 | 4 | 4 | 4 | 4 |
| Ordinary archived failures, 4 | 0 | 1 | 1 | 1 |
| Moderate low archived failures, 4 | 0 | 0 | 0 | 0 |
| Total, 12 | 4 | 5 | 5 | 5 |

The two interventions moved actual early joint angles by up to 0.277 and 0.200 rad relative to stock, respectively. The sole positive causal contrast was 4450. Stock touched the ball at y = −1.563 m, then faulted into its own half. The temporal action struck 10.4 ms later and 26.9 mm deeper. Preview struck 6.25 ms later and 16.6 mm deeper. Both physically landed. All three tapes replayed independently from the original launch with identical events and zero full-CAD overlap in 3,028 total solver states. [Physical witness](audit-4450.json)

The training receipt includes all 36 motions, none censored. Its 1,582 causal rows retain tick, stereo position and velocity, own encoders, previous and sent references, and normalized issued rates. Legal or failed outcomes are attached to the whole treatment trajectory, not repeated as independent credits at each frame. Identity, projected height, native contact, and the exact fork tick are evaluator metadata, not input features. [Reusable training observations](training.json.gz)

The frozen selector estimates local return advantage relative to stock using only the common RGB and encoder context at the moment when the three controllers may begin to differ. Positive and negative contrasts are both eligible. A lone contrast cannot establish a transferable advantage. Agreement by two preparations on one physical rescue is still one offer and permits only a narrow local pilot. Leave-one-release-out recomputation assesses how little evidence this sample provides. [One-sample-per-offer readout](uplift.json)

## Release-disjoint continuous games

The frozen selector was exercised on four separate complete launches from the held direct course, disjoint from all 12 fitted releases and excluded from the current TT44 fresh roster. Scores were **13 vs 16 stock**. It improved 11884 from 4 to 7 and reduced 7901 from 11 to 5. Both variants landed the first changed ball in both games, and the physical far impact and near bounce on that particular ball matched stock. The divergence in scores came later, after the changed returns led the reactive opponent to deliver different paths. Both unchanged-policy games reproduced the stock commands. This is evidence against treating local first-return rescue as sufficient for rally improvement. [Held games](held.json.gz)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
out=artifacts/tt44_paired_credit
python -m scripts.experimental.tt44_paired_credit plan
python -m scripts.experimental.tt44_paired_credit collect
python -m scripts.experimental.tt44_paired_credit_audit --ids 4450 --output "$out/audit-4450.json"
python -m scripts.experimental.tt44_paired_credit_fit fit
python -m scripts.experimental.tt44_paired_credit_fit play
python -m pytest -q tests/test_tt44_paired_credit.py
```
