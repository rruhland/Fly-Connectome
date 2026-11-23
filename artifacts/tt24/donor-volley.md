# From repeated bouncing to a first-flight partner return

We tested whether a reward-trained repeated-bounce head helped the same stock SO-101 learn a different physical goal. In the main court, one incoming ball must reverse its lateral travel after a legal face impact and cross an upward receiving aperture on the entry side. The aperture is 165 mm above the initial blade center, 80 mm to the entry side, 110 mm wide and 140 mm deep. A powered delivery also requires upward blade contact-point speed above 55 mm/s. The partner is represented by a static receiving region. Motions after the first encounter are evaluated physically with the last servo targets held. The head receives no second encounter or further decisions. [Runner](../../scripts/experimental/tt24_donor_volley.py) · [Original volley task and independently trained reference](tt24_volley_source.md)

## Observation qualification

The frozen trained volley reference and stationary control received 12 paired task runs in each preflight. Before opening target practice, the reference needed at least four physical deliveries and at least three more than stationary. The original elevated gate and the bounded-speed two-receiver variant both failed with the stock wrist observer. A diagnostic on the original wrist seeds found that the frozen volley reference made nine powered deliveries with true metric observations, compared with three using the wrist. The frozen bounce head made zero in that diagnostic. The true-state observation was used only for this diagnostic and native referee parity.

| Observer and receiver | Ready volley routes | Ready powered | Stationary routes |
|---|---:|---:|---:|
| Wrist, elevated mirror gate | 3/12 | 3/12 | 0/12 |
| Wrist, bounded-speed two receivers | 3/12 | 2/12 | 5/12 |
| Two fixed RGB cameras, elevated mirror gate | 10/12 | 10/12 | 0/12 |

The final row uses two surveyed stationary cameras to triangulate the ball, a causal free-flight velocity estimate, and the existing public encoder fit for blade position and normal. The optical differentiator accounts for gravity during incoming free flight. This setup opens a test of the motor transfer with public observations. It requires fixed cameras in addition to encoders. Three independent clean reference cases per profile agreed with the existing native referee within numerical tolerance. [Original wrist tapes](donor-volley-preflight.json.gz) · [Two-receiver wrist tapes](donor-volley-two-receivers-preflight.json.gz) · [Clean diagnostic](donor-volley-clean-diagnostic.json.gz) · [Public stereo feasibility](donor-volley-stereo-preflight.json.gz)

## Eight physical practice attempts per learner

Two independent streams used four paired perturbations on distinct launches. Each pair scored two physical episodes from the same release, one with either sign of the same random parameter direction. The optimizer used only the difference in the resulting physical score. The direction sequence and update rule were identical for the three initializations. The scrambled head permutes and flips sensory columns and preserves the original per-motor weight norms and singular values. The learners received no demonstrated actions or hidden state. The stationary and ready-volley heads received no practice updates.

After eight attempts per arm per stream, each frozen result ran on 20 disjoint held feeds. The score combines the true receiving gate with bounded shaping for actual flight, direction, proximity and paddle lift. Tentative weights were selected solely from practice, and replay and holdout outcomes did not update them.

| Initialization and training | Held routes, out of 40 | Powered | Mean shaped score |
|---|---:|---:|---:|
| Repeated bounce with eight attempts | 1 | 1 | .853 |
| Blank with eight attempts | 0 | 0 | .736 |
| Scrambled bounce with eight attempts | 0 | 0 | .740 |
| Frozen repeated bounce | 1 | 0 | .730 |
| Frozen scrambled bounce | 0 | 0 | .700 |
| Stationary | 0 | 0 | .501 |
| Ready volley reference | 26 | 26 | 2.137 |

None of the 48 reward-used practice episodes produced an accepted physical delivery. The bounce-initialized updates raised the shaped score on the held set, with one powered delivery across both streams and the same accepted-route count as the frozen bounce head. This budget did not establish an advantage in learning partner returns. Every target motor tape replays to the same physical metrics. [Practice and held tapes](donor-volley-stereo.json.gz)

## Return to bouncing

The two updated heads were frozen and tested on 12 fresh wrist-only ordinary bounce drops. They averaged 1.75 and 7.08 longest consecutive rebounds. Reloading the preserved bounce head averaged 15.00 and reached 18. The source checkpoint's bytes are unchanged. The preservation result comes from recalling the saved source head and does not establish bounce retention in the adapted parameters. [Wrist return tapes](donor-volley-stereo.json.gz)

## Reproduce

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
python -m scripts.experimental.tt24_donor_volley --preflight-only
python -m scripts.experimental.tt24_donor_volley --two-receivers --preflight-only
python -m scripts.experimental.tt24_donor_volley --stereo
```
