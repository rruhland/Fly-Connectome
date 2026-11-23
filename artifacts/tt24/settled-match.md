# A settled bounce calibration carries into live table tennis

The experiment transfers four previously measured stock five-channel command maps. Two were dense remaps, and the other two added off-axis shear. Each map came from one 87-frame, 1.45-second contact-free body exercise in the solo-bounce fixture. The exercise used a 12-frame wait followed by five four-frame probe pulses, each with eleven frames to settle. The fitter saw measured encoder effects. The stored relative matrix errors span 0.210% to 0.697%. [Calibration stream 0](motor-settle-0.json.gz) · [Calibration stream 2](motor-settle-2.json.gz)

The match uses the same direct, previously learned `retina_rate` five-joint readout and responsive far arm as the earlier bounce-to-match benchmark. That benchmark used a different five-drop body decoder. The present test compares unchanged requests through the new wiring, inverse of the settled encoder estimate, and inverse of the exact wiring. The exact matrix belongs to the simulated interface and the explicitly ideal benchmark. The carried condition receives only the earlier estimate. The mapping assumes the same stock five-joint command order and normalization. Physical normalized rate clipping applies to every condition.

The near arm receives actual two-camera RGB and encoders and issues every command from release. No action tape, prescribed opening, target fitting or match reward informs the transferred decoder. There are four fresh scene seeds per map and each is paired across the three conditions. Each completed point is recorded after exact replay of both physical arms and the legal-return referee. [Earlier bridge](bounce-to-match.json.gz) · [Frozen plan](settled-match-plan.json) · [Match runner](../../scripts/experimental/tt24_settled_match.py)

## Fresh physical match scores

| Control interface | Own legal returns | Far legal replies | Complete exchanges | Longest alternating run | Point wins |
|---|---:|---:|---:|---:|---:|
| Inverse of the measured map | 12 | 10 | 10 | 15 | 2 |
| Uncompensated new wiring | 1 | 0 | 0 | 1 | 1 |
| True inverse | 11 | 11 | 11 | 14 | 0 |

On seed `1656923015`, the measured-map controller won with eight learner returns and seven far replies. The uncompensated arm lost without a return. This seed supplied seven of the ten measured-map exchanges. Two maps had zero complete exchanges under the true inverse on their four sampled releases. The 95% paired bootstrap interval for complete exchanges gained over the uncompensated arm was 0.00 to 1.56 per release. The sample establishes immediate transfer on actual points, with sparse opportunities and concentration in one winning rally. All 48 treatment tapes reproduced the legal scores and terminal outcomes. [Aggregate and retention checks](settled-match-results.json.gz) · [Map 0](settled-match-map-0.json.gz) · [Map 1](settled-match-map-1.json.gz) · [Map 2](settled-match-map-2.json.gz) · [Map 3](settled-match-map-3.json.gz)

The 10.67-second matched video replays those two complete points and freezes each panel after its recorded terminal decision. The audit checks both action transformations, both stereo camera digests, the initial broadcast frame and the referee traces. [Matched video](settled-match-witness.mp4) · [Video audit](settled-match-witness.json)

The original carried solo controller was reloaded after the match. On one previously archived wrist release per map, it reproduced every issued rate and full physical result, with longest streaks of 21, 22, 22 and 20. The frozen native Pong readout also reproduced its camera, action and score receipts on six previously archived starts in each of its three comparison arms. None of these recall checks changed the motor or perception weights. [Retention data](settled-match-results.json.gz)

## Reproduce

```sh
export PYTHONPATH=src LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
python -m scripts.experimental.tt24_settled_match --workers 2
python -m scripts.experimental.tt24_settled_match_movie
```

The runner writes each verified point before continuing so interrupted trials resume without discarding completed evidence.
