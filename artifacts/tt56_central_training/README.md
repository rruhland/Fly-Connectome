# Training on actual central ball variation

The two previously coached central movements and the 40 natural optical examples are fixed sources. The new practice roster contains four other releases. The four held releases are different again and do not contribute a template to the new optical actor. The older exact forecourt case 4214 is not used.

Each case starts by replaying the same previously achieved outgoing stroke. Fresh independently seeded standard TT37 draws are admitted by a fixed, generous neighborhood of the requested goal to limit scout cost. They are not clipped or altered. Training eligibility and distance between variants use absolute RGB trajectory estimates on the later inbound flight. At most two optically distinct deliveries per release enter matched actual comparisons. The early shared arm body comes from the archived applied commands, while the begin point is triggered by publicly observed inbound motion. This is normal-attained-body practice, not whole-game qualification.

The ordinary optical and encoder controller runs first. On a missed offer, a bounded true-state teacher compares direct signed candidate paths, then submits at most two. Only an actually landed and physically answered return, independently replayed with strict CAD, can authorize a new observed movement. At least two release-disjoint acquisitions are required before opening the fixed held set. The held controller uses only the actual RGB track, own encoders, signed reference history and the resulting templates. It retargets the learned contact pose by small regularized position and face corrections and retains native model screening after normal control finds no robust option. No target, teacher clock or true plant state enters the deployed comparison. [Preassigned roles and draws](protocol.json)

## Fixed nearby-draw practice

All 48 scouts yielded a usable later RGB track. The eight admitted offers were physically different. The stock receiver answered four, and missed the other four at actual deep heights .9893, 1.0017, 1.0033 and 1.0058 m. Across these misses the controller's independently rated local motor search produced 170 predicted contacts in 576 trials, but no legal prediction. It supplied no candidate to physically try, so no teacher score or learned transfer is claimed. [Complete first physical course](training.json.gz)

The follow-up uses a practice-only no-render prefilter on fresh full-law draws to find tougher actual trajectories before spending stereo frames. Later RGB readings still select the physical comparisons. It tests a separate earlier body and both signed faces instead of expanding the same narrow waveform perturbations.

## Earlier signed-face practice

All 128 prefilter draws produced actual receiving-line crossings under the original TT37 law. Eight were truly below 1.00 m and within the 20 cm deep central lane. Twelve independent targets were paid for as stereo scouts, compared with 48 in the fixed neighborhood. Later RGB admitted seven distinct actual offers. The ordinary arm answered one and missed six. The earlier signed-geometric teacher tried 1,152 native trajectories across those misses, predicting 511 contacts and zero legal returns. This cost is separate from the earlier 576 native trials. No model-qualified candidate existed for actual reward practice. The two mechanisms therefore yielded no newly rewarded motion across the ten missed opportunities. The held roster remained unopened and no full game was run. These are finite attempts from the stated bodies, not a bound on the stock arm's absolute reach. [All 128 unmodified actual scout draws](scout-native.json.gz) · [Active optical protocol](active-protocol.json) · [Actual offers and early signed coach results](active-training.json.gz) · [Unopened validation receipt](held.json.gz)

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt56_central_training train
python -m scripts.experimental.tt56_central_training_active
python -m scripts.experimental.tt56_central_training train --protocol artifacts/tt56_central_training/active-protocol.json --out artifacts/tt56_central_training/active-training.json.gz
python -m scripts.experimental.tt56_central_training held --earned-training artifacts/tt56_central_training/active-training.json.gz
python -m pytest -q tests/test_tt56_central_training.py
```
