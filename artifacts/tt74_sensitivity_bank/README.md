# Centered SO-101 receiving sensitivity bank

On ten new continuous opponent streams, both ordinary and conditional play earned **54 legal returns on 64 actual offers**, with 57 contacts per condition. Every recorded command and referee event agreed across the pair. The bank issued no correction and acquired no new reward entry. This is a failure to activate the capability, not evidence that active deployment preserves ordinary returns. [All 20 one-launch tapes](games.json.gz) · [Paired results and native replay](report.json)

There were 54 public decision windows. The gate deferred to 47 stock-supported decisions and found no sufficiently supported local transport once. The other six windows supplied 24 distinct bank proposals to the public twin. None was nominally legal. Subsequent ordinary actions still scored the same real delivery in one of those six windows and missed the other five. Eliminating the nominal gate by itself would therefore also expose an ordinary success. This diagnoses an action-selection bottleneck on these continuous points, not lack of the local physical authority shown in TT67. [Local matched-reply result](../tt67_action_sensitivity/README.md)

For both modes, low deliveries below 1.035 m scored 6/9 left, 9/15 middle, and 3/4 right. The 36 other deliveries all scored. The longest 17-return point and the earlier 13-return point independently reproduced the native events. Their combined 24,349 CAD checks had no penetrations, and minimum positive clearance was 15.64 mm.

## Calibration and selection

The full source plan reserves 57 release-disjoint ordinary paid strokes. The bank kept 34 independently rooted local response laws on the original centered fixture. The 170 native source worlds supplied 1,870 nominal and paired signed action probes. Twelve roots are left, ten middle and twelve right in optical contact-plane coordinates. Twenty-five of the 34 meet the fixed .032 m/rad lateral/vertical authority threshold. None of the ten prospective game release IDs entered this bank. [Frozen releases and sources](protocol.json) · [Source bank and fit evidence](bank.json.gz)

Training selects physically returned and answered archive windows using earlier public RGB search position and support. It does not select by the subsequent exact impact radius or simulator contact time. The short finite-difference model needs only the source's observed entering state and its actually issued path. A conservative measured optical approach support bound uses its nominal blade geometry before the predicted contact window.

At assessment, the ordinary player receives its own standard, uninterrupted camera and encoder stream. A sensitive root comes from those current sensor values, not from a release, leg, target, exact simulator state or recorded future. The gate keeps any ordinarily proposed move with a nominal legal forecast. Falling back to the bank requires optical and encoder similarity, enough lateral/vertical actuator authority, a measured nontrivial displacement, and at least 7 mm of predicted gap reduction. The best of up to eight candidates must be legal in the public nominal twin and in at least one of its two error settings. It may issue one direct correction per incoming ball.

If a genuine emitted correction earns a legal return, the adaptive bank can retain the actually sent joint-reference shape, rooted in its own prior public observations. An inherited measured Jacobian is marked down in authority. Nothing estimates a new Jacobian from one binary reward, and no extra practice, replays, or privileged clues enter the actor's rule.

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt74_sensitivity_bank plan --feeds /tmp/tt25-rallies.json
python -m scripts.experimental.tt74_sensitivity_bank fit
python -m scripts.experimental.tt74_sensitivity_bank games
python -m scripts.experimental.tt74_sensitivity_report --verify
python -m pytest -q tests/test_tt74_sensitivity_bank.py
```

The plan embeds each measured initial ball, so execution from the checked-in protocol does not need the separate recording export.
