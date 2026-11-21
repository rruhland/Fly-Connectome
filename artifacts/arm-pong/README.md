# Pong through an arm

[Watch the three-earned-point match](../../docs/reconstruction/assets/2026-11-21-arm-three-earned-points.webp) · [Watch the first scoring-course match](../../docs/reconstruction/assets/2026-11-21-arm-scoring-match.webp) · [Watch the wiring reversal](../../docs/reconstruction/assets/2026-11-21-arm-plays-pong.webp) · [Read the results](../../docs/reconstruction/2026-11-21-pong-through-an-arm.md)

The saved Pong strategy contains its camera-crossing evidence and shot values. The arm checkpoint contains a local model of how issued joint commands move the visible fingertip. Neither contains button wiring. The replays show the first scheduled final scoring match, the earliest match in which all three winning points followed a return, and the first adaptive run of the hidden rewiring course. Add `--earned-points 3` to the scoring replay command to select the all-earned win. The compressed action tape can reproduce its physics without either policy.

## Reproduce

Run from the repository root with the project dependencies installed. Replay generation also uses Pillow. These examples use Bash. Each training or qualification output directory should be a fresh path.

```sh
export PYTHONPATH=src:scripts/experimental
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/experimental/button_pong_skill.py --out /tmp/direct-pong --seed 91729821 --opponent-speed .35 --interception 72 --placement 288 --evaluation 96 --batch 24
python scripts/experimental/run_pong_button_arm_pilot.py --output /tmp/arm-reaches.json
python scripts/experimental/arm_pong_rewire.py artifacts/arm-pong/direct-pong-trained.json /tmp/arm-rewire.json
python scripts/experimental/render_arm_pong_rewire.py artifacts/arm-pong/direct-pong-trained.json /tmp/arm-rewire.json /tmp/arm-pong.webp /tmp/joints.json.gz
python scripts/experimental/arm_pong_timing_run.py artifacts/arm-pong/direct-pong-trained.json /tmp/arm-timing.json
python scripts/experimental/arm_pong_camera_rate.py artifacts/arm-pong/direct-pong-trained.json /tmp/arm-camera-rate.json
```

The main course also loads the interception-only state next to the scoring state. Extract the experiment records, then run either command using the archived training directory. The longer check uses the independently registered schedule seed.

```sh
unzip -q artifacts/arm-pong/2026-11-21-pong-through-arm.zip -d /tmp/pong-arm-records
python scripts/experimental/run_arm_pong_matches.py /tmp/pong-arm-records/source-practice/source-trained.json /tmp/arm-qual --matches 32 --seconds 20 --first-to 3 --workers 3 --gear source
python -c 'import run_arm_pong_matches as m; m.QUALIFICATION_ENTROPY = 92367439481; m.main()' /tmp/pong-arm-records/source-practice/source-trained.json /tmp/arm-long --matches 16 --seconds 120 --first-to 3 --workers 1 --gear source --arms both_live arm_only_live
python scripts/experimental/arm_pong_arrival.py artifacts/arm-pong/direct-pong-trained.json /tmp/arrival-test.json --test release --workers 3
python scripts/experimental/render_arm_pong_match.py artifacts/arm-pong/direct-pong-trained.json /tmp/arrival-test.json /tmp/arm-scoring.webp
python scripts/experimental/arm_pong_archived_checkpoint.py /tmp/archived-arm
python scripts/experimental/arm_pong_calibration_run.py artifacts/arm-pong/direct-pong-trained.json /tmp/shot-calibration.json --selected optical_091 --matches 16 --workers 3
```

[Final scoring comparison](arm-scoring-results.json) · [Direct-practice checkpoint](direct-pong-trained.json) · [Direct-practice results](direct-pong-results.json) · [Reaching-practice states and timing](2026-11-20-rgb-button-arm-pilot.json) · [Matched game records](2026-11-21-pong-through-arm.zip) · [Wiring-challenge joint-action tape](rewire-joint-tape.json.gz) · [Shot calibration records](2026-11-21-original-gearing-shot-calibration.zip)
