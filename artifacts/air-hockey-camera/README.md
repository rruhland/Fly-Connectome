# Air hockey, Pong adaptation and reaching from RGB

[Full experiment report](../../docs/reconstruction/2026-11-20-air-hockey-and-camera.md) · [Side-by-side hockey replay](../../docs/reconstruction/assets/2026-11-20-hockey-selfplay.webp) · [Hockey-to-Pong replay](../../docs/reconstruction/assets/2026-11-20-hockey-to-pong.webp) · [Moving-camera arm replay](../../docs/reconstruction/assets/2026-11-20-rgb-arm-motor-adaptation.webp)

Download [the offline hockey match](hockey-replay.html) using GitHub’s **Download raw file** button and open it in a browser. It contains the actual camera frames from the first registered first-to-seven match against the tracking guard. Use the policy selector to switch between the starting policy and the policy trained against the laneguard. The recording includes scoreless points and both policies’ complete matches. The self-play animation above shows the later adaptation to an equally mobile rival.

## Learned states and receipts

| Files | Contents |
|---|---|
| [Hockey checkpoint](hockey-checkpoint.json) · [source, training and two held-out cohorts](air-hockey-forward-and-practice.zip) | Pong-derived wall estimate and shot values from 400 hockey points. |
| [Self-play checkpoint](hockey-selfplay-checkpoint.json) · [registered training, fresh points and paired RGB replay](hockey-selfplay.zip) | Acquired approach and shot tactics after another 360 practice points, with two equally mobile frozen rivals. |
| [Scored matches and head-to-head checks](hockey-matches.zip) | Points, seeds, action hashes, goals, own touches and timeouts. |
| [Direction transfer](hockey-direction-transfer.zip) · [shot transfer](shot-placement-transfer.zip) | Separate controls for knowledge of motor polarity and the striker-offset to outgoing-angle relation. |
| [Hidden wall transfer](../m2/2026-11-20-air-hockey-pong-wall-transfer.zip) · [learned update rate](changed-pong-update-rate.zip) | Clear-camera revision timing, all practice and matched target outcomes, plus two independent covered-camera replications. The update-rate bundle includes its own practice inputs and the original clear-view batch runner. |
| [RGB arm results](rgb-arm-results.zip) | Registered schedules, all physical reaches and per-stream costs, replay actions and verification, plus moved and stable camera tests of the previous carriage weights. |

[SHA-256 hashes](sha256.json)

## Reproduction

Run from the root of the repository using the project's Python environment. Specify new paths for outputs from runners that use exclusive creation.

```bash
export PYTHONPATH=src:scripts/experimental OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p /tmp/hockey-research
python scripts/experimental/run_hockey_matches.py artifacts/air-hockey-camera/hockey-checkpoint.json /tmp/hockey-research/guard.json --opponent guard
python scripts/experimental/render_hockey_match.py artifacts/air-hockey-camera/hockey-checkpoint.json /tmp/hockey-research/guard.json /tmp/hockey-research/match.html
python scripts/experimental/hockey_selfplay.py artifacts/air-hockey-camera/hockey-checkpoint.json /tmp/hockey-research/opponent-practice --workers 2
unzip -q artifacts/air-hockey-camera/changed-pong-update-rate.zip -d /tmp/hockey-research/rate
python scripts/experimental/occluded_pong_transfer.py /tmp/hockey-research/rate/reverse-confirmation/practice.json /tmp/hockey-research/replication.json --seed 238950
python scripts/experimental/run_camera_arm_transfer.py /tmp/hockey-research/arm.json --first-seed 812340 --streams 64 --courses both unchanged
python scripts/experimental/registered_arm_preferences.py /tmp/hockey-research/registration.json --seed 712640
```

The camera runs use a planar simulated arm and an RGB renderer with explicitly painted features. All learning is local and non-backpropagating. The hardware was never connected.
