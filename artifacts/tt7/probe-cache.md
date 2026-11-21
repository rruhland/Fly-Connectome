# Sharing calibration images across matched receiving trials

TT6 and the TT7 receiving trials accept `--probe-cache DIR`. Use the same directory for groups with matching installations, cameras, calibration schedules, and probe commands. A standard run captures the probes live.

The cache stores the reset image and each resulting stereo RGB image as lossless uint8 arrays, one archive per probe. Each run still makes its own recall and fit decisions. It generates the same public commands and builds its trackers and motor maps from the RGB images for every probe it actually requests. A later group that needs more probes captures the missing trajectories as it reaches them. A warm run advances MuJoCo and the command transport through the original actions with rendering disabled.

The key covers the compiled MuJoCo model, including joint axes, camera calibration and appearance seed, wiring and disturbances, rendering implementation and driver, ordered calibration seeds, and exact command bytes. Native MuJoCo rendering, cached hall rendering, and the live-reflection renderer use separate keys. Writers lock individual trajectories and publish completed archives by atomic rename. An unreadable archive is recaptured.

## Reproduce the parity check

Use MuJoCo 3.17.0 and the default 192 × 144 stereo hall. The course uses seed 54782309, three installations, and two receiving points at each installation. It requests 8, 5, and 8 calibration trajectories, totaling 735 stereo images including resets.

```bash
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
work=$(mktemp -d)
for pass in plain cold warm; do
  cache_args=()
  if [[ "$pass" != plain ]]; then cache_args=(--probe-cache "$work/cache"); fi
  time python -m scripts.experimental.tt6_continuous_trial "$work/$pass.json.gz" \
    --seed 54782309 --course offset gantry offset-b --phases ordinary mixed \
    --predictive --cached-render --reflections "${cache_args[@]}"
done
sha256sum "$work/"*.json.gz
```

## Result

| Run | Process CPU | Wall time |
| --- | ---: | ---: |
| Live probes | 38.50 s | 224.59 s |
| Empty cache | 42.73 s | 149.39 s |
| Warm cache | 24.11 s | 53.87 s |

The warm run used 37% less process CPU than the live run. Concurrent experiments shared the host during this check, so the wall times also reflect changing CPU contention. The warm run reused all 21 archives, totaling 28.2 MiB.

All three compressed receipts matched byte for byte. They recorded the same calibration decisions and fits, six action tapes and physical outcomes, two legal returns, and one receiving point. The shared SHA-256 was `a93f6724a637aee636fc13ea7ea49a3b41056db8a6dacbaeed65932534e93241`.
