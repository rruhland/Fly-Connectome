# Direct receiving with an 80 g camera payload

The frozen SO-101 receiver played the first 12 starts from the published direct-receive schedule again using both priors. An 80 g ballast sat on the live arm’s camera housing. The player planned with the nominal 35 g camera and mount assembly and the same dry movement catalogue. Both fixtures used native wrist and fixed RGB and the same end-line scoring. [Matched baseline](direct-receive.md) · [Payload receipt](direct-receive-payload.json.gz)

| Priors | Nominal arm, starts with a legal return | Loaded arm, starts with a legal return | Successes lost / gained | Loaded arm, points won |
| --- | ---: | ---: | ---: | ---: |
| Gantry | 9/12 | 7/12 | 2 / 0 | 3/12 |
| Nominal | 9/12 | 9/12 | 1 / 1 | 2/12 |

With the payload, the nominal priors produced one point with two legal returns. The gantry priors produced zero. Each prior took a median of 36 control frames among loaded-arm successes. Point wins were the same as on the matched unmodified starts. The payload changed individual outcomes for both priors. The results cover this load and these starts. [Per-point outcomes](direct-receive-payload.json.gz)

The ballast was a separate rigid inertial body with the inertia of a 36 × 22 × 18 mm box, located 27 mm, −8 mm and −12 mm along the camera’s local axes. The fixture used the original optical and collision geometry. The simulated camera assembly weighed 115 g and its center of mass shifted 43.3 mm. Applying the same command tape to both arms changed the blade trajectory by up to 13.7 mm over 32 frames. [Mass, center of mass, and inertia audit](direct-receive-payload.json.gz)

The player used RGB ball coordinates and the public control clock. The evaluator supplied the held starts and kept the baseline outcomes, true load and physics state. Both branches ran with zero target practice or stroke bank. The audit reproduced the 24 loaded tapes and the 24 original tapes on the matched subset. [Trial and physical replay](../../scripts/experimental/tt19_direct_receive_payload.py)

```bash
MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt19_direct_receive_payload \
  artifacts/tt19 artifacts/tt19/factory-dry-catalog.npz
```
