# Online optical control of the SO-101

After 24 charged setup frames, the player carrying TT17 experience landed more legal own serves than the cold player on three independent port layouts and game streams. The difference appeared with the original scorer and with strict end-line scoring. On the strict matched games, both players landed one legal return and lost 1–11. The cold player landed its return five control frames earlier. [Summary and checkpoint hashes](online-summary.json.gz)

| Stream and controller | Start | Final scores | Legal own serves | Legal returns | First serve / return, charged frames |
| --- | --- | --- | ---: | ---: | --- |
| 29413852, clipped inverse, original scorer | Carried, 24 setup frames | 2–11 | 3/6 | 0 | 166 / — |
| 29413852, clipped inverse, original scorer | Cold, 24 setup frames | 0–11 | 1/5 | 0 | 265 / — |
| 670448313, speed objective, original scorer | Carried, 24 setup frames | 0–11, 0–11 | 2/10 | 0 | 136 / — |
| 670448313, speed objective, original scorer | Cold, 24 setup frames | 0–11, 0–11 | 0/10 | 1 | — / 65 |
| 670448313, speed objective, original scorer | Carried, zero setup frames | 1–11 | 0/6 | 1 | — / 157 |
| 670448313, speed objective, original scorer | Body frozen after two points, 24 setup frames | 0–11 | 0/5 | 0 | — / — |
| 886230749, clipped inverse, strict scorer | Carried, 24 setup frames | 1–11 | 1/6 | 1 | 184 / 63 |
| 886230749, clipped inverse, strict scorer | Cold, 24 setup frames | 1–11 | 0/6 | 1 | — / 58 |
| 886230749, clipped inverse, strict scorer | Carried with nominal ball law, 24 setup frames | 0–11 | 1/5 | 1 | 448 / 61 |

The strict branch uses the compact end line when judging a long bolley's interception. Each matched comparison has the same public court dimensions, hidden motion-port layout, opening-ball seeds and 30 Hz control clock. Each policy plays its own chronological scored stream. Setup uses a short pulse schedule selected with its initial settling curve and charges each applied frame. It holds the public gripper channel idle, also throughout all games. Warm and cold can select different setup commands within the same budget. The two-game branches continue from the saved endpoint on the same physical fixture using a new point schedule. [Simulation and referee](../../scripts/experimental/tt19_so101.py) · [Evaluation protocol](../../scripts/experimental/tt19_online_trial.py)

The player observes images from the actual wrist sensor and two fixed views. Camera registration uses visible landmarks and blade orientation uses rim marks. Motor learning trains a local quadratic map of optical blade position and inclination from integrated, time-filtered public commands. The primary estimator fits short changes in blade pose and scores candidate transport lags before updating. A tested alternative fits absolute pose changes from the opening blade image. The body models start with zero command gains. The carried branch uses a settling curve from the TT17 gantry, a fitted 3D flight law from sequential gantry RGB, and prior optical racket-impulse samples. The cold branch uses generic timing and ball dynamics. The nominal-flight ablation keeps the carried settling and impulse fit while replacing only the ball law. [Optical inverse model](../../scripts/experimental/tt19_online_body.py) · [Flight forecast source](flight-source.json.gz) · [Vision tracker](../../scripts/experimental/tt19_view.py)

On the strict warm game, the blade contacted all six opening rival serves. Five failed to land legally. Cross-table flight remains the main bottleneck on that stream. The fitted ball law moved the first legal serve from charged frame 448 to 184 with one legal serve each. The speed objective in the second stream explicitly plans forward blade velocity around contact. It produced one legal return in the cold branch while both warm games had zero returns. All listed games ended in losses and none recorded two legal returns in one point. [Strict warm tape](online-886230749-warm-wrist+fixed-setup24-strict.json.gz) · [Nominal ball-law tape](online-886230749-nominal-flight-wrist+fixed-setup24-strict.json.gz) · [Speed objective tapes](online-670448313-warm-wrist+fixed-setup24.json.gz)

The endpoint keeps the old gantry model bank, gain transform, optical calibration and Pong state. New optical bounce and impulse evidence flows into the shared world fields, and the original gantry world stays available under its own checkpoint key. The RGB replay reproduced every command on all 12 strict warm points and the final motor state. Every scored tape in the summary also passed physical replay. [Addressed and shared retention tests](retention.md)

## Learning to register blade pitch at impact

The pitch-registration branch compares intended and observed blade pitch near the estimated impact time, when the blade stays near the planned contact position. It retains a correction per stroke role across points and games. Positive point reward gives the correction a smaller update. The inverse controller receives the pitch correction alongside the contact target from the optical flight plan. Setup and the opening point transcript matched the plain clipped inverse before this feedback. [Policy and replay](../../scripts/experimental/tt19_online.py)

The first strict continuation at seed 886230749 scored 3–11 with two legal returns and two legal own serves in six attempts. The paired plain continuation scored 2–11 with zero returns and two legal own serves in six attempts. Their first games each had one legal return, with one legal own serve from the plain controller and zero from registration. On the independent strict seed 418062733, both controllers recorded one return in the first game and zero in the second. The registered controller landed two legal own serves in each game, and the plain controller landed one in each. Every game ended in a loss. The two extra legal returns at seed 886230749 came on distinct points. [Registered first stream](online-886230749-warm-wrist+fixed-setup24-strict-register.json.gz) · [Plain continuation](online-886230749-warm-wrist+fixed-strict-continued1.json.gz) · [Registered second stream](online-418062733-warm-wrist+fixed-setup24-strict-register.json.gz) · [Plain second stream](online-418062733-warm-wrist+fixed-setup24-strict.json.gz)

Physical replay reproduced all 213 scored points in the online comparisons. RGB replay reproduced the 12-point strict warm baseline and the 26-point registered continuation, including every command, motor update and final pitch correction. [RGB replay receipt](online-rgb-audit.json.gz)

## Reproduce the strict comparison

Use the table-tennis environment with MuJoCo 3.17 and headless EGL. The renderer uses a decimated visual mesh of the stock arm while the scorer uses its original plant dynamics.

```sh
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
for mode in warm cold nominal-flight; do
  python -m scripts.experimental.tt19_online_trial artifacts/tt19 \
    --seed 886230749 --mode "$mode" --solver clipped \
    --setup-frames 24 --strict-endline --games 1
done
python -m scripts.experimental.tt19_online_trial artifacts/tt19 --rgb --verify \
  'artifacts/tt19/online-886230749-warm-wrist+fixed-setup24-strict.json.gz'
python -m scripts.experimental.tt19_online_trial artifacts/tt19 \
  --seed 886230749 --mode warm --solver register \
  --setup-frames 24 --strict-endline --games 2
python -m scripts.experimental.tt19_online_trial artifacts/tt19 \
  --seed 886230749 --mode warm --solver clipped --game-offset 1 --games 1 --strict-endline \
  --resume 'artifacts/tt19/online-886230749-warm-wrist+fixed-setup24-strict-game1-player.json.gz'
```
