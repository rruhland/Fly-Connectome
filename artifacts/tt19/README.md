# Table tennis with the SO-101 follower

[Watch the SO-101 play its first game, 11–7](so101-first-game.mp4) · [Full 13–11 deuce game with the practiced server](so101-deuce-game.mp4) · [Wrist camera close-up](so101-wrist.png) · [Continual player checkpoint](continual-so101-player.json.gz)

The stock-scale SO-101 follower plays alternating-service table tennis on a 1.60 × 0.848 m table. The wrist camera moves with the arm and sees the gripper tip and paddle. Two fixed cameras help locate the approaching ball. The first-game film shows the player that used zero target serving or receiving practice, including all 18 points at 30 simulated control frames per second. All 981 original RGB captures, decoded optical inputs and physical outcomes reproduced. The deuce film shows a different player with a practiced server. [Body and model limitations](so101.md) · [First-game verification](so101-first-game-replay.json.gz) · [Deuce-game verification](so101-deuce-game-replay.json.gz) · [Earlier 11–5 film](so101-full-game.mp4)

The first-game player uses public robot geometry, a nominal physics model, a supplied table-tennis scoring objective, and ball and impact priors from the gantry. In a separate simulator it forecast candidate receiving strokes and tested service movements on synthetic tosses. **With zero target practice it won 2/4 held games**, scoring 11–7, 11–7, 10–12 and 3–11, with 36 legal returns and 31/33 legal serves. On the same four schedules nominal ball and racket priors also won two, with 25 legal returns and two points with multiple returns. Gantry priors produced 36 legal returns and five points with multiple returns. A 12-distinct-serve online correction selected a different stroke and won 1/4, with 37 legal returns and 27/36 legal serves. [Four-game comparison](factory-serve-summary.json.gz) · [Factory serves and learning budgets](factory-serve.md) · [Matched nominal control](factory-serve-nominal.json.gz) · [All retained skills](continual-so101-audit.json.gz)

## How to interpret the games

The complementary target-practice player learned serving and receiving strokes by trying joint commands and retaining useful outcomes. It used public point rewards and an engineered measure of shot placement derived from ball motion in RGB. One player retrieves practiced strokes based on the incoming ball. The first-game player tests possible receiving strokes in a separate copy of the public factory model. Both approaches use calibrated color vision, a known grasp, public service order and structured stroke proposals. The wrist-only test had poor ball range coverage. All published wins used the wrist and fixed stereo. [Practice and complete-game comparisons](match-learning.md) · [Factory-model player](model-rollout.md) · [Direct stroke learning](stroke.md)

With no SO-101 stroke bank or target practice, the factory-model receiver returned 17/24 novel rival serves using gantry priors and 15/24 on the same starts using nominal priors. Median time to a legal landing was 36 control frames among successes in both conditions. The carried player also had three points with more than one legal return. [Zero-practice receiving test](direct-receive.md)

Gantry experience transferred a usable ball-flight forecast to the arm and sped up the first legal own serve in a matched short-calibration run. Direct transfer of the tested native Pong movement gains gave the same six first returns in 36 starts as neutral gains. A prior trained on which gantry shots won failed to improve early SO-101 return selection. Learning the opening strokes directly from experience took extensive SO-101 practice. The factory-model player could instead win using offline simulation alone. Keeping earlier table models in the checkpoint preserved play on the gantry and on native Pong. [Online warm and cold comparison](online.md) · [Pong transfer](pong-forward.md) · [Gantry reward transfer](utility-transfer.md) · [Return to prior games](retention.md)

With 80 g added to the camera housing, first returns with gantry priors fell from 9/12 to 7/12 on matched incoming balls. A 24-frame optical probe correctly identified the added load from four known possibilities. Updating the internal receiving model with the identified mass still produced 7/12. Correct identification alone did not improve the next stroke in this test. [Load test](direct-receive-payload.md) · [Optical adaptation and game outcomes](direct-receive-payload-adapt.md)

## Replaying a saved player

From the repository root, use the table-tennis Python environment with MuJoCo and EGL. The first command below the environment setup restores the continual checkpoint and plays its full 11–7 game. The second restores the high-practice player for its 11–9 game. Both verify the motion catalogue digest. [Continual checkpoint loader and gameplay](../../scripts/experimental/tt19_carry_factory.py) · [Film verifier](../../scripts/experimental/tt19_view_full_game.py)

```sh
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl
export LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt19_carry_factory artifacts/tt19 --play 930477119
python -m scripts.experimental.tt19_match_player artifacts/tt19 \
  artifacts/tt19/factory-dry-catalog.npz my-replay \
  --resume artifacts/tt19/match-full-rgb-player.json.gz --native --seeds 36325598
```
