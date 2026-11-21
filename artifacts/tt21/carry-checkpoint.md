# Carrying SO-101 drive timing alongside gantry and Pong

The bundle adds one address to the existing TT19 continual checkpoint. Every original checkpoint field, including the gantry motor maps, perception state, role model, numeric Pong profiles and TT19 factory SO-101 policy, is preserved. The added address stores the TT21 strict compact station, single-eye control configuration, factory serve commands, flight and impulse laws, and one integer drive lead and seven accumulated shift losses per eye. It pins the public dry stroke catalogue by path and SHA-256. The carry uses explicit task and body addresses. [Portable checkpoint](carry-checkpoint-player.json.gz) · [Loader](../../scripts/experimental/tt21_carry_checkpoint.py) · [Pre-TT21 checkpoint](../tt19/continual-so101-player.json.gz)

The saved state comes from the first delayed point in the ordinary left and wrist source games. Each learned a four-control-frame lead at the boundary after point five. The outside camera used 44 ordinary frames on the changed point, and the wrist used 34. Their shift-loss vectors were fitted independently from their own command tape and RGB motion. The package uses the original two-eye-tested `PassiveDelay` observer and reads its existing fits. The later `OnsetDelay` estimate and the stereo rally recovery policy remain independent experiments. [Source games](game-delay.json.gz) · [Method](game-delay.md) · [Stereo recovery](recovery.md)

## Resume an ordinary point

From the repository root, with the MuJoCo environment active, the following loads the bundle, registers the specified unmarked camera from new native RGB, and starts the point with its saved lead. The simulated four-frame queue is supplied by the fixture. The policy receives RGB and operates its saved drive observer independently of the queue setting.

```sh
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt21_carry_checkpoint point --eye wrist --seed 1428607538 --role -1 --fixture-delay 4
```

The `resume_drive` context manager accepts successive `point(seed, role, actuation)` calls. It retains the updated drive and camera state in the session. Each call starts a new point and the fixture supplies a fresh command queue. The saved address starts at a point boundary and initializes optics from the selected camera. To move the bundle, keep the public catalogue at its recorded path under the chosen asset root, or supply that root with `--assets`. Training receipts and the pre-TT21 player are provenance references with hashes. The drive loader uses the bundled state and the matching stroke catalogue.

## Reproduce the paired regressions

```sh
python -m scripts.experimental.tt21_carry_checkpoint build
python -m scripts.experimental.tt21_carry_checkpoint gantry
python -m scripts.experimental.tt21_carry_checkpoint pong
python -m scripts.experimental.tt21_carry_checkpoint arm
```

The gantry return reproduced the complete pre-TT21 7–11 game, including 18 legal returns and eight legal own serves in eight attempts. All 18 native RGB digests, action tapes and referee outcomes agreed with the archive. Every point also passed a physical replay. [Gantry comparison](carry-checkpoint-gantry.json.gz)

Native Pong reproduced all 180 routed action and outcome tapes across 36 matches, and all 180 with the shared-only control. Routed performance stayed at 129/160 first returns, 195/206 later returns and 41 earned points. The full and shared Pong state and its rank-one profile had matching hashes. [Native Pong comparison](carry-checkpoint-pong.json.gz)

SO-101 restoration reproduced one qualified legal return per eye from a fresh optical registration. The outside point won and the wrist point lost. RGB, optical observations, requested and applied actions, plans, lead, referee events and scores matched their archived tapes. The observer processed each new point and retained the four-frame estimate. The archived eye-carry comparison used a different full source file hash. The selected source exposure records matched exactly before the native point replay. [SO-101 comparison](carry-checkpoint-arm.json.gz) · [Reference points](game-eye-carry.json.gz)

The evaluation measures retention through explicit task addressing. The SO-101 observational timing estimate is used by the SO-101 controller. A beneficial backward-transfer claim would require a separate test showing that using that new experience improves one of the older policies.
