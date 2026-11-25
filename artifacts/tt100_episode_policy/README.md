# Whole-point credit for a compact search policy

TT46 context search changed a local joint residual and credited each offered shot. TT41's episodic critic still ranked individual motion fragments. TT45's tactical planner and TT57's shot strategy chose outgoing effects. TT42's checked execution kept or reopened a particular forecast after observing innovations. None of those directly optimized a common search policy over complete games while holding the memory and ranking law fixed. TT100 tests two such weights and gives them no credit until a point ends.

The first weight shifts when stereo-derived inbound y first permits ordinary search. The second scales the random direct-joint proposals used by that search. The existing proposal budget, hardware, joint and slew limits, stereo estimator, incumbent interval, uncertainty tests, ranker, opponent and reference feedback are unchanged. A zero setting is the literal ordinary actor. Each pair runs from one measured opening per arm. Common assigned continuous partner seeds reduce shared noise, but later delivered balls can legitimately differ after different successful outgoing shots.

The installed motor library was paid for on the .12 m translated TT73 clamp. Both policies play back on the original centered fixture with that same frozen bank. Six calibration release IDs are outside the installed motor practice and both TT97 and TT80 release pools. The first six untouched TT97 assignments with distinct release IDs form the held comparison. Every TT80 release appears in installed/source training, so those streams stay untouched. [Preassigned episodes and rules](protocol.json)

Three antithetic blocks each run two native launch assignments with plus and minus whole-point policies. A scaled difference of log-return utilities moves the common weights only between blocks. Opposite signs across releases shrink the step. No per-shot reward shaping, synthetic success, simulator truth, rerun lottery, GPU training, or backpropagation enters the update.

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt100_episode_policy plan --feeds /tmp/tt25-rallies.json
python -m scripts.experimental.tt100_episode_policy practice
python -m scripts.experimental.tt100_episode_policy held
```

The protocol embeds the measured initial balls. Re-running from the sealed plan does not need the original launch export.
