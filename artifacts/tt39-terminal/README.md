# Inverse control of the moving paddle

This study tests whether a body-only inverse controller can create more legal opportunities on the usual low central misses. The direct stock search found the same four whole-game results after the terminal fallback was added, including the natural 24-return, 49-alternating development point. Its entire command and referee tapes matched the archived elastic control bit for bit. The new fallback proposed no nominally legal shot on those points. [Candidate game tape](terminal-development.json.gz) · [Archived exact control](../tt38/joint-mpc-extended.json.gz)

## What the inverse controller actually controls

A separate plant has ball contacts disabled and starts from a body estimate tethered to the joint encoders and previously issued actuator references. A native batch integrates the forward body under the existing reference slew and velocity-dependent force envelope. It measures the actual endpoint of the moving blade, including its center, forward-facing normal and world-frame linear velocity. With 10 or 15 reference parameters across two or three temporal basis functions, repeated finite differences estimate which smooth changes can achieve the requested pose and speed through the face. There is no backpropagation, live simulation state, or fixed recorded stroke at play time. An independent camera-initialized optical court must find a legal landing before any new schedule can be issued. [Body-only integration](../../scripts/experimental/tt39_terminal_body.cpp) · [Terminal inversion and motor gate](../../scripts/experimental/tt39_terminal_servo.py)

The optimizer is distinct from applying a paddle Jacobian directly to reference commands. It differentiates a body *trajectory* that has already responded to spring, force and slew limits. The first full games retained ten modeled endpoint solutions, but median center error remained 64 mm. Median normal-speed mismatch fell from 1.58 to .94 m/s. Nominal pose-and-velocity progress was not enough to make those shots legal. Four bit-matched central failure forks at the original search windows retained seven endpoint solutions and issued none. They reproduced their original referee streams from the action tapes. [Targeted suffix assessment](forks.json.gz)

## Earlier depths and alternate reachable postures

The second check used two-sided finite differences and an extra early windup basis. It limited requested full endpoints to static configurations whose face neighborhood contained the forecast ball. The earlier-horizon check also probed before the stock search gate, screened alternate inverse-kinematics seeds against travel from the current joint configuration, and optimised normal speed as well as tilt. It used a 29-tick rather than 21-tick maximum horizon and required a nominal legal result plus at least one legal optical stress case. Neither method found an issuable stroke on the four targeted failures. Early body configuration matters. In a separate nominal setup check, some spatially close solutions on very low central balls required a joint-branch change exceeding 3 rad, much more travel than the observed flight gave the arm. [Contact-compatible suffixes](reachable-forks.json.gz) · [Earlier suffixes](early-forks.json.gz) · [Earlier target and travel screen](../../scripts/experimental/tt39_terminal_early.py)

The early-fork roster is a diagnostic set of known failed receives. Its prefix and probe times are selected from archived physical points. No future event or simulator measurement is supplied to the new decision. The pair stays bit-matched until a privately legal candidate changes the actuator command. A modified continuation runs normally without a reset.

## What changes when the targets come from actual legal shots

The low contacts in successful practice are not well represented by the analytic target family. The empirical target bank registered 271 physically rewarded contact locations against adjacent measured encoder samples. Five have modeled paddle-center heights below 1.015 m. Some low successes used a more open face and lower normal speed than the analytic center-first proposal. At play it retrieves only a contact-relative position, face and speed target from estimated optical motion. It then recomputes a new joint trajectory from the current body. The bank has no motor tape. [Paired legal endpoint labels](empirical-endpoints.json) · [Endpoint readout](../../scripts/experimental/tt39_terminal_examples.py)

With these targets, the arm-only solver could achieve the intended moving pose. Across five retained proposals, median center error fell from 14.5 to 4.2 mm, median face vector error from .0443 to .0062, and median normal-speed error from .882 to .169 m/s. The optical twin touched the ball in 13 of 15 evaluated settings, but none yielded a legal candidate. The gate therefore left all four physical points unchanged. These results distinguish accurate motor inversion at a specified free-body time from solving the actual moving contact and resulting ball impulse. They do not establish a new late-return skill. [Empirical-target forks](empirical-forks.json.gz)

## Reproduce

Use the pinned measured launches and the original stock opponent.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
out=artifacts/tt39-terminal
python -m scripts.experimental.tt39_terminal_servo terminal development "$out/terminal-development.json.gz" --limit 4 --cap 200
python -m scripts.experimental.tt39_terminal_fork "$out/forks.json.gz" --limit 4
python -m scripts.experimental.tt39_terminal_reachable "$out/reachable-forks.json.gz" --limit 4
python -m scripts.experimental.tt39_terminal_early "$out/early-forks.json.gz" --limit 4
python -m scripts.experimental.tt39_terminal_examples collect "$out/empirical-endpoints.npz"
python -m scripts.experimental.tt39_terminal_examples forks "$out/empirical-forks.json.gz" --limit 4
python -m pytest -q tests/test_tt39_terminal_servo.py
```
