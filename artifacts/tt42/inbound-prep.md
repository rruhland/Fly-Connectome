# Earned joint preparation from the first inbound images

This readout learns how actual joint references moved over the next six control ticks in practice. It responds to the current optical ball position and velocity plus measured joints while the ball is still on the far half. It is not the earlier outbound motor habit, a stored stroke tape, or an earlier call to the late contact planner.

Three practice banks supply successful and unsuccessful issued movements. A local comparison gives at most one vote to an approach. The policy chooses a single preparation mode with enough nearby successful support to beat the ordinary direct field's projected move. The small within-mode shrinkage does not average opposite joint movements. It nudges bounded actuator rates before saturation and shuts off at y = +.06 m, before the original late direct controller and private validator. The latter run unchanged on the resulting physically attained body state.

Actual legal near returns add positive labels to the early local bank. A terminal miss on a legally delivered ball adds a negative label if the necessary inbound images and actual six-tick command history are available. Censored offers and court caps do not add a negative label. The current decision never uses a future image or command. The six-tick label is read only after the actual command was issued and the physical reward was observed. Repeated frame labels are not independent trials.

## Off-policy transfer

The held motion set is taken from a separate physical practice archive. Releases found in the fit data were excluded completely. Errors are Euclidean five-joint reference displacement errors in radians, averaged per release before averaging across releases. “Stock” is a six-tick slew-limited projection of the baseline one-step reference, not a real replay of how the stock policy would replan.

| Release-disjoint frames | Count | Stock | Nearest rewarded movement | Gated, mode-preserving readout | Stationary |
| --- | ---: | ---: | ---: | ---: | ---: |
| All, 22 releases | 938 | .391 | .186 | .348 | .398 |
| Actually awarded, 17 releases | 779 | .394 | .175 | .350 | .410 |
| Awarded and selected for a change, 14 releases | 180 | .485 | .207 | .263 | .515 |

A nearest successful example reconstructs the saved motion better than the guarded policy, but that is an imitation result. It cannot tell whether substituting that motion would actually return more balls. The gated policy uses physical successes and misses and avoids replacing the incumbent outside supported modes. [Off-policy evidence](inbound-prep-off-policy.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt42_inbound_prep audit artifacts/tt42/inbound-prep-off-policy.json
python -m scripts.experimental.tt42_inbound_prep live artifacts/tt42/inbound-prep-live.json.gz --online
python -m pytest -q tests/test_tt42_inbound_prep.py
```
