# Approved visual baseline promotion implementation plan

The user approved the hybrid probabilistic candidate on 2026-09-27 and made
online efficiency, continual transfer and low compute the governing priorities.
The original M1 design remains research history, not the acceptance contract.

Goal: ship the verified candidate as `fly_connectome.vision`, with its pinned
checkpoint and a production stream entrypoint. Preserve its numerical behavior;
do not build a new predictor, neural adapter or M1B learner in this promotion.

1. Extract only runtime dependencies into observation, association, memory,
   dynamics and state modules. Remove experiment generators and legacy model
   variants from the production dependency chain. Package the approved checkpoint.
   Verify full output parity against the research candidate, online updates,
   missing samples, checkpoint replay and CPU/sensor shape validation.
2. Add a generic event/frame stream CLI and documented public API. The new vision
   entrypoint uses the approved baseline by default. Preserve old graph training
   commands as explicitly legacy research paths. Verify CLI and installed wheel
   outside the repository (no `scripts/` or ignored checkpoints required).
3. Run relevant/full tests and the existing held-out checkpoint acceptance against
   the production class. Inspect code review findings before committing/pushing.
4. Record promotion and write the remaining M1A.5 and M1B decision plan. Separate
   demonstrated transfer from untested continual transfer/forgetting. Recommend
   the smallest decision-learning loop and evidence gates for recurrent additions.

No production motor behavior changes in this task. Inputs remain every-sample
grayscale frames plus events, on CPU, at a declared fixed camera cadence. Missing
frames are explicit. Checkpoints save learned parameters and start a new scene;
they do not claim exact mid-scene resume. No backprop is introduced.

Completed: runtime extraction, bundled checkpoint/API/CLI, full output and online
credit parity, 216-scene production replay, installed-wheel test outside the repo,
review and float64 sensor regression fix. Final full suite: 700 passed, four skipped.
The continuation plan is `2026-09-27-m1a5-to-m1b-roadmap.md`.
