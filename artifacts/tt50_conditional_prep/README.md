# Conditional direct prep from actual precontact opponent RGB

This comparison uses forecasts from the frozen far-paddle RGB predictor at ordinary camera exposure times. It does not look at the opponent's requested placement, true far-paddle state, or the actual future incoming ball to decide whether to change a motor command.

The receiving challenge starts at the original native outgoing state after the 42nd arm stroke and just before a new TT37 placement request. One diagnostic retains the original hard source and five other streams are kept outside these motor fits. Every condition uses the same measured first physical launch and the same original histories up to that point, the physical motors and two cameras, the existing 768 earned fragments, and the strong direct player through the same receiving score. For each stream, the recorded referee checks if actual far impact and near bounce match before any comparison of scores. This is a comparison of finishing motions from common physical starts, not an assessment of full games from scratch. [Predefined streams](plan.json)

- **Stock** receives the same camera update schedule with no precontact motor change.
- **Continuous map** borrows a bounded vector of early joint-reference shifts from a local affine readout. The readout fits one observation per distinct native awarded preparation. The context contains a forecast at y = −1.10 m and the arm's current encoders and issued references. The old field resumes at incoming optical detection. [Direct reference fit](motor.json)
- **Conditional body only** can use a stronger route for a supported leftward low estimate. It requires a qualifying outgoing fit at gate −0.45 or 0 m, a causal installed-rate margin and an independent kinematic CAD route screen. A later ordinary prediction can veto it. It hands off to the existing actor on the first observed inward direction.
- **Conditional body and learned stroke** adds two physically rewarded and answered shallow movement templates, one from the exact hard source and one from a different previous training reply. The templates were independently replayed while rendering real stereo and logging own encoders and controls. A new incoming optical fit must support a matching local template, and its direct references must pass the existing independent camera-initialized physical model before committing. [Optical and issued-ref teacher strips](optical-skill.json.gz)

The optical gate deadline is physical, not an evaluation-time hidden label. The original hard source was successfully answered when delayed preparation began at outward gate 0, but came too late for that taught family at around gate 0.45. Thus only the earlier gates are permitted to open the strong body phase. Actual inbound motion can cancel a route. Actuator limits and the physical referee remain unchanged. Without a supported forecast the actor uses the ordinary controls. [Physical timing bracket](../tt49_terminal_blind/handoff.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt50_conditional_training map
python -m scripts.experimental.tt50_conditional_training extract
python -m scripts.experimental.tt50_conditional_prep plan
python -m scripts.experimental.tt50_conditional_prep run
python -m pytest -q tests/test_tt50_conditional_prep.py
```
