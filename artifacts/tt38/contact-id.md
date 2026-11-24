# Identifying contact-model error from issued commands

A camera-built twin can predict that a return will land even though the real ball comes off the blade too low. This study asks whether the actual issued motion and subsequent stereo trajectory can identify a repeatable correction before the next stroke. The correction is an **effective disturbance** to the private model, not a direct measurement of an unmarked ball's spin. [Implementation](../../scripts/experimental/tt38_contact_id.py)

## Matched optical replay

We replayed archived camera/encoder command tapes and confirmed their native referee event digests separately. The learner did not use those events, scores, release IDs, private live poses or actual spin as labels. Approximately 100–160 ms before the optically estimated −1.56 m approach, it saved a private `OpticalClinic(body=True)` state populated only from the stereo trajectory, encoders and remembered actuator references. It then tested zero and positive/negative disturbances to depth, height, two velocity components and three angular components. Every hypothesis consumed the same actually issued and bounded joint references in the isolated model. No hypothetical reference was sent to the physical arena. [Train observations](contact-id-fit-observations.json) · [Disjoint held observations](contact-id-held-observations.json)

The label was signed stereo velocity across a detected ball–blade deflection. That includes some balls still moving backward and not scoring. A passage with both silhouettes visible and at least 13 cm clearance in lateral/vertical center distance could instead label a plain visual miss. No evidence meant abstention, not a failure label. A matched signed outcome measured velocity and contact position error. It also penalized a predicted hit on a clear miss and a predicted miss when the cameras measured a deflection. The local predictor reads current optical ball motion and encoders plus a measured rebound impulse when a six-frame optical window supports it. It chooses one bounded hypothesis rather than averaging mutually inconsistent contacts. [Sensor metrics and diagnostic limits](contact-id-sensor-results.json)

## Physical held control

On fresh launches the same near planner, nominal arm, cameras, geometry, motor bounds and frozen movement archive were used in both conditions. A paired branch added the learned disturbance only to its separate camera-initialized twin. The frame-matching fit has no access to these games' outcomes before or during play. This is a model-assisted controller, not the direct joint pupil. [Frozen motor settings and nominal commands](contact-id-live-nominal.json.gz) · [Adjusted commands](contact-id-live-local.json.gz)

## Reproduce

Use the pinned TT37 tapes and the optical flight checkpoint. Run sequentially to avoid duplicating the private solver initialization.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
for phase in collect fit plan nominal local summary; do
  python -m scripts.experimental.tt38_contact_id "$phase"
done
python -m pytest -q tests/test_tt38_contact_id.py
```
