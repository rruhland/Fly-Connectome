# Identifying contact-model error from issued commands

A camera-built twin can predict that a return will land even though the real ball comes off the blade too low. This study asks whether the actual issued motion and subsequent stereo trajectory can identify a repeatable correction before the next stroke. The correction is an **effective disturbance** to the private model, not a direct measurement of an unmarked ball's spin. [Implementation](../../scripts/experimental/tt38_contact_id.py)

## Matched optical replay

We replayed archived camera/encoder command tapes and confirmed their native referee event digests separately. The learner did not use those events, scores, release IDs, private live poses or actual spin as labels. Approximately 100–160 ms before the optically estimated −1.56 m approach, it saved a private `OpticalClinic(body=True)` state populated only from the stereo trajectory, encoders and remembered actuator references. It then tested zero and positive/negative disturbances to depth, height, two velocity components and three angular components. Every hypothesis consumed the same actually issued and bounded joint references in the isolated model. No hypothetical reference was sent to the physical arena. [Train observations](contact-id-fit-observations.json) · [Disjoint held observations](contact-id-held-observations.json)

The label was signed stereo velocity across a detected ball–blade deflection. That includes some balls still moving backward and not scoring. A passage with both silhouettes visible and at least 13 cm clearance in lateral/vertical center distance could instead label a plain visual miss. No evidence meant abstention, not a failure label. A matched signed outcome measured velocity and contact position error. It also penalized a predicted hit on a clear miss and a predicted miss when the cameras measured a deflection. The local predictor reads current optical ball motion and encoders plus a measured rebound impulse when a six-frame optical window supports it. It chooses one bounded hypothesis rather than averaging mutually inconsistent contacts. [Sensor metrics and diagnostic limits](contact-id-sensor-results.json)

The ten calibration tapes yielded 28 signed optical effects. After fitting, an independent referee check classified those as 19 legal returns and nine nonlegal touches. None of the fit cases met the stricter condition for a clear no-contact miss. Nine release-disjoint tapes supplied 15 held deflections and one clear miss. Every condition agreed on touch or miss in these 16 samples. We therefore compare departure error on the same 15 contacts. [Fit receipt](contact-id-fit-observations.json) · [Disjoint holdout receipt](contact-id-held-observations.json)

| Private twin under the actually issued commands | Mean velocity error | Vertical bias |
| --- | ---: | ---: |
| Nominal | 1.196 m/s | −0.331 m/s |
| Choose a hypothesis in the local sensory neighborhood | 1.107 m/s | +0.362 m/s |
| Fit one constant hypothesis across seen cases | **0.738 m/s** | +0.197 m/s |
| Choose with the held outgoing observation already known | 0.333 m/s | −0.092 m/s |

The fit-only constant moved the modeled ball 14 mm toward the near arm. It is an effective depth correction, not a measurement of camera misalignment. The local model lowered error on eight held contacts and raised it on seven. The outcome-aware row measures what the sampled hypotheses can express and is not available during play. The first source block simulated the available longer tape prefix. The later blocks stopped at 25 control ticks, sufficient for the departure measurement. No predicted landing labels entered fitting or sensor accuracy. [Exact comparison](contact-id-sensor-results.json)

## Physical held control

Across four releases absent from the fit, the same near planner, nominal arm, cameras, geometry, motor bounds and frozen movement archive were used. Each case had a new continuation seed. The two adjusted branches added a disturbance only to their separate camera-initialized twin. The constant was added as a comparison after the recorded-action check, before reading live comparison outcomes. This is model-assisted control, not the direct joint pupil. [Frozen motor settings and nominal commands](contact-id-live-nominal.json.gz) · [Local correction](contact-id-live-local.json.gz) · [Constant correction](contact-id-live-global.json.gz)

The nominal twin made seven legal arm returns and the local correction made three. Both landed the same three opening replies. All four lost later returns came from one point on release 12380, which fell from five arm returns to one. The local correction's lower recorded-action error did not transfer to this planner's return counts. [Paired games](contact-id-live-results.json)

## Reproduce

Use the pinned TT37 tapes and the optical flight checkpoint. Run sequentially to avoid duplicating the private solver initialization.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
for phase in collect fit plan nominal local global summary audit; do
  python -m scripts.experimental.tt38_contact_id "$phase"
done
python -m pytest -q tests/test_tt38_contact_id.py
```
