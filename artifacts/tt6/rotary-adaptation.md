# Six-axis rotary stroke follow-up

The [original mixed-condition run](combined-rotary.json.gz) returned none of seven serves at the first rotary installation. Four contacts all bounced on the learner's half. The original response learner retained its initial swing speed through the visit. The earlier [ordinary-play run](rotary.json.gz) returned five of six serves on a different installation.

A replay of the short-horizon controller exposed a feedback failure. Near contact in serve `957910535`, the camera located the racket around 4 cm above tabletop while the model put it around 26 cm above tabletop. The model rejected the sighting when the discrepancy exceeded its innovation gate. The alternative accepts two spatially coherent sightings, corrects the model residual, and dampens channels with small learned effects on center and pitch. It also leads the stroke using camera-estimated actuator latency. The alternative runs through the existing arm and wiring. It consumes the same stereo RGB, previous requests, and public reward.

The [exact repeated course](rotary-adaptation-audit.json.gz) used the original installation, serve seeds, hidden intervention schedule, and native reflected stereo cache. The short-horizon feedback controller with a modest faster and higher stroke returned **4/7** serves, including three ordinary and one delayed serve. It made seven legal returns, two rallies with multiple player returns, and won three points. It returned none of two mixed serves or the one softened-racket serve. The improved course was used for diagnosis and setting selection.

Two paired held courses evaluated whether learned stroke settings transfer to fresh serves and a separate wiring. Practice selected a setting from public point reward and a forecast based on observed outbound RGB ball samples. Each held comparison reused the same fresh camera body calibration and starts across conditions. The frozen stroke comparison retained the alternative feedback servo and camera timing adaptation.

| Practice / held seed | Carried stroke, serves returned | Frozen stroke, serves returned | Original predictive, serves returned |
| --- | ---: | ---: | ---: |
| `76041983` / `89170063` | 2/8 | 3/8 | 1/8 |
| `83169851` / `46419231` | 1/8 | 1/8 | 3/8 |

The initial [practice and held receipt](rotary-adaptation.json.gz) used its recorded three profiles and chose a setting after unequal exploration. The [confirmation receipt](rotary-adaptation-confirm.json.gz) tested a wider, earlier stroke and balanced each profile over four hidden conditions during practice. Carried settings returned zero mixed or softened serves on either held course. The paired results leave transferable stroke selection and robust rotary play under the combined intervention unresolved.

Keep the predictive controller as the shared default. A [hybrid test](rotary-adaptation-hybrid.json.gz) that added the large-innovation recovery to the predictive solver still returned 0/7 on the original mixed course. The improvement in the exact rerun therefore belongs to the tested combination of feedback, solver, and stroke settings. Stroke selection under hidden changes needs a context signal grounded in observations. Camera timing can supply evidence before contact. Evidence about rubber response arrives after the ball departs, so useful carry should be tested across successive shots under a hidden response that can persist.

```sh
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=.:src:scripts/experimental
python -m scripts.experimental.tt6_rotary_adapt_audit /tmp/rotary-audit.json.gz
python -m scripts.experimental.tt6_rotary_adapt_audit /tmp/rotary-hybrid.json.gz --predictive
python -m scripts.experimental.tt6_rotary_adapt_trial /tmp/rotary-hold.json.gz \
  --practice-seed 83169851 --held-seed 46419231
```
