# Source-world learning with the measured receiving body

A small ball-free movement fit improved the policy on source worlds with an identified receiving-body response. On 500 fresh fitted-body worlds, the original source policy averaged 15.91 verified rebounds and the body-aligned policy averaged **20.71**. On 500 fresh stress worlds with the same body-model bank, it improved from 8.03 to **14.54**. This comparison measures the simulated body alignment. Physical-arm deployment is a separate evaluation. [Frozen body-aligned policy](world-prior-body.npz) · [Per-world outcomes](world-prior-body-source.json)

| 500 paired holdouts per group | Original prior | Body-aligned prior | Idle |
| --- | ---: | ---: | ---: |
| Fitted-body source worlds | 15.908 | 20.714 | 2.150 |
| Fitted-body stress worlds | 8.028 | 14.538 | 1.268 |
| Original idealized body, retention | 19.718 | 19.452 | 1.962 |

The movement fit used `goal` and `measured` from the public ball-free play recording. It consumed six 144-frame runs. Two complete 144-frame runs were held out. Pose observations came from the public RGB and encoder registration using a supplied vendor-link forward model. The simulator diagnostic columns were excluded. The recorded body commands contained zero ball contacts and zero target game reward observations. [Movement recording](world-arm-public-play.json) · [Fitted model and selected parameters](world-prior-body-model.json)

The fitted model predicts five-axis displacement from current task-space pose, recent measured displacement and the previous requested pose error. Cross-axis coefficients were learned together. Delay and ridge strength were selected by leaving complete calibration runs out and comparing 1-, 6- and 18-frame movement forecasts. The selected one-tick delay and ridge penalty of 0.5 yielded normalized errors of 0.061, 0.346 and 0.368 on the two held runs. The original source-style actuator forecast yielded 0.161, 0.457 and 0.483. The exposed absolute normal commands stay in the gravity-aligned frame. The fitted body uses measured initial tilt and fitted actuator bias directly. [Body adaptation code](../../scripts/experimental/tt24_world_prior_body.py)

An optional native entry point keeps the original collision and flight physics and swaps in the measured response model with empirically bounded rates and pose excursions. Six full fits with different observed start states and six leave-one-calibration-out fits make up the plant bank. A source world seed deterministically assigns a bank entry. The original source entry point reproduced the existing 3,000 paired qualification rollouts after adding the hook. [Native simulation](../../scripts/experimental/tt24_world_prior_native.cpp)

The same 42-coefficient source policy was initialized from `world-prior.npz` and refined by reward-based perturbation search on the fitted cheap worlds. It used 618,240 new source training rollouts and 181,016,832 control frames, plus 8,000 development evaluations with 2,263,763 frames. The three frozen qualification groups comprised 4,500 rollouts with 894,719 frames. Checkpoint selection took about 13.8 seconds on two CPU workers. These are source-world exposures. The physical input to this stage is the eight ball-free motion runs, including the two held movement forecasts. [Training exposure and held seed receipt](world-prior-body-source.json)
