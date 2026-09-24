# Learned state over local correlation primitives transfers and predicts

This opt-in experiment gave 24 competitive, unlabeled visual units fixed *low-level* ON/OFF event coincidences at eight nearby one- and two-pixel offsets. Local Hebbian/homeostatic plasticity formed the unit dictionary; local predictive error trained its next-primitive synapses. The offset bank is a sensory input and fixed control, **not** the final motion representation. No direction, speed, shape, object, or game label entered learning. The same 96 diverse training episodes and 48 new-shape test cases were used as the previous experiment.

| Held-out measure | Learned dictionary + causal credit | Random dictionary + causal credit | Learned dictionary + shuffled future credit |
| --- | ---: | ---: | ---: |
| Post-hoc direction probe | 47/48 | 28/48 | 47/48 |
| Next-event F1, threshold 0.5 | **0.669** | 0.238 | 0.042 |
| One-pixel/frame next-event F1 | **0.707** | 0.196 | 0.041 |
| Two-pixel/frame next-event F1 | **0.643** | 0.261 | 0.043 |
| Next-primitive F1 | **0.450** | 0.040 | 0.010 |

The learned code passes the registered local forecast gate at both speeds. Its event F1 on training scenes was **0.671**, nearly the same as 0.669 held out. Each tested shape/contrast/speed cell exceeded 0.58 F1. An evaluation-only sparse linear decoder on the frozen learned code achieved **0.918** held-out event F1, versus **0.283** for a random-code decoder, and 0.963 on training. This passes the registered capacity gate and shows that a better local predictive readout may close much of the remaining gap. The fixed event-correlation forecaster still scored **0.971** on these simple constant-translation scenes; the learned model has not surpassed that control.

This is a meaningful learned visual-state result, but it is **not yet a working generic M1A system**. The current code is selected at sites with active correlation input and its prediction does not yet feed back into inference. The suite contains one moving connected pattern at a time, with constant speed and no occlusion, crossings, texture, or sensor noise. The next decisive tests should freeze this trained learner and challenge it with multiple independent movers, partial occlusion, speed changes, and noise, keeping the same low-level fixed control. That will expose whether learned state and local prediction provide something reusable beyond the easy translation pattern before proposing production integration.

The local predictor remains below its own frozen-code capacity ceiling. If broader scenes preserve a useful latent but forecasts weaken, investigate a better local credit/readout mechanism and actual recurrent state. No production M1A code changed.
