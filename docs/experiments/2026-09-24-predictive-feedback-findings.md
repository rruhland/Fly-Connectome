# Ungated predictive feedback sustains state but causes false alarms

The successful 96-episode correlation-input latent learner was retrained from the archived seed and frozen. One opt-in inference variant added 0.5 times its prior thresholded primitive prediction to the next observed primitive input. The no-feedback arm reproduced 0.669 held-out event F1. No extra learning or labels were used.

| Event F1 | No feedback | Predicted-input feedback | Fixed sensory control |
| --- | ---: | ---: | ---: |
| Original unseen single patterns | 0.669 | 0.600 | 0.971 |
| Two independent movers | 0.681 | 0.619 | 0.977 |
| Crossings | 0.583 | 0.486 | 0.817 |
| Speed changes | 0.634 | 0.515 | 0.893 |
| Brief disappearance | 0.430 | 0.357 | 0.716 |
| Sensor-bit noise | 0.487 | 0.418 | 0.798 |

Feedback preserved active latent units on **56/88** blank-primitive frames, versus zero without feedback, but quiet-target false-alarm pixels rose from **97 to 417**. It missed every preregistered benefit and non-regression gate. This is the expected failure of using one ungated variable for both hidden motion continuity and predicted *visible* events: activity useful as internal memory is emitted as a false camera event when the pattern disappears or sensory evidence is sparse.

Do not promote this feedback variant. The next opt-in architecture should separate a locally persistent latent transition state from the local observation/event-emission prediction. Both may be learned with local predictive error and eligibility, with no backpropagation or object labels. Train with generic appearance/disappearance and interruption scenes as well as ordinary motion, then compare to the no-feedback learner and fixed control on the archived clean/robustness/horizon suites. Require a reduction in quiet false alarms and an improvement on missing evidence without sacrificing clean translation. The successful feedforward-selected latent remains the checkpoint baseline. Production M1A was unchanged.
