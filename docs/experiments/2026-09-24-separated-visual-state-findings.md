# Persistent latent state survived gaps, but shared emission credit underfit

The opt-in two-path circuit used the frozen 24-unit learned dictionary and trained on 96 diverse clean scenes plus 32 generic interruption/noise scenes made only from training shapes. Its local recurrent pathway predicted latent activity; observed and predicted-only latent sources fed separate visible-event synapse banks. Both emission banks, however, received the **same scalar local event error**. A matched no-recurrence emission arm and a primitive-prediction arm saw the identical 128 episodes. Held-out shapes remained unseen.

| Event F1 | Archived learner | Matched primitive learner | Recurrent two-path | Recurrent without imagined emission | No-recurrence emission | Fixed sensory control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 0.621 | 0.552 | 0.452 | 0.396 | 0.563 | 0.909 |
| Unseen single patterns | 0.669 | 0.595 | 0.528 | 0.463 | 0.640 | 0.971 |
| Independent movers | 0.681 | 0.623 | 0.539 | 0.467 | 0.668 | 0.977 |
| Crossings | 0.583 | 0.540 | 0.497 | 0.447 | 0.564 | 0.817 |
| Speed changes | 0.634 | 0.603 | 0.484 | 0.429 | 0.583 | 0.893 |
| Brief disappearance | 0.430 | 0.421 | 0.366 | 0.366 | 0.446 | 0.716 |
| Sensor-bit noise | 0.487 | 0.473 | 0.399 | 0.400 | 0.489 | 0.798 |

The recurrent state was active on **216/488** blank-primitive frames, demonstrating that state could persist without observed events. Quiet-target false-alarm pixels fell from 97 in the archived learner to **81** in the recurrent model; the matched no-recurrence arm had 91. But recurrent event recall and F1 fell in every family, missing all registered gain and non-regression gates. The run took 177 seconds with the frozen ablation included.

Zeroing only the imagined-to-event weights after training did **not** recover the no-recurrence arm: clean held-out F1 fell further from 0.528 to 0.463, while disappearance stayed at 0.366. Thus the imagined emission channel provides some useful positives and is not, by itself, the source of the failure. The observed emission pathway also underfits its training data when learned alongside recurrence. A plausible explanation is **credit interference**: both emission banks receive the same combined prediction error even when one pathway represents observed evidence and the other represents predicted-only state. This is an inference from the ablation, not a direct proof.

Do not promote this recurrent circuit. One bounded opt-in follow-up can train observed and imagined emission synapses against **separate local residuals** and arbitrate overlapping outputs locally, keeping the state transition, training corpus, tests, and controls fixed. If that fails, stop tuning this two-bank design and reconsider how latent continuity should be represented. The no-recurrence emission arm is a useful control but does not beat the archived learner. Production M1A remains unchanged.
