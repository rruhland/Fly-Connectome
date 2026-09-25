# Online polarity-conditioned local learning works across directions, but underfits

The [registered opt-in experiment](2026-09-25-polarity-local-prediction-protocol.md) indexed separate zero-initialized local event-emission kernels by each learned history unit and the **currently observed raw ON/OFF polarity**. It retained the same one-pass delayed local event-error update, 96 clean plus 32 interruption/noise episodes, 5×5 field, learning rate 0.5, and 0.5 scoring threshold. Matched controls used the old unsplit emitter, a random history dictionary with the same split emitter, and trace reset. No backpropagation, labels, offline readout weights, or production code entered training.

| Exact t=10→11 event F1 | Archived | Learned unsplit | Learned polarity split | Random polarity split | Learned trace reset |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training interruptions | 0.000 | 0.000 | **0.488** | 0.103 | 0.000 |
| 144 held-out interruptions | 0.000 | 0.027 | **0.289** | 0.088 | 0.000 |

The learned split readout yielded nonzero true positives in **all four** held-out directions: up 0.271 F1 (93 TP), down **0.022** (6 TP), left 0.397 (150 TP), and right 0.397 (144 TP). It improved over both registered controls by more than 0.10 overall, preserved ordinary unseen single-pattern F1 at 0.669, and raised quiet-target false-alarm pixels only from 97 to 103. Those are meaningful online gains from local plasticity. The candidate still missed the registered 0.40 overall gate and generalized poorly in the down direction. Its matched opposite-history pair reached only 0.174 F1, so context dependence alone is not enough.

The frozen polarity-conditioned capacity ceiling was 0.959 training / 0.923 held out. The online 0.488 / 0.289 result therefore leaves substantial learnable information unused. The original training corpus included only 16 interrupted episodes; its shape/direction/speed/contrast combinations are sparse. A single predeclared follow-up should add a **balanced generic interruption exposure** from all four training shapes, four directions, two speeds, and both contrasts, train learned/random/unsplit controls with the same additional episodes, and rerun the full 144-case gate. Keep architecture, learning rate, horizon, and threshold fixed. If transfer remains poor, change the local objective or eligibility rather than expanding a narrow data sweep.

This Python opt-in run took 631 seconds while evaluating roughly ten thousand model-frames across candidates. It is not an acceptable production throughput result; a successful architecture would still need exact-behavior software optimization before promotion. Production M1A remains unchanged, pending a concrete validated revision and user approval.
