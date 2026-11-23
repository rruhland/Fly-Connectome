# Channel-neutral feedback on the physical donors

The universal readout starts from zero on the same three physical donor arms. Each donor observation contains 11 metric channels for ball position and motion and for blade pose. The donor observation extractor supplies those quantities directly from simulation, with position measured relative to blade home and the last two channels giving the upward blade normal's horizontal components. The feedback features treat the channel values alike.

The same 30 signed, five-tick motor probes provide unlabelled observations and the measured response to each motor direction. Six stationary candidates from the first training round also supply temporal variation for preprocessing. These six candidates count within that round's practice budget. The fit scales every channel by the same robust rule and computes principal components of observed frame differences. The learned feedback then uses a bias, all 11 transformed channels, and all 66 pairwise products. The action readout fits all five joints from donor scores. Search directions use temporal variance and measured motor response without selecting named sensory axes. [Feature implementation](../../scripts/experimental/tt24_universal_native.cpp) · [Training and evaluation](../../scripts/experimental/tt24_universal_trial.py)

The original and universal searches use the same physical launches, candidate counts, per-drop limits, objective, validation roster, actuators, and rebound referee. The original has 100 learned coefficients and named feature groups for exploration. The universal version has 390 learned coefficients and data-selected exploration groups. The comparison tests both changes under the matched experience budget. Stock-arm runs use the stored fused observer with wrist images and joint encoders. Its earlier public RGB and encoder calibration remains fixed during motor learning and held tests.

The largest absolute loading of the first empirical temporal component was the vertical ball-velocity channel (0.94). A channel reorder accompanied by the same numerical transform reorder reproduced 64 sampled commands within 7×10⁻¹⁶. The six reused neutral scores matched the original archive exactly, and the training, validation, and donor-held seed sets were disjoint.

## Matched donor-held tests

The paired searches each used 18,000 practice drops and 8,664 donor-validation drops, plus the 150 motor-probe ticks. The universal search used 1,873,500 practice ticks and the original used 2,651,727. Early endings account for the different realized tick counts. The universal readout selected at round 65 remained the best through round 120. These checkpoints were fixed before the held runs.

| Donor | Stationary mean consecutive | Universal mean consecutive | Original mean consecutive | Universal and original ≥10 |
| --- | ---: | ---: | ---: | ---: |
| First | 2.50 | 10.75 | 15.63 | 5/8 and 8/8 |
| Second | 2.00 | 3.75 | 11.25 | 1/8 and 6/8 |
| Third | 2.25 | 5.63 | 10.13 | 1/8 and 3/8 |
| All 24 drops | 2.25 | 6.71 | 12.33 | 7/24 and 17/24 |

The universal readout produced at least one ten-rebound run on each donor and reached 19 on the first donor. All five joints were active during held execution. Each mode's action tapes replayed exactly through the original physics and referee. [Frozen universal weights and transform](universal-donor.npz) · [Matched outcomes and tapes](universal-donor.json.gz)

## Frozen stock-arm transfer

The donor result met the original gate for stock evaluation. On the same 16 stock drops, the universal controller beat stationary control with clean metric observation and fell below it with the fused wrist observer. The original readout tolerated that observer on this cohort.

| Stock observation | Mean consecutive | Maximum | At least 10 |
| --- | ---: | ---: | ---: |
| Stationary | 2.56 | 5 | 0/16 |
| Universal, clean metric state | 6.63 | 15 | 6/16 |
| Universal, wrist RGB and encoders | 2.06 | 5 | 0/16 |
| Original, clean metric state | 12.19 | 19 | 12/16 |
| Original, wrist RGB and encoders | 11.94 | 17 | 12/16 |

All 152 donor and stock action tapes replayed exactly. The universal controller received no stock motor practice or teacher actions. Run `python -m scripts.experimental.tt24_universal_trial` to reproduce training and held evaluation. [Results](universal-donor.json.gz)
