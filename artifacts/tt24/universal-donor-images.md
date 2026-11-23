# Generic feedback from rendered donor experience

The public residual fit improved the generic controller only modestly on stock wrist images. Its imaging bench sampled unobstructed motion, so a second fit used actual donor wrist images during physical contacts. The original generic policy drove 18 donor drops, supplying 3,284 paired clean and optical observations. The optical observer used the known simulated donor camera and blade pose to isolate errors in the visible ball estimate and its causal motion history. The stock observer stayed fixed. [Implementation](../../scripts/experimental/tt24_universal_donor_images.py)

An anchored, closed-form fit adjusted the same 390 generic coefficients. Targets came from the original reward-learned controller's own clean donor readout. This fit used no other policy's actions. The fit and checkpoint were fixed before running stock. On nine separate donor drops driven by their rendered wrist observations, the successive means were 1.89 for the initial generic policy, 4.33 for the public-residual fit, and 7.00 for the donor-image fit. The last condition reached streaks of 15 and 16. [Donor and paired stock trials](universal-donor-images.json.gz)

## Paired stock test

The new checkpoint ran on the same 16 held stock drops as the public-residual test. Scores count consecutive referee-verified rebounds.

| Frozen policy and observation | Mean | Maximum | At least 10 |
| --- | ---: | ---: | ---: |
| Stationary | 2.56 | 4 | 0/16 |
| Initial generic, clean state | 5.69 | 15 | 5/16 |
| Public-residual fit, clean state | 6.44 | 17 | 5/16 |
| Donor-image fit, clean state | 3.69 | 12 | 1/16 |
| Initial generic, wrist RGB and encoders | 1.63 | 3 | 0/16 |
| Public-residual fit, wrist RGB and encoders | 2.06 | 8 | 0/16 |
| Donor-image fit, wrist RGB and encoders | 3.63 | 16 | 2/16 |

This fit improved the observed wrist score and reduced the score with clean input. Every new action tape replayed exactly in the original stock fixture. Reloading the exported pupil reproduced the complete wrist rollout on seed 274197531, with 16 consecutive rebounds. [Exported checkpoint](universal-donor-images.npz) · [Paired action tapes](universal-donor-images.json.gz)

## Independent stock confirmation

The exported checkpoints were loaded for 16 new stock starts. The donor-image fit averaged 3.06 from wrist RGB and encoders, with a maximum of 10. The initial generic readout averaged 2.00, the public-residual update averaged 2.38, and the stationary arm averaged 2.44. With clean inputs the donor-image and initial generic means were 5.81 and 4.81.

Across the two wrist cohorts, the donor-image fit averaged 3.34 and produced three streaks of at least ten (12, 16, and 10). The corresponding means were 1.81 for the initial controller, 2.22 for the public-residual update, and 2.50 for stationary control. The paired difference against stationary was 0.84 rebounds. A cohort-stratified bootstrap gave a 95% interval of −0.19 to 2.06 for that difference. All 96 confirmation tapes replayed exactly. [Independent confirmation](universal-donor-images-confirm.json.gz)

Run `python -m scripts.experimental.tt24_universal_donor_images` to reproduce the donor-image fit and paired stock evaluation. Add `--confirm` to run the stored checkpoints on the second stock cohort.
