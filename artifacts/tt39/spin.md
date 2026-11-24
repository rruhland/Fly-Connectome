# What stereo can infer about an unmarked ball’s spin

A long calibrated RGB arc reveals some rotation perpendicular to the ball’s travel. The component along its velocity is much less identifiable. A camera-only fit can recover parts of the angular state that an impulse-only readout misses, but letting that fit also replace the translational state degraded the estimate just after the bounce.

## Causal observer

The optional observer takes timestamps and triangulated RGB ball centers. It integrates gravity, drag, decay and the linearized Magnus term over earlier exposures, with a shared position and estimated pre/post velocities when the images have already shown a tabletop reversal. The table height and restitution priors come from the 134 observed practice bounces. Across the bounce it fits incoming angular velocity and adds `R × Δ(mv) / I` to the estimate for the outgoing angular velocity. It does not treat that impulse *change* as the total spin. The optional coupled fit also adds a soft effective surface-slip constraint. It does not assume sticking.

The fitting rule is regularized least squares. Practice calibration and the choice of angular regularization and effective slip use delayed **RGB position** only. The practice and held image traces have disjoint opening and opponent seeds, with 10 practice release groups and 24 held release groups across the source bank. A future stereo position 67 ms later is a label, never an observer input. The independent simulator audit reads angular velocity and p/v only after parameters and the cases are frozen. All variants use the same purely optical bounce times and samples.

## Held prediction

| Decision phase | Paired cases | Delayed position error, matched zero-spin | Fitted spin and p/v | Earlier TT38 p/v and impulse | Same TT38 p/v and fitted spin |
| --- | ---: | ---: | ---: | ---: | ---: |
| Descent | 34 | 4.00 mm | 3.73 mm | 4.89 mm | 4.87 mm |
| First visible rebound | 30 | 13.84 mm | 15.41 mm | 5.21 mm | 5.22 mm |
| Settled after observed bounce | 42 | 7.45 mm | 8.36 mm | 3.37 mm | 3.33 mm |
| Independent visible rise, no clean bounce spanning fit | 42 | 3.92 mm | 3.88 mm | 3.76 mm | 3.76 mm |

The fitted-spin columns show the arc variant without the soft slip constraint. The tiny drop in descent error versus the matched zero-spin fit was uncertain (release-bootstrap 95% interval about −0.70 to +0.15 mm). Post-bounce TT38 translational estimates were substantially better than the joint fit. For example, its early-rebound velocity error was 0.062 m/s versus 0.204 m/s for the new arc fit. Keeping that better TT38 p/v identical and substituting latent spin changed 67 ms position errors by under 0.07 mm at every phase. The small changes do not justify altering the translational observer.

In the independent angular audit, the long descending fit recovered top-spin variation along x with correlation 0.63 and reduced transverse angular error from 33.7 to 28.6 rad/s. The early visible bounce fit recovered spin around z (correlation 0.96, mean absolute error 4.1 versus 12.1 rad/s at zero). But a direction close to the axis of flight retained nearly its entire prior uncertainty. No comparison established full vector identification.

## Does the angular estimate matter at contact?

A separate holdout replay used nine issued physical action tapes from the contact-calibration study, none overlapping the spin-fit releases. Causal snapshot timing came from the existing optical approach estimator. At each snapshot the private simulator kept the same RGB-derived p/v, encoder-initialized body and exact issued joint references for every arm. Only the ball's angular velocity changed. Subsequent stereo images measured the departure and independently recorded physical contacts checked classification. An explicitly evaluator-only ablation substituted actual angular velocity while still leaving p/v and body optical. It is not an upper bound on a jointly noisy model and supplies no training labels.

| Angular input to identical private state | Contact classification | Mean departure error on 15 common RGB-scored contacts |
| --- | ---: | ---: |
| Zero spin | 16 / 17 | 1.91 m/s |
| Old impulse spin | 16 / 17 | 1.95 m/s |
| Image arc with impulse update | 16 / 17 | 1.62 m/s |
| Image arc, impulse and soft slip | 16 / 17 | 1.62 m/s |
| Actual spin, evaluator only | 16 / 17 | 1.86 m/s |

All variants produced the **same** contact classification. The coupled estimate reduced mean departure error by 0.33 m/s versus the old impulse, but the median paired change was a 0.015 m/s worsening, with five improvements and eight worsenings greater than 0.01 m/s. The mean gain comes from reducing large errors. Leaving one release out gave mean changes from −0.72 to −0.01 m/s. This is enough to justify a small live check, not to infer a general gain from spin alone.

## Paired live check

The same four random-opponent starts and continuation seeds were played using the unchanged optical mentor's archived nominal comparator. The experimental arm retained the old 30 Hz p/v algorithm, identical issued-reference search settings and initial memory. Additional frames were read only by the angular branch, which fell back to the old impulse estimate unless a fitting tabletop reversal was already visible. This was an optical mentor with a private contact model, **not** the planner-free direct-joint pupil. The first physical far collision records matched exactly across arms.

The coupled spin arm scored 0, 2, 1 and 1 near-side legal returns, versus 1, 0, 1 and 5 under the old angular readout (4 vs. 7 total). One point improved, one tied and two worsened. The absolute readiness refit difference was below 3e−16 rad. All eight legal-score histories independently rescored and their saved action tapes reproduced the full physical event logs exactly.

**Result:** some rotation is observable and predicted departure can improve under fixed issued actions. It did not improve contact classification or live play in this test. Keep the existing translational and motor defaults. The arc estimate remains a diagnostic, not a deployable change.
