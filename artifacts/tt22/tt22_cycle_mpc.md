# TT22 rechecking a planned stroke when an optical fit arrives

The clamped SO-101 still uses the calibrated fixed stereo pair and its own issued commands to reconstruct its body and the ball. The same privileged four-axis physical planner plays the far side. The intervention uses the existing causal inverse fit and its optical bounce inference. A fit qualifies when the modeled ball presents a receiving window with at least four control frames of lead. An eligible event may occur once before a near bounce and once after it in a given incoming leg.

The near controller applies the remaining commands of an existing stroke in an independent strict model built from that optical estimate. It keeps the tape if the model referee credits a legal return. If the tape fails or an earlier search left no tape, the controller makes one fresh proposal search across its full motor grid. The normal planner retains its scheduling outside these events. The new use of the referee exists inside the model. Live referee events and physical positions never enter the MPC policy. [Controller and trial](../../scripts/experimental/tt22_cycle_mpc.py)

| Previously evaluated ordinary seed | Original inverse scheduler | Fit-triggered MPC |
| --- | ---: | ---: |
| 2119291602 | 4 | 4 |
| 265865819 | 1 | 1 |
| 509321423 | 2 | 3 |
| 1213749908 | 3 | 6 |
| Total legal alternating returns | **10** | **14** |

The near arm made six versus eight returns. The far arm made four versus six. Two of four points became longer and both conditions won two points. The controller acted on six eligible fit arrivals. It kept the same commands in all three suffix validations and installed two new tapes from three searches. The first two points had identical action tapes under both controllers. There were no ten-return points.

On seed 509321423, the ordinary search had failed when a pre-bounce fit arrived at frame 45, with 10.25–14.25 control frames to feasible contact in its model. The extra search installed a new tape. A post-bounce fit rechecked it at frame 52 and kept it. The near arm contacted the real ball at frame 58 and landed its second legal return at frame 72, flipping the point from loss to win. On 1213749908, an extra search at frame six restored an opening tape. Later fits at frames 48 and 60 kept a planned second reception. This point grew from three to six returns and ended in a loss instead of a win.

The cutoff for a new four-start prospective cohort was fixed before replay. The old four had to show a positive total change and more lengthened than shortened points. The four prospective seeds were drawn before evaluating the old points.

The four new optical starts produced 2, 0, 0 and 2 legal alternating returns with the original scheduler and 0, 0, 0 and 5 with the new one. The near arm made two versus three legal returns. None reached ten. The new scheduler accepted three old tapes and made no extra full searches on this cohort. There was one longer point, one shorter point and two ties.

A separate evaluator replaced the near arm’s visual estimate with the simulator’s exact ball trajectory on a previously known 32-return start. The ordinary optical controller reached zero and the privileged intervention reached 27. Accurate flight information can make this point playable for the existing body. Those 27 use simulator state and are excluded from all optical learning results.

[Native trajectories, event counters, and exact replay results](cycle-mpc.json.gz)
