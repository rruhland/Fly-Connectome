# Learned local history transfers across changed interruption timing

The [registered frozen-weight timing test](2026-09-25-gap-timing-transfer-protocol.md) retrained the same balanced 192-episode local learner, saved those experimental weights to `runs/2026-09-25-balanced-interruption-models.pt`, and evaluated three windows on 48 unseen-shape/direction/speed/contrast trajectories apiece. Only the three-frame `(7, 8, 9)` gap appeared in training; the early two-frame and late four-frame gaps were held out. No weight changed during evaluation. The checkpoint was loaded successfully after the run.

| Exact reappearance-to-next-event F1 | Archived | Learned unsplit | Learned sign split | Random sign split | History reset | Fixed raw correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Early 2-frame gap | 0.000 | 0.111 | **0.427** | 0.154 | 0.000 | 0.000 |
| Familiar 3-frame gap | 0.000 | 0.114 | **0.434** | 0.175 | 0.000 | 0.000 |
| Late 4-frame gap | 0.000 | 0.101 | **0.380** | 0.105 | 0.000 | 0.000 |

Both new windows clear the registered 0.30 F1 and 0.10 margin gates. The learned candidate made 285/720 target-event true positives in the early window and 231/720 in the late window; resetting its local history trace at reappearance removed all true positives. Thus the earlier gain is **not restricted to the exact training gap duration or position in the episode**.

The distinction across phases matters. Just before each gap, the learned candidate equaled the archived predictor (0.650–0.697 F1), while a fixed raw-event correlation control scored 0.994 on this simple visible translation. On the transition following reappearance, the learned candidate scored 0.408–0.417, above archived 0.354 but below fixed raw 0.663. During quiet targets inside gaps, learned sign-split produced 520–521 false-alarm pixels versus archived 456 and fixed raw 364; this passes the registered bound of twice archived plus one pixel per scored frame. The near-constant total as gap length grows suggests most quiet errors occur immediately after disappearance, but a frame-resolved audit would be needed to confirm that inference.

This is evidence for a genuinely learned, reusable short-gap visual history signal under local learning without backpropagation or semantic object labels. It is **not** yet evidence of a complete M1A world model: visible-motion forecasts remain weaker than a simple fixed sensory control, and the broader robust-family occlusion score improved only slightly in the prior test. The next high-value opt-in experiment should evaluate a fixed sensory primitive and learned temporal-history forecast together across full generic scenes, using the saved weights, while checking whether the learned component adds value beyond the fixed primitive without a false-alarm cost. Any production integration still requires a concrete revision and user approval.
