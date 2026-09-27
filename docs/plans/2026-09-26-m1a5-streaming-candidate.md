# Opt-in streaming candidate, not production promotion

Assemble the components with demonstrated capacity into one causal interface:
`step(events, visible_frame, learn=False)`. Frames are declared sensor input,
not a hidden training teacher. A missing frame is an unavailable observation,
not a blank image. Output persistent anonymous entity hypotheses, whether each
was actually observed this sample, observation age, and separate horizon forecasts.
No Pong metadata, object categories, reward or motor interface.

Retain the generic frame observation state and local patch associations. Feed
only five consecutive actual observed centers into separate direct-horizon local
associations. Store local credit until the corresponding current observation
arrives. No propagated or forecast center becomes a dynamics target. Learning
can be disabled without modifying learned parameters.

Attach empirical endpoint calibration as a separate declared readout: estimate
the probability that the same track is **visually reidentified at the endpoint**,
and an empirical residual radius conditional on reidentification. This is not a
probability of hidden existence. Unavailable camera endpoints are censored.
Confidence strengths from the original tracker are not calibrated probabilities.

Use bounded buffers and remove expired internal tracks without reusing external
identity numbers. Verify streaming/assembled-sequence agreement, prefix causality,
no update on missing evidence, save/reload continuity, and bounded memory under
repeated births. Measure throughput including observation, association, online
credit and forecast. Keep measured-connectome input optional until paired ablation
establishes useful contribution. Production `src/` remains unchanged.
