# One broader-experience test before rejecting the forecasting family

The combined candidate retains both contextual relations, but the new 216-scene
camera/dynamics transfer gate fails (eight-frame 13.10 pixels versus two-step
replay 11.22). Keep this failure. No tuning to that set.

Train the same contextual direct learner online on 192 generic observed streams:
32 per registered family, random heading over the circle, scale 0.6–1.6, phase,
polarity, and dot/square appearance. Use the declared noisy grayscale/frame+event
sensor. Keep the 128 competitive prototypes and all learning constants fixed.
The accepted contextual memory is retained separately, not relearned from this
single-entity motion dataset. No direction/speed/family labels enter learning.

Then freeze it and evaluate newly reserved phases 11/13, headings 22.5/67.5/112.5
degrees, scales .85/1.35, and the L/T/line motifs. Keep the same error gates and
report observation coverage. This is one experience-vs-capacity comparison, not
a training-budget or learning-rate sweep. If it fails, stop prototype variants
and reconsider the downstream dynamics architecture.
