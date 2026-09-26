# Sensory information versus world-state bottleneck

The frozen evidence-state comparison failed noisy localization for both
representations. Surface association passed all clean localization, continuing
reacquisition and vanished-target gates. Do not tune it further in this run.

Hold that surface architecture fixed and change only the available sensor
information: event-only baseline (already measured), visible images every eight
frames, and visible images every frame. The latter is an appearance-information
ceiling, not an event-only candidate. Sparse images are an explicitly different
sensor contract, not a free teacher. Neither arm supplies hidden object masks.

At an image arrival, anchor contrast to current visible image minus its spatial
median; this uses no simulator background label. Between arrivals use the same
learned event-only observer. This is a full observation update rather than the
previous .35 blend. Frames themselves are noiseless in this synthetic audit;
that assumption must not be silently transferred to real cameras. Keep all
event corruption, model settings, splits and gates fixed. Train the same local
context association with the declared sensor. Compare aligned/shuffled/frozen
learning in each arm. Report frame-input cost and runtime.

One run, no sensor-cadence sweep. If accurate visible appearance cannot rescue
the state, reject the state architecture. If only the all-frame ceiling works,
prioritize an event-only reconstruction or explicit sensor-contract decision,
not further associative credit tweaks. If sparse frames pass, retain a hybrid
candidate for broader generalization; do not declare event-only M1A.5 complete.
