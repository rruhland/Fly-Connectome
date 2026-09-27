# Noise-aware cross-sensor observation, one bounded candidate

Pure contrast normalization recovers low-contrast motion but creates hundreds of
false tracks from independent noise on an otherwise blank camera. Stop its long
run and preserve a 12-frame reproducer. A robust Gaussian noise floor (MAD/0.6745,
three-sigma segmentation) removes most proposals but still permits accidental
moving associations; it is not by itself an accepted state.

Test cross-sensor temporal consistency rather than tune that threshold: an image
region may remain stationary without an event, but a moving match requires an
event within its local 5x5 neighborhood. The fixed event-correlation primitive
checks observation agreement, not a final learned representation or object label.
No-event image motion therefore cannot count as an observation in this candidate;
this is an explicit joint sensor contract, not frame-only inference.

Require the previous binary/grayscale/outage identity gates, the blank-camera
false-track test, and noisy event dropout transfer. Preserve failed gain-only
results. No neighborhood/noise-threshold sweep. Local associations and forecasts
remain unchanged.
