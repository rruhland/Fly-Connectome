# Bounded grayscale sensor correction

The streaming binary-image candidate fails all 12 lower-contrast and all 12
illumination-transfer identities because segmentation assumes unit contrast.
Test one explicit opt-in gain-control stage: subtract the current image median,
divide by its 99.9th percentile absolute contrast (maximum for a vanishing
percentile), then clip to the existing signed unit range. This changes sensor
normalization, not entity labels, topology or predictive targets. It is global
image gain control, not a claim of biological synapse locality.

Rerun the identical frozen binary/low-contrast/illumination/outage streaming audit.
Check uniform images and independent sensor noise for false confirmed moving
hypotheses. Preserve baseline outputs. No percentile/noise-threshold sweep.
The result cannot establish natural-camera transfer beyond these interventions.
