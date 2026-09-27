# Fresh integrated-candidate transfer gate

Do not treat repeatedly inspected development scores as final acceptance. Load
the combined checkpoint without retraining and test new L/T/line motifs, phases
7/9, angles 30/60/120 degrees and scales 0.75/1.5. Add independent image noise
(sigma .02), grayscale contrast .35, drifting illumination and event corruption
(15% dropped events, .0005 false-event rate; seeds starting 120000). All six
registered dynamics families remain, so family-specific degradation cannot be
hidden in the aggregate. Model receives arrays only; truth is scorer-only.

Compare identical actually observed histories against constant velocity, two-step
replay, persistence and acceleration. Count unavailable histories with the same
64-pixel penalty. Require at least 20% lower aggregate error than the best fixed
control at horizons 4/8, with no family worse by more than one pixel at horizon8.
The one-pixel tolerance reflects rasterized centers, not an omitted failed family.
No tuning against this run. This gate is new-camera synthetic transfer, not a
natural-image or real-robotics claim.
