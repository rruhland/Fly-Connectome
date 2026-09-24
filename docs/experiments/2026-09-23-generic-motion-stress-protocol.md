# Generic local-motion stress screen

Freeze the clean-translation learner and fixed competitive controls. Test next-event prediction on trajectories outside training: a direction change, a speed change, two same-polarity movers that cross, a mover passing an occupied visual region, and independent false/missing event-camera events. The generative paths are labels for evaluation only; predictors receive ON/OFF event maps and no path, object, or game metadata. Compare predictions with the next **clean visual event map** so random sensor events are not counted as desired forecasts.

This is a failure-finding screen, not a tuning search. Report event F1 per condition and compare the learned competitive version with fixed competitive and uncompetitive versions. Do not infer object identity from event F1: a separate grouping/continuity test is required. The script does not change production M1A.
