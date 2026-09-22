# Preserve event-balanced amplitude; audit temporal selectivity

Diagnostic only; hold the saved split event-balanced coefficients and histories
fixed. Do not refit amplitudes or install weights/gates in the neural model.

1. Partition held-out forecasts by the existing target-frame phase: ON, OFF,
   inter-event quiet offsets and blank. Retain signed amplitude, alarm counts,
   squared error and clipping. Plot ON/OFF profiles against target-frame age
   since the preceding event (forecasts were issued one frame earlier).
2. Exhaustively fit one closed window on abs(existing postsynaptic sensory
   current), retaining original forecast amplitude inside and zero outside.
   No clock, target label, tempo or phase enters the window input. Optimize the
   minimum retained signed event amplitude over ON/OFF and fitting tempos,
   under <=5% quiet alarms separately at every fitting tempo. Tie-break by mean
   retention then fewer alarms. 80% retention is a diagnostic preservation
   reference, not a new M1A acceptance criterion.
3. Fit a single shared window on the unchanged trials1-14 of2/4/6; evaluate on
   trials20-29 and unseen3 (trials1-29). Separately fit tempo-specific windows
   on each known tempo's training trials as diagnostic controls; these use
   external tempo to choose a window and are not a candidate neural mechanism.
   Never fit a window to tempo3 or select one based on its outcome.
4. Keep source checksums/reference coefficients, report limitations and next
   interpretation. Do not infer that a failed one-dimensional window rules out
   every existing local-state timing mechanism.

Window boundary convention: choose midpoint margins between included/excluded
training values. This preserves the same training-window membership. Added a
regression check for margins before the final report. Follow-up sensitivity:
1%relative Gaussian perturbation of window input only, seed9092, no refitting.
This is diagnostic selector sensitivity, not noisy neural evaluation.
