# Capped block-RMS conditioning on explicitly easier Pong

Freeze this separate allocation before either 30k endpoint evaluation. It does
not alter the active raw+cap protocol, its source, allocation or results. Generic
readout diagnostics on existing data found useful tracking information but a
large scale disparity: context mean squared norm .1508 versus metadata63.62,
while an isolated diagnostic teacher fit used substantial context contributions.
The raw+cap6000 readout's median context action variation was .00262 versus
metadata mean action bias .574. This motivates one conditioning intervention,
not oracle-derived candidate weights or new policy labels.

Use the existing **fixed-block-rms-v1:energy-matched:uncentered** rule on actual
raw block mean energies from the already-counted, score-free1024-transition Pong
calibration (environment849375169, random action686753009, opponent.25). Bounds
are [.125,8], with actual pre-cap mean total energy preserved. Scales are
current1.6435266412013219, future1.696438875304546,
outside8, relations1.3275753320059096, context8, metadata.5339910770515155.
Preserve missing zeros and raw structured visual transport. Apply the existing
**fixed radial squared-norm cap256 after scaling**, not before. The fitted
calibration has three of1024 pre-cap rows above256; the cap reduces mean total
energy by about .18%. Pin the complete fit/config and source calibration manifest.
No score, simulator label or privileged fit chooses these scales.

Train one **from-zero** policy for exactly **30,000 transitions**, same seed
172900127, opponent speed.25, alpha.001, lambda.995, gamma.95 per physical second,
epsilon.2, score-only terminal ±1, frozen prior/vision, legal actions, exact camera
cadence, explicit resets and2400-tick limits. The only intervention relative to
raw+cap is the declared fixed block scaling. Reuse unchanged shared worker and
policy transform composition. Save6000/15000/30000 checkpoints descriptively;
**only30000 is evaluated**, with no early selection or budget extension.

Fresh evaluation uses the same24 development serves904271833 and20s limits.
Reuse all four already-verified observation-independent controls: initial/down
9wins(8natural+1own-hit), up10(8+2), random9(8+1), no-op10(8+2), all zero truncation.
The development screen remains **at least14total wins AND6own-hit-followed wins**.
Truncations count as nonwins. Report losses, truncations, completion, contacts,
score per rally, durations and physical-time return. This is .25-only development,
not original.45 success or final acquisition proof. Holdout seeds stay untouched.

Between the raw+cap and block-RMS+cap fixed30k endpoints, select only among arms
passing both development thresholds. Select most total wins; ties select raw+cap.
If neither qualifies, preserve both failures and do not allocate replication
automatically. Three-seed replication, visual-input ablation and final evaluation
need a separate frozen allocation. This comparison rule is declared before
either endpoint, including if one arm fails numerically.

The unchanged numerical abort is nonfinite update or any committed record with
max(|q|,|q_next|,q_spread)>1000. Retain last valid journal and classify the arm as
failed; do not replace seeds, lower alpha or evaluate a substitute checkpoint.
Per-arm maximum **30k learning +28,800 evaluation**,35 aggregate wall minutes,
200-transition durable chunks, up to five chunks/interpreter160s work/180s kill.
Two independent workers may overlap for wall-time efficiency with one Torch/BLAS
thread each. Source signatures, unchanged worker hash, calibration/config, full
source snapshots, journals, frozen weights and process costs are mandatory.
