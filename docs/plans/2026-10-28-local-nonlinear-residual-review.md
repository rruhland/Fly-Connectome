# Draft review: localized nonlinear policy residual

Status: proposal only; not approved or implemented. The current opening-retention
trial is still running. Do not replace that outcome with an architecture claim.

## Problem and hypothesis

The existing linear head can learn complete reward-earned later-return sequences,
but changes often disrupt many other rallies. Preserving all earlier successful
actions did not produce a verified fit within three bounded attempts. Soft score
preservation made fitting fast, but its first whole-bank candidate regressed
409→391 returns. Those results do not prove that the linear policy is incapable.

A small nonlinear residual may express a correction in a limited region of visual
experience without changing decisions across the wider state distribution. The
current inputs already combine generic visual positions, directional velocities,
forecasts and image context; a nonlinear readout can combine these quantities.
It cannot reconstruct information lost by the encoder or distinguish identical
encoded inputs requiring different behavior.

## Concrete bounded proposal

Keep the existing frozen vision, scaler, actuator and learned linear head.
Add64 fixed radial-basis features of the same legal visual input. Choose centers
from existing TRAIN observations without reward or physics labels, and set a
single bandwidth from their distance distribution. Calibrate activity on TRAIN
only. Record nonlinear variation beyond a fitted linear map; do not repeat the
previous nearly linear64-tanh branch without checking its redundancy.

Initialize all new output weights to zero: the complete carried policy is exactly
unchanged. First train only the3×64 residual output weights using the existing
own-reward donor trajectories and no backpropagation. The centers and bandwidth
remain fixed during this comparison. No score/ball-coordinate/opponent-state
inputs enter the policy. Existing raw vision features remain its sole sensor data.

Use the saved shortest new later-return donor and known opening-prefix retention
contract for a single bounded comparison. Verify initialization action parity,
record fit time and inference overhead, and test actual fixed TRAIN sentinel,
full160 if useful, then new games. A fitted witness is insufficient for acceptance.
Compare against the preserved3ac7f87f head; keep every old checkpoint immutable.
No architecture/bandwidth sweep follows automatically. Proposed execution budget:
10minutes including fitting and native checks; stop this branch on clear global
regression or unavailable useful nonlinear variation.

## Approval boundary

This adds a nonlinear policy branch, so it is a major architecture experiment
reserved for user review. The earlier approval concerned a specific64-feature
residual test already completed; it is not treated as blanket approval for this
new construction. Await a concrete parent recommendation after the active trial
and the Sol review before requesting approval.
