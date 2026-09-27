# Probabilistic M1A.5 review bundle

`probabilistic-candidate-v1.zip` contains the tested checkpoint, exact runtime
dependency closure, evidence JSONs, SHA-256 manifest and production-review report.
Implementation source revision: `4eb2451`.

Install this repository and its dependencies in Python 3.11+, extract the archive
into a separate directory, then run `python smoke.py` there. This verifies the
bundled runtime imports and checkpoint against the installed Fly-Connectome core.
Use `ProbabilisticVisualState.load('candidate.pt')` from that directory.

This is a **review artifact**, not automatic production activation. Required
input is every-sample grayscale frames plus events. The passing contract is
probabilistic spatial belief with calibrated marginal intervals. Older event-only
and single-point forecast gates remain failed. Read the bundled README before
cutover. No motor/reward learning is enabled.

The source and reproduction commands are under `scripts/`; the complete decision
record is `docs/plans/2026-09-26-m1a5-production-review.md`.

Archive SHA-256:
`aebcf0b81ed7f856fc71823eff01bfb4abd0141ffb849a50e35406516dee7337`
