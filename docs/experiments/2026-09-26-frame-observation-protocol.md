# Align observation inference with the declared frame sensor

Shared-region handling passes the changed-camera identity/noise/outage audit.
Before interpreting the downstream forecast experiment, fix its supervision:
only actually matched current observations may provide displacement targets;
propagated tracker positions are not observed teacher values. Preserve the
provisional result separately and rerun the baseline with corrected eligibility.

Test a separate frame-aware state with three structural changes:

- Every compact region in an available current image is an observation, even
  when it generates no event. Sensor outages provide no new observations.
- A newly born history has unknown velocity until two positions have been
  observed. Its first match uses the existing spatial admissibility region as
  a broad location prior, not a zero-velocity prediction with false certainty.
- Confirmed stationary entities remain in the state. Shared regions preserve
  separate moving or stationary histories rather than overwriting identities.

All numeric association settings, learned patch associations, and gates stay
fixed. This candidate is explicitly frames+events, not an event-only repair.
Test stopping/restarting and first-motion acquisition in unit fixtures. Rerun
the existing camera audit, contextual regressions and corrected forecast test.
No parameter sweeps. Any full M1A.5 claim still requires calibrated uncertainty,
the connectome ablation/integration review and streaming performance.
