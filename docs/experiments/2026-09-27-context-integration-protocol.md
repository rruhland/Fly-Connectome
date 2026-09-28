# Approved context integration: execution ledger

Historical ledger for the initially blocked run. The subsequent authorized fix
passed and was promoted; see [the follow-up findings](2026-09-27-context-consensus-findings.md).

The user approved integration and promotion, including the documented two-seed
retention exception. No change to the tested mechanism is authorized by this run.

- [x] Production eight-step / append-all four-step fallback and version-2 storage.
- [x] Exact experimental parity, including active camera credit and outages.
- [x] Active full-camera regression: all gates except outage support pass.
- [x] One boundary control: restoring only context memory restores outage support.
- [ ] Default promotion: blocked by the failed outage-support gate.
- [x] Installed-package check and independent review; final suite recorded in findings.
- [ ] Standalone latency/growth benchmark: deferred after unexpected regression;
  do not treat concurrent-test replay timing as an isolated online measurement.

Before running evaluation: train once on 192 scenes, seed 291027, six original
generic motion families with dot/square shapes. Train through actual camera
observations only. Reload the learned checkpoint into a fresh scene. Require
active long banks. No seed selection, parameter sweep or held-out retraining.

Run the unchanged 216-scene transfer split and its original aggregate, family,
85–95% marginal calibration and >=95% observation coverage gates, plus noisy
crossings, outages, identical prefixes, visible contradiction, both contextual
relations across rotations, and 600 noisy blank frames. Historical replay-score
equality is descriptive only because the predictor deliberately changed; exact
production-versus-experiment parity is separately tested. A failed material gate
blocks default promotion and is reported rather than tuned away.

Version-1 checkpoint behavior and bundled artifact remain preserved. Historical
comparison scripts must explicitly use that baseline after factory promotion.

Review added a non-vacuous cached-forecast availability gate while the run was
active. The saved summary includes it from the completed recovery result:
zero missing forecasts and 900 cached evaluations. No simulation was rerun to
obtain a better result. The one additional run replaces only context memory with
the original prior, preserving learned dynamics; it is diagnosis, not a new
production candidate or a waiver of the failed gate.
