# Eight-step conditional retrieval with four-step fallback

Approved experimental continuation; production remains unchanged. Use the previous
five seeds, 2,000-sample A/B/A segments, fixed probes and 3.5 NLL criterion. Retention
screen remains <=.2 nat degradation with no loss of forecast availability. Do not
sweep context lengths, bank sizes or mixing weights.

Two new arms share an append-all four-step learner initialized from the bundled
prior, preserving the useful previous no-deletion result. Each adds an initially
empty specialist. One specialist uses the last eight observed displacements; the
matched control uses only the last four, but learns at exactly the same origin
times as the eight-step arm. Both need nine consecutive actual observations and
32 recorded specialist endpoints before using specialist predictions. Otherwise
they use the full four-step learner. No eight-step keys are fabricated from the
old checkpoint, and no hidden positions, domain identities or future data enter
forecast inputs. Same normalization function, nearest-32 mixture, outcome credit
and no-backprop rule; only context length differs between specialists.

The full four-step learner continues learning all eligible observed endpoints.
Specialist credit uses the history captured at issue time, never reconstructed at
the deadline. Overall calibration uses the distribution actually issued. Shadow
four-step NLL on the same probe origins measures the exact append-all control;
its calibration is not claimed because only the issued mixture is calibrated.
Previously recorded reservoir and append-all curves are reference controls, with
their source file identified. Verify shadow scores reproduce append-all exactly.

Report overall and specialist-eligible NLL, specialist-use fraction, coverage,
immediate retention, fresh-at-switch acquisition and return learning. Stratification
matters: short scenes offer only a limited number of eight-step query histories.
Keep no-history/fallback samples in the overall scores. All memory sizes/runtime
are measurements, not hard rejection criteria. Count the extra bank explicitly.

A positive result must improve actual held-out prediction and retention, not merely
separate training keys or beat a deliberately cold control. If neither specialist
helps, report it and reassess; do not extend this into another parameter sweep.
