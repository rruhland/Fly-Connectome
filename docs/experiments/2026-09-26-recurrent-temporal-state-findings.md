# Full-resolution recurrent temporal state: one-run decision

**Decision:** Stop this latent family and reassess M1A with the user. The single registered opt-in run failed the [joint advance gate](../plans/2026-09-26-m1a-recurrent-temporal-state-pivot.md). Do not tune another recurrent decay, radius, update rate, or emission gate against these held-out streams. Production M1A remains unchanged.

The experiment kept the previously learned event-credibility observer frozen. A full-resolution visual population used ON/OFF sensory evidence at three intrinsic decay rates, nearby recurrent state, and per-site activity normalization. Its 3×3 recurrent transition and separate 5×5 visible-event emission weights received only local prediction error times presynaptic activity. There was no backpropagation, Pong information, object label, or motor reward. All 256 mixed-cadence generic training episodes were unlabeled. Frozen-transition, time-shuffled transition-credit, no-recurrence, and state-reset arms used the same data; the last two reused the aligned arm's learned emission weights. Scoring used the same held-out full-frame next-meaningful-event ranking as the prior slow/diverse/equal-pathway comparisons. Immediate next-frame scores were recorded separately.

| Held-out top-32 recall | Aligned | Frozen transition | Shuffled transition | No recurrence | State reset | Required aligned |
|---|---:|---:|---:|---:|---:|---:|
| Clean generic | .288 | .531 | .545 | .445 | .302 | ≥.617 |
| Corrupted generic | .162 | .460 | .427 | .365 | .139 | ≥.572 |
| Changed speed | .420 | .398 | .501 | .423 | .440 | ≥.376 |
| Changed position | .379 | .351 | .467 | .406 | .390 | — |
| Changed shape | .395 | .409 | .510 | .486 | .410 | — |
| Two separated entities | .264 | .304 | .383 | .334 | .314 | ≥.307 |
| Crossing entities | .344 | .333 | .401 | .453 | .406 | — |
| Pong, native cadence | .200 | .360 | .330 | .328 | .283 | ≥.312 |
| Pong, four-step cadence | .473 | .571 | .588 | .541 | .519 | — |

Native Pong top-8 was .079 aligned, barely above the .072 threshold, but frozen transition reached .219. The aligned model did not beat shuffled credit on any required transfer split. Resetting the state **improved** changed-speed top-32 from .420 to .440. It also improved native Pong from .200 to .283, so the learned autonomous state is not causally useful under these conditions. The immediate next-frame metric told the same broad story: clean generic .161 aligned versus .524 frozen; native Pong .136 versus .372. A good changed-speed score alone would have hidden the transfer failure.

The aligned state did persist through quiet camera frames: mean summed activity was 10.46 on clean generic quiet frames, versus 2.53 for frozen transition and zero with state reset. Yet its clean next-event ranking was much worse. Quiet-frame emitted forecast mass was 2.19 aligned versus 2.97 frozen, so the main failure is **not** a simple increase in quiet emission; the autonomous state appears to carry activity that harms localization. This does not prove local recurrence is intrinsically unsuitable. It rejects this particular polarity/decay population, local sensory-target transition credit, and separate one-frame emission objective as a promising next production architecture.

The run took 161.4 seconds to train the frozen observer and three 20,480-frame recurrent arms, then 12.9 seconds to evaluate all controls and transfers (174.3 seconds total). That is about 127 training camera frames per second for the three-arm experiment, inclusive of observer preparation. The training teacher for the frozen observer still assumes paired rendered intensity as described in the plan. No second variant or held-out tuning was run. Reproduce with `.venv/Scripts/python scripts/run_recurrent_temporal_state.py`; [raw scores and per-case counts](2026-09-26-recurrent-temporal-state-results.json) include the immediate, quiet, and reset measurements.
