# Event-centric local credit reveals a weak transferable four-frame signal

**Status:** Opt-in M1A experiment; no production model or measured graph changed. All learning is local and uses no backpropagation, privileged scene labels, Pong-specific inputs, or motor reward.

Following the [pre-registered comparison](2026-09-25-event-centric-competition-proposal.md), we kept persistent ON/OFF observation as separate evidence while a locally learned recurrent state used only event-linked motion activity as active sources. We compared the previous independent local-error readout with a new per-site winning-unit readout. The latter learns a shared 5×5 conditional future-event distribution from delayed local error and divides overlapping source contributions rather than summing them. Training used 64 generic 80-frame fractional-motion episodes. Both four-frame and next-event targets had temporally shuffled-credit controls. The native test was four unseen 120 Hz Pong camera streams; no native target affected training or calibration.

The default 0.5 threshold still failed the four-frame gate. On held-out generic fractional motion, the competitive head scored F1 .000 (0 true positives, 17 false positives); on native Pong it scored .015 (4 true positives, 33 false positives, 510 misses). Its quiet native false-alarm rate was 0.024 pixels per frame. The independent head scored .027 generic and .000 native. Next-event results remained weak and do not test quiet time.

The frozen *four-frame* ranking was more informative:

| Head | Generic top-eight signed recall | Native top-eight signed recall |
| --- | ---: | ---: |
| Competitive local credit | .112 | .195 |
| Independent local credit | .133 | .126 |
| Shuffled competitive credit | .054 | .058 |

This is a learned location signal at the intended horizon, not merely a one-frame next-event result. Ranking alone cannot satisfy M1A: the native competitive head still missed 510 of 514 target event pixels at its direct threshold.

As a **diagnostic**, we fit one scalar event-budget gain per head on generic *training* episodes, using the ratio of future event count to forecast probability mass. At inference the frozen gain and current forecast mass selected the highest-scoring sites; the future target never determined the budget. The competitive head reached native four-frame F1 .059 (34 true positives, 606 false positives, 480 misses), versus .010 for independently calibrated shuffled credit; generic held-out F1 was .047 versus .011 shuffled. Native quiet false alarms were 0.85 predicted pixels per quiet frame. This output step is an offline calibration probe, not an accepted local biological readout.

The probability mass only partially distinguishes time. On native streams the competitive head's mean mass was 1.53 when an event was due and 0.75 on quiet target frames; on held-out generic streams it was 1.93 versus 1.54. A separate local event-hazard mechanism is therefore a more focused next question than another threshold sweep.

We also repeated the same state/head/budget procedure with a fresh local motion dictionary learned on generic native-cadence streams. It reduced native competitive F1 to .033 and top-eight recall to .161; generic held-out F1 was .031. This matched comparison does not support another dictionary-data variant as the next priority.

**Decision:** Do not promote this architecture to production. The four-frame spatial signal justifies one opt-in factorization test: learn a local *event-due hazard* from causal recurrent/age state separately from the conditional signed spatial distribution, then multiply them at inference. Train the hazard at the same four-frame delay with quiet and active targets; compare to shuffled credit, fixed generic budget, and persistence on held-out generic scenes and zero-shot native Pong. Require both a meaningful F1 gain and lower quiet alarms without sacrificing the spatial rank. If the hazard cannot distinguish due from quiet, revisit the state and temporal objective rather than tuning gains.

Reproduction: `python scripts/run_event_centric_competition.py`, `python scripts/run_event_budget_probe.py`, and `python scripts/run_event_centric_native_encoder.py`. Their correspondingly named JSON files contain counts, precision, recall, ranks, controls, and elapsed times.
