# Native-cadence generic encoder did not rescue transfer

**Status:** Opt-in M1A architecture experiment. Production code and measured connectome graph are unchanged.

The [observation-horizon study](2026-09-25-observation-horizon-findings.md) found that a learned next-event head ranks future generic events but ranks native Pong events badly. To test whether the older fixed encoder was the main bottleneck, this experiment trained a fresh 24-unit local dictionary on 64 generic 80-frame streams. Each stream contained one to three unlabeled subpixel-moving patterns, varied shape, horizontal and vertical fractional velocity, arbitrary position and background, optional static pattern, and generic boundary reflection. The learner received only signed event-camera input and a four-frame local correlation trace; no scene parameters or Pong state entered learning. The dictionary made 9,087 local updates. The same locally learned recurrent state and delayed event heads were then trained on those streams. Evaluation used unseen generic motifs/seeds, independent and crossing scenes, and four native 120 Hz Pong streams.

| Frozen evaluation | Four-frame learned F1 | Next-event learned F1 | Next-event top-eight signed recall |
| --- | ---: | ---: | ---: |
| Held-out generic fractional motion | .000 | .036 | .305 |
| Generic independent objects | .000 | .007 | — |
| Generic crossing objects | .000 | .015 | — |
| Native Pong | .000 | .000 | .012 |

The four-frame head emitted no thresholded events on any of these domains. The next-event head emitted 128 native false positives with zero true positives. Its ranking on held-out generic targets was better than chance but substantially worse than the previous sparse-trained encoder's held-out dot top-eight recall of .669. Native top-eight recall remained near zero (.012). Simply replacing the dictionary training distribution therefore did not solve the representation/forecast problem. This result does not establish that generic native-cadence training is inherently ineffective: the same recurrent state and independently clamped local-error head were reused, and long streams may dilute updates over increasingly many persistent observation sources.

**Decision:** Stop trace, threshold, and encoder-data variants in this family. The next opt-in experiment should test a genuinely event-centric locally learned hidden state with bounded spatial competition and a normalized future-event distribution, while retaining persistent observation as separate evidence. Its pass criteria must include held-out generic multi-object scenes, a nontrivial fixed four-frame forecast, quiet-frame false alarms, and improved native event-location ranking. Only a successful opt-in result should be considered for the measured optic-lobe interface and a user-approved production architecture revision. No motor learning belongs to this M1A gate.

Reproduction: `python scripts/run_native_cadence_encoder.py`; raw counts, timing, and rankings are in [the result JSON](2026-09-25-native-cadence-encoder-results.json). The complete run took 177 seconds, of which the local dictionary fit took 4.4 seconds.
