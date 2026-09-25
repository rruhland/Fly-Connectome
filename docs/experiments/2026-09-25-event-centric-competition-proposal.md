# Opt-in event-centric competition test

**Scope:** Experimental M1A only. The user has authorized experiments without per-run approval; any in-place production architecture revision still requires a concrete proposal and approval. No backpropagation, privileged training labels, Pong-specific model input, or motor learning.

**Question:** Did persistent observed OFF trails and independently summed synaptic outputs make a locally learned forecast diffuse, even when the event-linked motion state contains useful information?

Train one recurrent hidden transition on generic 120 Hz-like fractional-motion event streams, using the previously learned motion code. Maintain the persistent signed observation as a separate evidence state; do not activate every known pixel as a recurrent source. Compare two delayed local heads on the identical recurrent state:

1. The existing independent local-error head.
2. A competitive head with one winning hidden unit per retinotopic site, a shared 5×5 conditional future-event distribution learned by delayed local error, and divisive normalization of overlapping source contributions. Quiet targets reduce the local probabilities. This is a local probabilistic readout, not a handcoded object or direction bank.

Train both heads for four-frame and next-event targets; add frozen, repeated-event, and temporally shuffled-credit controls. Evaluate held-out generic fractional-motion and multi-object scenes plus zero-shot native Pong. Report signed F1, quiet-frame false alarms for four-frame targets, top-eight signed target recall, precision, and the number of active forecasts. Next-event F1 alone cannot pass because its target excludes quiet frames.

**Pass gate:** A meaningful fixed four-frame gain over repeated-event and shuffled controls on held-out generic scenes, with bounded quiet alarms and improved native ranking. If the competitive head only improves next-event generic scores or stays silent on four-frame targets, stop this architecture family and reassess temporal credit/state capacity before production promotion.
