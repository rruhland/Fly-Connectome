# Predictive memory versus temporal context: bounded comparison

Experiment only; no production defaults/checkpoints change. Reuse the five seeds,
A/shared-B/changed-B streams, independent probes, 2,000 samples per segment and
3.5 NLL target from the preceding continual audit. No threshold or bank-size sweep.

Arms start from identical bundled priors: production 2,048-example reservoir;
that same reservoir plus a 512-example FIFO of recent observed endpoints; append
all new observations without replacement (diagnostic, not a compute-budget veto).
Recent+stable retrieval pools both banks using the same nearest-32 distance rule;
recent examples can occur in both banks, an explicit recency weighting rather
than independent evidence. Stable-bank random admission is unchanged. Calibration
always uses the mixture issued before endpoint observation. Frozen post-A probe
scores are the retention reference, and fresh-at-B controls use each arm's prior.

Measure B acquisition, A immediately after B, and A after return training. Record
bank sizes and runtime. A better memory must preserve old performance without
merely refusing B learning. Append-all distinguishes replacement from retrieval
interference: if it still deteriorates, deletion is not necessary for deterioration.
It does not isolate every difference between mechanisms or prove aliasing.

Separately group observed histories by raw four-step, normalized four-step and
raw eight-step keys (rounded to 1e-4). Use actual same-ID future observed positions,
never simulator targets. Within each exact-key group, compare mean future offsets
from A versus changed B. Report overlap support and cross-domain mean gaps >2
pixels for raw histories. Normalized histories use the production scale/basis for
outcomes too; their gaps/spreads and threshold 2 are in normalized displacement
units (equivalent pixels at reference speed 1 pixel/sample), not actual pixel
errors. Compare raw4 versus raw8 in common pixel units. Grouping
uses domain labels for diagnosis only, never predictive input. Eight-step keys
are evaluated on the same eligible origin samples as four-step keys. Lack of
overlap is not proof that eight steps solve prediction; this is an identifiability
screen, not a fitted long-history learner. Also report within-domain outcome spread.

Preserve mixed/negative results and stop after this comparison. Recommend a concrete
next step based on whether replacement, retrieval mixing, or temporal aliasing is
supported. No biological changes, backpropagation, game semantics or M1B actions.
